"""Visit-level constrained-beam evaluation for EHR SimpleSFT."""

import hashlib
import json
import os
import random
import statistics
import tempfile
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ehr_data_Qwen3 import (
    EHRVisitEvalDataset,
    VISIT_END_TOKEN,
    VISIT_START_TOKEN,
)


EVALUATION_VERSION = "ehr-simple-sft-visit-eval-v1"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_ks(ks):
    if isinstance(ks, str):
        value = ks.strip().strip("()[]")
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    elif isinstance(ks, int):
        parsed = [ks]
    else:
        parsed = [int(value) for value in ks]
    parsed = sorted(set(parsed))
    if not parsed or parsed[0] <= 0:
        raise ValueError(f"ks must contain positive integers, received {ks!r}")
    return tuple(parsed)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_vocab_hash(tokenizer):
    payload = json.dumps(
        tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(path)}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def atomic_write_jsonl(path, records):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(path)}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def update_split_json(path, split, value):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"existing output is not a JSON object: {path}")
    else:
        payload = {}
    payload[split] = value
    atomic_write_json(path, payload)


def load_sid_index(index_path):
    with open(index_path, "r", encoding="utf-8") as handle:
        sid_index = json.load(handle)
    if not isinstance(sid_index, dict) or not sid_index:
        raise ValueError(f"SID index must be a non-empty JSON object: {index_path}")

    disease_to_sid = {}
    for disease_id, tokens in sid_index.items():
        if not isinstance(disease_id, str):
            raise ValueError("all disease IDs in the SID index must be strings")
        if not isinstance(tokens, list) or not tokens or not all(
            isinstance(token, str) and token for token in tokens
        ):
            raise ValueError(f"invalid SID token path for {disease_id!r}")
        disease_to_sid[disease_id] = "".join(tokens)

    sid_to_disease = {}
    sid_to_all_diseases = defaultdict(list)
    for disease_id in sorted(disease_to_sid):
        sid = disease_to_sid[disease_id]
        sid_to_all_diseases[sid].append(disease_id)
        sid_to_disease[sid] = disease_id

    collision_groups = {
        sid: disease_ids
        for sid, disease_ids in sid_to_all_diseases.items()
        if len(disease_ids) > 1
    }
    overwritten_disease_ids = sorted(
        disease_id
        for disease_ids in collision_groups.values()
        for disease_id in disease_ids[:-1]
    )
    return {
        "sid_index": sid_index,
        "disease_to_sid": disease_to_sid,
        "sid_to_disease": sid_to_disease,
        "collision_groups": collision_groups,
        "overwritten_disease_ids": overwritten_disease_ids,
    }


def atomic_token_id(tokenizer, token):
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        raise ValueError(f"tokenizer preflight failed: {token!r} is missing from the vocabulary")
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"tokenizer preflight failed: {token!r} encodes as {len(token_ids)} tokens"
        )
    if int(token_ids[0]) != int(vocab[token]):
        raise ValueError(
            f"tokenizer preflight failed: {token!r} does not encode to its vocabulary ID"
        )
    return int(token_ids[0])


def build_prefix_constraint(sid_index, tokenizer, eos_token_id=None):
    """Build dynamic SID token paths and prefix -> allowed-next-token table."""
    eos_token_id = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    all_sid_tokens = sorted(
        {token for tokens in sid_index.values() for token in tokens}
    )
    token_to_id = {
        token: atomic_token_id(tokenizer, token) for token in all_sid_tokens
    }
    visit_token_ids = {
        token: atomic_token_id(tokenizer, token)
        for token in (VISIT_START_TOKEN, VISIT_END_TOKEN)
    }
    all_special_token_ids = list(token_to_id.values()) + list(visit_token_ids.values())
    if len(all_special_token_ids) != len(set(all_special_token_ids)):
        raise ValueError("tokenizer preflight failed: SID/visit tokens do not have unique IDs")

    prefix_to_allowed = defaultdict(set)
    path_to_sid = {}
    sid_depths = set()
    for disease_id in sorted(sid_index):
        sid_tokens = sid_index[disease_id]
        path = tuple(token_to_id[token] for token in sid_tokens)
        sid = "".join(sid_tokens)
        previous_sid = path_to_sid.get(path)
        if previous_sid is not None and previous_sid != sid:
            raise ValueError(
                f"distinct SID strings map to the same token-ID path: {previous_sid}, {sid}"
            )
        path_to_sid[path] = sid
        sid_depths.add(len(path))
        for position, next_token_id in enumerate(path):
            prefix_to_allowed[path[:position]].add(next_token_id)
        prefix_to_allowed[path].add(int(eos_token_id))

    return {
        "prefix_to_allowed": {
            prefix: sorted(token_ids)
            for prefix, token_ids in prefix_to_allowed.items()
        },
        "path_to_sid": path_to_sid,
        "sid_token_ids": token_to_id,
        "visit_token_ids": visit_token_ids,
        "sid_depths": sorted(sid_depths),
        "max_sid_depth": max(sid_depths),
        "eos_token_id": int(eos_token_id),
    }


def make_prefix_allowed_tokens_fn(prefix_to_allowed, prompt_width, eos_token_id):
    """Constrain only tokens generated after this left-padded batch's prompt width."""

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        del batch_id
        generated = tuple(int(value) for value in input_ids[prompt_width:].tolist())
        return prefix_to_allowed.get(generated, [eos_token_id])

    return prefix_allowed_tokens_fn


def _strip_at_eos(token_ids, eos_token_id):
    result = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == eos_token_id:
            break
        result.append(token_id)
    return tuple(result)


def parse_ranked_beams(
    completion_ids,
    sequence_scores,
    path_to_sid,
    sid_to_disease,
    eos_token_id,
):
    """Sort beams, validate exact token paths, and retain each best unique SID."""
    if len(completion_ids) != len(sequence_scores):
        raise ValueError("beam sequences and sequence scores are not aligned")

    ranked = sorted(
        zip(sequence_scores, completion_ids),
        key=lambda pair: float(pair[0]),
        reverse=True,
    )
    invalid_count = 0
    duplicate_count = 0
    seen_paths = set()
    seen_sids = set()
    predictions = []
    invalid_paths = []
    for score, token_ids in ranked:
        path = _strip_at_eos(token_ids, eos_token_id)
        if path in seen_paths:
            duplicate_count += 1
        else:
            seen_paths.add(path)
        sid = path_to_sid.get(path)
        if sid is None:
            invalid_count += 1
            invalid_paths.append(list(path))
            continue
        if sid in seen_sids:
            continue
        seen_sids.add(sid)
        predictions.append(
            {
                "rank": len(predictions) + 1,
                "sid": sid,
                "disease_id": sid_to_disease[sid],
                "score": float(score),
            }
        )

    return {
        "predictions": predictions,
        "invalid_count": invalid_count,
        "duplicate_count": duplicate_count,
        "invalid_paths": invalid_paths,
    }


def recall_at_k(ground_truth, rank_lists, ks=(10, 20, 30, 40)):
    if len(ground_truth) != len(rank_lists):
        raise ValueError("ground truth and rank lists are not aligned")
    metrics = {}
    for k in parse_ks(ks):
        per_visit = []
        for true_ids, predicted_ids in zip(ground_truth, rank_lists):
            true_set = set(true_ids)
            if not true_set:
                raise ValueError("Recall@K is undefined for an empty ground-truth set")
            pred_set = set(predicted_ids[:k])
            per_visit.append(len(true_set & pred_set) / len(true_set))
        metrics[f"recall@{k}"] = float(np.mean(per_visit))
    return metrics


def weighted_f1_at_true_cardinality(ground_truth, rank_lists, disease_ids):
    if len(ground_truth) != len(rank_lists):
        raise ValueError("ground truth and rank lists are not aligned")
    disease_ids = sorted(disease_ids)
    disease_to_col = {
        disease_id: index for index, disease_id in enumerate(disease_ids)
    }
    y_true = np.zeros((len(ground_truth), len(disease_ids)), dtype=np.int8)
    y_pred = np.zeros_like(y_true)

    for row, (true_ids, predicted_ids) in enumerate(zip(ground_truth, rank_lists)):
        unique_true = list(dict.fromkeys(true_ids))
        unique_predictions = list(dict.fromkeys(predicted_ids))
        unknown = [value for value in unique_true if value not in disease_to_col]
        if unknown:
            raise ValueError(f"ground truth contains diseases missing from the index: {unknown}")
        for disease_id in unique_true:
            y_true[row, disease_to_col[disease_id]] = 1
        for disease_id in unique_predictions[: len(unique_true)]:
            if disease_id not in disease_to_col:
                raise ValueError(f"prediction missing from the disease index: {disease_id}")
            y_pred[row, disease_to_col[disease_id]] = 1

    return float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )


def collision_impact(ground_truth, overwritten_disease_ids):
    overwritten = set(overwritten_disease_ids)
    unrecoverable_occurrences = 0
    affected_visits = 0
    affected_sample_ids = []
    for sample_id, disease_ids in ground_truth:
        count = sum(disease_id in overwritten for disease_id in disease_ids)
        unrecoverable_occurrences += count
        if count:
            affected_visits += 1
            affected_sample_ids.append(sample_id)
    return {
        "unrecoverable_gt_occurrences": unrecoverable_occurrences,
        "affected_visit_count": affected_visits,
        "affected_sample_ids": affected_sample_ids,
    }


def validate_ground_truth(samples, disease_to_sid):
    for sample in samples:
        disease_ids = sample["ground_truth_disease_ids"]
        sids = sample["ground_truth_sids"]
        if len(disease_ids) != len(set(disease_ids)):
            raise ValueError(
                f"sample {sample['sample_id']}: duplicate ground-truth disease IDs"
            )
        for disease_id, sid in zip(disease_ids, sids):
            expected_sid = disease_to_sid.get(disease_id)
            if expected_sid is None:
                raise ValueError(
                    f"sample {sample['sample_id']}: {disease_id} is missing from the SID index"
                )
            if sid != expected_sid:
                raise ValueError(
                    f"sample {sample['sample_id']}: SID mismatch for {disease_id}: "
                    f"visit={sid}, index={expected_sid}"
                )


def _left_pad_batch(samples, pad_token_id, device):
    prompt_width = max(len(sample["input_ids"]) for sample in samples)
    input_ids = []
    attention_mask = []
    for sample in samples:
        padding = prompt_width - len(sample["input_ids"])
        input_ids.append([pad_token_id] * padding + list(sample["input_ids"]))
        attention_mask.append([0] * padding + list(sample["attention_mask"]))
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        prompt_width,
    )


def generate_ranklists(
    model,
    tokenizer,
    samples,
    constraint,
    sid_to_disease,
    batch_size,
    num_beams,
    length_penalty,
    required_candidates,
    stop_on_failure=False,
):
    device = next(model.parameters()).device
    ranklists = []
    diagnostics = {
        "invalid_candidate_count": 0,
        "duplicate_beam_count": 0,
        "candidate_shortage_visit_count": 0,
    }
    failures = []

    for start in tqdm(range(0, len(samples), batch_size), desc="constrained beam"):
        batch = samples[start : start + batch_size]
        input_ids, attention_mask, prompt_width = _left_pad_batch(
            batch, tokenizer.pad_token_id, device
        )
        prefix_fn = make_prefix_allowed_tokens_fn(
            constraint["prefix_to_allowed"],
            prompt_width,
            constraint["eos_token_id"],
        )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
                length_penalty=length_penalty,
                max_new_tokens=constraint["max_sid_depth"] + 1,
                prefix_allowed_tokens_fn=prefix_fn,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=constraint["eos_token_id"],
            )

        completion_ids = generated.sequences[:, prompt_width:].detach().cpu().tolist()
        if generated.sequences_scores is None:
            raise RuntimeError("transformers did not return beam sequence scores")
        sequence_scores = generated.sequences_scores.detach().float().cpu().tolist()

        for batch_index, sample in enumerate(batch):
            begin = batch_index * num_beams
            end = begin + num_beams
            parsed = parse_ranked_beams(
                completion_ids[begin:end],
                sequence_scores[begin:end],
                constraint["path_to_sid"],
                sid_to_disease,
                constraint["eos_token_id"],
            )
            candidate_count = len(parsed["predictions"])
            diagnostics["invalid_candidate_count"] += parsed["invalid_count"]
            diagnostics["duplicate_beam_count"] += parsed["duplicate_count"]
            if candidate_count < required_candidates:
                diagnostics["candidate_shortage_visit_count"] += 1
            if parsed["invalid_count"] or candidate_count < required_candidates:
                failures.append(
                    {
                        "sample_id": sample["sample_id"],
                        "invalid_candidate_count": parsed["invalid_count"],
                        "unique_valid_candidate_count": candidate_count,
                        "required_candidate_count": required_candidates,
                        "invalid_token_paths": parsed["invalid_paths"][:5],
                    }
                )
            ranklists.append(
                {
                    "sample_id": sample["sample_id"],
                    "ground_truth_disease_ids": list(
                        sample["ground_truth_disease_ids"]
                    ),
                    "ground_truth_sids": list(sample["ground_truth_sids"]),
                    "predictions": parsed["predictions"],
                }
            )
            if failures and stop_on_failure:
                return ranklists, diagnostics, failures
    return ranklists, diagnostics, failures


def _preflight_payload(
    status,
    checked_visit_count,
    required_candidate_count,
    diagnostics,
    failures,
    stage="online_preflight",
):
    return {
        "status": status,
        "stage": stage,
        "CC": diagnostics["invalid_candidate_count"],
        "checked_visit_count": checked_visit_count,
        "required_unique_candidate_count": required_candidate_count,
        "checks": {
            "invalid_candidate_count_is_zero": diagnostics[
                "invalid_candidate_count"
            ]
            == 0,
            "every_visit_has_required_unique_candidates": diagnostics[
                "candidate_shortage_visit_count"
            ]
            == 0,
        },
        "diagnostics": diagnostics,
        "failures": failures,
    }


def main(
    base_model="./output_dir/mimic3_icd_name_path_0.1_simple_sft_Qwen3-1.7B/final_checkpoint",
    data_dir="./data/EHR/mimic3_icd_name_path_0.1",
    dataset_prefix="mimic3_icd",
    split="test",
    visit_file=None,
    result_dir="./results/mimic3_icd_name_path_0.1_simple_sft",
    batch_size=4,
    num_beams=40,
    cutoff_len=1024,
    length_penalty=0.0,
    ks=(10, 20, 30, 40),
    sample=-1,
    preflight_samples=100,
    seed=42,
):
    split = str(split).strip().lower()
    if split not in {"valid", "test"}:
        raise ValueError("split must be 'valid' or 'test'")
    ks = parse_ks(ks)
    batch_size = int(batch_size)
    num_beams = int(num_beams)
    cutoff_len = int(cutoff_len)
    sample = int(sample)
    preflight_samples = int(preflight_samples)
    seed = int(seed)
    if batch_size <= 0 or num_beams <= 0 or cutoff_len <= 0:
        raise ValueError("batch_size, num_beams, and cutoff_len must be positive")
    if preflight_samples <= 0:
        raise ValueError("preflight_samples must be positive")
    if num_beams < max(ks):
        raise ValueError(
            f"num_beams={num_beams} cannot supply the required Top-{max(ks)} rank-list"
        )

    visit_file = visit_file or os.path.join(data_dir, "visit_level", f"{split}.jsonl")
    index_path = os.path.join(data_dir, "index", f"{dataset_prefix}.index.json")
    disease_manifest_path = os.path.join(
        data_dir, "manifest", "disease_manifest.json"
    )
    required_paths = {
        "base_model": base_model,
        "visit_file": visit_file,
        "index_path": index_path,
        "disease_manifest_path": disease_manifest_path,
    }
    for name, path in required_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")

    os.makedirs(result_dir, exist_ok=True)
    preflight_path = os.path.join(result_dir, "evaluation_preflight.json")
    set_seed(seed)

    sid_data = load_sid_index(index_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    constraint = build_prefix_constraint(sid_data["sid_index"], tokenizer)
    if len(constraint["path_to_sid"]) < max(ks):
        raise ValueError("SID index contains fewer unique paths than the largest K")

    dataset = EHRVisitEvalDataset(
        visit_file=visit_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
    )
    samples = dataset.get_inputs_list()
    if not samples:
        raise ValueError(f"no visit samples loaded from {visit_file}")
    validate_ground_truth(samples, sid_data["disease_to_sid"])

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    preflight_count = min(preflight_samples, len(samples))
    required_candidates = max(ks)
    first_ranklists, first_diagnostics, first_failures = generate_ranklists(
        model=model,
        tokenizer=tokenizer,
        samples=samples[:preflight_count],
        constraint=constraint,
        sid_to_disease=sid_data["sid_to_disease"],
        batch_size=batch_size,
        num_beams=num_beams,
        length_penalty=float(length_penalty),
        required_candidates=required_candidates,
    )
    preflight_status = "passed" if not first_failures else "failed"
    update_split_json(
        preflight_path,
        split,
        _preflight_payload(
            preflight_status,
            preflight_count,
            required_candidates,
            first_diagnostics,
            first_failures,
        ),
    )
    if first_failures:
        raise RuntimeError(
            f"online preflight failed for {len(first_failures)} visit(s); "
            f"diagnostics written to {preflight_path}"
        )

    later_ranklists = []
    later_diagnostics = {
        "invalid_candidate_count": 0,
        "duplicate_beam_count": 0,
        "candidate_shortage_visit_count": 0,
    }
    if preflight_count < len(samples):
        later_ranklists, later_diagnostics, later_failures = generate_ranklists(
            model=model,
            tokenizer=tokenizer,
            samples=samples[preflight_count:],
            constraint=constraint,
            sid_to_disease=sid_data["sid_to_disease"],
            batch_size=batch_size,
            num_beams=num_beams,
            length_penalty=float(length_penalty),
            required_candidates=required_candidates,
            stop_on_failure=True,
        )
        if later_failures:
            combined_diagnostics = {
                key: first_diagnostics[key] + later_diagnostics[key]
                for key in first_diagnostics
            }
            update_split_json(
                preflight_path,
                split,
                _preflight_payload(
                    "failed",
                    preflight_count + len(later_ranklists),
                    required_candidates,
                    combined_diagnostics,
                    later_failures,
                    stage="post_preflight_strict_validation",
                ),
            )
            raise RuntimeError(
                f"strict beam validation failed for {len(later_failures)} visit(s); "
                f"diagnostics written to {preflight_path}"
            )

    ranklists = first_ranklists + later_ranklists
    diagnostics = {
        key: first_diagnostics[key] + later_diagnostics[key]
        for key in first_diagnostics
    }
    ground_truth = [record["ground_truth_disease_ids"] for record in ranklists]
    predicted = [
        [prediction["disease_id"] for prediction in record["predictions"]]
        for record in ranklists
    ]
    collision_stats = collision_impact(
        [
            (record["sample_id"], record["ground_truth_disease_ids"])
            for record in ranklists
        ],
        sid_data["overwritten_disease_ids"],
    )
    cardinalities = [len(set(values)) for values in ground_truth]
    metrics = {
        **recall_at_k(ground_truth, predicted, ks),
        "oracle_cardinality_weighted_f1": weighted_f1_at_true_cardinality(
            ground_truth, predicted, sorted(sid_data["disease_to_sid"])
        ),
        "visit_count": len(ranklists),
        "CC": diagnostics["invalid_candidate_count"],
        "diagnostics": diagnostics,
        "gt_cardinality": {
            "min": min(cardinalities),
            "max": max(cardinalities),
            "mean": float(np.mean(cardinalities)),
            "median": float(statistics.median(cardinalities)),
        },
        "collision_impact": collision_stats,
    }

    manifest = {
        "evaluation_version": EVALUATION_VERSION,
        "evaluation_script_sha256": sha256_file(os.path.abspath(__file__)),
        "checkpoint": os.path.abspath(base_model),
        "tokenizer_vocab_sha256": tokenizer_vocab_hash(tokenizer),
        "files": {
            "index": {
                "path": os.path.abspath(index_path),
                "sha256": sha256_file(index_path),
            },
            "disease_manifest": {
                "path": os.path.abspath(disease_manifest_path),
                "sha256": sha256_file(disease_manifest_path),
            },
            "visit_file": {
                "path": os.path.abspath(visit_file),
                "sha256": sha256_file(visit_file),
            },
        },
        "sample_limit": sample,
        "seed": seed,
        "no_thinking_protocol": {
            "enabled": True,
            "apply_chat_template_enable_thinking": False,
            "generation_start": "left-padded batch prompt width",
        },
        "constraint": {
            "type": "dynamic prefix-to-allowed-token table from index.json",
            "leaf_rule": "EOS only",
            "sid_depths": constraint["sid_depths"],
            "max_new_tokens": constraint["max_sid_depth"] + 1,
            "sid_token_count": len(constraint["sid_token_ids"]),
            "visit_token_ids": constraint["visit_token_ids"],
        },
        "beam": {
            "batch_size": batch_size,
            "num_beams": num_beams,
            "num_return_sequences": num_beams,
            "do_sample": False,
            "early_stopping": True,
            "length_penalty": float(length_penalty),
        },
        "ks": list(ks),
        "weighted_f1_cutoff": "per-visit Top-|GT| (oracle cardinality)",
        "collision_policy": {
            "sid_to_disease": "sorted disease ID iteration with later overwrite",
            "disease_count": len(sid_data["disease_to_sid"]),
            "unique_sid_count": len(sid_data["sid_to_disease"]),
            "collision_disease_count": len(
                sid_data["overwritten_disease_ids"]
            ),
            "overwritten_disease_ids": sid_data["overwritten_disease_ids"],
            "collision_groups": sid_data["collision_groups"],
            **collision_stats,
        },
    }

    ranklist_path = os.path.join(result_dir, f"{split}_ranklist.jsonl")
    atomic_write_jsonl(ranklist_path, ranklists)
    update_split_json(os.path.join(result_dir, "metrics.json"), split, metrics)
    update_split_json(
        os.path.join(result_dir, "evaluation_manifest.json"), split, manifest
    )
    print(json.dumps({split: metrics}, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"rank-list saved to {ranklist_path}")
    return metrics


if __name__ == "__main__":
    try:
        import fire
    except ImportError as exc:
        raise RuntimeError(
            "CLI execution requires fire; pure evaluator functions can be imported without it"
        ) from exc
    fire.Fire(main)
