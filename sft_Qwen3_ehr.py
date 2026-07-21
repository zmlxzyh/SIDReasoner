import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from peft import TrainableTokensConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig, TrainerCallback
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np 
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import hashlib
import tempfile
import wandb
from contextlib import contextmanager
"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from ehr_data_Qwen3 import (
    EHRSFTData,
    EHRSidSFTDataset,
    EHRSidItemFeatDataset,
    EHRFusionSeqRecDataset,
    EHRTitleHistory2SidSFTDataset,
    SidTextInterleaveDataset,
    SidTextInterleaveDataset_v2,
    SidTextInterleaveSequenceDataset,
    GeneralSFTReasonDataset,
    VISIT_START_TOKEN,
    VISIT_END_TOKEN,
)
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset


class MultiEvalTrainer(transformers.Trainer):
    """
    Runs the default evaluation and then iterates through any extra eval sets so every epoch
    produces loss numbers for the auxiliary datasets as well.
    """

    def __init__(self, *args, extra_eval_sets: Optional[Dict[str, HFDataset]] = None, **kwargs):
        self.extra_eval_sets = extra_eval_sets or {}
        super().__init__(*args, **kwargs)

    @contextmanager
    def _disable_callback(self, callback_cls):
        callbacks = self.callback_handler.callbacks
        removed = [cb for cb in callbacks if isinstance(cb, callback_cls)]
        if not removed:
            yield
            return
        self.callback_handler.callbacks = [cb for cb in callbacks if not isinstance(cb, callback_cls)]
        try:
            yield
        finally:
            self.callback_handler.callbacks = callbacks

    def evaluate(
        self,
        eval_dataset: Optional[HFDataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if not self.extra_eval_sets:
            return metrics

        for name, dataset in self.extra_eval_sets.items():
            if dataset is None:
                continue
            with self._disable_callback(EarlyStoppingCallback):
                extra_metrics = super().evaluate(
                    eval_dataset=dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{name}",
                )
            self.log(extra_metrics)
            metrics.update(extra_metrics)
        return metrics


class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json", extra_tokens=None):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None
        self.extra_tokens = list(extra_tokens or [])
        
    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
    
    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
            
        if self.indices is None:
            self._load_data()
        
        self.new_tokens = set(self.extra_tokens)
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        
        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _decode_tokens(tokens, tokenizer_ref):
    if not isinstance(tokens, (list, tuple)):
        return ""
    valid_ids = [tid for tid in tokens if isinstance(tid, int) and tid >= 0]
    if not valid_ids:
        return ""
    return tokenizer_ref.decode(valid_ids, skip_special_tokens=False)

def _preview_dataset(dataset, name, tokenizer_ref, max_samples=3):
    print(f"[Preview] {name}: displaying up to {max_samples} samples")
    preview_count = min(max_samples, len(dataset))
    for idx in range(preview_count):
        sample = dataset[idx]
        input_text = ""
        # label_text = ""
        if isinstance(sample, dict):
            if "input_ids" in sample:
                input_text = _decode_tokens(sample["input_ids"], tokenizer_ref)
            if "labels" in sample:
                # Filter label padding tokens (e.g., -100) before decoding for readability
                label_ids = [tid for tid in sample["labels"] if isinstance(tid, int) and tid >= 0]
                label_text = _decode_tokens(label_ids, tokenizer_ref)
        print(f"Sample {idx + 1}:")
        if input_text:
            print(f"  Input : {input_text}")
        if label_text:
            print(f"  Label : {label_text}")
            print(f"  Length: {len(label_ids)} tokens")
        print()


def _tokenizer_vocab_hash(tokenizer_ref):
    payload = json.dumps(
        tokenizer_ref.get_vocab(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_token_ids(tokenizer_ref, tokens):
    token_ids = {}
    for token in tokens:
        ids = tokenizer_ref.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"preflight failed: {token!r} is encoded as {len(ids)} tokens")
        token_ids[token] = ids[0]
    return token_ids


def _run_tokenizer_preflight(
    tokenizer_ref,
    special_tokens,
    sid_prediction_dataset,
    output_dir,
    base_model,
    rank,
    write_artifact,
):
    """Fail before training when token atomicity or assistant-only labels drift."""
    if not special_tokens:
        raise ValueError("preflight failed: no SID/visit tokens were collected")
    if len(sid_prediction_dataset) == 0:
        raise ValueError("preflight failed: the T5 training dataset is empty")

    token_ids_before = _atomic_token_ids(tokenizer_ref, special_tokens)
    vocab_hash_before = _tokenizer_vocab_hash(tokenizer_ref)

    sample = sid_prediction_dataset[0]
    labels = list(sample["labels"])
    supervised_positions = [
        position for position, token_id in enumerate(labels) if token_id != -100
    ]
    if not supervised_positions:
        raise ValueError("preflight failed: T5 sample has no supervised tokens")
    first_supervised = supervised_positions[0]
    last_supervised = supervised_positions[-1]
    expected_positions = list(range(first_supervised, last_supervised + 1))
    if supervised_positions != expected_positions:
        raise ValueError("preflight failed: T5 supervised labels are not contiguous")
    if any(token_id != -100 for token_id in labels[:first_supervised]):
        raise ValueError("preflight failed: system/user prompt is not fully masked")
    if last_supervised >= len(labels) - 1:
        raise ValueError(
            "preflight failed: no masked assistant ending suffix was found after the answer"
        )
    if any(token_id != -100 for token_id in labels[last_supervised + 1 :]):
        raise ValueError("preflight failed: assistant ending suffix is supervised")

    target_sid = str(sid_prediction_dataset.data.iloc[0]["item_sid"]).strip()
    expected_response_ids = tokenizer_ref.encode(
        target_sid, add_special_tokens=False
    )
    supervised_ids = [labels[position] for position in supervised_positions]
    if supervised_ids != expected_response_ids:
        raise ValueError(
            "preflight failed: T5 labels do not exactly match the target SID; "
            f"target={target_sid!r}"
        )

    # Each torchrun rank uses a private temporary directory, so save/reload
    # validation has no shared-filesystem race.
    with tempfile.TemporaryDirectory(prefix=f"ehr-tokenizer-preflight-rank-{rank}-") as temp_dir:
        tokenizer_ref.save_pretrained(temp_dir)
        reloaded = AutoTokenizer.from_pretrained(
            temp_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        token_ids_after = _atomic_token_ids(reloaded, special_tokens)
        vocab_hash_after = _tokenizer_vocab_hash(reloaded)

    if token_ids_before != token_ids_after:
        raise ValueError("preflight failed: special-token IDs changed after tokenizer reload")
    if vocab_hash_before != vocab_hash_after:
        raise ValueError("preflight failed: tokenizer vocabulary changed after reload")

    artifact = {
        "base_model": base_model,
        "sid_token_count": sum(
            token not in {VISIT_START_TOKEN, VISIT_END_TOKEN}
            for token in special_tokens
        ),
        "visit_tokens": [VISIT_START_TOKEN, VISIT_END_TOKEN],
        "special_token_count": len(special_tokens),
        "tokenizer_vocab_size": len(tokenizer_ref),
        "tokenizer_vocab_sha256": vocab_hash_before,
        "checks": {
            "all_sid_and_visit_tokens_atomic": True,
            "token_ids_stable_after_save_reload": True,
            "vocabulary_stable_after_save_reload": True,
            "system_and_user_prompt_masked": True,
            "supervised_labels_equal_target_sid": True,
            "assistant_ending_suffix_masked": True,
        },
        "sample_target_sid": target_sid,
    }
    if write_artifact:
        manifest_dir = os.path.join(output_dir, "manifest")
        os.makedirs(manifest_dir, exist_ok=True)
        artifact_path = os.path.join(manifest_dir, "tokenizer_preflight.json")
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"Tokenizer preflight passed; artifact saved to {artifact_path}")
    return artifact



def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(
    # model/data params
    base_model: str = "Qwen/Qwen3-1.7B",
    data_dir: str = "./data/EHR/mimic3_icd_name_path_0.1",
    dataset_prefix: str = "mimic3_icd",
    train_file: str = None,
    eval_file: str = None,
    output_dir: str = "./output_dir/mimic3_icd_name_path_0.1_simple_sft_Qwen3-1.7B",
    sample: int = -1,
    seed: int = 42,
    category: str = "EHR",
    
    # training hyperparams
    batch_size: int = 1024,
    micro_batch_size: int = 1,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    # llm hyperparams
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "SIDReasoner_EHR",
    wandb_run_name: str = "mimic3_icd_name_path_0.1_simple_sft_Qwen3-1.7B",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    train_from_scratch: bool = False,
    sid_index_path: str = None,
    item_meta_path: str = None,
    llm_generated_data_path: str = None,
    llm_generated_sequence_path: str = None,
    general_reasoning_path: str = None,
    mask_assistant: bool = True,   # Whether only the target response is used for loss calculation
    train_new_token_embeddings_only: bool = False,
):
    set_seed(seed)
    os.environ['WANDB_PROJECT'] = wandb_project

    train_file = train_file or os.path.join(data_dir, "code_level", "train.csv")
    eval_file = eval_file or os.path.join(data_dir, "code_level", "valid.csv")
    sid_index_path = sid_index_path or os.path.join(
        data_dir, "index", f"{dataset_prefix}.index.json"
    )
    item_meta_path = item_meta_path or os.path.join(
        data_dir, "index", f"{dataset_prefix}.item.json"
    )
    required_paths = {
        "train_file": train_file,
        "eval_file": eval_file,
        "sid_index_path": sid_index_path,
        "item_meta_path": item_meta_path,
    }
    for path_name, path in required_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path_name} does not exist: {path}")
    if not mask_assistant:
        raise ValueError("EHR SimpleSFT requires --mask_assistant True")

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    main_rank = rank == 0 and local_rank == 0
    print(
        f"EHR dataset={data_dir}, prefix={dataset_prefix}, train={train_file}, "
        f"valid={eval_file}"
    )

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Tokenizer length: {len(tokenizer)}")
    
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split('.')[0],
            extra_tokens=[VISIT_START_TOKEN, VISIT_END_TOKEN],
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            existing_vocab = set(tokenizer.get_vocab().keys())
            tokens_to_add = [tok for tok in new_tokens if tok not in existing_vocab]
            if tokens_to_add:
                print(f"Adding {len(tokens_to_add)} new tokens to tokenizer")
                tokenizer.add_tokens(tokens_to_add)
                model.resize_token_embeddings(len(tokenizer))
                num_new_tokens = len(tokens_to_add)
            else:
                print("All candidate tokens already exist in the tokenizer; skipping addition.")
                num_new_tokens = 0
        else:
            num_new_tokens = 0
    else:
        new_tokens = []
        num_new_tokens = 0

    if train_new_token_embeddings_only:
        if num_new_tokens > 0:
            vocab_size = len(tokenizer)
            new_token_indices = list(range(vocab_size - num_new_tokens, vocab_size))
            print(f"Restricting training to new token ids.")
            peft_config = TrainableTokensConfig(
                token_indices=new_token_indices,
                target_modules=["embed_tokens"],
                init_weights=True,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    else:
        print("Full fine-tuning enabled: attention blocks, FFNs, and embeddings remain trainable.")

    if num_new_tokens == 0 and train_new_token_embeddings_only:
        print("No new tokens added; the entire model will remain trainable.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = (trainable_params / total_params) * 100 if total_params > 0 else 0.0
    print(f"Trainable parameters: {trainable_params} / {total_params} ({percent:.4f}%)")
        
    train_datasets = []
    train_dataset_names = []
    train_data1 = EHRSidSFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, mask_assistant=mask_assistant)
    train_datasets.append(train_data1)
    train_dataset_names.append("T5/EHRSidSFTDataset")
    train_data2 = EHRSidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, mask_assistant=mask_assistant)
    train_datasets.append(train_data2)
    train_dataset_names.append("T1+T2/EHRSidItemFeatDataset")
    train_data3 = EHRFusionSeqRecDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, mask_assistant=mask_assistant)
    train_datasets.append(train_data3)
    train_dataset_names.append("T3/EHRFusionSeqRecDataset")
    train_data4 = EHRSFTData(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, mask_assistant=mask_assistant)
    train_datasets.append(train_data4)
    train_dataset_names.append("T4/EHRSFTData")
    train_data5 = EHRTitleHistory2SidSFTDataset(train_file=train_file, item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, mask_assistant=mask_assistant)
    train_datasets.append(train_data5)
    train_dataset_names.append("T6/EHRTitleHistory2SidSFTDataset")

    if llm_generated_data_path is not None:
        train_data7 = SidTextInterleaveDataset_v2(json_file=llm_generated_data_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed)
        train_datasets.append(train_data7)
        train_dataset_names.append("SidTextInterleaveDataset_v2")
    if llm_generated_sequence_path is not None:
        train_data8 = SidTextInterleaveSequenceDataset(csv_file=llm_generated_sequence_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed)
        train_datasets.append(train_data8)
        train_dataset_names.append("SidTextInterleaveSequenceDataset")
    if general_reasoning_path is not None:
        train_data9 = GeneralSFTReasonDataset(train_file=general_reasoning_path, tokenizer=tokenizer, max_len=3072,  sample=60000, seed=seed)
        train_datasets.append(train_data9)
        train_dataset_names.append("GeneralSFTReasonDataset")
    
    train_data = ConcatDataset(train_datasets)

    _run_tokenizer_preflight(
        tokenizer_ref=tokenizer,
        special_tokens=new_tokens,
        sid_prediction_dataset=train_data1,
        output_dir=output_dir,
        base_model=base_model,
        rank=rank,
        write_artifact=main_rank,
    )

    if main_rank:
        for ds, name in zip(train_datasets, train_dataset_names):
            _preview_dataset(ds, name, tokenizer)

    val_data_sid_prediction = EHRSidSFTDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, test=False, mask_assistant=True)
    val_data_title2sid_translation = EHRSidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, task_type='title2sid', test=False, mask_assistant=True)
    val_data_sid2title_translation = EHRSidItemFeatDataset(item_file=item_meta_path, index_file=sid_index_path, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, task_type='sid2title', test=False, mask_assistant=True)
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    sample_frac = 1
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=seed).select(range(int(sample_frac * len(hf_train_dataset))))
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data_sid_prediction] for k in val_data_sid_prediction[0].keys()}).shuffle(seed=seed)
    hf_val_dataset = hf_val_dataset.shuffle(seed=seed)
    # additional eval set for translation performance
    hf_eval_dataset_title2sid_translation = HFDataset.from_dict({k: [v[k] for v in val_data_title2sid_translation] for k in val_data_title2sid_translation[0].keys()}).shuffle(seed=seed)
    hf_eval_dataset_title2sid_translation = hf_eval_dataset_title2sid_translation.shuffle(seed=seed)
    hf_eval_dataset_sid2title_translation = HFDataset.from_dict({k: [v[k] for v in val_data_sid2title_translation] for k in val_data_sid2title_translation[0].keys()}).shuffle(seed=seed)
    hf_eval_dataset_sid2title_translation = hf_eval_dataset_sid2title_translation.shuffle(seed=seed)

    extra_eval_sets = {
        "alignment_reconstruction/title2sid": hf_eval_dataset_title2sid_translation,
        "alignment_reconstruction/sid2title": hf_eval_dataset_sid2title_translation,
    }

    print(hf_train_dataset)
    print(hf_val_dataset)
    print(hf_eval_dataset_title2sid_translation)
    print(hf_eval_dataset_sid2title_translation)

    # eval_step = 0.05
    trainer = MultiEvalTrainer(
        # deepspeed=deepspeed,
        model=model,
        # train_dataset=hf_train_dataset.select(range(128)),
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        extra_eval_sets=extra_eval_sets,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            # eval_strategy="steps",
            # eval_steps=2,
            # save_strategy="steps",
            # save_steps=2,
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            output_dir=output_dir,
            save_total_limit=10,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False

    # evaluate first before training
    # trainer.evaluate()

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # trainer.save_model(output_dir)
    
    model.state_dict()
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    fire.Fire(train)
