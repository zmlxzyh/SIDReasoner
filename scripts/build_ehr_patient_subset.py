#!/usr/bin/env python3
"""Build a deterministic patient-level subset of the processed MIMIC-III data.

The processed EHR exports intentionally omit patient and visit identifiers.  This
script therefore recovers the patient grouping from the trusted PyHealth pickle
and the authoritative ``*.pt`` split indices, verifies that those samples align
with the processed visit-level exports, and only then filters the public data.

Real patient/visit identifiers are used in memory for auditing and are never
written to the subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


SPLITS = ("train", "valid", "test")
RAW_DATASET_FILE = "mimic3_box_dataset_1.pkl"
PROVENANCE_MANIFESTS = (
    "disease_manifest.json",
    "embedding_manifest.json",
    "sid_manifest.json",
    "sid_collision_report.json",
)
AUTHORITATIVE_HISTORY_FIELDS = (
    "history_disease_id_visits",
    "history_disease_text_visits",
    "history_sid_visits",
)
JSON_CSV_FIELDS = (
    "history_item_title",
    "history_item_id",
    "history_item_sid",
    "history_disease_id_visits",
    "history_disease_text_visits",
    "history_sid_visits",
    "ground_truth_disease_ids",
    "ground_truth_sids",
)
DISALLOWED_IDENTITY_FIELDS = {
    "patient_id",
    "visit_id",
    "adm_time",
    "admission_time",
    "absolute_time",
}
SELECTION_NAMESPACE = "ehr-patient-subset-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible patient-level subset of mimic3_icd_name_path."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the deterministic selection without writing.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def disease_id(code: Any) -> str:
    normalized = str(code).strip().replace(".", "").upper()
    if not normalized:
        raise ValueError("Encountered an empty ICD-9-CM code")
    return f"ICD9CM:{normalized}"


def unique_in_order(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def contains_disallowed_identity_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in DISALLOWED_IDENTITY_FIELDS:
                return True
            if contains_disallowed_identity_field(child):
                return True
    elif isinstance(value, list):
        return any(contains_disallowed_identity_field(child) for child in value)
    return False


def stable_patient_score(patient_id: str, split: str, seed: int) -> bytes:
    # Use the two printable bytes ``\\0`` as the field separator.  Keeping this
    # serialization explicit makes the selected cohort stable across shells,
    # platforms, and future refactors of this script.
    payload = "\\0".join(
        (SELECTION_NAMESPACE, str(seed), split, patient_id)
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def selected_patient_count(patient_count: int, fraction: float) -> int:
    if patient_count <= 0:
        raise ValueError("A split must contain at least one patient")
    return min(
        patient_count,
        max(1, math.floor(patient_count * fraction + 0.5)),
    )


def validate_fraction(fraction: float) -> None:
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("--fraction must be finite and in the interval (0, 1]")


def validate_source_layout(source_dir: Path, raw_dir: Path) -> None:
    require_file(raw_dir / RAW_DATASET_FILE)
    for split in SPLITS:
        require_file(raw_dir / f"{split}set.pt")
        require_file(source_dir / "visit_level" / f"{split}.jsonl")
        require_file(source_dir / "code_level" / f"{split}.csv")
    if not (source_dir / "index").is_dir():
        raise FileNotFoundError(f"Missing index directory: {source_dir / 'index'}")
    if not (source_dir / "info").is_dir():
        raise FileNotFoundError(f"Missing info directory: {source_dir / 'info'}")
    require_file(source_dir / "index" / "mimic3_icd.item.json")
    require_file(source_dir / "index" / "mimic3_icd.index.json")
    require_file(source_dir / "info" / "mimic3_icd.info.tsv")


def load_raw_dataset(raw_dir: Path) -> tuple[Any, dict[str, list[int]]]:
    # The pickle is trusted project data and requires pyhealth to be installed.
    with (raw_dir / RAW_DATASET_FILE).open("rb") as handle:
        dataset = pickle.load(handle)

    split_indices = {
        split: [
            int(index)
            for index in torch.load(
                raw_dir / f"{split}set.pt",
                map_location="cpu",
                weights_only=False,
            )
        ]
        for split in SPLITS
    }
    return dataset, split_indices


def validate_raw_splits(dataset: Any, split_indices: dict[str, list[int]]) -> None:
    index_sets = {split: set(indices) for split, indices in split_indices.items()}
    expected_indices = set(range(len(dataset.samples)))
    if set().union(*index_sets.values()) != expected_indices:
        raise ValueError("Raw split indices do not cover dataset.samples exactly")

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if index_sets[left] & index_sets[right]:
                raise ValueError(f"Raw sample indices overlap: {left} vs {right}")

    patient_sets = {
        split: {
            str(dataset.samples[index]["patient_id"])
            for index in split_indices[split]
        }
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if patient_sets[left] & patient_sets[right]:
                raise ValueError(f"Raw patient IDs overlap: {left} vs {right}")


def sid_maps(
    item_metadata: dict[str, dict[str, Any]],
    sid_index: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, str]]:
    if set(item_metadata) != set(sid_index):
        raise ValueError("item.json and index.json contain different disease IDs")
    disease_to_sid = {
        item_id: "".join(tokens) for item_id, tokens in sid_index.items()
    }
    disease_to_title = {
        item_id: str(metadata["title"])
        for item_id, metadata in item_metadata.items()
    }
    return disease_to_sid, disease_to_title


def expected_raw_record(sample: dict[str, Any]) -> tuple[list[list[str]], list[str]]:
    condition_history = sample["cond_hist"]
    if not condition_history or condition_history[-1] != []:
        raise ValueError("Raw cond_hist must end in the empty current-visit slot")
    history = [
        [disease_id(code) for code in visit]
        for visit in condition_history[:-1]
    ]
    targets = [
        disease_id(code) for code in unique_in_order(sample["icd9_code"])
    ]
    return history, targets


def validate_history_alignment(
    record: dict[str, Any],
    disease_to_sid: dict[str, str],
    disease_to_title: dict[str, str],
) -> None:
    id_visits = record["history_disease_id_visits"]
    text_visits = record["history_disease_text_visits"]
    sid_visits = record["history_sid_visits"]
    if not isinstance(id_visits, list) or not id_visits:
        raise ValueError(f"History is empty for {record.get('sample_id')}")
    if not (len(id_visits) == len(text_visits) == len(sid_visits)):
        raise ValueError(f"History visit counts differ for {record.get('sample_id')}")

    for visit_index, (ids, texts, sids) in enumerate(
        zip(id_visits, text_visits, sid_visits)
    ):
        if not (len(ids) == len(texts) == len(sids)):
            raise ValueError(
                f"History fields differ in visit {visit_index} for "
                f"{record.get('sample_id')}"
            )
        expected_texts = [disease_to_title[item_id] for item_id in ids]
        expected_sids = [disease_to_sid[item_id] for item_id in ids]
        if texts != expected_texts or sids != expected_sids:
            raise ValueError(
                f"History ID/text/SID mapping differs for {record.get('sample_id')}"
            )

    ground_truth_ids = record["ground_truth_disease_ids"]
    ground_truth_sids = record["ground_truth_sids"]
    if len(ground_truth_ids) != len(ground_truth_sids):
        raise ValueError(f"Ground-truth lengths differ for {record.get('sample_id')}")
    if ground_truth_sids != [
        disease_to_sid[item_id] for item_id in ground_truth_ids
    ]:
        raise ValueError(f"Ground-truth SID mapping differs for {record.get('sample_id')}")


def audit_source_visits(
    source_dir: Path,
    dataset: Any,
    split_indices: dict[str, list[int]],
    disease_to_sid: dict[str, str],
    disease_to_title: dict[str, str],
    fraction: float,
    seed: int,
) -> dict[str, dict[str, Any]]:
    selections: dict[str, dict[str, Any]] = {}

    for split in SPLITS:
        indices = split_indices[split]
        visit_path = source_dir / "visit_level" / f"{split}.jsonl"
        patient_to_sample_ids: dict[str, list[str]] = defaultdict(list)
        sample_id_to_patient: dict[str, str] = {}

        with visit_path.open("r", encoding="utf-8") as handle:
            row_count = 0
            for position, (raw_index, line) in enumerate(
                zip(indices, handle, strict=False), start=1
            ):
                row_count = position
                record = json.loads(line)
                if contains_disallowed_identity_field(record):
                    raise ValueError(f"Identity field found in {visit_path}:{position}")

                expected_sample_id = f"mimic3:{split}:{position:06d}"
                if record.get("sample_id") != expected_sample_id:
                    raise ValueError(
                        f"Unexpected sample_id at {visit_path}:{position}: "
                        f"{record.get('sample_id')} != {expected_sample_id}"
                    )
                if record.get("split") != split:
                    raise ValueError(f"Wrong split at {visit_path}:{position}")

                sample = dataset.samples[raw_index]
                expected_history, expected_targets = expected_raw_record(sample)
                if record.get("history_disease_id_visits") != expected_history:
                    raise ValueError(
                        f"Processed history does not align with raw sample {raw_index}"
                    )
                if record.get("ground_truth_disease_ids") != expected_targets:
                    raise ValueError(
                        f"Processed targets do not align with raw sample {raw_index}"
                    )
                validate_history_alignment(
                    record, disease_to_sid, disease_to_title
                )

                patient_id = str(sample["patient_id"])
                sample_id_to_patient[expected_sample_id] = patient_id
                patient_to_sample_ids[patient_id].append(expected_sample_id)

            if row_count != len(indices):
                raise ValueError(
                    f"Visit row count differs for {split}: {row_count} != {len(indices)}"
                )
            if handle.readline():
                raise ValueError(f"Extra visit-level rows found for split {split}")

        patient_ids = list(patient_to_sample_ids)
        keep_count = selected_patient_count(len(patient_ids), fraction)
        ranked_patient_ids = sorted(
            patient_ids,
            key=lambda patient_id: (
                stable_patient_score(patient_id, split, seed),
                patient_id,
            ),
        )
        selected_patients = set(ranked_patient_ids[:keep_count])
        selected_sample_ids = {
            sample_id
            for patient_id in selected_patients
            for sample_id in patient_to_sample_ids[patient_id]
        }
        expected_code_rows = sum(
            len(record_targets)
            for raw_index in indices
            if str(dataset.samples[raw_index]["patient_id"]) in selected_patients
            for _, record_targets in [expected_raw_record(dataset.samples[raw_index])]
        )

        # Completeness is evaluated against every source record for each patient.
        for patient_id in selected_patients:
            expected_ids = set(patient_to_sample_ids[patient_id])
            if not expected_ids <= selected_sample_ids:
                raise ValueError(f"Partial patient selection detected in {split}")
        if any(
            patient_id not in selected_patients
            and sample_id in selected_sample_ids
            for sample_id, patient_id in sample_id_to_patient.items()
        ):
            raise ValueError(f"Unselected patient record leaked into {split}")

        selections[split] = {
            "selected_patients": selected_patients,
            "selected_sample_ids": selected_sample_ids,
            "sample_id_to_patient": sample_id_to_patient,
            "source_patient_count": len(patient_ids),
            "source_visit_count": len(indices),
            "selected_patient_count": len(selected_patients),
            "selected_visit_count": len(selected_sample_ids),
            "expected_code_rows": expected_code_rows,
            "selected_sample_id_sha256": sha256_lines(selected_sample_ids),
        }

    selected_patient_sets = {
        split: selections[split]["selected_patients"] for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if selected_patient_sets[left] & selected_patient_sets[right]:
                raise ValueError(f"Selected patients overlap: {left} vs {right}")

    return selections


def selection_summary(
    selections: dict[str, dict[str, Any]], fraction: float, seed: int
) -> dict[str, Any]:
    return {
        "fraction": fraction,
        "seed": seed,
        "selection_algorithm": "lowest SHA-256 rank per split",
        "selection_namespace": SELECTION_NAMESPACE,
        "rounding": "floor(patient_count * fraction + 0.5), minimum 1",
        "splits": {
            split: {
                "source_patients": selections[split]["source_patient_count"],
                "selected_patients": selections[split]["selected_patient_count"],
                "source_visits": selections[split]["source_visit_count"],
                "selected_visits": selections[split]["selected_visit_count"],
                "selected_code_rows": selections[split]["expected_code_rows"],
                "selected_sample_id_sha256": selections[split][
                    "selected_sample_id_sha256"
                ],
            }
            for split in SPLITS
        },
    }


def copy_tree_with_hashes(source: Path, destination: Path) -> dict[str, str]:
    shutil.copytree(source, destination)
    hashes: dict[str, str] = {}
    for source_path in sorted(path for path in source.rglob("*") if path.is_file()):
        relative_path = source_path.relative_to(source)
        destination_path = destination / relative_path
        source_hash = sha256_file(source_path)
        destination_hash = sha256_file(destination_path)
        if source_hash != destination_hash:
            raise ValueError(f"Copied metadata hash differs: {relative_path}")
        hashes[relative_path.as_posix()] = source_hash
    return hashes


def build_visit_subset(
    source_dir: Path,
    temp_dir: Path,
    selections: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, int]]:
    records: dict[str, dict[str, dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    output_dir = temp_dir / "visit_level"
    output_dir.mkdir(parents=True)

    for split in SPLITS:
        selected_sample_ids = selections[split]["selected_sample_ids"]
        records[split] = {}
        source_path = source_dir / "visit_level" / f"{split}.jsonl"
        destination_path = output_dir / f"{split}.jsonl"
        with source_path.open("r", encoding="utf-8") as source_handle, (
            destination_path.open("w", encoding="utf-8", newline="")
        ) as destination_handle:
            for line in source_handle:
                record = json.loads(line)
                sample_id = record["sample_id"]
                if sample_id not in selected_sample_ids:
                    continue
                if sample_id in records[split]:
                    raise ValueError(f"Duplicate visit-level sample_id: {sample_id}")
                destination_handle.write(line.rstrip("\r\n") + "\n")
                records[split][sample_id] = record

        counts[split] = len(records[split])
        if set(records[split]) != selected_sample_ids:
            missing = sorted(selected_sample_ids - set(records[split]))[:5]
            raise ValueError(f"Missing selected visit rows in {split}: {missing}")
        if counts[split] != selections[split]["selected_visit_count"]:
            raise ValueError(f"Unexpected selected visit count for {split}")

    return records, counts


def flatten(visits: list[list[Any]]) -> list[Any]:
    return [item for visit in visits for item in visit]


def parse_json_csv_field(row: dict[str, str], field: str) -> Any:
    try:
        return json.loads(row[field])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Invalid JSON in CSV field {field} for {row.get('sample_id')}"
        ) from error


def validate_code_row(
    row: dict[str, str],
    split: str,
    visit_record: dict[str, Any],
    disease_to_sid: dict[str, str],
    disease_to_title: dict[str, str],
) -> int:
    if row.get("split") != split:
        raise ValueError(f"Wrong CSV split for {row.get('sample_id')}")

    parsed = {field: parse_json_csv_field(row, field) for field in JSON_CSV_FIELDS}
    for field in AUTHORITATIVE_HISTORY_FIELDS:
        if parsed[field] != visit_record[field]:
            raise ValueError(
                f"CSV {field} differs from visit-level data for {row.get('sample_id')}"
            )
    if parsed["ground_truth_disease_ids"] != visit_record["ground_truth_disease_ids"]:
        raise ValueError(f"CSV GT IDs differ for {row.get('sample_id')}")
    if parsed["ground_truth_sids"] != visit_record["ground_truth_sids"]:
        raise ValueError(f"CSV GT SIDs differ for {row.get('sample_id')}")

    if parsed["history_item_id"] != flatten(
        visit_record["history_disease_id_visits"]
    ):
        raise ValueError(f"Flattened history IDs differ for {row.get('sample_id')}")
    if parsed["history_item_title"] != flatten(
        visit_record["history_disease_text_visits"]
    ):
        raise ValueError(f"Flattened history titles differ for {row.get('sample_id')}")
    if parsed["history_item_sid"] != flatten(visit_record["history_sid_visits"]):
        raise ValueError(f"Flattened history SIDs differ for {row.get('sample_id')}")

    try:
        target_index = int(row["target_index"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid target_index for {row.get('sample_id')}") from error
    ground_truth_ids = visit_record["ground_truth_disease_ids"]
    ground_truth_sids = visit_record["ground_truth_sids"]
    if not 0 <= target_index < len(ground_truth_ids):
        raise ValueError(f"target_index out of range for {row.get('sample_id')}")
    expected_item_id = ground_truth_ids[target_index]
    if row.get("item_id") != expected_item_id:
        raise ValueError(f"item_id differs for {row.get('sample_id')}")
    if row.get("item_sid") != ground_truth_sids[target_index]:
        raise ValueError(f"item_sid differs for {row.get('sample_id')}")
    if row.get("item_sid") != disease_to_sid[expected_item_id]:
        raise ValueError(f"item/index SID differs for {row.get('sample_id')}")
    if row.get("item_title") != disease_to_title[expected_item_id]:
        raise ValueError(f"item title differs for {row.get('sample_id')}")
    return target_index


def maximize_csv_field_size() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def build_code_subset(
    source_dir: Path,
    temp_dir: Path,
    selections: dict[str, dict[str, Any]],
    visit_records: dict[str, dict[str, dict[str, Any]]],
    disease_to_sid: dict[str, str],
    disease_to_title: dict[str, str],
) -> dict[str, int]:
    maximize_csv_field_size()
    counts: dict[str, int] = {}
    output_dir = temp_dir / "code_level"
    output_dir.mkdir(parents=True)

    for split in SPLITS:
        selected_sample_ids = selections[split]["selected_sample_ids"]
        source_path = source_dir / "code_level" / f"{split}.csv"
        destination_path = output_dir / f"{split}.csv"
        target_indices: dict[str, list[int]] = defaultdict(list)
        row_count = 0

        with source_path.open("r", encoding="utf-8", newline="") as source_handle, (
            destination_path.open("w", encoding="utf-8", newline="")
        ) as destination_handle:
            reader = csv.DictReader(source_handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {source_path}")
            if set(reader.fieldnames) & DISALLOWED_IDENTITY_FIELDS:
                raise ValueError(f"Identity column found in {source_path}")
            missing_fields = set(JSON_CSV_FIELDS) - set(reader.fieldnames)
            if missing_fields:
                raise ValueError(f"Missing CSV fields in {source_path}: {missing_fields}")
            writer = csv.DictWriter(
                destination_handle,
                fieldnames=reader.fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()

            for row in reader:
                sample_id = row.get("sample_id", "")
                if sample_id not in selected_sample_ids:
                    continue
                target_index = validate_code_row(
                    row,
                    split,
                    visit_records[split][sample_id],
                    disease_to_sid,
                    disease_to_title,
                )
                writer.writerow(row)
                target_indices[sample_id].append(target_index)
                row_count += 1

        if set(target_indices) != selected_sample_ids:
            missing = sorted(selected_sample_ids - set(target_indices))[:5]
            raise ValueError(f"Selected visits without code rows in {split}: {missing}")
        for sample_id, indices in target_indices.items():
            expected = list(
                range(len(visit_records[split][sample_id]["ground_truth_disease_ids"]))
            )
            if indices != expected:
                raise ValueError(
                    f"Code-level target reaggregation differs for {sample_id}: "
                    f"{indices} != {expected}"
                )
        if row_count != selections[split]["expected_code_rows"]:
            raise ValueError(
                f"Unexpected code-level count for {split}: {row_count} != "
                f"{selections[split]['expected_code_rows']}"
            )
        counts[split] = row_count

    return counts


def collect_source_hashes(source_dir: Path, raw_dir: Path) -> dict[str, Any]:
    return {
        "raw": {
            RAW_DATASET_FILE: sha256_file(raw_dir / RAW_DATASET_FILE),
            **{
                f"{split}set.pt": sha256_file(raw_dir / f"{split}set.pt")
                for split in SPLITS
            },
        },
        "processed": {
            "visit_level": {
                split: sha256_file(source_dir / "visit_level" / f"{split}.jsonl")
                for split in SPLITS
            },
            "code_level": {
                split: sha256_file(source_dir / "code_level" / f"{split}.csv")
                for split in SPLITS
            },
        },
        "source_manifests": {
            path.name: sha256_file(path)
            for path in sorted((source_dir / "manifest").glob("*.json"))
        },
    }


def generated_data_hashes(temp_dir: Path) -> dict[str, dict[str, str]]:
    return {
        "visit_level": {
            split: sha256_file(temp_dir / "visit_level" / f"{split}.jsonl")
            for split in SPLITS
        },
        "code_level": {
            split: sha256_file(temp_dir / "code_level" / f"{split}.csv")
            for split in SPLITS
        },
    }


def write_manifests(
    source_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    temp_dir: Path,
    fraction: float,
    seed: int,
    selections: dict[str, dict[str, Any]],
    visit_counts: dict[str, int],
    code_counts: dict[str, int],
    metadata_hashes: dict[str, dict[str, str]],
    source_hashes: dict[str, Any],
    data_hashes: dict[str, dict[str, str]],
    disease_to_sid: dict[str, str],
) -> None:
    manifest_dir = temp_dir / "manifest"
    manifest_dir.mkdir(exist_ok=True)

    source_preprocessing = load_json(
        source_dir / "manifest" / "preprocessing_manifest.json"
    )
    source_export = load_json(source_dir / "manifest" / "export_manifest.json")
    source_validation = load_json(
        source_dir / "manifest" / "validation_report.json"
    )
    source_collision = load_json(
        source_dir / "manifest" / "sid_collision_report.json"
    )

    selected_patient_counts = {
        split: selections[split]["selected_patient_count"] for split in SPLITS
    }
    source_patient_counts = {
        split: selections[split]["source_patient_count"] for split in SPLITS
    }
    selected_sample_hashes = {
        split: selections[split]["selected_sample_id_sha256"] for split in SPLITS
    }
    created_at = datetime.now(timezone.utc).isoformat()
    variant_name = output_dir.name

    subset_manifest = {
        "schema_version": 1,
        "dataset": variant_name,
        "source_dataset": source_dir.name,
        "source_dir": source_dir.as_posix(),
        "raw_dir": raw_dir.as_posix(),
        "created_at": created_at,
        "sample_unit": "patient",
        "selection_scope": "independent_per_split",
        "selection_algorithm": "lowest SHA-256 rank per split",
        "selection_namespace": SELECTION_NAMESPACE,
        "rounding": "floor(patient_count * fraction + 0.5), minimum 1",
        "fraction": fraction,
        "seed": seed,
        "sample_ids_preserved": True,
        "identity_fields_written": False,
        "history_order_policy": {
            "stored_order": "source_preserved",
            "train_consumption": "shuffle diseases within each visit dynamically",
            "valid_test_consumption": "sort diseases within each visit by disease_id",
            "visit_chronology": "preserved",
        },
        "source_counts": {
            "patients": source_patient_counts,
            "visits": {
                split: selections[split]["source_visit_count"] for split in SPLITS
            },
            "code_level": source_preprocessing["code_level_counts"],
        },
        "selected_counts": {
            "patients": selected_patient_counts,
            "visits": visit_counts,
            "code_level": code_counts,
        },
        "selected_sample_id_sha256": selected_sample_hashes,
        "source_hashes": source_hashes,
        "output_data_hashes": data_hashes,
        "metadata_hashes": metadata_hashes,
    }

    preprocessing_manifest = dict(source_preprocessing)
    preprocessing_manifest.update(
        {
            "dataset_variant": variant_name,
            "source_dataset": source_dir.name,
            "subset_fraction": fraction,
            "subset_seed": seed,
            "subset_unit": "patient",
            "subset_scope": "independent_per_split",
            "source_patient_counts": source_patient_counts,
            "source_visit_counts": {
                split: selections[split]["source_visit_count"] for split in SPLITS
            },
            "source_code_level_counts": source_preprocessing["code_level_counts"],
            "patient_counts": selected_patient_counts,
            "visit_counts": visit_counts,
            "code_level_counts": code_counts,
            "selected_sample_id_sha256": selected_sample_hashes,
            "identity_fields_written": False,
            "history_order_policy": subset_manifest["history_order_policy"],
        }
    )

    export_manifest = dict(source_export)
    export_manifest.update(
        {
            "dataset_variant": variant_name,
            "source_dataset": source_dir.name,
            "subset_fraction": fraction,
            "subset_seed": seed,
            "sample_unit": "patient",
            "patient_counts": selected_patient_counts,
            "visit_counts": visit_counts,
            "code_level_counts": code_counts,
            "identity_fields_written": False,
            "sample_ids_preserved": True,
            "list_encoding": "JSON",
            "authoritative_history_fields": list(AUTHORITATIVE_HISTORY_FIELDS),
            "item_sha256": metadata_hashes["index"]["mimic3_icd.item.json"],
            "index_sha256": metadata_hashes["index"]["mimic3_icd.index.json"],
            "info_sha256": metadata_hashes["info"]["mimic3_icd.info.tsv"],
        }
    )

    warnings = list(source_validation.get("warnings", []))
    collision_count = int(source_collision.get("collision_disease_count", 0))
    if collision_count and not any("collision" in warning.lower() for warning in warnings):
        warnings.append("full SID collisions remain; see sid_collision_report.json")

    validation_report = {
        "valid": True,
        "dataset_variant": variant_name,
        "source_dataset": source_dir.name,
        "subset_fraction": fraction,
        "subset_seed": seed,
        "patient_counts": selected_patient_counts,
        "visit_counts": visit_counts,
        "code_level_counts": code_counts,
        "patient_leakage_count": 0,
        "identity_fields_written": False,
        "sample_ids_preserved": True,
        "selected_sample_id_sha256": selected_sample_hashes,
        "checks": {
            "raw_split_coverage_valid": True,
            "raw_patient_splits_disjoint": True,
            "processed_to_raw_mapping_valid": True,
            "selected_patient_records_complete": True,
            "selected_patient_splits_disjoint": True,
            "visit_level_schema_valid": True,
            "history_id_text_sid_alignment_valid": True,
            "code_level_reaggregation_valid": True,
            "code_and_visit_sample_ids_match": True,
            "identity_fields_absent": True,
            "metadata_hashes_match_source": True,
            "selection_is_deterministic": True,
        },
        "disease_count": len(disease_to_sid),
        "unique_sid_count": len(set(disease_to_sid.values())),
        "collision_disease_count": collision_count,
        "collision_group_count": len(source_collision.get("collision_groups", [])),
        "require_unique_sid": False,
        "warnings": warnings,
    }

    dump_json(manifest_dir / "subset_manifest.json", subset_manifest)
    dump_json(
        manifest_dir / "preprocessing_manifest.json", preprocessing_manifest
    )
    dump_json(manifest_dir / "export_manifest.json", export_manifest)
    dump_json(manifest_dir / "validation_report.json", validation_report)


def copy_provenance_manifests(source_dir: Path, temp_dir: Path) -> None:
    destination_dir = temp_dir / "manifest"
    destination_dir.mkdir(parents=True, exist_ok=True)
    for filename in PROVENANCE_MANIFESTS:
        source_path = source_dir / "manifest" / filename
        require_file(source_path)
        shutil.copy2(source_path, destination_dir / filename)
        if sha256_file(source_path) != sha256_file(destination_dir / filename):
            raise ValueError(f"Copied provenance manifest hash differs: {filename}")


def build_subset(
    source_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    fraction: float,
    seed: int,
    selections: dict[str, dict[str, Any]],
    disease_to_sid: dict[str, str],
    disease_to_title: dict[str, str],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to mix data: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )

    try:
        metadata_hashes = {
            "index": copy_tree_with_hashes(source_dir / "index", temp_dir / "index"),
            "info": copy_tree_with_hashes(source_dir / "info", temp_dir / "info"),
        }
        copy_provenance_manifests(source_dir, temp_dir)
        visit_records, visit_counts = build_visit_subset(
            source_dir, temp_dir, selections
        )
        code_counts = build_code_subset(
            source_dir,
            temp_dir,
            selections,
            visit_records,
            disease_to_sid,
            disease_to_title,
        )
        source_hashes = collect_source_hashes(source_dir, raw_dir)
        data_hashes = generated_data_hashes(temp_dir)
        write_manifests(
            source_dir=source_dir,
            raw_dir=raw_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            fraction=fraction,
            seed=seed,
            selections=selections,
            visit_counts=visit_counts,
            code_counts=code_counts,
            metadata_hashes=metadata_hashes,
            source_hashes=source_hashes,
            data_hashes=data_hashes,
            disease_to_sid=disease_to_sid,
        )
        os.rename(temp_dir, output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "output_dir": output_dir.as_posix(),
        "patient_counts": {
            split: selections[split]["selected_patient_count"] for split in SPLITS
        },
        "visit_counts": visit_counts,
        "code_level_counts": code_counts,
        "selected_sample_id_sha256": {
            split: selections[split]["selected_sample_id_sha256"] for split in SPLITS
        },
    }


def main() -> None:
    args = parse_args()
    validate_fraction(args.fraction)
    source_dir = args.source_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    validate_source_layout(source_dir, raw_dir)

    item_metadata = load_json(source_dir / "index" / "mimic3_icd.item.json")
    sid_index = load_json(source_dir / "index" / "mimic3_icd.index.json")
    disease_to_sid, disease_to_title = sid_maps(item_metadata, sid_index)

    dataset, split_indices = load_raw_dataset(raw_dir)
    validate_raw_splits(dataset, split_indices)
    selections = audit_source_visits(
        source_dir=source_dir,
        dataset=dataset,
        split_indices=split_indices,
        disease_to_sid=disease_to_sid,
        disease_to_title=disease_to_title,
        fraction=args.fraction,
        seed=args.seed,
    )

    if args.dry_run:
        print(
            json.dumps(
                selection_summary(selections, args.fraction, args.seed),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = build_subset(
        source_dir=source_dir,
        raw_dir=raw_dir,
        output_dir=output_dir,
        fraction=args.fraction,
        seed=args.seed,
        selections=selections,
        disease_to_sid=disease_to_sid,
        disease_to_title=disease_to_title,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
