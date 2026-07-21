import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
import random
from tqdm import tqdm
import os
import copy
import torch.nn.functional as F

# ASSISTANT_PREFIX = "<|im_start|>assistant"
ASSISTANT_PREFIX = "</think>"


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


def _find_subsequence(sequence, pattern):
    limit = len(sequence) - len(pattern) + 1
    for start in range(limit):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return -1


def mask_assistant_response_only(
    tokenizer,
    messages,
    assistant_response,
    max_len=None,
    mask_eos=True,
):
    """
    Build labels so that only assistant_response tokens contribute to loss.
    Everything before and after is masked.
    """

    # --- 1. raw text ---
    raw_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    # --- 2. locate assistant_response start ---
    pos = raw_text.rfind(assistant_response)
    if pos == -1:
        raise ValueError("assistant response not found in raw_text")

    # --- 3. tokenize prefix ---
    prefix_ids = tokenizer.encode(raw_text[:pos], add_special_tokens=False)
    prefix_len = len(prefix_ids)

    # --- 4. tokenize assistant_response itself ---
    response_ids = tokenizer.encode(assistant_response, add_special_tokens=False)
    response_len = len(response_ids)

    # --- 5. full tokenized sequence ---
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )
    if "input_ids" in full_ids:
        full_ids = full_ids["input_ids"]

    # --- 6. labels = mask prefix + keep response + mask suffix ---
    total_len = len(full_ids)
    labels = [-100] * total_len

    response_start = prefix_len
    if mask_eos:
        response_end = prefix_len + response_len
        if response_end > total_len:
            raise ValueError("response range exceeds total length")
        labels[response_start:response_end] = full_ids[response_start:response_end]

    else:
        labels[response_start:] = full_ids[response_start:]

    # --- 7. attention mask ---
    attention_mask = [1] * total_len

    # --- 8. truncate if needed ---
    if max_len is not None and total_len > max_len:
        full_ids = full_ids[-max_len:]
        attention_mask = attention_mask[-max_len:]
        labels = labels[-max_len:]

    return full_ids, attention_mask, labels


VISIT_START_TOKEN = "<visit_start>"
VISIT_END_TOKEN = "<visit_end>"


def _parse_nested_json_list(value, field_name, sample_index):
    """Parse and validate a JSON-encoded list of visits."""
    try:
        parsed = json.loads(value) if isinstance(value, str) else copy.deepcopy(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"sample {sample_index}: invalid JSON in {field_name}: {exc}"
        ) from exc

    if not isinstance(parsed, list):
        raise ValueError(f"sample {sample_index}: {field_name} must be a list of visits")
    for visit_index, visit in enumerate(parsed):
        if not isinstance(visit, list):
            raise ValueError(
                f"sample {sample_index}: {field_name}[{visit_index}] must be a list"
            )
        if not all(isinstance(item, str) for item in visit):
            raise ValueError(
                f"sample {sample_index}: {field_name}[{visit_index}] must contain strings"
            )
    return parsed


def _ordered_ehr_history(row, sample_index, seed):
    """Load the authoritative nested history fields and order them together."""
    disease_id_visits = _parse_nested_json_list(
        row["history_disease_id_visits"], "history_disease_id_visits", sample_index
    )
    disease_text_visits = _parse_nested_json_list(
        row["history_disease_text_visits"], "history_disease_text_visits", sample_index
    )
    sid_visits = _parse_nested_json_list(
        row["history_sid_visits"], "history_sid_visits", sample_index
    )

    if len({len(disease_id_visits), len(disease_text_visits), len(sid_visits)}) != 1:
        raise ValueError(
            f"sample {sample_index}: history visit counts are not aligned: "
            f"ids={len(disease_id_visits)}, texts={len(disease_text_visits)}, "
            f"sids={len(sid_visits)}"
        )

    split = str(row.get("split", "train")).strip().lower()
    training = split == "train"
    rng = random.Random(seed * 1_000_003 + sample_index)
    ordered_ids = []
    ordered_texts = []
    ordered_sids = []

    for visit_index, (ids, texts, sids) in enumerate(
        zip(disease_id_visits, disease_text_visits, sid_visits)
    ):
        if not (len(ids) == len(texts) == len(sids)):
            raise ValueError(
                f"sample {sample_index}: history visit {visit_index} is not aligned: "
                f"ids={len(ids)}, texts={len(texts)}, sids={len(sids)}"
            )
        records = list(zip(ids, texts, sids))
        if training:
            rng.shuffle(records)
        else:
            records.sort(key=lambda record: record[0])

        ordered_ids.append([record[0] for record in records])
        ordered_texts.append([record[1] for record in records])
        ordered_sids.append([record[2] for record in records])

    return {
        "disease_id_visits": ordered_ids,
        "disease_text_visits": ordered_texts,
        "sid_visits": ordered_sids,
        "split": split,
    }


def serialize_sid_history(history_sid_visits):
    blocks = []
    for visit in history_sid_visits:
        blocks.append(
            f"{VISIT_START_TOKEN}\n{', '.join(visit)}\n{VISIT_END_TOKEN}"
        )
    return "\n".join(blocks)


def build_ehr_sid_prediction_messages(history_sid_visits, assistant_response=None):
    """Build the shared T5/visit-evaluation SID prediction conversation."""
    history_text = serialize_sid_history(history_sid_visits)
    prompt = (
        "The patient's chronological diagnosis history is:\n"
        f"{history_text}\n"
        "Return exactly one disease SID."
    )
    messages = [
        {
            "role": "system",
            "content": "Predict one possible diagnosis SID in the next visit from the chronological SID history.",
        },
        {"role": "user", "content": prompt},
    ]
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})
    return messages


def serialize_text_history(history_text_visits):
    return "\n".join(
        f"Visit {visit_index}: {'; '.join(visit)}"
        for visit_index, visit in enumerate(history_text_visits, start=1)
    )


def _fit_ehr_history_to_context(
    history_visits,
    build_messages,
    tokenizer,
    max_len,
    add_generation_prompt,
    task_name,
    sample_index,
    chat_template_kwargs=None,
):
    """Drop complete oldest visits until the rendered chat fits the context."""
    kept_visits = [list(visit) for visit in history_visits]
    while True:
        messages = build_messages(kept_visits)
        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors=None,
            **(chat_template_kwargs or {}),
        )
        if isinstance(tokenized, dict):
            tokenized = tokenized["input_ids"]
        if len(tokenized) <= max_len:
            return kept_visits, messages, tokenized
        if not kept_visits:
            raise ValueError(
                f"{task_name} sample {sample_index}: prompt and answer exceed max_len={max_len} "
                "even after all history visits were removed"
            )
        kept_visits = kept_visits[1:]


def _encode_ehr_sample(
    tokenizer,
    messages,
    assistant_response,
    tokenized,
    max_len,
    mask_assistant,
    test,
    task_name,
    sample_index,
):
    if len(tokenized) > max_len:
        raise ValueError(
            f"{task_name} sample {sample_index}: sequence length {len(tokenized)} "
            f"exceeds max_len={max_len}"
        )

    attention_mask = [1] * len(tokenized)
    if test:
        return {"input_ids": tokenized, "attention_mask": attention_mask}

    if mask_assistant:
        input_ids, attention_mask, labels = mask_assistant_response_only(
            tokenizer=tokenizer,
            messages=messages,
            assistant_response=assistant_response,
            max_len=None,
            mask_eos=True,
        )
    else:
        input_ids = list(tokenized)
        labels = list(tokenized)

    if not (len(input_ids) == len(attention_mask) == len(labels)):
        raise ValueError(f"{task_name} sample {sample_index}: encoded fields are not aligned")
    if len(input_ids) > max_len:
        raise ValueError(
            f"{task_name} sample {sample_index}: encoded sequence exceeds max_len={max_len}"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _validate_ehr_dedup(dedup):
    if dedup:
        raise ValueError(
            "EHR datasets do not support dedup=True: recurring diagnoses are valid targets"
        )



class EHRSFTData(Dataset):
    def __init__(
        self,
        train_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        K=4,
        dedup=False,
        mask_assistant: bool = True,
    ):
        _validate_ehr_dedup(dedup)
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.seed = seed
        self.dedup = False
        self.mask_assistant = mask_assistant
        self.get_inputs()  
    def __len__(self):
        return len(self.data)
    

    def generate_prompt(self, data_point):
        return f"""{data_point["input"]}"""


    def get_history(self, row, idx=0):
        history = _ordered_ehr_history(row, idx, self.seed)
        history["output"] = str(row["item_title"]).strip()
        return history
    
    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx], idx)
        target_item = history["output"]
        assistant_response = target_item if not self.test else ""

        def build_messages(text_visits):
            history_text = serialize_text_history(text_visits)
            prompt = (
                "The patient's chronological diagnosis history is:\n"
                f"{history_text}\n"
                "Return one possible disease description for the next visit."
            )
            messages = [
                {
                    "role": "system",
                    "content": "Predict one possible diagnosis in the next visit from the chronological diagnosis history.",
                },
                {"role": "user", "content": prompt},
            ]
            if not self.test:
                messages.append({"role": "assistant", "content": assistant_response})
            return messages

        _, messages, tokenized = _fit_ehr_history_to_context(
            history["disease_text_visits"],
            build_messages,
            self.tokenizer,
            self.max_len,
            self.test,
            "T4/EHRSFTData",
            idx,
        )
        return _encode_ehr_sample(
            self.tokenizer,
            messages,
            assistant_response,
            tokenized,
            self.max_len,
            self.mask_assistant,
            self.test,
            "T4/EHRSFTData",
            idx,
        )
    

    
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            # print(inputs[-1])
            
        self.inputs = inputs
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i], i))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]
 

class EHRSidSFTDataset(Dataset):
    def __init__(
        self,
        train_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        K=4,
        dedup=False,
        mask_assistant: bool = True,
    ):
        _validate_ehr_dedup(dedup)
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.seed = seed
        self.dedup = False
        self.mask_assistant = mask_assistant
        self.get_inputs()  
    
    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        return f"""{data_point["input"]}"""

    def get_history(self, row, idx=0):
        history = _ordered_ehr_history(row, idx, self.seed)
        history["output"] = str(row["item_sid"]).strip()
        return history
    
    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx], idx)
        target_item = history["output"]
        assistant_response = target_item.strip() if not self.test else ""

        def build_messages(sid_visits):
            response = None if self.test else assistant_response
            return build_ehr_sid_prediction_messages(sid_visits, response)

        _, messages, tokenized = _fit_ehr_history_to_context(
            history["sid_visits"],
            build_messages,
            self.tokenizer,
            self.max_len,
            self.test,
            "T5/EHRSidSFTDataset",
            idx,
        )
        return _encode_ehr_sample(
            self.tokenizer,
            messages,
            assistant_response,
            tokenized,
            self.max_len,
            self.mask_assistant,
            self.test,
            "T5/EHRSidSFTDataset",
            idx,
        )


    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            
        self.inputs = inputs
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i], i))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]


class EHRVisitEvalDataset(Dataset):
    """One no-thinking generation prompt per visit-level JSONL record."""

    def __init__(
        self,
        visit_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        seed=0,
    ):
        self.data = []
        with open(visit_file, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{visit_file}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{visit_file}:{line_number}: each JSONL record must be an object"
                    )
                self.data.append(record)

        if sample > 0:
            self.data = self.data[:sample]
        sample_ids = [str(record.get("sample_id", index)) for index, record in enumerate(self.data)]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{visit_file}: sample_id values must be unique")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seed = seed
        self.get_inputs()

    def __len__(self):
        return len(self.data)

    def get_history(self, row, idx=0):
        history = _ordered_ehr_history(row, idx, self.seed)
        if history["split"] == "train":
            raise ValueError(
                f"visit sample {idx}: evaluation records must use a valid/test split"
            )
        ground_truth_sids = row.get("ground_truth_sids")
        ground_truth_disease_ids = row.get("ground_truth_disease_ids")
        if not isinstance(ground_truth_sids, list) or not all(
            isinstance(value, str) for value in ground_truth_sids
        ):
            raise ValueError(
                f"visit sample {idx}: ground_truth_sids must be a list of strings"
            )
        if not isinstance(ground_truth_disease_ids, list) or not all(
            isinstance(value, str) for value in ground_truth_disease_ids
        ):
            raise ValueError(
                f"visit sample {idx}: ground_truth_disease_ids must be a list of strings"
            )
        if len(ground_truth_sids) != len(ground_truth_disease_ids):
            raise ValueError(
                f"visit sample {idx}: ground-truth SID/disease counts are not aligned"
            )
        if not ground_truth_sids:
            raise ValueError(f"visit sample {idx}: ground truth cannot be empty")

        order = sorted(
            range(len(ground_truth_disease_ids)),
            key=lambda position: ground_truth_disease_ids[position],
        )
        history["ground_truth_disease_ids"] = [
            ground_truth_disease_ids[position] for position in order
        ]
        history["ground_truth_sids"] = [ground_truth_sids[position] for position in order]
        history["sample_id"] = str(row.get("sample_id", idx))
        return history

    def pre(self, idx):
        history = self.get_history(self.data[idx], idx)

        def build_messages(sid_visits):
            return build_ehr_sid_prediction_messages(sid_visits)

        _, _, input_ids = _fit_ehr_history_to_context(
            history["sid_visits"],
            build_messages,
            self.tokenizer,
            self.max_len,
            True,
            "EHRVisitEvalDataset",
            idx,
            chat_template_kwargs={"enable_thinking": False},
        )
        return {
            "sample_id": history["sample_id"],
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "ground_truth_sids": history["ground_truth_sids"],
            "ground_truth_disease_ids": history["ground_truth_disease_ids"],
        }

    def get_inputs(self):
        self.inputs = [self.pre(index) for index in tqdm(range(len(self.data)))]

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]


class EvalSidDataset(Dataset):

    def __init__(
        self,
        train_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        K=4,
        dedup=False,
        mask_assistant: bool = True,
    ):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.mask_assistant = mask_assistant
        self.get_inputs()  


    def __len__(self):
        return len(self.data)

    
    def generate_prompt(self, data_point):
        return f"""{data_point["input"]}"""

    def get_history(self, row):
        row['history_item_sid'] = eval(row['history_item_sid'])
        L = len(row['history_item_sid']) 
        history = ""
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]      
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": # f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {history}",
                "output": target_item + '\n',
                "dedup": target_item_sid == last_history_item_sid}
    
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
Can you predict the next possible item that the user may expect?
"""
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history_for_prompt = history.copy()
        history_for_prompt['output'] = ''
        prompt = self.generate_prompt(history_for_prompt)

        assistant_response = target_item if not self.test else ""
        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": prompt},
        ]

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        attention_mask = [1] * len(tokenized)

        if self.test:
            prefix_prompt = "<think>\n</think>\n\n"
            prefix_prompt_ids = self.tokenizer.encode(prefix_prompt)
            tokenized = tokenized + prefix_prompt_ids
            attention_mask = attention_mask + [1] * len(prefix_prompt_ids)

            len_prompt = self.max_len + len(prefix_prompt_ids)

            if len(tokenized) >= len_prompt:
                print(len(tokenized))
                tokenized = tokenized[-len_prompt:]
                attention_mask = attention_mask[-len_prompt:]
            return {
                "input_ids": tokenized,
                "attention_mask": attention_mask,
            }

        messages.append({"role": "assistant", "content": assistant_response})
        input_ids, attention_mask, labels = mask_assistant_response_only(
            tokenizer=self.tokenizer,
            messages=messages,
            assistant_response=assistant_response,
            max_len=self.max_len,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    

    
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            
        self.inputs = inputs
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]



class EHRSidItemFeatDataset(Dataset):
    def __init__(
        self,
        item_file,
        index_file,
        tokenizer=None,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        task_type=None, # select from ['sid2title', 'title2sid'] to indicate the task type, set to None for both
        mask_assistant: bool = True,
    ):
        """Disease name/SID bidirectional alignment dataset (T1 and T2)."""
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.mask_assistant = mask_assistant

        missing_metadata = sorted(set(self.indices) - set(self.item_feat))
        if missing_metadata:
            raise ValueError(
                f"{len(missing_metadata)} diseases in index_file are missing from item_file; "
                f"first={missing_metadata[0]}"
            )

        records = []
        sid_counts = {}
        title_counts = {}
        for disease_id in sorted(self.indices):
            sid_tokens = self.indices[disease_id]
            if not isinstance(sid_tokens, list) or not sid_tokens:
                raise ValueError(f"invalid SID token list for {disease_id}")
            combined_sid = "".join(sid_tokens)
            title = str(self.item_feat[disease_id]["title"]).strip()
            records.append((disease_id, combined_sid, title))
            sid_counts[combined_sid] = sid_counts.get(combined_sid, 0) + 1
            title_counts[title] = title_counts.get(title, 0) + 1

        self.sid_collision_group_count = sum(count > 1 for count in sid_counts.values())
        self.sid_collision_disease_count = sum(
            count for count in sid_counts.values() if count > 1
        )
        self.duplicate_title_group_count = sum(count > 1 for count in title_counts.values())
        self.duplicate_title_disease_count = sum(
            count for count in title_counts.values() if count > 1
        )
        if self.sid_collision_group_count:
            print(
                f"[warn] accepted {self.sid_collision_group_count} full-SID collision groups "
                f"covering {self.sid_collision_disease_count} diseases"
            )
        if self.duplicate_title_group_count:
            print(
                f"[warn] accepted {self.duplicate_title_group_count} duplicate-title groups "
                f"covering {self.duplicate_title_disease_count} diseases"
            )

        # Sorted disease IDs make the accepted overwrite behavior reproducible.
        self.sid2title = {}
        self.title2sid = {}
        for _, combined_sid, title in records:
            self.sid2title[combined_sid] = title
            self.title2sid[title] = combined_sid
        
        # Create data samples
        self.data = []
        
        # Create sid2title samples
        if task_type is None or task_type == 'sid2title':
            for sid in sorted(self.sid2title):
                title = self.sid2title[sid]
                self.data.append({
                    'task': 'sid2title',
                    'input': sid,
                    'output': title
                })
        
        # Create title2sid samples
        if task_type is None or task_type == 'title2sid':
            for title in sorted(self.title2sid):
                sid = self.title2sid[title]
                self.data.append({
                    'task': 'title2sid',
                    'input': title,
                    'output': sid
                })
        
        if sample > 0 and sample < len(self.data):
            self.data = random.Random(seed).sample(self.data, sample)
        
        if self.tokenizer is not None:
            self.get_inputs()
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt(self, data_point):
        if data_point['task'] == 'title2sid':
            prompt = f'Which disease has the name "{data_point["input"]}"?'
        else:  # sid2title
            prompt = f'What disease does {data_point["input"]} represent?'
        return prompt
    
    def pre(self, idx):
        if self.tokenizer is None:
            return self.data[idx]

        data_point = self.data[idx]
        user_prompt = self.generate_prompt(data_point)
        assistant_response = str(data_point["output"]).strip() if not self.test else ""

        if data_point["task"] == "title2sid":
            system_prompt = "You map ICD disease names to semantic identifiers."
        else:
            system_prompt = "You map semantic identifiers to ICD disease names."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if not self.test:
            messages.append({"role": "assistant", "content": assistant_response})

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        if isinstance(tokenized, dict):
            tokenized = tokenized["input_ids"]
        return _encode_ehr_sample(
            self.tokenizer,
            messages,
            assistant_response,
            tokenized,
            self.max_len,
            self.mask_assistant,
            self.test,
            f"T1/T2/EHRSidItemFeatDataset/{data_point['task']}",
            idx,
        )
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
        self.inputs = inputs
    
    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else [self.pre(i) for i in range(len(self))]
    
    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)

class EHRFusionSeqRecDataset(Dataset):
    def __init__(
        self,
        train_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
        mask_assistant: bool = True,
    ):
        """SID visit history to target disease-name dataset (T3)."""
        _validate_ehr_dedup(dedup)
        # Load sequence data
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.seed = seed
        self.dedup = False
        self.mask_assistant = mask_assistant
        # Build a deterministic SID-to-name mapping. Accepted collisions overwrite
        # in sorted disease-ID order, matching the documented first-version policy.
        self.sid2title = {}
        for disease_id in sorted(self.indices):
            if disease_id not in self.item_feat:
                raise ValueError(f"missing disease metadata for {disease_id}")
            combined_sid = "".join(self.indices[disease_id])
            self.sid2title[combined_sid] = str(
                self.item_feat[disease_id]["title"]
            ).strip()
        self.get_inputs()
    
    def __len__(self):
        return len(self.data)
    
    def get_history(self, row, idx=0):
        history = _ordered_ehr_history(row, idx, self.seed)
        target_sid = str(row["item_sid"]).strip()
        history["target_sid"] = target_sid
        history["target_title"] = self.sid2title.get(
            target_sid, str(row["item_title"]).strip()
        )
        return history
    
    def pre(self, idx):
        history_data = self.get_history(self.data.iloc[idx], idx)
        target = history_data['target_title']
        assistant_response = target if not self.test else ""

        def build_messages(sid_visits):
            history_text = serialize_sid_history(sid_visits)
            prompt = (
                "The patient's chronological diagnosis history is:\n"
                f"{history_text}\n"
                "Return one possible disease description for the next visit."
            )
            messages = [
                {
                    "role": "system",
                    "content": "Predict one possible diagnosis in the next visit from the chronological SID history.",
                },
                {"role": "user", "content": prompt},
            ]
            if not self.test:
                messages.append({"role": "assistant", "content": assistant_response})
            return messages

        _, messages, tokenized = _fit_ehr_history_to_context(
            history_data["sid_visits"],
            build_messages,
            self.tokenizer,
            self.max_len,
            self.test,
            "T3/EHRFusionSeqRecDataset",
            idx,
        )
        return _encode_ehr_sample(
            self.tokenizer,
            messages,
            assistant_response,
            tokenized,
            self.max_len,
            self.mask_assistant,
            self.test,
            "T3/EHRFusionSeqRecDataset",
            idx,
        )
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.pre(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs
    
    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []
    
    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)


class EHRTitleHistory2SidSFTDataset(Dataset):
    def __init__(
        self,
        train_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
        mask_assistant: bool = True,
    ):
        """Disease-name visit history to target SID dataset (T6)."""
        _validate_ehr_dedup(dedup)
        # Load sequence data
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.seed = seed
        self.dedup = False
        self.mask_assistant = mask_assistant
        
        # Build item_id to semantic ID mapping
        self.id2sid = {}
        for disease_id in sorted(self.indices):
            sid_tokens = self.indices[disease_id]
            if not isinstance(sid_tokens, list) or not sid_tokens:
                raise ValueError(f"invalid SID token list for {disease_id}")
            self.id2sid[disease_id] = "".join(sid_tokens)
        
        self.get_inputs()
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt(self, data_point):
        return f"""{data_point["input"]}"""
    
    def get_history(self, row, idx=0):
        history = _ordered_ehr_history(row, idx, self.seed)
        target_disease_id = str(row["item_id"])
        if target_disease_id not in self.id2sid:
            raise ValueError(f"target disease {target_disease_id} has no SID")
        target_sid = self.id2sid[target_disease_id]
        csv_target_sid = str(row["item_sid"]).strip()
        if target_sid != csv_target_sid:
            raise ValueError(
                f"target SID mismatch for {target_disease_id}: index={target_sid}, "
                f"csv={csv_target_sid}"
            )
        history["output"] = target_sid
        history["target_sid"] = target_sid
        return history
    
    def pre(self, idx):
        history_data = self.get_history(self.data.iloc[idx], idx)
        target_output = history_data['output']
        assistant_response = target_output if not self.test else ""

        def build_messages(text_visits):
            history_text = serialize_text_history(text_visits)
            prompt = (
                "The patient's chronological diagnosis history is:\n"
                f"{history_text}\n"
                "Return exactly one disease SID."
            )
            messages = [
                {
                    "role": "system",
                    "content": "Predict one possible diagnosis SID in the next visit from the chronological diagnosis history.",
                },
                {"role": "user", "content": prompt},
            ]
            if not self.test:
                messages.append({"role": "assistant", "content": assistant_response})
            return messages

        _, messages, tokenized = _fit_ehr_history_to_context(
            history_data["disease_text_visits"],
            build_messages,
            self.tokenizer,
            self.max_len,
            self.test,
            "T6/EHRTitleHistory2SidSFTDataset",
            idx,
        )
        return _encode_ehr_sample(
            self.tokenizer,
            messages,
            assistant_response,
            tokenized,
            self.max_len,
            self.mask_assistant,
            self.test,
            "T6/EHRTitleHistory2SidSFTDataset",
            idx,
        )
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.pre(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i], i))
        return temp
    
    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []

    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        result = self.pre(idx)
        return result if result is not None else {"input_ids": [], "attention_mask": [], "labels": []}


class SidTextInterleaveDataset(Dataset):
    def __init__(
        self,
        train_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        seed=0,
        category="",
        dedup=False,
    ):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.category = category
        self.dedup = dedup

        self.get_inputs()

    def __len__(self):
        return len(self.inputs)

    def get_history(self, row):
        history_item_sid = eval(row["history_item_sid"])
        history_item_title = eval(row["history_item_title"])

        history_segments = []
        for sid, title in zip(history_item_sid, history_item_title):
            history_segments.append(f"{sid}, its title is \"{title}\"")

        history_text = "; ".join(history_segments)

        category_label = self.category if self.category else "items"
        input_text = f"The user has interacted with: {history_text}."

        target_sid = str(row.get("item_sid", ""))
        target_title = str(row["item_title"])

        # Next token in LM training is part of same sequence
        if target_sid:
            target_part = f" The user is interested in: {target_sid}"
        else:
            target_part = f' The user is interested in: "{target_title}"'

        return input_text + target_part

    def pre(self, idx):
        row = self.data.iloc[idx]

        full_text = self.get_history(row)

        # Tokenize as plain text
        tokenized = self.tokenizer.encode(
            full_text,
            add_special_tokens=False
        )

        # Truncate
        if len(tokenized) > self.max_len:
            tokenized = tokenized[-self.max_len:]

        # LM target = shift left by 1
        labels = tokenized.copy()

        attention_mask = [1] * len(tokenized)

        return {
            "input_ids": tokenized,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_inputs(self):
        self.inputs = []
        for i in tqdm(range(len(self.data))):
            res = self.pre(i)
            if res is not None:
                self.inputs.append(res)

    def __getitem__(self, idx):
        return self.inputs[idx]



# This class reads LLM-generated coherent data mixing sid and natural language.
class SidTextInterleaveDataset_v2(Dataset):
    def __init__(
        self,
        json_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        seed=0,
    ):
        self.json_data = json.load(open(json_file, 'r'))
        random.seed(seed)

        if sample > 0:
            self.json_data = self.json_data.sample(sample, random_state=seed)

        self.data = {}
        real_id = 0
        for item_idx, item_meta in self.json_data.items():
            if "llm_stage2" in item_meta:
                self.data[real_id] = item_meta['llm_stage2']
                real_id += 1
    
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_inputs()


    def __len__(self):
        return len(self.data)

    def pre(self, idx):
        item_desc = self.data[idx]
        # Truncate <think> ... <\think> parts if exists
        if "</think>" in item_desc:
            item_desc = item_desc.split("</think>")[-1].strip()

        # Tokenize as plain text
        tokenized = self.tokenizer.encode(
            item_desc,
            add_special_tokens=False
        )
        # Truncate
        if len(tokenized) > self.max_len:
            tokenized = tokenized[-self.max_len:]
            print(f"Truncated sequence at idx {idx} to max_len {self.max_len}. Original length was {len(tokenized)}.")

        # LM target = shift left by 1
        labels = tokenized.copy()
        attention_mask = [1] * len(tokenized)
        return {
            "input_ids": tokenized,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_inputs(self):
        self.inputs = []
        for i in tqdm(range(len(self.data))):
            res = self.pre(i)
            if res is not None:
                self.inputs.append(res)

    def __getitem__(self, idx):
        return self.inputs[idx]





# This class reads LLM-generated coherent data mixing sid and natural language.
class SidTextInterleaveSequenceDataset(Dataset):
    def __init__(
        self,
        csv_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        seed=0,
    ):
        self.csv_data = pd.read_csv(csv_file)
        random.seed(seed)

        if sample > 0:
            self.csv_data = self.csv_data.sample(sample, random_state=seed)

        self.data = self.csv_data['integrated_narrative'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_inputs()


    def __len__(self):
        return len(self.data)

    def pre(self, idx):
        item_desc = self.data[idx]
        if item_desc is None:
            return None
        
        if "</think>" in item_desc:
            item_desc = item_desc.split("</think>")[-1].strip()

        # Tokenize as plain text
        tokenized = self.tokenizer.encode(
            item_desc,
            add_special_tokens=False
        )
        # Truncate
        if len(tokenized) > self.max_len:
            tokenized = tokenized[-self.max_len:]
            print(f"Truncated sequence at idx {idx} to max_len {self.max_len}. Original length was {len(tokenized)}.")

        # LM target = shift left by 1
        labels = tokenized.copy()
        attention_mask = [1] * len(tokenized)
        return {
            "input_ids": tokenized,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_inputs(self):
        self.inputs = []
        for i in tqdm(range(len(self.data))):
            res = self.pre(i)
            if res is not None:
                self.inputs.append(res)

    def __getitem__(self, idx):
        return self.inputs[idx]




# This dataset is used for reasoning activation task.
# The learning objective is to generate reasoning and answer given user history.
class ReasoningActivationDataset(Dataset):
    def __init__(
        self,
        reasoning_train_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
    ):
        """
        Fusion dataset combining sequence recommendation with item features.
        Uses semantic IDs for user history, outputs item titles or descriptions.
        
        Args:
            train_file: Path to CSV file with sequence data
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)
        
        # Load sequence data
        self.data = pd.read_csv(reasoning_train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        # Build sid2title and sid2description mappings
        self.sid2title = {}
        self.sid2description = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                description = self.item_feat[item_id]['description']
                
                processed_description = self._process_description(description, title)
                
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.sid2description[combined_sid] = processed_description
        
        self.get_inputs()
    
    
    def _process_description(self, description, title):
        """
        Process description according to the requirements:
        1. If description is empty, use title
        2. If description is a list, select the longest one
        3. If the longest in list is also empty, use title
        
        Args:
            description: The description field from item_feat
            title: The title field from item_feat
        
        Returns:
            str: Processed description
        """
        # Check if description is empty or None
        if not description or description == '':
            return title
        
        # Check if description is a list (either actual list or string representation)
        if isinstance(description, list):
            # It's already a list
            desc_list = description
        elif isinstance(description, str) and description.startswith('[') and description.endswith(']'):
            try:
                # Try to parse string representation of list
                desc_list = eval(description)
            except:
                # If parsing fails, treat as regular string
                return description if description.strip() else title
        else:
            # Regular string description
            return description if description.strip() else title
        
        # If we have a list, find the longest non-empty item
        if desc_list:
            # Filter out empty strings and find the longest
            non_empty_descriptions = [desc for desc in desc_list if desc and desc.strip()]
            if non_empty_descriptions:
                # Return the longest description
                longest_desc = max(non_empty_descriptions, key=len)
                return longest_desc
            else:
                # All descriptions in list are empty, use title
                return title
        else:
            # Empty list, use title
            return title
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt_title(self, history):
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Let's think step by step before making recommendation. Directly output the item SID after thinking."
    
    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        history_str = ", ".join(history_item_sid)
        
        target_sid = row['item_sid']
        reasoning = row['reasoning_path']
        # return None if reasoning is empty or nan
        if pd.isna(reasoning) or reasoning.strip() == "":
            return None


        # if reasoning.strip() start with <think>, then we need to remove content between <think> and </think>
        if reasoning.strip().startswith("<think>"):
            if "</think>" in reasoning:
                reasoning = reasoning.split("</think>")[-1].strip()
            else:
                return None
        
        # Use the new sid2title and sid2description mappings
        if target_sid in self.sid2title:
            target_title = self.sid2title[target_sid]
        else:
            target_title = target_sid
        
        # Check for deduplication
        last_history_sid = history_item_sid[-1] if history_item_sid else None
        is_duplicate = target_sid == last_history_sid
        
        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_sid": target_sid,
            "dedup": is_duplicate,
            "reasoning": reasoning,
        }
    
    def generate_formatted_prompt(self, prompt, response):
        return f"""{prompt}"""
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Can you recommend the next item for the user based on their interaction history?
"""  
        # tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history_data = self.get_history(self.data.iloc[idx])
        if history_data is None:
            return None
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        # Randomly choose between title and description tasks
        prompt = self.generate_prompt_title(history_data['history_str'])
        target = history_data['target_sid']
        # print("fusion prompt: ", prompt)

        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        assistant_response = f"<think>\n{history_data['reasoning'].strip()}\n</think>\n\n{target}"

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": formatted_prompt},
        ]
        messages.append({"role": "assistant", "content": assistant_response})

        input_ids, attention_mask, labels = mask_assistant_response_only(
            tokenizer=self.tokenizer,
            messages=messages,
            assistant_response=assistant_response,
            max_len=self.max_len,
            mask_eos=False,
        )


        if len(input_ids) > self.max_len:
            print(f"Sequence length {len(input_ids)} exceeds max_len {self.max_len}")
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.pre(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs
    
    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []
    
    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)




# This dataset is used for reasoning recommendation task.
# The learning objective is to generate reasoning then answer given user history.
class Reasoning_Eval_Dataset(Dataset):
    def __init__(
        self,
        data_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
    ):
        """
        Fusion dataset combining sequence recommendation with item features.
        Uses semantic IDs for user history, outputs item titles or descriptions.
        
        Args:
            train_file: Path to CSV file with sequence data
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)
        
        # Load sequence data
        self.data = pd.read_csv(data_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        # Build sid2title and sid2description mappings
        self.sid2title = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']                                
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
        
        self.get_inputs()
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt_title(self, history):
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Let's think step by step before making recommendation. Directly output the item SID after thinking."
    
    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        history_str = ", ".join(history_item_sid)
        
        target_sid = row['item_sid']
        
        # Use the new sid2title and sid2description mappings
        if target_sid in self.sid2title:
            target_title = self.sid2title[target_sid]
        else:
            target_title = target_sid
        
        # Check for deduplication
        last_history_sid = history_item_sid[-1] if history_item_sid else None
        is_duplicate = target_sid == last_history_sid
        
        return {
            "history_str": history_str,
            "target_title": target_title,
            "output": target_sid,
            "dedup": is_duplicate,
        }
    
    def generate_formatted_prompt(self, prompt, response):
        return f"""{prompt}"""
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Can you recommend the next item for the user based on their interaction history?
"""  
        # tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        # Randomly choose between title and description tasks
        prompt = self.generate_prompt_title(history_data['history_str'])
        target = history_data['output']
        # print("fusion prompt: ", prompt)

        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        assistant_response = f"{target}"

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": formatted_prompt},
        ]

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if self.test else False,
            tokenize=True,
            return_tensors=None,
        )
        attention_mask = [1] * len(tokenized)

        if len(tokenized) >= self.max_len:
            print(len(tokenized))
            tokenized = tokenized[self.max_len:]
            attention_mask = attention_mask[self.max_len:]
        return {
            "target": target,
            "input_ids": tokenized,
            "attention_mask": attention_mask,
        }

    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.pre(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs
    
    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)



class GeneralSFTReasonDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        self.data = []
        with open(train_file, 'r') as f:
            for line in f:
                data_sample = json.loads(line.strip())
                self.data.append(eval(data_sample["messages"]))
        random.seed(seed)
        if sample > 0:
            self.data = random.sample(self.data, sample)
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.cnt = 0
        self.get_inputs()  
        
    
    def __len__(self):
        return len(self.inputs)

    
    def pre(self, idx):
        prompt_messages = []
        for message in self.data[idx]:
            # print(f"message: {message['content']}")
            # message["content"] = eval(message["content"])
            if message["role"] == "user":
                if type(message["content"]) == list:
                    prompt_messages.append({"role": "user", "content": message["content"][0]["text"]})
                else:
                    prompt_messages.append({"role": "user", "content": message["content"]})
            elif message["role"] == "system":
                if type(message["content"]) == list:
                    prompt_messages.append({"role": "user", "content": message["content"][0]["text"]})
                else:
                    prompt_messages.append({"role": "system", "content": message["content"]})
        try:
            processed_template = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Error processing messages: {self.data[idx]}")
            print(f"Error processing idx {idx}: {e}")
            raise e
        try:
            input_ids = self.tokenizer.encode(processed_template)
        except Exception as e:
            return None
        
        if idx == 0:
            print(f"General data example: {[processed_template]}")
        
        attention_mask = [1] * len(input_ids)

        if self.test:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        golden_output = ""
        for elm in self.data[idx]:
            if elm["role"] == "assistant":
                if type(elm["content"]) == list:
                    golden_output = elm["content"][0]["text"]
                else:
                    golden_output = elm["content"]

        if golden_output == "":
            print(f"No assistant response found in idx {idx}")
            return None
        
        if idx == 0:
            print(f"General data golden output example: {[golden_output]}")
        # golden_output = sample["messages"]
        try:
            golden_tokens = self.tokenizer.encode(golden_output)
        except Exception as e:
            return None
        
        golden_tokens = golden_tokens + [self.tokenizer.eos_token_id]
        
        input_prompt_len = len(input_ids)
        input_ids = input_ids + golden_tokens
        attention_mask = [1] * len(input_ids)
        labels = [-100] * input_prompt_len + input_ids[input_prompt_len:]
        
        # if len(input_ids) >= self.max_len:
        #     print(len(input_ids))

        return {
            "input_ids": input_ids[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            if self.pre(i) is not None:
                inputs.append(self.pre(i))
        self.inputs = inputs
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]
