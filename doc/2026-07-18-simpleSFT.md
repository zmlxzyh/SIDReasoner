# 生成式 EHR 预测：Simple SFT 方案

> 更新日期：2026-07-21（并入 [`2026-07-21-sft-solution.md`](./2026-07-21-sft-solution.md) 的 11 条决策）
> 本文描述不使用富语料、不生成 reasoning、也不进行 RL 的生成式 EHR 最小完整闭环。  
> 具体问题的决策与理由见 [`2026-07-21-sft-solution.md`](./2026-07-21-sft-solution.md)；本文正文已按该决策更新。
> 代码以 SIDReasoner 的 Stage 1 为主体，保留 Case 1–6 的基础 SID 对齐与预测任务，删除富语料和推理任务；评测阶段使用合法 SID 约束的 beam search 生成疾病 rank-list，并计算 Recall@K 和 weighted-F1。

## 1. 方案范围

### 1.1 目标

给定患者在当前 visit 之前的 ICD 诊断历史：

```text
历史 visit 1 → 历史 visit 2 → ... → 历史 visit T-1
```

预测当前 visit 的完整 ICD 疾病集合：

```text
GT = {disease A, disease B, disease C, ...}
```

每个疾病先映射为固定长度 Semantic ID（SID）：

```text
ICD9CM:4280 → <a_17><b_93><c_4>
```

训练时，一个当前 visit 的多疾病集合拆成多条单目标 completion：

```text
同一历史 → SID_A
同一历史 → SID_B
同一历史 → SID_C
```

推理时不逐个疾病独立判断，而是对同一个 visit prompt 运行一次受约束 beam search，得到按生成分数排序的疾病 SID rank-list，再与完整 GT 集合比较。

### 1.2 本版本包含与不包含的内容

包含：

- ICD 疾病文本与 SID 的双向对齐。
- 历史疾病文本/SID到目标疾病文本/SID的多任务 SFT。
- SID token 扩词表和全参数微调。
- 合法 SID trie 与受约束 beam search。
- visit-level rank-list、Recall@K 和 weighted-F1。

不包含：

- 疾病级 LLM 富语料。
- 患者级病程叙事或推理语料。
- `<think>...</think>` reasoning 监督。
- Stage 2 reasoning activation。
- VERL/TRL、GRPO、reward 或其他 RL。
- procedures、drugs 和时间模态；第一版只使用 ICD 诊断历史。

### 1.3 与 SIDReasoner 的关系

本方案相当于只保留 SIDReasoner Stage 1 中的基础 Case 1–6：

| SIDReasoner 数据集 | 医疗任务 | 输入 → 输出 | 是否保留 |
|---|---|---|---|
| `SidItemFeatDataset` | T1 疾病文本 → SID | disease text → SID | 是 |
| `SidItemFeatDataset` | T2 SID → 疾病文本 | SID → disease text | 是 |
| `FusionSeqRecDataset` | T3 SID 历史 → 疾病文本 | SID history → target text | 是 |
| `SFTData` | T4 文本历史 → 疾病文本 | text history → target text | 是 |
| `SidSFTDataset` | T5 SID 历史 → SID | SID history → target SID | **是，主任务** |
| `TitleHistory2SidSFTDataset` | T6 文本历史 → SID | text history → target SID | 是 |
| `SidTextInterleaveDataset_v2` | 疾病/商品富语料 | rich text LM | 否 |
| `SidTextInterleaveSequenceDataset` | 患者/用户叙事 | narrative LM | 否 |
| `GeneralSFTReasonDataset` | 通用推理保持 | general reasoning | 否 |

后续如需 reasoning 和 RL，应在这个 Simple SFT 闭环通过后再接入，不应让富语料生成阻塞基础 SID 预测验证。

## 2. 整体流水线

```text
MIMIC-III / MIMIC-IV pkl + 原始 split
  │
  ├─ ① EHR 预处理
  │    visit-level 历史 + 当前完整 ICD 集合
  │    code-level 单目标拆分记录
  │
  ├─ ② 疾病语义与 SID
  │    disease.item.json
  │    disease.index.json
  │    disease.info.tsv
  │
  ├─ ③ Simple Multi-task SFT
  │    T1–T6，不含富语料/推理/RL
  │
  └─ ④ Evaluation
       一 visit 一 prompt
       constrained beam search → SID rank-list
       SID → disease_id / ICD
       Recall@K + oracle-cardinality weighted-F1
```

## 3. 开始 SFT 前应已具备的数据

Simple SFT 不负责从原始 MIMIC 表重新构造患者记录，也不负责训练 RQ-VAE/RQ-KMeans。开始本阶段前，应该已经获得以下稳定产物。

实际目录（以 `mimic3_icd_name_path_0.1` 为准；决策见 solution 文档问题 9）：

```text
SIDReasoner/data/EHR/mimic3_icd_name_path_0.1/     # <data_dir>：唯一需要切换的顶层版本目录
├── manifest/
│   ├── disease_manifest.json
│   ├── preprocessing_manifest.json
│   └── validation_report.json                     # 数据自检报告（非 tokenizer 校验）
├── index/
│   ├── mimic3_icd.item.json                       # 内部文件前缀固定 mimic3_icd（不含 _name_path_0.1）
│   └── mimic3_icd.index.json
├── info/
│   └── mimic3_icd.info.tsv
├── code_level/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── visit_level/
    ├── train.jsonl
    ├── valid.jsonl
    └── test.jsonl
```

不同规模版本（如全量 `mimic3_icd_name_path`）**内部结构与文件命名完全一致，仅规模不同**：切换只改顶层 `data_dir`，其余路径全部派生（`{data_dir}/index/{prefix}.index.json` 等），`prefix` 默认 `mimic3_icd`。MIMIC-IV 使用相同结构、`prefix` 改为 `mimic4_icd`；第一轮实验建议分别训练 MIMIC-III 和 MIMIC-IV，不要直接混合。`tokenizer` 相关校验不在此目录，改为训练启动 preflight（见 §12.2）。

### 3.1 数据验收条件

开始 SFT 前必须满足：

1. 沿用原始 train/valid/test 患者级 split，患者集合互斥。
2. 输入只取 `cond_hist[:-1]`，当前目标取 `icd9_code`（MIMIC-III）或 `icd_code`（MIMIC-IV）。
3. 每个 visit 的目标 ICD 已去重。
4. 每个疾病具有稳定 `disease_id`；MIMIC-IV 使用 `system:code_norm`。
5. 历史与目标疾病并集全部拥有疾病文本和 SID。
6. `disease_id ↔ SID` 一一对应且无完整 SID 碰撞。
7. 所有 SID token、`<visit_start>` 和 `<visit_end>` 加入 tokenizer 后都是原子 token。
8. code-level 行可以按 `sample_id` 无损回聚为原 visit 的完整 GT 集合。
9. patient ID、visit ID 和绝对时间不进入 prompt；内部 `sample_id` 不包含患者身份信息。
10. 所有列表/嵌套列表使用 JSON 编码，读取时使用 `json.loads`，不使用 `eval`。

数据规模与 MIMIC 字段操作见 [`2026-07-18-data.md`](./2026-07-18-data.md)。

## 4. 疾病元数据：`*.item.json`

MiniOneRec 的：

```text
Industrial_and_Scientific.item.json
```

以 `item_id` 为键，保存商品 `title`、`description`、`brand` 和 `categories`。医疗版保持相同的“ID → 语义元数据”结构，但键改为稳定疾病 ID。

推荐的 `mimic3_icd.item.json`：

```json
{
  "ICD9CM:4280": {
    "title": "Congestive heart failure, unspecified",
    "description": "Congestive heart failure, unspecified",
    "system": "ICD9CM",
    "code": "428.0",
    "code_norm": "4280",
    "parent_code": "428",
    "parent_code_norm": "428",
    "categories": [
      "Diseases of the circulatory system",
      "Heart failure"
    ]
  }
}
```

字段约定：

- JSON key 和 `disease_id` 均为 `system:code_norm`。
- `title` 是标准疾病英文名称，用于直接复用 SIDReasoner 的 `title` 相关数据集。
- `description` 在 Simple SFT 中不使用 LLM 扩写。第一版可以等于 `title`，或由名称、父类和 ontology path 按固定模板拼接。
- `categories` 可以保存可追溯的 ontology path，但不能混入训练集频次、目标共现或其他标签统计。
- MIMIC-III 也建议使用 `ICD9CM:` 前缀，避免未来与 MIMIC-IV 合并时冲突。

不建议只使用 `4280` 作为全局疾病 ID；更不能使用疾病名称作为主键，因为名称可能重复或发生版本变化。

## 5. 疾病 SID：`*.index.json`

MiniOneRec 的：

```text
Industrial_and_Scientific.index.json
```

格式是：

```json
{
  "0": ["<a_236>", "<b_231>", "<c_226>"],
  "1": ["<a_42>", "<b_80>", "<c_160>"]
}
```

医疗版保持完全相同的值结构：

```json
{
  "ICD9CM:4280": ["<a_17>", "<b_93>", "<c_4>"],
  "ICD9CM:4019": ["<a_8>", "<b_21>", "<c_77>"]
}
```

读取后可构造：

```python
import json

with open("mimic3_icd.index.json", encoding="utf-8") as handle:
    disease_to_sid_tokens = json.load(handle)

disease_to_sid = {
    disease_id: "".join(tokens)
    for disease_id, tokens in disease_to_sid_tokens.items()
}
# 决策（solution 问题 1）：接受少量完整 SID 碰撞，不因此中断。
# 按 disease_id 排序确定化覆盖，保证每次运行“保留者/丢弃者”一致、可复现。
sid_to_disease = {}
for disease_id in sorted(disease_to_sid):
    sid_to_disease[disease_to_sid[disease_id]] = disease_id

n_collision = len(disease_to_sid) - len(sid_to_disease)
if n_collision:
    print(f"[warn] full SID collision: {n_collision} disease(s) unrecoverable in eval")
```

> 第一版接受完整 SID 碰撞（实测 0.1 为 2 组 / 4 病）与 title 重复（14 组 / 28 病），不去碰撞、不改 `item.json`；理由与护栏见 solution 文档问题 1。这里**不再硬 assert 唯一**，改为确定化覆盖 + 告警，并在 `evaluation_manifest.json` 记录不可恢复的 GT 记录数。

### 5.1 SID 覆盖与碰撞检查

```python
def validate_sid_index(item_meta, sid_index, expected_disease_ids):
    assert set(item_meta) == set(sid_index)
    assert set(sid_index) == set(expected_disease_ids)

    # 完整 SID 唯一性：第一版接受碰撞，仅告警计数，不 assert（solution 问题 1）
    full_sids = ["".join(tokens) for tokens in sid_index.values()]
    n_collision = len(full_sids) - len(set(full_sids))
    if n_collision:
        print(f"[warn] {n_collision} full-SID collision(s) accepted")

    layer_count = {len(tokens) for tokens in sid_index.values()}
    assert len(layer_count) == 1

    for tokens in sid_index.values():
        assert all(token.startswith("<") and token.endswith(">") for token in tokens)
```

RQ-SID 的层级是量化路径，不应默认解释为 ICD 本体层级。第一版**接受少量完整 SID 碰撞**（不用作医学相似度或 reward）。

## 6. 合法 SID 列表：`*.info.tsv`

MiniOneRec 的 info 文件每行格式为：

```text
<full_sid>\t<title>\t<item_id>
```

医疗版示例：

```text
<a_17><b_93><c_4>\tCongestive heart failure, unspecified\tICD9CM:4280
<a_8><b_21><c_77>\tEssential hypertension, unspecified\tICD9CM:4019
```

该文件可直接供 `evaluate_Qwen3.py` 风格的前缀树构造逻辑读取。不过推荐把 `index.json` 作为唯一权威映射，`info.tsv` 由 `item.json + index.json` 确定性生成，避免三份文件内容漂移。

## 7. visit-level 数据

visit-level 数据是一条患者历史对应一个完整当前疾病集合，是 RL 之外也必须保留的基础数据。评测只能使用这一粒度。

推荐 JSONL 记录：

```json
{
  "sample_id": "mimic3:test:000001",
  "split": "test",
  "history_disease_id_visits": [
    ["ICD9CM:4019", "ICD9CM:25000"],
    ["ICD9CM:41401"]
  ],
  "history_disease_text_visits": [
    ["Essential hypertension, unspecified", "Diabetes mellitus without complication"],
    ["Coronary atherosclerosis of native coronary artery"]
  ],
  "history_sid_visits": [
    ["<a_8><b_21><c_77>", "<a_61><b_2><c_18>"],
    ["<a_14><b_62><c_9>"]
  ],
  "ground_truth_disease_ids": [
    "ICD9CM:4280",
    "ICD9CM:42731"
  ],
  "ground_truth_sids": [
    "<a_17><b_93><c_4>",
    "<a_14><b_90><c_31>"
  ]
}
```

`sample_id` 是内部随机或顺序标识，不应编码原始 `patient_id` 或 `visit_id`。

## 8. code-level SFT CSV

### 8.1 为什么需要拆成单目标行

SIDReasoner/MiniOneRec 的生成目标是一个 SID。若当前 visit 的 GT 为：

```text
[SID_A, SID_B, SID_C]
```

则构造三行：

```text
相同历史 → SID_A
相同历史 → SID_B
相同历史 → SID_C
```

所有行共享 `sample_id` 和完整 `ground_truth_sids`。第一版采用“全部拆分”，不做每 epoch 单目标随机采样。

### 8.2 推荐字段

与 `Industrial_and_Scientific_5_2016-10-2018-11.csv` 对应的兼容字段：

| Amazon 字段 | 医疗含义 |
|---|---|
| `user_id` | `sample_id`；不是患者真实 ID |
| `history_item_title` | 历史疾病文本 |
| `item_title` | 当前子目标疾病文本 |
| `history_item_id` | 历史 `disease_id` |
| `item_id` | 当前子目标 `disease_id` |
| `history_item_sid` | 历史疾病 SID |
| `item_sid` | 当前子目标完整 SID |

EHR 还必须增加以下字段：

| 字段 | 作用 |
|---|---|
| `history_disease_id_visits` | 保留嵌套 visit 边界的疾病 ID |
| `history_disease_text_visits` | 保留嵌套 visit 边界的疾病文本 |
| `history_sid_visits` | 保留嵌套 visit 边界的 SID |
| `ground_truth_disease_ids` | 当前 visit 完整疾病集合 |
| `ground_truth_sids` | 当前 visit 完整 SID 集合 |
| `target_index` | 当前子目标在完整 GT 中的位置 |

CSV 表头建议：

```text
sample_id,split,history_item_title,item_title,history_item_id,item_id,history_item_sid,item_sid,history_disease_id_visits,history_disease_text_visits,history_sid_visits,ground_truth_disease_ids,ground_truth_sids,target_index
```

其中所有列表字段都是合法 JSON 字符串。概念示例：

```csv
sample_id,split,item_id,item_title,item_sid,target_index
mimic3:train:000001,train,ICD9CM:4280,"Congestive heart failure, unspecified",<a_17><b_93><c_4>,0
```

实际文件还应包含完整历史和完整 GT 字段。CSV 仅用于兼容 SIDReasoner 的现有数据入口；如果不需要与 pandas CSV 直接对接，JSONL 或 Parquet 更适合嵌套 EHR 字段。

### 8.3 兼容字段与权威字段

措辞（决策见 solution 问题 11）：EHR code-level CSV **不是**与 Amazon 完全相同的 schema，而是**保留原版五个 Dataset 所需全部消费字段的兼容超集（consumer-compatible superset）**——差异点为 `sample_id + split`（而非 `user_id`）、列表用合法 JSON（而非 Python repr）、并额外含 `history_*_visits` / `ground_truth_*` / `target_index` 等 visit-level 权威字段；无需额外复制 `user_id`。

Amazon 兼容字段中的历史通常是扁平列表，会丢失 visit 边界。医疗实现应遵循：

- `history_*_visits` 是权威字段。
- `history_item_*` 是为复用旧类生成的派生字段。
- prompt 必须由权威嵌套字段经过统一 serializer 生成。
- 不应直接把 Amazon 原版 `SidSFTDataset.get_history()` 原封不动用于 EHR。

## 9. 历史序列化

推荐 SID 历史格式：

```text
<visit_start>
<a_8><b_21><c_77>, <a_61><b_2><c_18>
<visit_end>
<visit_start>
<a_14><b_62><c_9>
<visit_end>
```

推荐文本历史格式：

```text
Visit 1: Essential hypertension, unspecified; Diabetes mellitus without complication
Visit 2: Coronary atherosclerosis of native coronary artery
```

统一规则：

1. visit 之间严格保持时间顺序。
2. 训练阶段在样本物化时对每条样本各做**一次** visit 内打乱（逐样本确定化种子），之后固定为静态数据，减少对任意 ICD 排序的过拟合；不做每 epoch 重新打乱（决策见 solution 问题 2）。由于同一 visit 历史会因多个目标疾病出现在多行、逐行独立打乱，单个 epoch 内模型已见到同一历史的多种顺序。
3. valid/test 使用确定性排序，例如按 `disease_id` 排序。
4. 超长样本从最旧 visit 开始整段删除，优先保留近期历史。
5. 不允许从一个 SID 中间截断，也不允许留下不闭合的 visit 边界。
6. T3/T5、训练、验证和评测共用同一个 SID serializer；T4/T6 共用同一个 text serializer。

建议实现两个纯函数：

```python
serialize_sid_history(history_sid_visits, training: bool, seed: int) -> str
serialize_text_history(history_text_visits, training: bool, seed: int) -> str
```

不要让每个 Dataset 类各自拼一套 prompt 历史，否则训练和评测容易发生模板漂移。

## 10. 六个基础 SFT 任务

以下 prompt 是医疗版建议模板。具体措辞可以调整，但同一任务的 train/valid 必须一致。

### 10.1 T1：疾病名称 → SID

```text
System:
You map ICD disease names to semantic identifiers.

User:
Which disease has the name "Congestive heart failure, unspecified"?

Assistant:
<a_17><b_93><c_4>
```

来源：`item.json`（取 `title` 作为 name，**不拼 ICD code**）`+ index.json`。每个疾病一条。措辞改用 “name”（决策见 solution 问题 5）；14 组重复 title 造成的“同名两 SID”歧义按问题 1 接受不处理。

### 10.2 T2：SID → 疾病名称

```text
System:
You map semantic identifiers to ICD disease names.

User:
What disease does <a_17><b_93><c_4> represent?

Assistant:
Congestive heart failure, unspecified
```

来源：`item.json + index.json`。每个疾病一条。

### 10.3 T3：SID 历史 → 目标疾病文本

```text
System:
Predict one possible diagnosis in the next visit from the chronological SID history.

User:
The patient's chronological diagnosis history is:
<visit_start> ...SID history... <visit_end>
Return one possible disease description for the next visit.

Assistant:
Congestive heart failure, unspecified
```

来源：code-level CSV。每个拆分子目标一条。

### 10.4 T4：文本历史 → 目标疾病文本

```text
System:
Predict one possible diagnosis in the next visit from the chronological diagnosis history.

User:
The patient's chronological diagnosis history is:
Visit 1: ...
Visit 2: ...
Return one possible disease description for the next visit.

Assistant:
Congestive heart failure, unspecified
```

来源：code-level CSV。每个拆分子目标一条。

### 10.5 T5：SID 历史 → 目标 SID（主任务）

```text
System:
Predict one possible diagnosis SID in the next visit from the chronological SID history.

User:
The patient's chronological diagnosis history is:
<visit_start> ...SID history... <visit_end>
Return exactly one disease SID.

Assistant:
<a_17><b_93><c_4>
```

这是最终评测任务，验证和 checkpoint 选择应重点关注该任务。

### 10.6 T6：文本历史 → 目标 SID

```text
System:
Predict one possible diagnosis SID in the next visit from the chronological diagnosis history.

User:
The patient's chronological diagnosis history is:
Visit 1: ...
Visit 2: ...
Return exactly one disease SID.

Assistant:
<a_17><b_93><c_4>
```

T6 直接连接可读疾病语义和 SID 输出空间，可以作为 T5 的重要辅助任务。

## 11. Dataset 实现方式

建议新增医疗专用数据类，而不是在原类中到处添加 `if category == EHR`：

```text
SIDReasoner/ehr_data_Qwen3.py
├── EhrSidItemFeatDataset       # T1 / T2
├── EhrSidHistoryToTextDataset  # T3
├── EhrTextHistoryToTextDataset # T4
├── EhrSidSFTDataset            # T5
├── EhrTextHistoryToSidDataset  # T6
└── EhrVisitEvalDataset         # visit-level 评测
```

这些类可以复用 `data_Qwen3.py` 中：

- `mask_assistant_response_only`；
- Qwen3 `apply_chat_template`；
- assistant-only label masking；
- `input_ids / attention_mask / labels` 返回协议。

必须修改的地方（决策见 solution 问题 6）：

- 用 `json.loads` 替换 `eval`。
- 从嵌套权威字段（`history_sid_visits` / `history_disease_text_visits`）构造历史，**不用**扁平 `history_item_*`；经统一 serializer 生成 prompt。
- **visit 级截断**：超长时整段删除最旧 visit，绝不切断 SID、不留不闭合 visit 边界；不再用 `input_ids[-max_len:]`。
- 训练集在 `pre()` 内做**一次性** visit 内打乱（问题 2，逐样本确定化种子）；valid/test 按 `disease_id` 确定排序。
- **全任务禁用 dedup**（问题 3）：EHR 慢病复发是合法目标，不移植原类“target==末位历史即丢弃”的逻辑，不启用 `dedup=True`。
- T1/T2 用 `title` 作为 name 输入，不拼 ICD code（问题 5）。
- 在 `EhrVisitEvalDataset` 中返回完整 `ground_truth_sids`，而不是单个 `item_sid`。
- 评测数据一 visit 一行，不读取 code-level 重复行。

## 12. 多任务 SFT 训练

### 12.1 基于 `sft_Qwen3.py` 的最小改造

建议新增：

```text
SIDReasoner/sft_Qwen3_simple_ehr.py
SIDReasoner/sft_Qwen3_simple_ehr.sh
```

训练入口复用 SIDReasoner 的：

- `TokenExtender`；
- Qwen3/Qwen 系列 `AutoModelForCausalLM`；
- `resize_token_embeddings`；
- `MultiEvalTrainer`；
- `DataCollatorForSeq2Seq`；
- DDP/torchrun、W&B、early stopping 和 checkpoint 保存。

只保留以下组合：

```python
train_datasets = [
    EhrSidSFTDataset(...),             # T5，主任务
    EhrSidItemFeatDataset(...),        # T1 + T2
    EhrSidHistoryToTextDataset(...),   # T3
    EhrTextHistoryToTextDataset(...),  # T4
    EhrTextHistoryToSidDataset(...),   # T6
]
train_data = ConcatDataset(train_datasets)
```

明确不传：

```text
llm_generated_data_path
llm_generated_sequence_path
general_reasoning_path
reasoning_train_file
```

也不运行：

```text
sft_reasoning_activation.py
RL_training_script.sh
```

### 12.2 词表扩展

从 `mimic3_icd.index.json` 收集所有分层 SID token，再加 visit token：

```python
sid_tokens = sorted({
    token
    for tokens in sid_index.values()
    for token in tokens
})
special_tokens = sid_tokens + ["<visit_start>", "<visit_end>"]

tokens_to_add = [
    token for token in special_tokens
    if token not in tokenizer.get_vocab()
]
tokenizer.add_tokens(tokens_to_add)
model.resize_token_embeddings(len(tokenizer))
```

保存 tokenizer 后必须重新加载并检查（作为**训练启动 preflight artifact**，决策见 solution 问题 8）：

```python
tokenizer.save_pretrained(out_dir)
tk = AutoTokenizer.from_pretrained(out_dir)
for token in special_tokens:                       # 768 SID + 2 visit = 770 个
    assert len(tk.encode(token, add_special_tokens=False)) == 1
```

preflight 在扩词表 + `resize_token_embeddings` 之后、`trainer.train()` 之前运行，四项检查任一不过即**报错终止**，结果落盘 `manifest/tokenizer_preflight.json`：

1. 768 个 SID token 和两个 visit token 均为单 token。
2. label 只覆盖 assistant response 本身（`mask_eos=True`），**不监督** assistant 结束后缀。
3. `mask_eos=True` 后，answer 之后的 chat template 结束后缀（如 `<|im_end|>`、EOS）确实**全部为 `-100`**，既没有被监督、也不含下一轮模板内容。
4. 保存并重载 tokenizer 后结果不变。

其中 2/3 用一条样本 decode 可视化确认：labels 中非 `-100` 段解码后正好是回答本身（**不含**结束符），其后的结束后缀全部为 `-100`。

第一版建议像 SIDReasoner 一样进行全参数 SFT，而不是只训练新增 token embedding。SID 历史预测不仅需要学习新 token，还需要 attention 和 FFN 适应纵向 EHR 序列模式。

### 12.3 loss 和 masking

- 所有任务只对 assistant response 计算交叉熵。
- system/user prompt 的 label 全部为 `-100`。
- **T1–T6 统一 `mask_eos=True`（沿用原始 codebase 默认），不监督 assistant 结束后缀（结束标记 / EOS）**；即目标只覆盖疾病名称或完整 SID 本身，回答之后的 chat template 后缀不参与 loss。
- 不允许将 ground-truth 完整集合放进 system/user prompt。
- valid/test 不做 visit 内随机打乱。

当前 `data_Qwen3.py` 的 `mask_assistant_response_only` 默认 `mask_eos=True`，该分支只保留 response 本身并遮掉 chat template 后缀。**Simple SFT 直接沿用这一默认**（`ehr_data_Qwen3.py` 的 `_encode_ehr_sample` 固定 `mask_eos=True`），保持与原 SIDReasoner 一致、便于对齐复现。

由此带来的权衡（明确接受）：模型不会通过监督学到“在答案后主动输出 EOS 停止”。这对本版本可接受，因为——

- **T5（主任务、唯一评测任务）** 通过受约束 beam search + SID trie 在完整 SID 后强制 EOS（§13.3），不依赖模型自行学会停止；
- **T1–T4/T6** 的自由文本输出在 Simple SFT 阶段不进入 beam 评测，只作辅助 loss，不需要精确的停止行为。

后续若接入富语料 / reasoning（需要模型自主生成并停止），再对相应任务显式改用 `mask_eos=False`（参考 `ReasoningActivationDataset` 已用 `mask_eos=False`），不影响本版本决策。

### 12.4 多任务采样

**第一版决策（solution 问题 4）**：这是小流量测试，以跑通代码为主，直接用 `ConcatDataset` 让数据集自然大小决定占比，**不做**加权采样 / 行复制。注意：0.1 保留全量 item 但 code-level 只有约 10%，自然比例下 T1/T2 占比偏大（各约 11%），因此 0.1 训练曲线只验证代码正确、不能外推全量。下表的显式比例**推迟到正式实验**再引入（届时 EHR code-level 远大于翻译数据、MIMIC-IV 展开超 300 万行，需惰性读取或在线采样，不宜物化多份 list）：

| 任务 | 初始采样比例 |
|---|---:|
| T5 SID 历史 → SID | 50% |
| T1 文本 → SID | 10% |
| T2 SID → 文本 | 10% |
| T3 SID 历史 → 文本 | 10% |
| T4 文本历史 → 文本 | 10% |
| T6 文本历史 → SID | 10% |

这是工程初始值，不是最终结论。至少应做：

- T5-only；
- T1+T2+T5；
- T1–T6 full simple SFT。

三个消融，确认辅助任务是否真正提高主任务 rank-list 指标。

对于 MIMIC-IV，Dataset 应按行惰性读取或按 task 在线采样，不应像当前 `sft_Qwen3.py` 那样先把全部样本转换成多份 Python list，再整体 `HFDataset.from_dict`。

### 12.5 验证集与 checkpoint

继续使用 `MultiEvalTrainer`：

- 主验证集（held-out）：T5 `SID history → SID`，读 code-level valid（验证集病人）。
- 额外监控集：T1 `name → SID`、T2 `SID → name`。

注意（决策见 solution 问题 7）：T1/T2 额外集读的是**与训练同一份** `item.json`/`index.json`，是**映射重建/记忆**，非泛化验证；因此其 metric key 命名为 `alignment_reconstruction/title2sid_loss` 与 `alignment_reconstruction/sid2title_loss`，**不参与** early stopping / checkpoint 选择（`MultiEvalTrainer` 评测额外集时已临时禁用 EarlyStoppingCallback）。

early stopping / 选最优 checkpoint **只看 T5 held-out `eval_loss`**；正式实验的最佳 checkpoint 应额外运行 valid visit-level beam search，按 `Recall@10` 或预先确定的主指标选择，避免 token-level loss 与集合排序能力不一致。

### 12.6 初始训练配置

可参考 SIDReasoner Stage 1：

```text
base model                  Qwen3-1.7B 或实验指定模型
precision                   bfloat16
optimizer                   adamw_torch
learning rate               1e-5 ～ 3e-4，按全参/LoRA和 batch 调整
assistant-only mask         true
train_new_embeddings_only   false
group_by_length             false（第一轮）
max sequence length         1024 起步，按历史覆盖率调整
save/eval strategy          epoch
early stopping patience     3
```

正式训练前先运行 32–128 条样本 overfit test，确认：

- loss 能快速下降；
- T1/T2 能互相映射；
- T5 能输出完整 SID；
- decode labels 后监督段正好是回答本身、其后的结束后缀（EOS）为 `-100`（`mask_eos=True`，不监督结束符）；评测端由 trie 在完整 SID 后强制 EOS，不依赖模型自行停止；
- decode 后没有把 prompt 纳入 labels。

## 13. Simple SFT 推理

### 13.1 评测数据必须回到 visit-level

训练 CSV 中同一个 visit 会出现多行，但评测只能生成一次：

```text
错误：同一个 visit 按 GT 疾病重复评测 N 次
正确：一个 visit → 一个 prompt → 一个 rank-list → 完整 GT 集合
```

`EhrVisitEvalDataset` 应读取 `visit_level/valid.jsonl` 或 `visit_level/test.jsonl`，每个元素返回：

```python
{
    "sample_id": str,
    "input_ids": list[int],
    "attention_mask": list[int],
    "ground_truth_sids": list[str],
    "ground_truth_disease_ids": list[str],
}
```

### 13.2 no-thinking 输出协议

本版本不生成 reasoning，assistant 最终只输出一个 SID：

```text
<a_x><b_y><c_z>
```

如果沿用 SIDReasoner/Qwen3 的 `EvalSidDataset` 模式，可以在 answer 起点注入空 thinking 区：

```text
<think>
</think>

```

然后只对其后的 SID 启用约束。更理想的实现是统一使用模型 chat template 的 no-thinking 开关，并动态定位 answer 起点；不要在代码中假设分隔符永远恰好占 3 个 token。

### 13.3 构建合法 SID trie

trie 的叶子是 `index.json` 中全部完整疾病 SID：

```text
root
├── <a_17>
│   ├── <b_93>
│   │   ├── <c_4> → EOS
│   │   └── <c_8> → EOS
│   └── <b_20> ...
└── <a_8> ...
```

生成规则：

1. answer 开始后只允许合法一级 SID token。
2. 给定已生成 prefix，只允许 index 中存在的下一层 token。
3. 完整 SID 后只允许 EOS。
4. trie 从 `index.json` 动态构建，不硬编码三层；即使第一版为三层，代码也读取实际层数。
5. tokenizer 或 checkpoint 更换后重新构建并测试。

**第一版决策（solution 问题 10）**：受约束 beam 的 `prefix_allowed_tokens_fn` / trie 与 no-think 处理**直接沿用 `evaluate_Qwen3.py` 原始代码**，先不改。原实现基于 hash + 固定 `prefix_index` 且按 3 层硬编码（EHR 恰为 3 层可用），改为显式 token trie 留作后续；若评测出现非法 SID 比例异常或候选不足 40，再回来处理。

### 13.4 beam search 生成 rank-list

建议初始配置：

```text
do_sample               false
num_beams               40
num_return_sequences    40   # = num_beams（决策见 solution 问题 10）
length_penalty          0.0
early_stopping          true
output_scores           true
return_dict_in_generate true
padding_side            left
```

当前 MIMIC-III/IV 每个 visit 的最大目标疾病数均为 39。为了计算 `K ∈ {10,20,30,40}` 和“预测数等于真实疾病数”的 weighted-F1，评测至少需要 40 个有效候选；当前 SIDReasoner 默认 `num_beams=10` 不够。

beam 输出后：

1. 按 `sequences_scores` 从高到低排序。
2. 标准化并解析完整 SID。
3. 删除重复 SID，保留首次出现的最高分候选。
4. 用 `sid_to_disease` 还原 `disease_id` 和 ICD。
5. 若去重后不足 40 个，扩大 beam 或继续回填低排名合法候选。

推荐结果格式：

```json
{
  "sample_id": "mimic3:test:000001",
  "ground_truth_disease_ids": ["ICD9CM:4280", "ICD9CM:42731"],
  "ground_truth_sids": ["<a_17><b_93><c_4>", "<a_14><b_90><c_31>"],
  "predictions": [
    {
      "rank": 1,
      "sid": "<a_17><b_93><c_4>",
      "disease_id": "ICD9CM:4280",
      "score": -1.23
    }
  ]
}
```

## 14. Recall@K

对第 `i` 个 visit：

- 真实疾病集合为 `Y_i`；
- rank-list 前 K 个去重疾病为 `P_i@K`。

单样本 Recall@K：

```text
Recall_i@K = |Y_i ∩ P_i@K| / |Y_i|
```

整体 Recall@K 是所有 visit 的宏平均：

```text
Recall@K = mean_i(Recall_i@K)
```

对应实现：

```python
import numpy as np


def recall_at_k(ground_truth, rank_lists, ks=(10, 20, 30, 40)):
    scores = []
    for k in ks:
        per_visit = []
        for true_ids, predicted_ids in zip(ground_truth, rank_lists):
            true_set = set(true_ids)
            pred_set = set(predicted_ids[:k])
            per_visit.append(len(true_set & pred_set) / len(true_set))
        scores.append(float(np.mean(per_visit)))
    return dict(zip(ks, scores))
```

该口径与 [`Plan/参考指标计算.py`](./Plan/参考指标计算.py) 的 `top_k_prec_recall` 一致。

## 15. weighted-F1

### 15.1 第一版的预测集合阈值

参考现有指标代码，第一版暂定对每个 visit 选择：

```text
预测疾病数 n_i = 当前 visit 的真实疾病数 |Y_i|
预测集合 P_i = rank-list 的 Top-n_i
```

然后把所有 visit 的真实集合和预测集合转为疾病多热矩阵，在疾病标签维度计算：

```python
f1_score(y_true, y_pred, average="weighted")
```

其中 weighted-F1 以每个疾病在真实标签中的 support 加权，头部疾病权重更大。

这不是固定概率阈值，而是使用真实标签基数的 oracle-cardinality cutoff。它适合和现有 EHR baseline 对齐，但推理时需要知道真实诊断数，不能直接视为可部署决策规则。实验报告中应明确命名为：

```text
oracle-cardinality weighted-F1
```

后续可以在 valid 集上另行选择固定 K、概率阈值或预测 cardinality 模型，但第一版先按当前患者真实诊断数截断。

### 15.2 实现

```python
import numpy as np
from sklearn.metrics import f1_score


def weighted_f1_at_true_cardinality(
    ground_truth,
    rank_lists,
    disease_ids,
):
    disease_to_col = {
        disease_id: index
        for index, disease_id in enumerate(disease_ids)
    }
    y_true = np.zeros(
        (len(ground_truth), len(disease_ids)), dtype=np.int8
    )
    y_pred = np.zeros_like(y_true)

    for row, (true_ids, predicted_ids) in enumerate(
        zip(ground_truth, rank_lists)
    ):
        true_ids = list(dict.fromkeys(true_ids))
        predicted_ids = list(dict.fromkeys(predicted_ids))

        for disease_id in true_ids:
            y_true[row, disease_to_col[disease_id]] = 1

        cutoff = len(true_ids)
        for disease_id in predicted_ids[:cutoff]:
            y_pred[row, disease_to_col[disease_id]] = 1

    return f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
```

`disease_ids` 应来自同一份全局 disease manifest，并保持固定顺序。不要分别从 GT 和预测临时构建两个列空间。

### 15.3 建议同时输出的检查指标

主指标按用户指定为 Recall@K 和 weighted-F1，同时建议记录：

- Precision@K。
- 非法 SID 比例；约束正确时应为 0。
- 重复 beam 比例。
- 去重后候选不足 K 的比例。
- head/mid/tail Recall@K。
- 按历史 visit 数和 GT 集合大小分层的 Recall@K。

这些指标用于定位问题，不改变第一版主结论口径。

## 16. 训练与评测数据不能混用的地方

| 阶段 | 数据粒度 | target |
|---|---|---|
| T1/T2 | disease-level | 一个疾病文本或 SID |
| T3–T6 SFT | code-level | 一个目标疾病 |
| validation loss | code-level | 一个目标疾病 |
| beam-search evaluation | visit-level | 完整疾病集合 |
| Recall@K / weighted-F1 | visit-level | 完整疾病集合 |

常见错误：

- 用 code-level test CSV 直接评测，导致同一 visit 被重复计算多次。
- 每行只保留一个子目标，丢失完整 GT。
- 用当前目标疾病数作为训练 prompt 的输入，造成标签信息泄漏。
- beam rank-list 没有去重就计算 Recall。
- 用原 MiniOneRec `calc.py` 的单目标 HR/NDCG 代替多标签 Recall/F1。
- 把疾病名称或 ICD code 当作 SID trie 的合法输出。

## 17. 代码复用与新增清单

### 17.1 可直接复用

| 文件 | 复用内容 |
|---|---|
| `sft_Qwen3.py` | 模型加载、扩词表、Trainer、DDP、保存 checkpoint |
| `data_Qwen3.py` | chat template、assistant-only mask、Dataset 返回协议 |
| `evaluate_Qwen3.py` | HF beam search 和 `prefix_allowed_tokens_fn` 总体思路 |
| `split.py` / `merge.py` | 多卡离线切分与结果合并思路 |
| MiniOneRec `rq/` | 疾病 embedding 的 RQ-VAE/RQ-KMeans 与 SID 生成 |

### 17.2 必须新增或改造

| 模块 | 原因 |
|---|---|
| EHR 预处理/产物生成 | Amazon CSV 不含 visit 边界和完整多标签 GT |
| `ehr_data_Qwen3.py` | 解析嵌套 visit、JSON 字段和医疗 prompt |
| `sft_Qwen3_simple_ehr.py` | 只组合 T1–T6，移除富语料与 reasoning 参数 |
| `evaluate_Qwen3_simple_ehr.py` | 一 visit 一次推理、返回完整 rank-list |
| SID trie | 从疾病 `index.json` 动态构造，不依赖固定 prompt token 数 |
| EHR metrics | 多标签 Recall@K 和 oracle-cardinality weighted-F1 |

## 18. 分阶段验收

### M0：数据货币冻结

产物：

- `item.json`；
- `index.json`；
- `info.tsv`；
- code-level CSV；
- visit-level JSONL；
- manifest 与验证报告。

验收：映射全覆盖、SID 无碰撞、患者无泄漏、拆分可回聚。

### M1：T1/T2 对齐闭环

只训练文本↔SID，确认模型能够识别疾病 SID，验证 tokenizer 保存/重载无误。

### M2：T5 主任务闭环

只训练 SID 历史→SID，跑通 valid visit-level constrained beam 和 Recall@K。

### M3：T1–T6 Simple SFT

加入全部基础辅助任务，比较 T5-only、基础对齐和完整 Simple SFT。

### M4：正式评测

在 test split 输出 Top-40 rank-list，计算：

- Recall@10/20/30/40；
- oracle-cardinality weighted-F1；
- 合法率、重复率和分层结果。

## 19. 第一版最终产物

```text
data/EHR/<dataset>/
├── manifest/*
├── index/*.item.json
├── index/*.index.json
├── info/*.info.tsv
├── code_level/{train,valid,test}.csv
└── visit_level/{train,valid,test}.jsonl

output_dir/<experiment>/
└── final_checkpoint/
    ├── model weights
    ├── tokenizer files
    └── config.json

results/<experiment>/
├── valid_ranklist.jsonl
├── test_ranklist.jsonl
├── metrics.json
└── evaluation_manifest.json
```

`evaluation_manifest.json` 至少记录：checkpoint、tokenizer hash、disease manifest hash、beam 参数、K、weighted-F1 cutoff 策略和评测脚本版本。

