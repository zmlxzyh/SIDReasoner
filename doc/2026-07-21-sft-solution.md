# Simple SFT 问题修复方案

> 更新日期：2026-07-21
> 本文针对 [`2026-07-18-simpleSFT.md`](./2026-07-18-simpleSFT.md) 与 [`2026-07-20-可能的问题.md`](./2026-07-20-可能的问题.md) 中暴露的问题，逐条给出**决策与落地做法**。
> 数据以 `data/EHR/mimic3_icd_name_path_0.1` 为准；任务与数据构造参考 `data_Qwen3.py` / `sft_Qwen3.py`。
> 结论以“决策”为准：标注“接受不处理”的条目表示第一版明确不改，只做必要护栏。

## 问题清单与状态

| # | 问题 | 状态 | 决策摘要 |
|---|---|---|---|
| 1 | SID 完整碰撞 + title 重复 | ✅ 已定 | 接受不处理，仅加护栏，避免崩溃与指标误读 |
| 2 | 动态打乱 vs `HFDataset.from_dict` 一次性物化冲突 | ✅ 已定 | 放弃每 epoch 动态；改为 `__getitem__` 内一次性打乱后固定为静态数据 |
| 3 | EHR 必须禁用 dedup（慢病复发是合法目标） | ✅ 已定 | 全任务固定 `dedup=False`，不移植原“target==末位历史即丢弃”逻辑 |
| 4 | 显式任务采样比例（50/10/10/10/10/10）无落地机制 | ✅ 已定 | 小流量测试以跑通为主，接受 `ConcatDataset` 自然比例，不做加权采样 |
| 5 | T1 模板 title/description 措辞不一致 + title 作 key 歧义 | ✅ 已定 | T1/T2 用 “name” 措辞；title 重复歧义按问题 1 接受不处理 |
| 6 | 原类扁平历史 / eval / 左截断 / 预物化 → 新 EHR Dataset | ✅ 已定 | 新写 EHR Dataset：读嵌套字段 + json.loads + visit 级截断 + 一次性打乱 |
| 7 | T1/T2 额外验证非 held-out，命名与 checkpoint 用途 | ✅ 已定 | 逻辑本就正确；仅把额外集重命名为 `alignment_reconstruction/*`，不参与选择 |
| 8 | tokenizer / EOS 校验作为训练前 preflight | ✅ 已定 | 训练启动时生成 preflight artifact，四项校验，不过即终止 |
| 9 | 文档目录、文件命名与实际不符 | ✅ 已定 | 统一用实际 `mimic3_icd_name_path_0.1`；路径仅由顶层版本目录决定，内部命名固定 |
| 10 | 评测 `num_beams≥40`、显式 trie、no-think 协议 | ✅ 已定 | 仅设 `num_beams=num_return_sequences=40`；trie/no-think 沿用原始代码 |
| 11 | “完全对齐 Amazon”措辞（实为兼容超集） | ✅ 已定 | 措辞改为“保留原版消费字段的兼容超集”，直接对齐 |

---

## 问题 1：SID 完整碰撞与 title 重复 —— 决策：接受不处理

### 1.1 背景与机制

`SidItemFeatDataset` 用普通 dict 建映射（`data_Qwen3.py:551-558`），key 分别是完整 SID 和 title：

- `sid2title[完整SID] = title`：两个疾病共享同一完整 SID（**SID 碰撞**）时，后写入者静默覆盖前者。
- `title2sid[title] = 完整SID`：两个疾病共享同一 title（**title 重复**）时，后写入者静默覆盖前者。

`FusionSeqRecDataset`（T3）取目标标题时是拿 CSV 的 `item_sid` 反查这份 `sid2title`（`data_Qwen3.py:808-812`），因此碰撞 SID 的目标行会被打上“同 SID 另一疾病的标题”作为标签。

### 1.2 实测影响（`mimic3_icd_name_path_0.1`）

碰撞规模极小，且集中在语义近乎同义的编码上。

SID 碰撞：**2 组 / 4 个疾病**

```
<a_68><b_128><c_0>  ICD9CM:9980  (Postoperative shock)
                    ICD9CM:99809 (Postoperative shock, other)
<a_68><b_128><c_3>  ICD9CM:9973  (Respiratory complications, not elsewhere classified)
                    ICD9CM:99739 (Other respiratory complications)
```

title 重复：**14 组 / 28 个疾病**，全部是“简写码 vs 带尾号细分码”共享同一标准名，例如
`ICD9CM:2794 / ICD9CM:27949`（Autoimmune disease, NEC）、`ICD9CM:4538 / ICD9CM:45389`（Acute venous embolism...）等。

对样本与各任务的实测波及：

| 影响面 | 数值（0.1） | 说明 |
|---|---|---|
| T2（sid→title）样本 | 4489（丢 2） | SID 碰撞组各丢一个成员 |
| T1（title→sid）样本 | 4477（丢 14） | title 重复组各丢一个成员 |
| `SidItemFeatDataset` 合计 | **8966**（= 4489+4477，非 4491×2=8982） | 与 `可能的问题.md` 第 2 点一致 |
| T3 目标为碰撞病、被打错标题的行 | train **8** / valid **1** / test **3** | 标签变成同 SID 兄弟病名字 |
| T3 目标为重复 title 病的行 | train 30 / valid 6 / test 22 | 输出层面标签不唯一，非“错误”，可接受 |

### 1.3 决策与理由

**第一版直接不处理**，不重训 SID、不做去碰撞、不改 `item.json`。理由：

1. **影响样本极小**：训练侧真正“错误标签”的仅 T3 的 8 行（占 train 7952 的 0.1%）；T1/T2 少训 16 个疾病。
2. **碰撞/重复疾病本身高度相似**：SID 碰撞的两对都是“术后休克 / 术后休克-其他”“呼吸系统并发症 / 其他呼吸系统并发症”这类近义细分码；title 重复也基本是 ICD 记录规范（尾号 9 细分 vs 简写码）造成的同名，语义几乎等价。
3. **少训这些碰撞疾病问题不大**：不影响主任务 T5 的目标空间正确性，也不改变第一版主结论口径。

### 1.4 “不处理”≠“可以不管代码”——最小护栏

“接受不处理”仍必须保证管线**不因碰撞崩溃、不静默误导指标**，需要以下三点护栏（这是本条目的实际落地内容）：

1. **放宽硬断言**。`simpleSFT.md` §5 示例里的
   `assert len(disease_to_sid) == len(sid_to_disease), "full SID collision"`
   在有碰撞时会直接抛错。改为**告警 + 计数**，不中断：

```python
disease_to_sid = {d: "".join(t) for d, t in disease_to_sid_tokens.items()}
sid_to_disease = {}
for disease_id in sorted(disease_to_sid):          # 按 disease_id 排序，保证覆盖结果确定
    sid_to_disease[disease_to_sid[disease_id]] = disease_id
n_collision = len(disease_to_sid) - len(sid_to_disease)
if n_collision:
    print(f"[warn] full SID collision: {n_collision} disease(s) unrecoverable in eval")
```

2. **保持 `require_unique_sid=false`，覆盖顺序确定化**。所有由碰撞/重复引发的 dict 覆盖，统一按 `disease_id` 升序写入，使每次运行得到相同的“保留者/丢弃者”，结果可复现。

3. **评测端明确记账**。碰撞组中被 `sid_to_disease` 丢弃的“失败者”疾病，其 GT 出现时在 Recall/F1 中**不可能被命中**（其 SID 会被解码成兄弟病）。这抬不动的天花板已量化为 test 仅 3 行，属可接受且有界的偏差。要求在 `evaluation_manifest.json` 中记录：`unique_sid_count`、`collision_disease_count`、以及“因碰撞不可恢复的 GT 记录数”，避免把这几分误差当成模型缺陷。

### 1.5 对主任务与评测的确认

- **T5（SID 历史→SID，主任务）**：碰撞疾病的训练目标是相同 SID，无“错误标签”，只是两个疾病共用一个输出，正确性不受影响。
- **评测（受约束 beam → SID → disease_id）**：trie 叶子按完整 SID 唯一，碰撞 SID 是单叶子；解码回 disease_id 时按 1.4 的确定化 `sid_to_disease` 取兄弟病之一。影响有界（test 3 行），已在 manifest 记账。
- **T1/T2/T3**：按 1.2 接受少量样本缺失与 T3 的 8 行标签偏差，不做修复。

### 1.6 复查方法（可复现）

```bash
cd data/EHR/mimic3_icd_name_path_0.1
python3 - <<'PY'
import json, csv, collections
item=json.load(open("index/mimic3_icd.item.json"))
index=json.load(open("index/mimic3_icd.index.json"))
sidmap=collections.defaultdict(list)
for d,s in index.items(): sidmap["".join(s)].append(d)
collide=set(d for m in sidmap.values() if len(m)>1 for d in m)
t2=collections.defaultdict(list)
for d,m in item.items(): t2[m["title"]].append(d)
duptitle=set(d for ids in t2.values() if len(ids)>1 for d in ids)
print("SID碰撞病:",sorted(collide)," 重复title病数:",len(duptitle))
for sp in ["train","valid","test"]:
    rows=list(csv.DictReader(open(f"code_level/{sp}.csv")))
    print(sp,"总行",len(rows),
          "目标为碰撞病行",sum(r["item_id"] in collide for r in rows),
          "目标为重复title病行",sum(r["item_id"] in duptitle for r in rows))
PY
```

---

## 问题 2：动态打乱 vs 一次性物化 —— 决策：静态一次性打乱

### 2.1 背景

`simpleSFT.md` §9 规则 2 原写“训练阶段每次取样动态打乱同一 visit 内的疾病顺序”。但 `sft_Qwen3.py:354`

```python
hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
```

会把 `ConcatDataset` **遍历一次、每条样本调用一次 `__getitem__`，然后冻结成静态 `HFDataset`**。因此“每 epoch 换一种顺序”的动态增广在现有管线下不会发生。

### 2.2 决策

**放弃“每 epoch 动态打乱”，改为“一次性静态打乱”**：在 `__getitem__`（或等价的 `pre()` 预物化）里对每条样本的历史 visit **内部**做一次随机打乱，物化时被固定下来；之后所有 epoch 都基于这份打乱后的静态数据训练。

这与现有 `from_dict` 物化管线天然吻合，**无需改动 Trainer / 训练框架**，也不需要把 torch `ConcatDataset` 直接塞给 Trainer。原“待定：动态 vs 直接喂 ConcatDataset”不再需要。

### 2.3 为什么静态就够（不损失主要正则化收益）

code-level 展开后，**同一个 visit 历史会因为多个目标疾病出现在多行**（GT 有 N 个疾病 → N 行共享同一历史）。只要对**每一行独立打乱**，这 N 行就会得到 N 种不同的历史排列——于是**在单个 epoch 内**，模型就已经见过同一历史的多种顺序。这基本覆盖了“动态打乱”想要的“打散对任意 ICD 排序的过拟合”的效果，静态一次性打乱足矣。

### 2.4 落地要求

1. **打乱粒度**：只打乱**每个 visit 内部**的疾病顺序；visit 之间严格保持时间先后（与 `simpleSFT.md` §9 规则 1 一致）。
2. **确定化种子**：用**按样本确定**的随机种子，而不是全局 RNG，保证冻结出的数据集可复现、DDP 各 rank 结果一致。推荐 `rng = random.Random(seed * LARGE_PRIME + idx)`，逐行独立（这样 2.3 的“多行多排列”才成立）。
3. **仅训练集打乱**：valid/test 不打乱，按 `disease_id` 确定排序（`simpleSFT.md` §9 规则 3）。
4. **序列化统一**：打乱后再进 `serialize_sid_history` / `serialize_text_history`，T3/T5 共用 SID serializer、T4/T6 共用 text serializer。

实现骨架（EHR Dataset 内部，示意）：

```python
def _ordered_visits(self, history_visits, idx):
    if not self.training:                       # valid/test 确定序
        return [sorted(v) for v in history_visits]
    rng = random.Random(self.seed * 1_000_003 + idx)   # 逐行确定化
    shuffled = []
    for visit in history_visits:                # 只打乱 visit 内部
        v = list(visit)
        rng.shuffle(v)
        shuffled.append(v)
    return shuffled                             # visit 间顺序不动
```

只要在 `pre()`/`__getitem__` 里调用它一次，`from_dict` 物化即把结果固定为静态数据。

### 2.5 对文档的影响

`simpleSFT.md` §9 规则 2 应从“训练阶段每次取样动态打乱”改写为“训练阶段在样本物化时对每条样本各做一次 visit 内打乱，之后固定为静态数据；valid/test 确定排序”。

---

## 问题 3：EHR 去重 —— 决策：全任务禁用 dedup

### 3.1 背景

原类的 dedup 判定是“目标 == 最后一个历史 item 就丢弃该样本”，用于 Amazon 过滤“复购同一商品”。EHR 里**慢性病复发是完全合法的预测目标**：实测 `mimic3:train:000017` 的目标 `ICD9CM:25063` 同时出现在历史 visit 和当前 GT 中。若沿用该逻辑，会把大量正确的复发样本误删。

### 3.2 决策

**所有 EHR 任务固定 `dedup=False`**，并且在新 EHR Dataset 中**不移植**原类基于“target==末位历史”的 dedup 分支。

### 3.3 落地要求

1. 构造各任务 Dataset 时不传 `dedup=True`；新 EHR 类可直接不实现该分支，避免被误用。
2. 历史里与目标重复的疾病**照常保留**，不做任何过滤。
3. 与 `simpleSFT.md` §8.1“全部拆分、不做单目标随机采样”保持一致：一个 visit 的 N 个 GT 疾病展开为 N 行，逐行独立打乱历史（见问题 2），互不丢弃。

---

## 问题 4：任务采样比例 —— 决策：接受自然比例（小流量跑通优先）

### 4.1 决策

第一版是**小流量测试，以跑通代码为主**，因此直接采用 `ConcatDataset` 的**自然大小比例**拼接六个任务，**不做**加权采样 / 行复制 / `IterableDataset`。`simpleSFT.md` §12.4 的 50/10/10/10/10/10 显式采样**推迟到正式实验**再引入。

### 4.2 需要知晓的偏差（不影响“跑通”目标）

- 0.1 保留了全量 item，但 code-level 只有约 10%，因此自然比例下 **T1/T2 占比偏大**（各约 11%），与全量数据（各约 1.2%）不同。
- 结论：0.1 的训练曲线只用于验证代码正确，**不能**直接外推到全量。此点已在 `2026-07-20-可能的问题.md` 第 4 点记录，这里沿用。

### 4.3 落地要求

- 直接 `train_data = ConcatDataset([...六个 EHR 任务...])`，与 `sft_Qwen3.py` 现有写法一致。
- 不需要为比例改动 `group_by_length` 或训练管线。

---

## 问题 5：T1/T2 措辞 —— 决策：使用 “name”

### 5.1 决策

T1（疾病文本→SID）prompt 统一用 **name**（标准疾病英文名，即 `item.json` 的 `title`），不再用歧义的 “description” 措辞：

```text
System:
You map ICD disease names to semantic identifiers.

User:
Which disease has the name "Congestive heart failure, unspecified"?

Assistant:
<a_17><b_93><c_4>
```

T2（SID→疾病文本）对称，输出标准 name：

```text
User:
What disease does <a_17><b_93><c_4> represent?

Assistant:
Congestive heart failure, unspecified
```

### 5.2 title 作 key 的歧义

T1 用 `name` 作为输入时，14 组重复 title（28 个疾病）会出现“同一输入对应两个合法 SID”的天然歧义。按**问题 1 的决策接受不处理**：不为消歧引入 ontology path，也不额外去重；`title2sid` 的确定化覆盖（按 `disease_id` 排序）保证可复现即可。影响样本极小，且这些多为 ICD 记录规范造成的同名近义码。

### 5.3 落地要求

- `simpleSFT.md` §10.1 的 prompt 从 “Which disease has the description ...” 改为 “Which disease has the name ...”，system 提示相应改为 “names”。
- T1 输入取 `item.json` 的 `title` 字段，**不**拼接 ICD code（避免任务退化为记 code）。

---

## 问题 6：新 EHR Dataset —— 决策：新写专用类，不复用原类历史行为

### 6.1 为什么原类不能原封不动用

原版 5 个 Dataset **能读** EHR 的 CSV（字段名齐全，多余列 pandas 忽略），但读出来的语义/行为有问题：

1. **扁平 `history_item_*` 丢失 visit 边界**：原类只读拍平列（如 `SidSFTDataset.get_history`，`data_Qwen3.py:296-305`），用 `, ` 连成流水，看不出哪些疾病属于同一次住院；EHR 需要的嵌套 `history_sid_visits` 根本没被读。
2. **使用 `eval`**：`eval(row['history_item_sid'])`（`data_Qwen3.py:297` 等），对我们的 JSON 列表虽能解析，但不安全、不规范。
3. **左侧硬截断切断 SID / visit**：`full_ids[-max_len:]`（`data_Qwen3.py:109-113` 及各 `pre()`）按 token 个数从左砍。SID 是多 token（`<a_17><b_93><c_4>` = 3 token），visit 边界也是 token，切口可能落在 SID 中间（留下 `<b_93><c_4>` 这种半个 SID）或砍掉 `<visit_start>` 只剩 `<visit_end>`（不闭合 visit），甚至砍掉 system/chat 模板开头。Amazon 历史短很少触发，EHR 长历史会经常触发。
4. **`__init__` 预物化**：原类在 `__init__` 调 `get_inputs()` 一次性 tokenize；本身与静态打乱不冲突，但打乱/禁 dedup/name 措辞都要落在新的 `pre()` 里。

### 6.2 决策

**新写 EHR 专用 Dataset 类**（放 `ehr_data_Qwen3.py`），复用原版的**任务思想、chat template、assistant-only mask、Trainer 返回协议**；**不复用**原版的扁平历史解析、`eval`、左截断、预物化行为。

对应 `simpleSFT.md` §11 的类划分：

```text
ehr_data_Qwen3.py
├── EhrSidItemFeatDataset       # T1 / T2（name ↔ SID）
├── EhrSidHistoryToTextDataset  # T3（SID 历史 → 目标 name）
├── EhrTextHistoryToTextDataset # T4（文本历史 → 目标 name）
├── EhrSidSFTDataset            # T5（SID 历史 → 目标 SID，主任务）
├── EhrTextHistoryToSidDataset  # T6（文本历史 → 目标 SID）
└── EhrVisitEvalDataset         # visit-level 评测
```

### 6.3 落地要求（把已定的 2/3/5 收拢进来）

1. **读嵌套权威字段**：历史一律用 `history_sid_visits` / `history_disease_text_visits`（`json.loads` 解析），不用扁平 `history_item_*`。
2. **`json.loads` 代替 `eval`**：所有列表/嵌套列表字段统一 `json.loads`。
3. **统一 serializer**：T3/T5 共用 SID serializer，T4/T6 共用 text serializer，训练/验证/评测同一套模板，避免漂移。
4. **visit 级截断**：超长时**整段删除最旧 visit**，绝不切断 SID、不留不闭合 visit 边界（详见 6.5）。
5. **一次性 visit 内打乱**（问题 2）：训练集在 `pre()` 里用逐样本确定化种子对每个 visit 内部打乱一次；valid/test 按 `disease_id` 确定排序。
6. **禁用 dedup**（问题 3）：不实现/不启用“target==末位历史即丢弃”。
7. **T1/T2 用 name**（问题 5）：输入取 `item.json` 的 `title`，不拼 ICD code。
8. **返回协议**：训练类返回 `input_ids/attention_mask/labels`；`EhrVisitEvalDataset` 额外返回完整 `ground_truth_sids/ground_truth_disease_ids`（评测一 visit 一行，读 `visit_level/*.jsonl`，不读 code-level 重复行）。

### 6.4 历史序列化骨架

```python
def serialize_sid_history(history_sid_visits, training, idx, seed):
    visits = _ordered_visits(history_sid_visits, training, idx, seed)  # 见问题 2
    blocks = []
    for visit in visits:
        blocks.append("<visit_start>\n" + ", ".join(visit) + "\n<visit_end>")
    return "\n".join(blocks)

def serialize_text_history(history_text_visits, training, idx, seed):
    visits = _ordered_visits(history_text_visits, training, idx, seed)
    lines = [f"Visit {i+1}: " + "; ".join(v) for i, v in enumerate(visits)]
    return "\n".join(lines)
```

`_ordered_visits`：训练用 `random.Random(seed*1_000_003+idx)` 只打乱每个 visit 内部、visit 间时序不动；valid/test 每个 visit 按 `disease_id` 排序。

### 6.5 visit 级截断骨架

在**序列化+tokenize 之后**判断长度；若超过 `max_len`，**从最旧 visit 整段丢弃**后重新序列化，循环直到放得下（始终保留 system/user 模板与答案）：

```python
def fit_history(visits, build_prompt_fn, tokenizer, max_len):
    kept = list(visits)
    while kept:
        ids = tokenizer.apply_chat_template(build_prompt_fn(kept), tokenize=True,
                                            add_generation_prompt=False)
        if len(ids) <= max_len:
            return kept, ids
        kept = kept[1:]          # 删掉最旧的一整个 visit，不切 SID/不留半开边界
    # 兜底：历史全删仍超长（极端），只保留最近一个 visit 或截到空历史
    return [], tokenizer.apply_chat_template(build_prompt_fn([]), tokenize=True,
                                             add_generation_prompt=False)
```

要点：**不再用 `input_ids[-max_len:]`**；截断以“整个 visit”为最小单位，保证不产生半个 SID、不产生不闭合的 `<visit_start>/<visit_end>`。

### 6.6 对文档的影响

`simpleSFT.md` §11「必须修改的地方」已列了大部分；本条把“新写类 + 读嵌套 + json.loads + visit 级截断 + 一次性打乱 + 禁 dedup + name 措辞”统一为落地口径。

---

## 问题 7：T1/T2 验证命名 —— 决策：逻辑不动，仅重命名

### 7.1 背景

`sft_Qwen3.py:339-341` 三个验证集来源不同：

- T5 验证 `val_data_sid_prediction` 读 `eval_file`（code-level **valid** CSV，来自 held-out 验证病人）——真正的 held-out。
- T1/T2 验证 `val_data_title2sid_translation` / `val_data_sid2title_translation` 读**与训练同一份** `item.json`/`index.json`，疾病全集训练时都见过，无 held-out 划分。

因此 T1/T2 的 “eval loss” 是**映射重建/记忆**，不是泛化，不应被当作验证泛化，也不应驱动 checkpoint 选择。

### 7.2 现有逻辑已正确

- checkpoint / early stopping 用 `metric_for_best_model="eval_loss"`（`sft_Qwen3.py:400`），即 T5 held-out valid loss。
- `MultiEvalTrainer` 评测 T1/T2 额外集时**临时禁用 EarlyStoppingCallback**（`sft_Qwen3.py:76-86`），只旁观、不参与选择。

结论：**逻辑本身正确，只是命名会误导**。

### 7.3 决策

保持逻辑不变，仅把额外集的 metric key 从 `eval_title2sid_loss` / `eval_sid2title_loss` 重命名为：

```text
alignment_reconstruction/title2sid_loss
alignment_reconstruction/sid2title_loss
```

即 `extra_eval_sets` 的 key 改为 `alignment_reconstruction/title2sid` 与 `alignment_reconstruction/sid2title`（`metric_key_prefix` 会据此生成 key）。

### 7.4 落地要求

1. checkpoint / early stopping 继续只用 T5 held-out valid `eval_loss`。
2. 正式实验的最佳 checkpoint 额外看 visit-level `Recall@10`（与 `simpleSFT.md` §12.5 一致）。
3. 文档/看板中说明 `alignment_reconstruction/*` 仅为监控指标，非泛化验证。

---

## 问题 9：命名对齐 —— 决策：以实际目录为准，路径仅随顶层版本目录变化

### 9.1 决策

文档与脚本统一使用**实际**命名 `data/EHR/mimic3_icd_name_path_0.1`，废弃 `simpleSFT.md` §3 里虚构的 `mimic3_icd/`、`disease.item.json`、`tokenizer_validation.json` 等名称。

### 9.2 版本切换只改顶层目录

不同版本/规模的数据集（如 `mimic3_icd_name_path`、`mimic3_icd_name_path_0.1` …）**内部目录结构与文件命名规则完全一致，仅规模不同**。因此路径应由**单一顶层版本目录**派生，切换数据集只改这一个值。

实际内部结构（固定）：

```text
<data_dir>/                       # 唯一需要切换的顶层目录，如 mimic3_icd_name_path_0.1
├── index/<prefix>.item.json      # prefix 固定为 mimic3_icd（mimic4 则为 mimic4_icd）
├── index/<prefix>.index.json
├── info/<prefix>.info.tsv
├── code_level/{train,valid,test}.csv
├── visit_level/{train,valid,test}.jsonl
└── manifest/*.json
```

注意：顶层目录名带 `_0.1` 后缀，但**内部文件前缀是 `mimic3_icd`**（不含 `_name_path_0.1`）。

### 9.3 落地要求

1. 训练/评测脚本只暴露 `data_dir`（顶层版本目录）+ `dataset_prefix`（默认 `mimic3_icd`，换源时改为 `mimic4_icd`），其余路径全部派生：

```python
item_meta_path  = f"{data_dir}/index/{dataset_prefix}.item.json"
sid_index_path  = f"{data_dir}/index/{dataset_prefix}.index.json"
info_path       = f"{data_dir}/info/{dataset_prefix}.info.tsv"
train_file      = f"{data_dir}/code_level/train.csv"
eval_file       = f"{data_dir}/code_level/valid.csv"
test_file       = f"{data_dir}/code_level/test.csv"
visit_eval_file = f"{data_dir}/visit_level/valid.jsonl"
visit_test_file = f"{data_dir}/visit_level/test.jsonl"
```

2. `sft_Qwen3.py` 现有的 `TokenExtender(dataset=os.path.basename(sid_index_path).split('.')[0])` 已能从 `mimic3_icd.index.json` 正确取到前缀 `mimic3_icd`，无需额外改动。
3. 切换到全量或 MIMIC-IV：只改 `data_dir`（必要时改 `dataset_prefix`），不改任何内部路径拼接逻辑。
4. `simpleSFT.md` §3 的目录示例与文件名按 9.2 更新，删除 `disease.*` / `tokenizer_validation.json` 虚构名（`tokenizer` preflight 归入问题 8）。

---

## 问题 11：措辞 —— 决策：直接对齐（已定）

将 `simpleSFT.md` 中“完全对齐 Amazon / 完全照搬 Amazon”改为：

> EHR code-level CSV 保留原版五个 Dataset 所需的全部消费字段，并增加 visit-level 权威字段；它是 **consumer-compatible superset（兼容超集）**，不是完全相同 schema。

差异点：`sample_id + split`（而非 `user_id`）、列表用合法 JSON（而非 Python repr）、额外含 `history_*_visits` / `ground_truth_*` / `target_index` 等 visit-level 权威字段。无需额外复制 `user_id`。

---

## 问题 8：tokenizer / EOS 校验 —— 决策：训练启动 preflight

### 8.1 决策

不再把 tokenizer 校验当作 SFT 前置数据产物（它依赖具体 base model/tokenizer），改为**训练启动时生成的 preflight artifact**，检查：

1. **768 个 SID token 和两个 visit token 均为单 token**（共 770 个，实测第 0/1/2 层各 256 个唯一 token）。
2. **label 只覆盖 assistant response 本身（`mask_eos=True`），不监督 assistant 结束后缀**。
3. **`mask_eos=True` 后，answer 之后的 chat template 结束后缀（`<|im_end|>` / EOS）确实全部为 `-100`**（既未被监督，也不含下一轮模板内容）。
4. **保存并重载 tokenizer 后结果不变**。

> 说明：第一版**沿用原始 codebase 的 `mask_eos=True` 默认，不监督结束后缀**（`ehr_data_Qwen3.py` 的 `_encode_ehr_sample` 固定 `mask_eos=True`）。因此 preflight 的取向与 simpleSFT.md 之前“医疗版用 `mask_eos=False`”的旧写法相反——本决策为准。权衡：模型不通过监督学习主动 EOS，但 T5 评测由受约束 beam + SID trie 强制在完整 SID 后 EOS（simpleSFT.md §13.3），T1–T4/T6 自由文本不进入 beam 评测，故可接受。后续富语料 / reasoning 任务如需模型自主停止，再对相应任务改用 `mask_eos=False`。

### 8.2 落地要求

- preflight 在扩词表 + `resize_token_embeddings` 之后、正式 `trainer.train()` 之前运行；**任一项不过即报错终止**，不进入训练。
- 结果落盘为 `manifest/tokenizer_preflight.json`（记录 base model、token 数、各校验通过与否、tokenizer hash）。
- 校验 1 的实现示例：

```python
special_tokens = sid_tokens + ["<visit_start>", "<visit_end>"]   # 768 + 2
tokenizer.save_pretrained(out_dir); tk = AutoTokenizer.from_pretrained(out_dir)
for tok in special_tokens:
    assert len(tk.encode(tok, add_special_tokens=False)) == 1, f"{tok} 不是单 token"
```

- 校验 2/3 用一条样本做 decode 可视化：确认 labels 中非 `-100` 段解码后正好是 assistant 回答**本身**（`mask_eos=True`，不含结束符），其后的结束后缀全部为 `-100`。

---

## 问题 10：评测配置 —— 决策：仅设 beam=40，其余沿用原始代码

### 10.1 决策

- **`num_beams = num_return_sequences = 40`**，`do_sample=False`。原因：每 visit 最大目标疾病数实测为 **39**（train/valid/test 均为 39），要算 `K∈{10,20,30,40}` 和 oracle-cardinality weighted-F1 至少需要 40 个有效候选；SIDReasoner 默认 `num_beams=10` 不够。
- **受约束 beam search 的实现（`prefix_allowed_tokens_fn` / trie）与 no-think 处理，第一版直接沿用 `evaluate_Qwen3.py` 原始代码**，先不改，应也没问题。

### 10.2 已知但本版暂不处理（记录备查）

以下是原始实现的潜在点，本版接受、留待后续需要时再改：

- trie 基于 hash + 固定 `prefix_index`，且 dataset 类里按 3 层硬编码；EHR 恰好 3 层可用。
- no-think 注入 `<think>\n</think>\n\n` 并假设其 token 数固定（`data_Qwen3.py:448-452`）；若换 tokenizer 需复核 answer 起点定位。

若第一版评测出现非法 SID 比例异常或候选不足 40，再回来处理 10.2。
