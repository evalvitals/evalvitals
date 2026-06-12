# deco_chair — 开放式描述下的 DeCo failure/success example（差异部分）

> 共同设计（场景机理、模型适配、统计纪律、protocol/Mode-2 用法、修复公式）见
> [`../deco_pope/DESIGN.md`](../deco_pope/DESIGN.md)；施工清单见本目录
> [`TODO.md`](TODO.md)。本文档只写 captioning 任务的不同之处。
> 这是论文 **Finding 2 的忠实复现**场景。定位同样是**诊断 loop 的输入**
> （cases + protocol + 容器），不向包内新增任何代码。

## 1. 任务与 case 单位

- prompt 固定为论文/OPERA 用的 `"Please help me describe the image in detail."`，
  greedy，`max_new_tokens=512`。
- **喂给 loop 的 case 单位是图像级**（与 `chair` analyzer 的约定对齐：
  `metadata["gt_objects"]` + observed caption）：
  - FAIL = caption 含 ≥1 个不在 GT 物体表中的 COCO 物体（CHAIR 匹配，含同义词表）；
  - PASS = caption 无幻觉 mention。
- **机理分析的最小单位是 mention**（manifest 里逐条记录）：
  - 幻觉 mention（FAIL 单元）vs **同一条 caption** 里被正确提到的在场物体
    （PASS 单元）——同图、同上下文，唯一差异是 mention 是否有视觉根据，
    正是论文 Fig.2 的对比方式（"people/umbrella" 18 层即激活 vs "bird/green"
    30 层才冒头）。统计 `cluster_by=image_id`。

## 2. 数据与冻结清单

- ~50 张 COCO val2014 图，**优先杂物密集的室内场景**（kitchen/office/dining——
  共现先验最强）；挑选标准写进 mine 脚本，不手工钦点。
- `data/gt_objects.json`：每图 GT 物体表，离线从 `instances_val2014.json` 提取
  （250MB 注释文件只进缓存不进库）；同义词表沿用 CHAIR 标准表
  （`data/chair_synonyms.json`，与 `analyzers/hallucination/chair.py` 的 vocab 约定对齐）。
- `mine_cases.py --model {key}`：生成 caption → CHAIR 匹配 → 每个 mention 记录
  `{surface, coco_category, mention_kind: hallucinated|grounded, token_index}`
  （`token_index` = mention 首 token 在全序列中的位置；greedy 下重喂
  `ids[:token_index]` 能精确复现产出该 token 时的分布——这是冻结清单必须
  greedy 的原因）→ 冻结 `data/cases/{model_key}.json`。
- 漂移校验：caption 是长生成，更易漂移——比对"幻觉 mention 集合"而非全文逐字。
- split：图像级 60/40 explore/validate，按是否含幻觉分层。

## 3. 机理探测规格（给 tier-(b) 探针 / GPU 侧 agent，不进包）

与 deco_pope 的两点不同：

1. **位置**：重喂 `prompt + caption[:mention)` 前缀，在 pos=-1 探测；
2. **目标 token 集**：
   - `out_token_ids` = mention 自己的首 token；
   - `gt_token_ids` = 该图在场 GT 物体的首 token 中、落在末层 top-p(0.9)
     候选集内的那些（论文 Eq.2 的前提；过滤方向对照
     [官方 repo](https://github.com/zjunlp/DeCo) 核实）；
   - grounded mention 作对照时角色互换。

预期信号与 deco_pope 同构（`activated_gt` 率、`delta_final`），另加形态断言：
幻觉 mention 的 GT 峰值层集中在 [0.55N, 0.85N]（复现"20–28 层激活"的形状）。

protocol description（M1/M5 锚定，run.py 内置）：模型在详细描述图像时提到
不存在但与场景高共现的物体；怀疑机理同 DeCo——中间层已含正确物体信息、
末层被语言先验覆盖。

## 4. 修复与验证（与 POPE 的主要差别：必须接管生成循环）

- POPE 单步重打分不够用：captioning 需要**逐步** DeCo（手写 greedy 循环 +
  每步重打分，KV cache 复用），实现放本目录 `deco_fix.py`（TODO.md Step 4）。
  超参直接沿用 deco_pope explore 上调好的 (α, 窗口)——同模型同机理，
  这本身是一次跨任务迁移检验。
- validate 图像重新生成 caption，对比：
  - **主效应**：CHAIR_S / CHAIR_I 下降（论文量级：greedy 下 LLaVA 45.0→37.8 /
    14.7→11.1；Qwen3-VL 基线低得多 → min_effect 设小、图像级配对 bootstrap）；
  - **机理一致性**：被消掉的幻觉 mention 应集中在 `activated_gt==True` 子集；
  - **"no free lunch" 护栏**（论文明示）：distinct-2、最长 n-gram 重复率、
    caption 长度、grounded mention 召回（不能把真物体也修没）——α 过大时
    这些先报警。

## 5. 预算

50 图 × 512 token 生成 ×2（基线+DeCo）+ 每 mention 1 次前缀 forward
（~3-8 mention/图）。8B 单卡几十分钟；stepwise DeCo 论文报 ~1.2× 延迟。

## 6. 文件清单

```
examples/deco_chair/
├── DESIGN.md          本文档
├── TODO.md            GPU 侧施工清单
├── mine_cases.py      caption 生成 + CHAIR 匹配 + mention 定位 → 冻结清单
├── run.py             冻结清单 → CaseBatch + Protocol → VLDiagnoseLoop → run_m4
├── config.yaml
├── Dockerfile / docker-compose.yml
└── data/
    ├── image_list.json        选图清单（含选择标准元数据）
    ├── gt_objects.json        每图 GT 物体（离线提取，提交进 git）
    ├── chair_synonyms.json    CHAIR 同义词表
    └── cases/                 冻结 manifest（每模型尺寸一份）

（运行后由 agent 产生，留在本目录：）
├── deco_probe.py      mention 级前缀重喂探针
└── deco_fix.py        stepwise DeCo 生成
```
