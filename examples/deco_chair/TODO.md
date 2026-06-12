# TODO — deco_chair 施工清单（在有 GPU 的机器上由 coding agent 执行）

> 设计依据：[`DESIGN.md`](DESIGN.md)（差异部分）+
> [`../deco_pope/DESIGN.md`](../deco_pope/DESIGN.md)（总纲）。
> 原则同 deco_pope：**不修改包内代码**，新逻辑全留本目录。
> **先完成 deco_pope 再做本例**——(α, 窗口) 超参与 token 等价集经验直接复用。

## Step 0 — 环境自检

同 `../deco_pope/TODO.md` Step 0。

## Step 1 — 完成 mine_cases.py 并挖出冻结清单

- [ ] `select_images()`：缓存 `instances_val2014.json`（zip ~250MB，URL 在文件
      顶部），按（类别数多、含 kitchen/furniture/electronics 超类）排序取前 50，
      写 `data/image_list.json`（记录选择标准）
- [ ] `extract_gt_objects()` → `data/gt_objects.json`（小文件，提交 git）
- [ ] `data/chair_synonyms.json`：取 CHAIR 标准同义词表（Rohrbach et al.；
      `analyzers/hallucination/chair.py` 若已带 vocab 处理则直接对齐它的格式）
- [ ] `chair_match()`：复用/对齐包内 chair analyzer 的匹配逻辑，**不要重新发明**
- [ ] `token_index_of()`：processor 的 offset mapping 把 char offset 映射到全序列
      token index（prompt+caption 一次性 tokenize；注意 chat template 包裹后的偏移）
- [ ] 跑 `python mine_cases.py --model qwen3-vl-2b-instruct --n-images 50`

**验收**：manifest 含每图 caption + mentions；幻觉 caption（FAIL）≥ 10 张，
不足→换更杂乱的图或加图；抽 3 个 mention 人工核对 `token_index` 复现性
（重喂 `ids[:k]` 的 greedy next-token == mention 首 token，必须 100% 命中）。

## Step 2 — 完成 run.py 的 loop 接线并跑通 M1→M5

- [ ] 图像级 case 构造：`metadata["gt_objects"]` 按 chair analyzer 约定填，
      observed = 冻结 caption，label 按是否含幻觉 mention
- [ ] judge 构造同 deco_pope；跑 loop，确认 M1 tier-(a) 选中 `chair`
- [ ] 漂移校验：重新 caption 抽样 3 张，比对幻觉 mention 集合（非全文）

**验收**：同 deco_pope Step 2（run_log.jsonl + 非空且贴题的假设）。

## Step 3 — mention 级机理探针（本例核心，tier-(a) 必然不够）

- [ ] 在本目录写 `deco_probe.py`（或配 WhiteboxProbeGenerator 让 loop 生成）：
      对 manifest 里每个 mention，重喂 `ids[:token_index]`，按总纲 §6 规格算
      逐层轨迹 → per-mention `s_supp / activated_gt / delta_final / gt_peak_layer`
- [ ] 注意：前缀重喂需要 token 级输入——确认 `Inputs`/backend 是否接受预 tokenize
      的 ids；不行就从 offsets 重建前缀字符串（greedy 下二者等价，ids 更稳）
- [ ] `gt_token_ids` 的候选集过滤（Eq.2 前提）方向对照官方 repo 核实

**验收**：
1. hallucinated vs grounded mention 的 `activated_gt` 率差 + `delta_final` 差
   （`cluster_by=image_id`）；
2. 形态：幻觉 mention 的 `gt_peak_layer` 直方图是否集中在 [0.55N, 0.85N]
   （复现论文"20–28 层"形状；不集中也照实记录）。

## Step 4 — M4 修复（stepwise DeCo）

- [ ] `deco_fix.py`：手写 greedy 循环 + 每步 DeCo 重打分（公式同总纲 §6；
      KV cache 复用；**沿用 deco_pope 调好的 α/窗口，不在本例再调参**）
- [ ] validate 图像重生成 caption；CHAIR_S/CHAIR_I 前后对比（图像级配对
      bootstrap，`min_effect` 见 config.yaml）

**验收**：
1. CHAIR 指标下降且统计显著；
2. 被消掉的幻觉 mention 集中在 `activated_gt==True` 子集（机理一致性）；
3. 护栏全绿：distinct-2 / 最长 n-gram 重复 / caption 长度漂移 < 20% /
   grounded mention 召回不降（论文 "no free lunch"：α 过大先伤这些）。

## Step 5 — 固化产物

同 deco_pope Step 5（manifest 进 git、运行记录追加在本文件末尾、4b/8b 重复）。

## 已知坑

- chat template 包裹后 caption 的 char→token 偏移会整体平移：offset mapping
  必须对**完整模板化序列**计算，不能只 tokenize caption
- CHAIR 匹配的复数/同义词（"people"→person、"bikes"→bicycle）：用标准表，
  别用子串匹配
- mention 首 token 可能是多 token 词的一部分：等价集只取首 token，
  且 grounded 对照也同样处理（保持对称）
- stepwise DeCo 的重复护栏：α 从 0.4 起步（config.yaml），论文明示 α 过大
  会产生重复/非典型描述
