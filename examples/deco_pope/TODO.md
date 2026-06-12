# TODO — deco_pope 施工清单（在有 GPU 的机器上由 coding agent 执行）

> 设计依据全部在 [`DESIGN.md`](DESIGN.md)，本清单只列"做什么、怎么验收"。
> 原则：**不修改 `evalvitals/` 包内任何代码**；所有新逻辑留在本目录。
> 建议先用 `qwen3-vl-2b-instruct` 走通全流程（失败多、显存小），再跑 4b/8b。

## Step 0 — 环境自检

- [ ] GPU 可见（`nvidia-smi`），bf16 显存：2B ≈ 5GB / 4B ≈ 9GB / 8B ≈ 17GB
- [ ] `pip install -e "/path/to/evalvitals[local,data]"`，transformers ≥ 4.57
- [ ] `pytest`（包自带 fast tests）通过，确认环境没坏
- [x] 判别模型（M2/M3/M5 的 judge）可用：默认 `ClaudeModel(model="claude-fable-5",
      effort="low")`（agy 配额已耗尽，2026-06-12 切换；测试可
      `--judge-model sonnet|haiku`），容器挂载见 docker-compose.yml

## Step 1 — 完成 mine_cases.py 并挖出冻结清单

- [ ] 实现 `fetch_pope_probes()`：下载 POPE 两个 split（URL 在文件顶部，
      **先核实路径是否仍有效**，并固定 commit hash），解析 JSON-lines
      （`{"question_id","image","text","label"}`，从 "Is there a X" 抽物体名），
      按 DESIGN §4.2 组装同图三元组
- [ ] 核实 greedy 参数透传方式（`do_sample=False, max_new_tokens=8`——查
      `models/backends/hf_local.py` 的 generate kwargs / RuntimeConfig）
- [ ] 核实 tokenizer 取法（`model.tokenizer` 还是 processor.tokenizer）
- [ ] 跑 `python mine_cases.py --model qwen3-vl-2b-instruct --n-images 100`
- [ ] manifest 里补记 transformers/torch 版本（漂移校验用）

**验收**：`data/cases/qwen3-vl-2b-instruct.json` 存在；yields 表里
explore 的 fail ≥ 15。不足→提高 `--n-images`；8B 仍不足→并入 AMBER
（DESIGN §8）。三元组完整率 > 90%（个别图缺 random probe 可丢弃整图）。

## Step 2 — 完成 run.py 的 loop 接线并跑通 M1→M5

- [ ] 按 `examples/qwen_loop_agy/run.py` 的样式补全 judge 构造（agy 或 API）
- [ ] `python run.py --model qwen3-vl-2b-instruct --smoke-test` 先过 schema 检查
- [ ] 真跑：确认漂移校验通过；`outputs/logs/run_log.jsonl` 有 M1/M2/M3/M5 事件；
      M1 tier-(a) 应自动选中 `pope`（cases 带 `pope_label`）等 analyzer

**验收**：loop 正常退出（resolved 或 max_cycles）；`report.final_hypotheses`
非空且至少一条假设谈及 absent-object / co-occurrence / prior 方向
（protocol 一致性检验在 M5 内部完成）。

## Step 3 — 机理探针（预期 tier-(a) 不够用时）

包内 `logit_lens` 不做 final norm 且只读单 case（DESIGN §3.4），逐层压制信号
大概率要靠：

- 优先：给 `ProbeAgent` 配 `WhiteboxProbeGenerator`（tier-(b)），让 loop 自己写探针；
- 否则：在本目录写 `deco_probe.py`，规格照 DESIGN §6 第一条逐条实现
  （final norm 后 unembed、全词表 softmax、token 等价集求和、窗口用深度比例、
  输出 per-case `s_supp / activated_gt / delta_final`，schema 对齐
  `findings["per_case"]` 以便 M2 收割）

**验收**：explore 上能产出每个 case 的 `activated_gt`；
报告 `activated_rate_fail` vs `activated_rate_pass`（**两者差距即本实验主结论**，
差距不显著 = 幻觉非 DeCo 型，照实记录，不算失败）。

## Step 4 — M4 修复（DeCo 单步重打分）

- [ ] 在本目录写 `deco_fix.py`：`deco_rescore_answer(model, inputs, alpha, window)`，
      公式照 DESIGN §6 第二条（候选 top-k20→top-p0.9 的**边界方向对照
      [官方 repo](https://github.com/zjunlp/DeCo) 核实**；二元任务单步即可，
      不要接管 generate）
- [ ] α 与窗口只在 explore 网格（config.yaml 给了网格）；
- [ ] 通过 `loop.run_m4(report, cases)` 的 `verify_fn` 注入，或独立脚本跑
      validate 一次性验证

**验收**（validate，一次性）：
1. FAIL→PASS 翻转率 > PASS→FAIL 回归率，`stats.compare(..., paired=True,
   cluster_by=image_ids, min_effect=config)` 给出 REJECT H0；
2. 翻转集中在 `activated_gt==True` 的 failure（机理一致性）；
3. present(yes) 对照组 accuracy 不降。

## Step 5 — 固化产物

- [ ] `data/cases/*.json` 提交进 git；`outputs/` 不提交（.gitignore）
- [ ] 在本文件末尾追加一节"运行记录"：模型、yields、activated_rate、
      修复前后指标、最终采纳/拒绝的假设
- [ ] 4b / 8b 重复 Step 1–4（每尺寸独立 manifest 与调参）

## 已知坑（提前避雷）

- POPE 原始 JSON 是 **JSON-lines** 不是数组；`image` 字段带 `COCO_val2014_` 前缀
- 同一物体 "Yes/No" 的 token 等价集只保留**单 token** 变体（多 token 变体在
  pos=-1 不可比）
- Qwen3-VL 是 DeepStack：早层（< 0.3N）轨迹不要拿去解读
- 窗口/层号一律从 config 读层数换算，2B 与 4B/8B 层数不同
- 漂移校验失败 ≠ 代码错：先查 transformers 版本是否与 manifest 记录一致
- transformers 4.57 的 processor **不接受 str 图片路径**（`make_list_of_images`
  无 str 分支）——`Inputs.image` 必须传 PIL（mine/run 均已如此）
- PyPI 默认 torch 2.12 是 cu130，driver 550（CUDA 12.4）带不动 → 装
  cu124 专用 wheel（torch 2.6.0，同 Dockerfile）

## 运行记录

### 2026-06-12 挖掘（Step 1 完成）

- 环境：RTX A6000、torch 2.6.0+cu124、transformers 4.57.6、bf16、greedy
  （`do_sample=False, max_new_tokens=8`）、seed 42
- 数据：POPE COCO adversarial/random 全部 **500 图**（三元组完整率 100%），
  commit 固定 `08d957b9`；三元组规则：各类型取文件序第一个可用 probe，
  random-absent 强制 ≠ adversarial-absent
- yields（cases = 1500/尺寸；FAIL 按 probe 类型细分）：

| model | adversarial | present | random | explore F/P | validate F/P |
|---|---|---|---|---|---|
| 2B | **41F**/459P | 59F/441P | 5F/495P | 55/845 | 50/550 |
| 4B | **35F**/465P | 70F/430P | 4F/496P | 61/839 | 48/552 |
| 8B | **50F**/450P | 68F/432P | 5F/495P | 71/829 | 52/548 |

- 全部尺寸 `parse_yes_no` 无法解析的回答 = 0
- **共现先验梯度成立**：同为 absent，adversarial 失败率是 random 的 ~8–10×
  （2B 8.2×、4B 8.8×、8B 10×）——与 DESIGN §4.2 的 H_deco 预期一致；
  present 假 No（12–14%）是并存的另一失败模式（漏检），留给 loop 区分
- explore fail 全部 ≥ 15 验收线（55/61/71）
- judge：agy 配额耗尽，2026-06-12 起默认
  `ClaudeModel(model="claude-fable-5", effort="low")`（探活通过）；
  测试可 `--judge-model sonnet|haiku`
