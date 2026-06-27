# LAMBDA 探索 × M2/M5 确证 集成设计

> 状态：**设计提案（未实现）**。本文档定义如何把 LAMBDA 式探索层（`evalvitals/analysis/`）安全地接入 M1→M5 诊断-修复链路，使系统**既能用 LLM 自由写代码探索、又只用已验证的统计引擎裁决**，最终服务于"失败样例 → 提假设 → 修复"。
>
> 配套参考：`/tealab-data/jiaqiliu/evalsmith/LAMBDA/LAMBDA_架构与设计原理.md`（LAMBDA 原理）。
>
> 一句话：**用 LAMBDA 提议、用 M5（gated on M2）验证、用 FixAgent 修复——永不用一张图来确证。** M2 作为"无偏 family-FDR 防火墙"保留，新增 **operationalization bridge** 把 LAMBDA 发现的复合/切片/形状信号编译成冻结的 per-case 信号，使其进入 M2 的 e-BH family、在留出集上确证。

---

## 1. 目标与设计原则

**终极目标**：从失败样例出发，**提出关于失败机理的假设**，并据此**修复**。一个"可修复级"的假设必须同时满足两条轴：

- **(i) 统计真实**——该信号确实区分 fail/pass，且在多重比较 / optional stopping 下成立。
- **(ii) 机制可操作**——干预这个信号能改变结果（FixAgent 用配对 e-value 验证修复）。

两条轴由不同部件擅长：**M2 管 (i)**（无偏边际-FDR 扫描 + 防确证偏倚），**LAMBDA 管 (ii)**（交互/切片/形状/复合谓词 + 图表，catalog 表达不了）。二者**正交互补**。

**核心原则：explore freely, confirm rigorously（探索自由、确证严谨）。**
- 探索侧（EXPLORE split）：LLM 自由写代码、出图、提候选——**只提议，不裁决**，无 e-value 权威。
- 确证侧（CONFIRM split，留出）：一切"是否显著 / 是否 SUPPORTED / 是否修好"只走已验证引擎（`compare()` → e-value → e-BH FDR），**且在提案没碰过的数据上算**。

---

## 2. 现有链路与关键依赖（已核实，file:line）

```
M1 ProbeAgent        跑 analyzers → probe_results（Result.findings["per_case"] 是关键载体）
  → M2 StatsAnalysisAgent   default_plan 在「全部 per_case 信号」上跑已验证工具族
                            → fdr_correct/e-BH → corrected_rejections.rejected_tools
                            + conclusion + evidence_chain
  → M3 DiagnosisAgent       读 M2 的 conclusion/evidence + raw findings → 提「可证伪假设」
  → M5 HypothesisTester     检验每个假设
  → M4 SurgeryAgent         设计实验确证
  → FixAgent                修复；配对 McNemar/e-value，候选族 e-BH，留出 CONFIRM split
```

三条**载荷依赖**（决定了"能不能删 M2"）：

1. **M2 的信号族只从 `per_case` 构造**。`build_stats_input`（[stats_tools.py:175](../eval_agent/stages/stats_tools.py)）从 `findings["per_case"]`（:202-212）收集 `{"analyzer.metric" -> {case_id -> value}}`；`default_plan` 只在这些 key 上建工具族。**一个信号若不在某 analyzer 的 `per_case` 输出里，它永远进不了 M2 的 family、拿不到 e-value、永远不可能出现在 `rejected_tools`。**
2. **M5 不自洽，gated on M2**。`_verdict_from_stats_results._decisive`（[hypothesis_tester.py:324-343](../eval_agent/stages/hypothesis_tester.py)）：一个带 e-value 的结果只有当 `r.tool ∈ corrected_rejections.rejected_tools` 才算 decisive。M5 是 M2 survivor set 的**消费者**。
3. **M3 优先读 M2 的 conclusion/evidence**（[diagnosis.py:402-411](../eval_agent/stages/diagnosis.py)）。

M5 的 label-free 回退路径也只从 `findings["per_case"]` 取信号（`_extract_per_case_signals`，[surgery.py:159](../eval_agent/stages/surgery.py)；被 hypothesis_tester.py:475 使用）。

**已具备的护栏（漏洞已修）**：
- **充分统计量 + 宿主重算决策**：生成工具只发 `sufficient`，`_reconstruct_decision` 宿主算 e-value（[stats_tool_generator.py:328](../eval_agent/stages/stats_tool_generator.py)）。LLM 提"算什么"，宿主决"是否拒绝"。
- **候选族 e-BH**：FixAgent `_ebh_survivors`（[fix_agent.py:560](../eval_agent/stages/fix_agent.py)，`from evalvitals.stats.ebh import ebh`）。
- **留出 CONFIRM split**：`VLDiagnoseLoop._split_explore_confirm`（[loop.py:760](../eval_agent/loop.py)，`confirm_split` 参数 :718）：确定性、`(label, probe_type)` 分层、按 id 不相交。
- **M5 fallback 多重性控制**（leak #6 已修）。

---

## 3. 核心设计判定

### 3.1 保留 M2，不要删（RECONCEIVE）

"删 M2 只留 M5" **三连击失败**：
1. **机械上立刻坏**——删了 M2，`corrected_rejections` 为空 → `_decisive` 对每个带 e-value 的假设返回非 decisive → **全部塌成 INCONCLUSIVE**。
2. **省不了，只是搬家**——M2 的三件事（全信号无偏扫描 / e-BH→gate / conclusion+evidence）都得搬进 M5。
3. **丢掉防火墙**——M2 在 M3 提假设**之前、独立**跑完，是**防确证偏倚的防火墙**。删了它就变成"只检验 LLM 看过数据后提的假设、还在同一批数据上"= 选择性推断；逐假设 e-value 救不了（控的是每次检验的错误率，不控"选哪些来检验"），且照样打印自信 SUPPORTED。

**仅当 M5 内化 M2 的全部保证**（对全信号族盲扫 + e-BH 重建 rejected_tools + 留出 confirm + 仍产 conclusion）才安全——那不是删 M2，是 **inline 了 M2**，边界没了但统计契约还在。结论：**保留 M2 作为防火墙，扩展其信号族，不删、不并进 M5。**

### 3.2 LAMBDA = 探索侧提议者，永不验证

LAMBDA explorer（[explorer.py](explorer.py)）是 **proposer-feeder**：自由 codegen EDA，产 observations / candidate_signals（每个带「确定性信号配方」+ suggested_test）/ charts，**不发 e-value、不发 reject**（schema 强制，:49）。它**增强/替代 M3（提议者）**，但 **proposing ≠ validating**——它的图表/代码统计**无裁决权**，否则就是"提议者在同一批数据上给自己打分"。

**图表只在探索侧**：喂 M3 的 prompt（机制语言）+ 人工引导下一轮 EDA。一张图都不进确证或修复验证。

### 3.3 不要"两条确证引擎合并"

"跑两条路、合并两个统计结果"几乎不成立：explorer 现在**不产 e-value**，没有第二个量可合并；要让它产，就得走充分统计量 → 它就**不再是第二个引擎**，而是同一个宿主核心。**正确分解**：
- **发现（测什么）= 两源**：LLM explorer ∪ catalog planner，在 EXPLORE split 上，并集去重。
- **裁决（是否真）= 一路**：所有候选进**同一个 e-BH family**，在 CONFIRM split 上算。**永不合并两个裁决。**

唯一合法的"合并 e-value"是罕见审计场景：两条路对**同一 null**、各自宿主算出真 e-value、在有权使用的 confirm 数据上——此时**只能算术平均** `E=(E₁+E₂)/2`（相关稳健）；**product/sum 非法**（重复计证，E[E] 爆过 1）；**同一 null 绝不进两次 e-BH**（先去重）。不同 null → 各作为独立成员进同一个 e-BH family（并集，不是算术）。

---

## 4. 目标架构

```
            ┌──────────────── EXPLORE split（选/提议）────────────────┐
输入 data ─┤  M1 analyzers ─────────────────────────────────────────┐ │
 _split_   │  LAMBDA explorer（自由 EDA / charts / candidate_signals │ │  并集 + 去重
 explore_  │    = 信号配方 + suggested_test；无 e-value）─────────────┘ │  （按 estimand）
 confirm   └──────────────────────────────────────────────────────────┘
   │                                  │
   │            ② OPERATIONALIZATION BRIDGE（核心新组件）
   │               把每个候选的「配方」编译成「冻结 per-case 提取器」
   │               → 作为 finding 落进 inp.per_case 的新 signal key
   ▼                                  ▼
CONFIRM split ── ③ M2 = 防火墙（保留）：validated catalog 跑「扩展后的信号族」
                    （catalog 信号 + 桥接来的 LAMBDA 信号）
                    → e-BH FDR → rejected_tools + conclusion/evidence_chain
                    【盲于、先于 M3；且在留出 CONFIRM split 上】
                                       ▼
                 ④ M3 = 提议者（增强）：读 (a) M2 survivor+evidence（什么真）
                    + (b) LAMBDA observations/charts（机制语言）→ 提可证伪假设
                                       ▼
                 ⑤ M5 = gate（契约不变）：假设 SUPPORTED ⟺ 其信号过了
                    M2 在 CONFIRM split 上的 e-BH（_decisive 仍读 rejected_tools）
                                       ▼
                 ⑥ M4 surgery + FixAgent：配对 e-value、候选族 e-BH、留出 CONFIRM 验证修复
```

**角色一句话**：谁提议=LAMBDA→M3；谁验证=M5（gated on M2）；图表去哪=只在探索侧；family-FDR 防火墙在哪=M2（独立、提案前、CONFIRM split）；留出=LAMBDA/EDA+M3 提案在 EXPLORE，M2 确证+FixAgent 验证在 CONFIRM。

---

## 5. Operationalization Bridge（核心新组件，接口草案）

这是**目前缺失、必须新建**的部件——把 LAMBDA 的丰富性接进验证引擎的**唯一通道**。一个假设是句话、不可直接检验；必须先被编译成**一个冻结、可复现的 per-case 信号函数**（case→value），作为 analyzer-grade finding 落进 `inp.per_case`。

### 5.1 数据契约

LAMBDA explorer 的 `candidate_signal` 扩展为携带一份**确定性配方**（仍无 e-value）：

```python
@dataclass
class SignalRecipe:
    name: str                      # 新 per-case 信号 key，如 "explored.small_and_peripheral"
    description: str               # 机制语言（给 M3 读）
    kind: str                      # "expr" | "code"
    # kind == "expr": 受限 DSL，仅对【已存在的 per_case 列】做交互/阈值/复合
    #   例: "(obj_size < 40) and (attention_focus_share < 0.3)"
    expr: str = ""
    # kind == "code": 沙箱 codegen，对每个 record 算一个值（用于全新信号）
    code: str = ""
    suggested_test: str = ""       # 建议路由到的具名工具，如 "signal_label_assoc"
```

### 5.2 编译：配方 → 冻结提取器 → per_case finding

```python
def compile_recipe(recipe: SignalRecipe, records) -> dict[str, float]:
    """把配方编译成 {case_id -> value}，即一个 per_case finding 条目。
    - kind == "expr": 在【已有 per_case 列】上求值（无 codegen，最安全、确定性）。
    - kind == "code": 在 ExperimentSandbox 跑，复用 leak-1 充分统计量契约的执行/解析路径；
      宿主只接受「每 case 一个标量」的形状，按内容 hash 冻结，禁止网络/写仓库。
    产物作为 Result.findings["per_case"] 的新条目，于是 build_stats_input 自动收进 inp.per_case。
    """
```

两种 `kind` 的取舍：
- **`expr`（首选）**：交互/阈值/切片**本质上是把复合谓词归约成一列**已测信号的布尔/连续函数——`signal_label_assoc`/`rank_corr`/`mcnemar_evalue` 就能测。无 codegen、确定性、可审计。**绝大多数 LAMBDA 发现应走这条。**
- **`code`（兜底）**：信号尚不是任何已测列时（如"答案为空/全空白"），用沙箱 codegen 算每 case 的值。复用 `_reconstruct_decision` 的沙箱+宿主校验路径，**只发标量、宿主冻结**。

### 5.3 注册进 M2 引擎并确证

```python
# 1) 把编译出的信号塞进一个「合成 analyzer」的 per_case finding
synthetic_findings["per_case"].append({"id": cid, recipe.name: value, ...})
# 2) 扩展 default_plan：为新信号加一条已验证工具
plan += [(run_stats_tool, "signal_label_assoc", {"signal": recipe.name})]
# 3) 在 CONFIRM split 上跑 M2：fdr_correct 把【catalog 信号 + 桥接信号】放进同一个 e-BH family
#    —— 于是「多提候选 → 多付多重性代价」自动成立
# 4) M5._decisive 不变：信号过了 e-BH → 进 rejected_tools → 假设可 SUPPORTED
```

**关键不变量**：信号在 **EXPLORE split** 上被发现/选中，在 **CONFIRM split** 上被编译求值 + 确证；二者 case id 不相交（避免 double-dip）。

---

## 6. 统计护栏（不可协商）

| 护栏 | 规则 | 现有支撑 |
|---|---|---|
| **无 double-dip** | 选在 EXPLORE、每个 e-value 算在冻结 CONFIRM | `_split_explore_confirm`（loop.py:760） |
| **宿主裁决** | LLM 写码只发充分统计量/标量信号；reject/e_value/p_value 一律宿主重算 | `_reconstruct_decision`（stats_tool_generator.py:328） |
| **统一 e-BH family** | catalog 信号 + 桥接信号进**同一个** `fdr_correct` 族；提更多候选→付更多多重性 | `fdr_correct`/`ebh`（stats_tools.py / stats/ebh.py） |
| **同 null 不双进** | 撞同一 estimand 先去重为一个（仅算术平均），再进 e-BH | 新增去重步 |
| **图表无权威** | charts/observations 只进 M3 prompt 与人工引导，绝不进确证/修复门 | explorer schema 无 reject（:49） |
| **M2 盲于假设** | M2 在 M3 提假设前、独立跑完整信号族 | 现有 M2→M3 顺序 |

---

## 7. 输出 schema（按 provenance 组装）

最终对象 `{observations, candidate_signals, charts, recommended_confirmatory_tests}`：

- **observations**——来自 explorer（描述性、未认证、来自 EXPLORE），自由拼接。
- **charts**——explorer 的 figures/tables ∪ 可选 `plot_effects`（confirmed 候选的 effect±CI forest 图）。描述性。
- **candidate_signals**——两源并集去重，每个**标注宿主裁决**：`source ∈ {explorer, catalog, both}`、`effect`、`ci`、`e_value`、`reject (= tool ∈ rejected_tools)`、`underpowered`、`host_adjudicated`、`descriptive_only (sufficient/recipe 不可归约时)`、`confirmed_on='confirm'(split id/seed)`。可选 `e_value_merged + merge_method='arithmetic_mean'`。
- **recommended_confirmatory_tests**——无法自动归约的候选（配方为空 / confirm 上无数据 / underpowered）+ explorer 显式建议的。**优雅降级通道：不静默丢弃、不过度宣称。**
- 整体盖 e-BH family 元数据（`method='e-BH', alpha, n_tested`），让读者看到多少假设在争 `m/(alpha·k)` 这条线。

---

## 8. 落地计划（分阶段，file-level）

**Phase A — 把 chat 的每个候选接上已有的宿主裁决（S/M，无新统计代码）**
- A1（S）`CandidateSignal`/`ExploratoryAnalysisReport` 加可选确证字段（`recipe`/`sufficient`/effect/ci/host 裁决），`to_dict` 向后兼容。 → [explorer.py](explorer.py) L80-140
- A2（S）explorer `_GENERATE_PROMPT` 加"每候选可附 `sufficient`/`recipe`；禁发 reject/e_value（被忽略）；只从 records.json（confirm 行）算"。 → explorer.py `_GENERATE_PROMPT`
- A3（S）解析器把 recipe/sufficient 透传进 `CandidateSignal`。 → explorer.py `_report_from_sandbox` L538
- A4（M）宿主裁决 pass：import `_reconstruct_decision`/`fdr_correct`；每候选算 StatsToolResult → `fdr_correct` → 标注 reject。 → 新函数 import 自 stats_tool_generator.py:328 + stats_tools.py `fdr_correct`

**Phase B — operationalization bridge + 双发现源接进循环（M/L，核心）**
- B1（M）`SignalRecipe` + `compile_recipe`（先做 `kind="expr"` 受限 DSL）。 → 新模块 `analysis/operationalize.py`
- B2（L）编排器：`_split_explore_confirm` 切分；EXPLORE 上跑 explorer + `default_plan`（两发现源）；按 estimand 并集去重；CONFIRM 上编译信号 + 跑 M2。 → 新模块 `analysis/fused_pipeline.py`，复用 loop.py:760 / stats_tools `default_plan`/`build_stats_input`
- B3（M）catalog 路由：候选若映射到具名工具 → `run_stats_tool` 进同一 family。
- B4（M）输出组装 + `recommended_confirmatory_tests` 兜底 + e-BH 元数据 + 可选 forest 图。

**Phase C — 把桥接信号接回 in-loop M2（L，让 VLDiagnoseLoop 也受益）**
- C1（L）让 M2 的 `StatsAnalysisAgent` 接受"额外 per_case 信号"输入，使 LAMBDA 发现的信号进 in-loop e-BH family，M3/M5 直接受益。 → stats_agent.py / probe_agent 注入点

**Phase D — 测试**
- 双盲守卫（e-value 全在 confirm，id 与 explore 不相交）；宿主裁决（喂 `reject=true/e_value=1e9` 必被丢弃重算）；默认不合并（不同 null → 两个独立 e-BH 成员）；同 null 才算术平均；确定性（同 batch+seed 复现 confirm 分区）；`expr` 编译正确性。

---

## 9. 明确不要做

- ❌ **删 M2 只留 M5**——删了 M5 的 gate，退化成 cherry-picking（§3.1）。要解决"不可表达假设永远 INCONCLUSIVE"，做法是**扩展信号族（operationalization bridge）**，不是删防火墙。
- ❌ **用图表/explorer 输出当确证**——它们是数据选出来的、未认证。只进 M3 prompt 与人工引导。
- ❌ **product/sum 合并 e-value**——相关 e-value 重复计证。只算术平均，且仅同 null。
- ❌ **同一 null 进两次 e-BH**——膨胀 m、给一个假设两次过线机会。先去重。
- ❌ **信任 LLM 自报 reject/e_value/p_value**——`_parse_result` 已丢弃 stdout 裁决、宿主重算，别绕过。
- ❌ **在 explore 数据上确证**——选与证必须分在 EXPLORE/CONFIRM。

---

## 10. 开放问题 / 风险

- **数据税**：留出 CONFIRM 缩小两边；`_split_explore_confirm` 在 ~4 case 以下 no-op，小 confirm 产更多 underpowered/descriptive。`confirm_split` 取值需权衡；underpowered 候选路由到 `recommended_confirmatory_tests` 而非过度宣称。
- **`expr` DSL 的表达力上限**：交互/阈值/复合够用，但真正全新的连续估计量需要 `kind="code"`（沙箱）。先做 `expr`，`code` 作兜底。
- **estimand 漂移**：两次调用（chart 与 test）可能指不同的 X。优先 C 式单脚本（chart 与 recipe 出自同一段代码、同一批 confirm 行），或加显式 estimand-identity 检查再合并。
- **沙箱是资源/路径隔离，非网络隔离**：explorer 的"禁网"只是 prompt 级。确证安全（裁决走宿主代码），但探索产物只在沙箱限额范围内可信，绝不当认证结果。
- **catalog 表达力**：M2 的 ~6 边际工具仍是 e-BH 防火墙的上限；bridge 的价值正是把交互/切片**归约成 catalog 能测的一列**，从而扩大防火墙覆盖面。

---

*依据：本仓库 stats_tools.py / stats_agent.py / stats_tool_generator.py / hypothesis_tester.py / diagnosis.py / surgery.py / fix_agent.py / loop.py / analysis/explorer.py（file:line 见正文），以及 LAMBDA_架构与设计原理.md。本设计由多智能体工作流（grounding → 设计面板 → 对抗性综合）推导得出。*
