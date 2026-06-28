# LAMBDA 图表/观察 → M3 + Dashboard 集成设计（A+B+C）

> 状态：**设计提案（未实现）**。本文档定义如何让 in-loop **M3（DiagnosisAgent）**在产出假设之外，也像 `evalvitals chat --dashboard` 那样**消费并输出 LAMBDA explorer 的图表/观察**，并为整条诊断循环渲染一个 dashboard。
> 配套：[DESIGN.md](DESIGN.md)（LAMBDA×M2 信号集成，Phase A–D 已实现）。本设计是其自然延伸——前者接的是**信号**(进 M2 确证)，本设计接的是**机制语言**(图表/观察，进 M3 提假设)。
> 一句话：**explorer 的图表/观察是描述性的"机制语言"，只进 M3 的 prompt 与 dashboard，永不进 M2/M5/修复门。** 复用 [DESIGN.md](DESIGN.md) 的护栏「图表无权威」。

---

## 1. 目标与原则

**目标**：在真实诊断循环里，M3 提假设时**看得到** explorer 在 EXPLORE 上自由 EDA 得到的图表与观察(机制语言)，从而提出**更有机制依据**的可证伪假设；并把"explore 图表 → M2 统计 → M3 假设 → M5/M4/Fix"渲染成一份 dashboard，达到 `chat --dashboard` 的可观察性。

**核心原则(不变量)**：
- **图表/观察 = 描述性、未认证。** 只进 **M3 的提假设 prompt** 与 **dashboard**。**绝不进 M2(确证)、M5(检验)、Fix(修复门)。** —— 这正是 [DESIGN.md](DESIGN.md) §4 第④步「M3 读 (b) LAMBDA observations/charts(机制语言)」+ 护栏「图表无权威」。
- **explorer 不在循环内重跑**(省钱 + 守 split 纪律)：循环**消费 Step 1 的 explore 报告**(`fused_report.json`)。explorer 只在 Step 1 的 EXPLORE 留出集上跑过；M3 读它的描述性产物，不碰 confirm 数据。
- **图表宿主渲染、确定性**：优先用宿主从 chart spec + CSV **确定性渲染** PNG，而非信任 LLM 写的 matplotlib 代码(可能缺包/有 bug/不确定)。
- **机制语言 informs *哪个* 假设,不 informs *是否* 为真**：chart 启发的假设照样走 M5(gated on M2) + M4 + e-BH 裁决。即使图表来自循环信号(如 deco_hallu 的 probe1≈label),下游严谨仍兜底(实测中语言先验假设被 M4 实验确证、best-of-N 修复被 e-BH 拒)。

---

## 2. 现状与三处缺口（已核实，file:line）

`evalvitals chat --dashboard` 的探索侧已能产图(`ExploratoryAnalysisReport.charts/plots/tables`)并有 dashboard([dashboard.py](dashboard.py) `launch_dashboard`/`load_session`)。但 in-loop M3 拿不到,三处缺口：

| 缺口 | 现状(file:line) |
|---|---|
| **A 渲染** | explorer 产 chart **spec**(`fused_report.json` 的 `charts`:`{name,kind,data(CSV),x,y,title}`),但 **venv 无 matplotlib** → 图未渲染(实测 caveat `plot skipped: No module named 'matplotlib'`);run 脚本无 `--dashboard`。 |
| **B 喂 M3** | `DiagnosisAgent.diagnose(analysis, model_name, prior_cycles)`([diagnosis.py:370](../eval_agent/stages/diagnosis.py))**无 explorer 入口**;prompt(:447 `_DIAGNOSE_PROMPT.format(...)`)无 explore 段;Phase C 桥接只注入 per_case **信号**,不注入 charts/observations。 |
| **C M3 出图** | M3 的 `DiagnosisResult` 不记录引用了哪些图;dashboard `load_session`(:dashboard.py)只读 chat 的 `turn_*/exploratory_report.json`,**不读循环 run**(`logs_m2_5/`)。 |

**已具备、可复用**：
- M3 已会把图当 image 传给多模态 judge：[diagnosis.py:456-461](../eval_agent/stages/diagnosis.py) `if "images" in sig: judge.generate(prompt, images=_figs)`;`ClaudeModel.generate(inputs, images, ...)` **接受 images** → **M3 能"看见"图表**(不只读文字描述)。
- explorer 报告已带 `observations/charts/caveats`;`write_report_artifacts` 已落 `figures/` `tables/` `exploratory_report.json`([chat.py](chat.py))。
- dashboard 框架已在([dashboard_app.py](dashboard_app.py) + `launch_dashboard`),仅缺"循环 run"加载器。

---

## 3. 目标架构

```
Step 1  run_fused.py
  explorer → FusedReport{observations, charts(spec), caveats}
  └─[A] render_chart_specs(charts, tables/, figures/) → 宿主确定性渲染 PNG
  → 落 fused_report.json + figures/*.png + tables/*.csv

Step 2  run_m2-5.py --explore-report outputs/fused/fused_report.json  [--dashboard]
  load ExploreContext(observations, charts{title,desc,figure_path}, caveats)
  → VLDiagnoseLoop(explore_report=ctx)            [B] 透传(不重跑 explorer)
      M1(replay) → 桥接信号 → M2(确证, 不见 explore_ctx)
      → M3 DiagnosisAgent.diagnose(analysis, explore_context=ctx)   [B 核心]
          prompt += explore_section(描述性, 标 UNCONFIRMED)
          judge.generate(prompt, images=[M2图] + [explore PNG])     [B] M3 看见图表
          → DiagnosisResult.referenced_charts = [...]               [C] 记录引用
      → M5(gated on M2) → M4 → Fix(e-BH)          ← explore_ctx 全程不进这些门
  → run_logger 落 explore charts + M3 引用         [C]
  └─[A/C] dashboard: load_loop_run(run_dir) → 渲染
          explore 图表 → M2 统计 → M3 假设(带引用) → M5/M4/Fix
```

**角色一句话**：图表谁产=explorer(Step1);谁渲染=宿主(确定性);谁看=M3(多模态,提假设);谁渲染成 dashboard=load_loop_run;谁**绝不**碰图表=M2/M5/Fix。

---

## 4. 三部分接口设计

### A — 渲染 + dashboard 依赖

```python
# evalvitals/analysis/charts.py（新）—— 宿主确定性渲染,不信任 LLM 的 matplotlib 代码
def render_chart_specs(
    charts: list[dict],        # explorer 的 chart spec: {name,kind,data(CSV相对路径),x,y,title}
    tables_dir: str | Path,    # CSV 所在目录(sandbox 拷出的 tables/)
    out_dir: str | Path,       # 输出 figures/
) -> list[dict]:
    """把每个 chart spec 从其 CSV 表确定性渲染成 PNG(bar/line/scatter)。
    返回带 figure_path 的 chart dict 列表;matplotlib 缺失则跳过并回退到纯文字描述。
    确定性、宿主侧、可审计 —— 不执行 LLM 写的绘图代码。"""
```
- env：venv 装 `matplotlib`;`pyproject.toml` 加 `[project.optional-dependencies] viz = ["matplotlib>=3.5"]`、`dashboard = ["streamlit>=1.30"]`。
- `run_fused.py` 调 `render_chart_specs` 把 spec 渲成图;加 `--dashboard` → `launch_dashboard(out_dir)`。

### B — 喂 M3（核心）

```python
# evalvitals/eval_agent/stages/diagnosis.py
@dataclass
class ExploreContext:
    """explorer 的描述性机制语言 —— 永不权威。只读,只进 M3 prompt。"""
    observations: list[str] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)   # {title, kind, description, figure_path}
    caveats: list[str] = field(default_factory=list)
    source: str = "lambda_explorer"                     # provenance

    @classmethod
    def from_report(cls, data: dict) -> "ExploreContext": ...  # 从 fused_report.json 装

class DiagnosisAgent:
    def diagnose(self, analysis, model_name="", prior_cycles=None,
                 explore_context: "ExploreContext | None" = None) -> DiagnosisResult:
        ...
        explore_section = _format_explore_section(explore_context)  # 见下,标 UNCONFIRMED
        prompt = _DIAGNOSE_PROMPT.format(..., explore_section=explore_section)
        _figs = [M2 figures] + [c["figure_path"] for c in (explore_context.charts if explore_context else []) if c.get("figure_path")]
        raw = self.judge.generate(prompt, images=_figs) if "images" in sig else self.judge.generate(prompt)
        ...
        result.referenced_charts = _extract_referenced(raw, explore_context)  # C
        return result
```

`_DIAGNOSE_PROMPT` 新增 `{explore_section}` 占位,渲染为**强标注的描述块**：
```
EXPLORATORY MECHANISM NOTES (free-form EDA on a HELD-OUT explore split — DESCRIPTIVE
and UNCONFIRMED; use ONLY to decide WHICH hypotheses to propose, NEVER as evidence
that one is true; every claim below must still be tested downstream):
  observations:
    - <obs 1> ...
  charts (attached as images):
    - [<title>] <description>
  caveats (the explorer's own warnings):
    - <caveat> ...
```

透传(不重跑 explorer)：
```python
# loop.py: VLDiagnoseLoop.__init__(..., explore_report: ExploreContext|None=None)
#   self._explore_context = explore_report
# loop.run 调 M3 处: self.diagnosis_agent.diagnose(stats_report, explore_context=self._explore_context)
# run_m2-5.py: --explore-report fused_report.json → ExploreContext.from_report(json) → VLDiagnoseLoop(explore_report=ctx)
```

### C — M3 出图 + dashboard

```python
# DiagnosisResult 加:
#   referenced_charts: list[str]      # M3 假设引用到的 chart title/observation
#   explore_context_used: bool
# run_logger.log_diagnosis 落 explore charts(图路径)+ M3 引用,进 run 目录。

# evalvitals/analysis/dashboard.py: 新增循环加载器
def load_loop_run(run_dir: str | Path) -> dict:
    """读 fused_report.json(explore 图表/观察)+ logs_m2_5/run_log.jsonl(M2 统计 / M3 假设+引用 /
    M5 / Fix),组装成统一"诊断故事"视图供 dashboard_app 渲染。"""
```
dashboard_app 加一个"loop run"视图：explore 图表 → M2 forest/统计 → M3 假设(每条标其引用的图/观察)→ M5 verdict → Fix 候选 + e-BH。

---

## 5. 护栏（不可协商，复用 + 新增）

| 护栏 | 规则 | 支撑 |
|---|---|---|
| **图表无权威** | explore_context 只进 M3 prompt + dashboard;**M2/M5/Fix 永不接收它** | explore_context 仅是 `DiagnosisAgent` 的参数,M2/M5/Fix 签名不含它 |
| **机制语言 ≠ 证据** | prompt 强标 UNCONFIRMED;M3 假设照走 M5(gated on M2)+M4+e-BH | 现有 M5/M4/e-BH 链路不变 |
| **不重跑 explorer** | 循环消费 Step1 报告;explorer 只见 EXPLORE 留出集 | `--explore-report` 透传,无 in-loop explorer |
| **宿主渲染** | 图从 spec+CSV 确定性渲染;不执行 LLM 绘图代码 | `render_chart_specs`(新) |
| **provenance** | 每张图/观察标 `source=lambda_explorer, exploratory`;dashboard 显式区分"探索(描述)"vs"确证(认证)" | ExploreContext.source + dashboard 分区 |

---

## 6. 落地计划（分阶段，file-level）

**A — 渲染 + dashboard 依赖**
- A1（S）venv 装 matplotlib;`pyproject.toml` 加 `viz`/`dashboard` extra。
- A2（M）`analysis/charts.py`:`render_chart_specs`(bar/line/scatter,从 CSV 确定性渲染,缺 matplotlib 优雅回退)。 → 新模块
- A3（S）`run_fused.py`:渲染 chart spec + `--dashboard` → `launch_dashboard`。

**B — 喂 M3（核心）**
- B1（M）`ExploreContext` dataclass + `_format_explore_section` + `_DIAGNOSE_PROMPT` 加 `{explore_section}`;`diagnose(..., explore_context=None)`;figures 合并传 `images=`。 → [diagnosis.py](../eval_agent/stages/diagnosis.py)
- B2（M）`VLDiagnoseLoop(explore_report=...)` + run 调 M3 处透传;`run_m2-5.py --explore-report`。 → [loop.py](../eval_agent/loop.py) / run_m2-5.py

**C — M3 出图 + dashboard**
- C1（S）`DiagnosisResult.referenced_charts`/`explore_context_used`;`run_logger.log_diagnosis` 落 explore 图 + 引用。
- C2（M）`dashboard.py: load_loop_run` + `dashboard_app` 的 loop-run 视图(explore 图 → M2 → M3 假设+引用 → M5/Fix)。

**D — 测试**
- 双盲守卫：构造带 explore_context 的 M3 调用,断言 explore_context **从不**出现在 M2/M5/Fix 的输入(签名 + 运行期)。
- 渲染:chart spec + CSV → PNG 确定性(同输入同图);matplotlib 缺失优雅回退到文字。
- M3 读取:explore_section 进 prompt、figures 进 `images=`;`referenced_charts` 正确抽取。
- 端到端(轻):run_fused → fused_report.json → run_m2-5 --explore-report → M3 prompt 含 explore_section(用 ScriptedJudge,无 GPU)。

---

## 7. 明确不要做

- ❌ **把图表/观察当确证**——它们是数据选出来的、未认证。只进 M3 prompt 与 dashboard。
- ❌ **在循环内重跑 explorer 去"现场出图"**——重引入发现-then-测同批 double-dip,且费钱。消费 Step1 留出集报告。
- ❌ **执行 LLM 写的绘图代码渲染**——用宿主 `render_chart_specs` 从 spec+CSV 确定性渲染(安全 + 确定性)。
- ❌ **让 explore_context 影响 M5/Fix 的门**——M3 用它选**哪个**假设,真假仍由 M5/M4/e-BH 定。

---

## 8. 开放问题 / 风险

- **循环信号的图会不会误导 M3?** 会启发(如 deco_hallu 的 probe1≈label),但下游 M5/M4/e-BH 兜底(实测:语言先验假设被 M4 确证、best-of-N 修复被 e-BH 拒)。可在 explore_section 里把 explorer 自己的循环 caveat 一并呈现给 M3。
- **多模态 judge 的图像支持**:`ClaudeModel.generate` 接受 `images=`,但 claude CLI 实际是否消费图像需实测;不支持则 M3 退化为读图的**文字描述**(仍有用)。
- **依赖**:matplotlib(渲染)+ streamlit(dashboard)是可选 extra,默认环境可能没有 → 优雅回退(无图则纯文字 explore_section;无 streamlit 则 dashboard 不可用但 run 正常)。
- **estimand/chart 一致性**:chart 的 CSV 表与 confirm 上的信号可能指不同切片;chart 仅描述性,不参与裁决,故无统计风险,但 dashboard 应标注"探索切片,非 confirm"。
- **chat 路径复用**:本设计的 `render_chart_specs` 也可回填到 `evalvitals chat`,让独立路径的图同样宿主渲染(去掉对 LLM matplotlib 代码的依赖)。

---

*依据:diagnosis.py(diagnose/_DIAGNOSE_PROMPT/images 传参,file:line 见正文)、analysis/{explorer,fused_pipeline,dashboard,dashboard_app,chat}.py、loop.py、cli_agent.py(ClaudeModel.generate 签名)。本设计是 [DESIGN.md](DESIGN.md) 的延伸(信号→M2 之外,补 机制语言→M3 + dashboard)。*
