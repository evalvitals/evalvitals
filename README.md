# EvalVitals

[![CI](https://github.com/evalvitals/evalvitals/actions/workflows/ci.yml/badge.svg)](https://github.com/evalvitals/evalvitals/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalvitals)](https://pypi.org/project/evalvitals/)
[![Python](https://img.shields.io/pypi/pyversions/evalvitals)](https://pypi.org/project/evalvitals/)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://evalvitals.github.io/evalvitals/)
[![License: CC0-1.0](https://img.shields.io/badge/license-CC0--1.0-green)](LICENSE)

EvalVitals is an sklearn-like toolkit for failure-case analysis in the era of
LLMs, VLMs, omni (text+image+audio+video) models, and agentic systems.

The package is organized around a small set of uniform contracts so researchers,
engineers, and automated agents can discover, compose, and run evaluations
programmatically:

| Contract | Role |
|---|---|
| `ModelSpec` | Model identity: family, Hugging Face repo, architecture traits, VLM/Omni/MoE/MLA caveats; its `modalities` (text/image/audio/video) follow from the components present. |
| `Backend` | Runtime: local Hugging Face internals, black-box API calls, or offline batch engines. |
| `Model` | A runnable object with `generate()` and `forward(capture=...) -> Trace`. |
| `Analyzer` | An sklearn-style estimator: `Analyzer(**params).run(model, data) -> Result`. |
| `Capability` | The vocabulary used to match analyzers to compatible model runtimes. |
| `FailureCase` | The central data unit for prompts, labels, provenance, and agent trajectories. |
| `Result` | Uniform output with human-readable summaries and structured findings. |

The key idea is simple: model identity is separate from runtime, and analyzers
connect to models by capability matching. The same spec can run through a
black-box API backend or a white-box local backend; only the capability set
changes.

## Quickstart

**Text attention (LLM):**

```python
import evalvitals
from evalvitals.analyzers.attention.summary import AttentionAnalyzer

model = evalvitals.load("qwen2.5-7b-instruct")
result = AttentionAnalyzer(layer=-1, top_k=5).run(
    model,
    "The Eiffel Tower is in",
)

print(result.summary())
print(result.findings)
```

**Relative attention (VLM) — "MLLMs Know Where to Look" ([arXiv 2502.17422](https://arxiv.org/abs/2502.17422), [code](https://github.com/saccharomycetes/mllms_know)):**

```python
from PIL import Image
from evalvitals import compose, Capability
from evalvitals.analyzers.attention import RelativeAttentionAnalyzer
from evalvitals.core.case import Inputs

# Load Qwen2.5-VL with white-box attention capture
model = compose("qwen2.5-vl-7b-instruct", "hf_local", want={Capability.ATTENTION})

# Run relative attention: ratio of task-specific vs generic image attention
result = RelativeAttentionAnalyzer(layer=22, top_k=5).run(
    model,
    Inputs(prompt="What color is the car?", image=Image.open("scene.jpg")),
)

print(result.summary())   # agent-readable findings
result.plot()             # (H, W) heatmap — requires evalvitals[viz]
```

Config-driven runs use the same contracts:

```python
from evalvitals import load_config, run

config = load_config("configs/qwen_attention.yaml")
result = run(config, "The Eiffel Tower is in")
```

```yaml
model: qwen2.5-7b-instruct
analysis: attention
analysis_kwargs:
  layer: -1
  top_k: 5
```

For explicit runtime selection:

```python
from evalvitals import Capability
from evalvitals.models import compose

model = compose(
    "qwen2.5-7b-instruct",
    "hf_local",
    want={Capability.ATTENTION},
)
```

## Documentation

- [Docs overview](docs/index.md)
- [Architecture](docs/architecture.md)
- [Quickstart](docs/quickstart.md)
- [Extending EvalVitals](docs/extending.md)
- [Roadmap](docs/roadmap.md)

```python
from evalvitals import list_specs, registry

list_specs()                                      # all registered model keys
registry.analyzers.list()                         # ['attention', 'rise', 'loop_detect', 'logit_lens', ...]
registry.analyzers.names_compatible_with(model)   # analyses runnable on this model (capability + modality)
```

An analyzer run on a model that lacks a required capability raises a clear
`CapabilityError` naming exactly what's missing.

## Backends & ModelSpec — run one model three ways

Identity lives in a `ModelSpec` (registered in `evalvitals/specs.py`); the runtime
is a backend. `compose()` combines them and negotiates capabilities up front.

```python
from evalvitals import compose, RuntimeConfig
from evalvitals.core import Capability

# 1) API / black-box (also covers a `vllm serve` endpoint). Reuse your own engine:
from evalvitals.models.backends import call_vision_api_generate_fn
rt = RuntimeConfig(generate_fn=call_vision_api_generate_fn(my_call_vision_api))
api_model = compose("qwen3-vl-8b-instruct", "api", rt)         # caps: GENERATE, TOOL_CALLS

# 2) Local white-box, full internals (forces eager when attention is requested):
wb = compose("qwen3-vl-8b-instruct", "hf_local", want={Capability.ATTENTION})

# 3) Wrong ask fails immediately, before any weights load:
compose("qwen3-vl-8b-instruct", "api", want={Capability.ATTENTION})   # -> CapabilityError
```

Module paths in a spec are *hints*: the white-box backend **discovers** the real
decoder-layer `ModuleList` at load time (`models/_discover.py`) instead of trusting
a hardcoded path — robust across transformers releases and the doubled-`.model.`
/ no-`.model` / fused-experts traps.

**Modality is a set, not a class fork.** A spec declares modalities by the
components it carries — `vision` adds `"image"`, `audio` adds `"audio"`, `video`
adds `"video"` — so an *omni* model (Qwen3-Omni reference) is just a spec with
more than one. Analyzers match on `model.modalities`, and `Inputs` carries
`image` / `audio` / `video` slots beside the prompt:

```python
omni = compose("qwen3-omni-30b-a3b-instruct", "api", rt)
omni.modalities    # frozenset({'text', 'image', 'audio', 'video'})
spec.is_omni       # True;  the audio-only Captioner -> {'text', 'audio'}
```

The thinker (text-emitting multimodal LM) is what failure analysis hooks; the
talker (speech out) is out of scope. Full multimodal generate and white-box
token maps over the audio/vision towers are Stage-2 (needs `transformers>=5.2.0`
+ `qwen_omni_utils`).

## Agent — one tool-calling loop, any backend

`Agent(wraps=handle)` is **backend-agnostic**: it needs only `GENERATE` +
`TOOL_CALLS` (checked up front), never internals — so the *same* loop drives an
API model and a local model. The single thing that varies is the
`ToolCallCodec` (auto-selected): OpenAI-native structured calls for the API,
Hermes-style `<tool_call>{…}</tool_call>` text parsing for local templates. Tool
execution goes through a pluggable `ToolExecutor` (swap in your `APIToolHandler`).

```python
from evalvitals import Agent, Tool, compose, RuntimeConfig
from evalvitals.models.backends import call_vision_api_chat_fn

search = Tool(name="search", description="web search",
              parameters={"type": "object", "properties": {"q": {"type": "string"}}},
              fn=my_search)

# API agent (reuse your engine): native tool-calls, OpenAI codec
api = compose("qwen3-8b", "api", RuntimeConfig(chat_fn=call_vision_api_chat_fn(call_vision_api)))
traj = Agent(api, tools=[search]).run("who won the 2022 world cup?")   # -> Trajectory

# Local agent: SAME Agent; tools rendered via the model's chat template, Qwen codec
local = compose("qwen3-8b", "hf_local")        # TOOL_CALLS granted only if spec.tool_calling
traj = Agent(local, tools=[search]).run("...")  # -> Trajectory (steps: USER→ACTOR→TOOL→…)
```

`TOOL_CALLS` is a **conditional** capability for local models: the backend
provides the channel, but it's granted only when the model's chat template
renders tools (`spec.tool_calling`, verified against the template at load). So
`compose(non_tool_model, "hf_local", want={TOOL_CALLS})` fails up front.

## Statistics & the pre-registered loop

`stats.compare` is the single entry point and **never returns a bare p** — it
gives an effect size + clustered-bootstrap CI, an anytime-valid e-value, a
corrected reject decision, and an underpowered flag:

```python
from evalvitals.stats import compare
r = compare(success_a, success_b, paired=True, alpha=0.05, min_effect=0.02, cluster_by=task_ids)
print(r.summary())   # [mcnemar + e-value] effect=+0.18 (B>A) CI=+0.07..+0.29, e=41.2 -> REJECT H0
```

The closed loop is **selective-inference-safe**: mine on `explore`, pre-register a
falsifiable contract, test once on `validate`, lock `confirm`:

```python
from evalvitals.eval_agent import EvalOrchestrator, PreregisteredHypothesis
hyp = PreregisteredHypothesis(predicate="cluttered scenes", statement="prompt B helps",
                              min_effect=0.03, split="validate")
report = EvalOrchestrator().run(cases, hyp, strategy_a, strategy_b)   # registers hash BEFORE unblinding
```

LOGPROBS are black-box-retrievable (OpenAI-style): wire `RuntimeConfig(logprobs_fn=...)`
and run `LogprobEntropyAnalyzer` (perplexity + predictive entropy) on an API model.

## Automated failure attribution

Two diagnosis loops are available.  `VLDiagnoseLoop` is the current
architecture for VL and LLM tasks.  `AutoDiagnoseLoop` is kept for
backward compatibility.

### `VLDiagnoseLoop` — M1 → M2 → M3 → M5 (current)

```
M1  ProbeAgent         protocol-guided analyzer selection + execute
M2  StatsAnalysisAgent stats tools + e-BH FDR correction + LLM evidence chain
M3  DiagnosisAgent     "AI scientist" hypothesis generation
M5  HypothesisTester   stats test + protocol consistency check
     ↑___________________________________|
     stop when M5 finds a verified, protocol-consistent hypothesis
```

M4 (`SurgeryAgent`) runs **after** the loop via `loop.run_m4()` to propose
(Plan A) or execute (Plan B) a targeted fix for the best verified hypothesis.

```python
from evalvitals import compose
from evalvitals.core.capability import Capability
from evalvitals.eval_agent import VLDiagnoseLoop, AgyModel, RunLogger
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol
from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
from evalvitals.eval_agent.stages.stats_agent import StatsAnalysisAgent
from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent

protocol = ExperimentProtocol(
    description="The VLM gives wrong left/right positions in spatial questions.",
    task_domain="spatial reasoning",
    success_criteria="Positions must match what is visible in the image.",
)

model = compose("qwen2.5-vl-7b-instruct", "hf_local",
                want={Capability.GENERATE, Capability.ATTENTION})
judge = AgyModel()   # or any Model with Capability.GENERATE

loop = VLDiagnoseLoop(
    model=model,
    probe_agent=ProbeAgent(max_analyzers=3),
    stats_agent=StatsAnalysisAgent(judge=judge),
    diagnosis_agent=DiagnosisAgent(judge=judge),
    max_cycles=3,
    protocol=protocol,
    run_logger=RunLogger(),
)
report = loop.run(failure_cases)

print(report.resolved)           # True when M5 finds a supported, consistent hypothesis
print(report.final_hypotheses)   # list[Hypothesis] — status SUPPORTED/REFUTED/INCONCLUSIVE

fix = loop.run_m4(report, failure_cases)   # post-loop fix proposal
```

`RunLogger()` above writes just the JSONL event log. For the full output
directory — `report/`, `figures/`, `artifacts/`, per-trial `fixes/`/`experiments/`
folders, `manifest.json` — construct a `RunContext` and pass `run_logger=ctx.logger`
instead; see [docs/architecture.md](docs/architecture.md) for the layout.

**`ExperimentProtocol`** is the human prior that anchors the loop.  M1 uses it
to select analyzers relevant to the task; M5 uses it to reject hypotheses that
drift from what the user was investigating:

```python
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

protocol = ExperimentProtocol(
    description="free text — what the experiment tests and what failure looks like",
    task_domain="spatial reasoning",      # short label
    success_criteria="what counts as a pass",
    failure_patterns="observations already noticed (optional)",
    target_modalities=frozenset({"text", "image"}),
)
```

**`ProbeAgent` / `StrategyProbe`** — M1 selects analyzers in two tiers:

- **Tier (a)** `StrategyProbe` ranks analyzers by diagnostic value for the detected model kind, guided by the protocol description via an LLM judge:

| Kind detected | Priority analyzers |
|---|---|
| VLM (image/video) | `pope`, `chair`, `attention`, `attention_rollout`, `attention_sink`, `prompt_contrast`, `mm_shap`, `logprob_entropy` |
| Agent (`TOOL_CALLS`) | `loop_detect`, `ignored_obs`, `first_error_judge`, `counterfactual` |
| LLM (text-only) | `attention`, `logit_lens`, `token_entropy`, `logprob_entropy`, `attention_sink`, `prompt_contrast`, `cka` |

- **Tier (b)** `ProbeGenerator` / `WhiteboxProbeGenerator` — when no standard analyzer covers the failure mode, an LLM or CLI agent writes a bespoke probe and runs it in a sandbox.

**`StatsAnalysisAgent`** (M2) runs a catalog of statistical tools
(`signal_label_assoc`, `mcnemar_evalue`, `bootstrap_diff`, `friedman`,
`rank_corr`, `single_rate_evalue`) over the analyzer findings, applies
e-BH FDR correction, and produces a `StatsAnalysisReport` with a structured
evidence chain for M3.

M2 can also be used independently from the diagnosis loop:

```python
from evalvitals.analysis import StatsAnalysisAgent

rows = [
    {"case_id": "c0", "label": "fail", "low_img_attn": 1},
    {"case_id": "c1", "label": "pass", "low_img_attn": 0},
]

report = StatsAnalysisAgent().analyze_records(
    rows,
    id_col="case_id",
    label_col="label",
    signal_cols=["low_img_attn"],
)
print(report.conclusion)
print([r.summary for r in report.stats_results])
```

For Lambda-style no-code exploration over an existing results directory, run a
single-shot exploration:

```bash
evalvitals explore /path/to/results \
  --backend antigravity \
  -q "Which failure patterns distinguish wrong answers from correct ones?" \
  --out evalvitals_explore_output \
  --dashboard          # optional: open the Streamlit dashboard when done
```

The run writes `exploratory_report.json`, the generated `analysis.py`,
stdout/stderr, and any rendered charts under `figures/` + `tables/`. The
exploratory report surfaces candidate signals; run `StatsAnalysisAgent` on
promoted signals when you need confirmatory effect/CI/e-value/FDR verdicts.
(The standalone console script `evalvitals-explore` is equivalent.)

To inspect the output as a Streamlit dashboard:

```bash
pip install -e ".[dashboard]"
evalvitals dashboard evalvitals_explore_output
```

See [docs/m2_analysis.md](docs/m2_analysis.md) for the standalone exploratory
analysis workflow, including the single-shot explore entry, dashboard review,
generated artifacts, and when to promote candidate signals into confirmatory
`StatsAnalysisAgent` tests.

**`HypothesisTester`** (M5) asks two questions per hypothesis:

1. *Statistical support* — does the signal group fail at a significantly higher
   rate than the control group? Consumes M2's FDR-corrected stats when present;
   falls back to a clustered-bootstrap `stats.compare` call.
2. *Protocol consistency* — does the hypothesis match what the user described?
   Keyword-based by default; an optional `judge=` runs an LLM critic.

**`SurgeryAgent`** (M4) has three verification strategies (first match wins):

1. **Injected `verify_fn`** — full caller control.
2. **`analyzer_params`** — re-run named analyzers with modified settings; returns before/after findings.
3. **Default label correlation** — extract per-case signals, split into signal vs. control, compare FAIL rates. When `SUPPORTED`, produces `new_data` for the next M1 cycle.

---

### `AutoDiagnoseLoop` — M1 → M2 → M3 → M4 (legacy)

```
M1  ProbeAgent     analyzer selection + execute
M2  AnalysisModule threshold rules → structured report
M3  DiagnosisAgent LLM judge proposes hypotheses
M4  SurgeryAgent   correlate signals, verify, refocus data
     ↑_____________|  (repeat until resolved or max_cycles)
```

```python
from evalvitals.eval_agent import AutoDiagnoseLoop, DiagnosisAgent

loop = AutoDiagnoseLoop(
    model=my_vlm,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    max_cycles=3,
    max_analyzers=4,
)
report = loop.run(failure_cases)
print(report.resolved, report.final_hypotheses)
```

Analyzers requiring constructor arguments are passed via `analyzer_overrides`:

```python
from evalvitals.analyzers.agent.counterfactual import CounterfactualReplay

loop = AutoDiagnoseLoop(
    model=my_agent_model,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    analyzer_overrides={"counterfactual": CounterfactualReplay(rerun_fn=my_rerun, n_replays=5)},
)
```

## Input Modes

There are two ways to trigger the diagnosis agent.

---

### Mode 1 — Container submission (reproduce a known failure)

Write a `run.py` that defines your failure cases as `FailureCase` objects,
then package it into a Docker container.  The agent runs the M1→M5 loop
inside the container and writes findings to `outputs/`.

**Step-by-step:**

1. **Define failure cases** in `run.py`:

```python
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

protocol = ExperimentProtocol(
    description="The model gives wrong left/right positions in spatial questions.",
    task_domain="spatial reasoning",
)

cases = CaseBatch([
    FailureCase(
        id="case_0",
        inputs=Inputs(prompt="Is the red box to the left or right of the blue box?",
                      image=my_image),
        expected="left",
    ),
    # ... more cases ...
])
```

2. **Run the loop** (same `run.py`):

```python
from evalvitals import compose
from evalvitals.core.capability import Capability
from evalvitals.eval_agent import VLDiagnoseLoop, AgyModel, RunLogger
from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
from evalvitals.eval_agent.stages.stats_agent import StatsAnalysisAgent
from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent

model = compose("qwen2.5-vl-7b-instruct", "hf_local",
                want={Capability.GENERATE, Capability.ATTENTION})
judge = AgyModel()

loop = VLDiagnoseLoop(
    model=model,
    probe_agent=ProbeAgent(max_analyzers=3),
    stats_agent=StatsAnalysisAgent(judge=judge),
    diagnosis_agent=DiagnosisAgent(judge=judge),
    max_cycles=2,
    protocol=protocol,
    run_logger=RunLogger(),
)
report = loop.run(cases)
print(report.final_hypotheses)
```

3. **Add a Dockerfile + docker-compose.yml** mirroring one of the concrete
   example directories under `examples/analyzer_demos/`, `examples/m2_statistics/`, or
   `examples/diagnosis_loops/`.

4. **Submit the container:**

```bash
docker compose up
```

Outputs (logs, analyzer artifacts, hypotheses) are written to `outputs/` in the
container, mounted to your local directory via the compose volume.

See `examples/diagnosis_loops/qwen_loop_agy/` and `examples/diagnosis_loops/qwen_video_temporal/` for complete
working examples.

---

### Mode 2 — Natural-language description (agent writes the container)

Describe the failure in plain English.  The scaffold generator produces a
ready-to-run Docker experiment with a `run.py`, `Dockerfile`, and
`docker-compose.yml`.  A CLI coding agent (Claude Code, Gemini CLI, …) can
write a fully customised `run.py`; otherwise a template is used as a starting
point.

**Python API:**

```python
from evalvitals.eval_agent import scaffold_from_description

out = scaffold_from_description(
    description="My VLM frequently confuses left and right when answering "
                "spatial relationship questions about images.",
    model_key="qwen2.5-vl-7b-instruct",
    output_dir="./my_experiment",
)
# cd my_experiment && docker compose up
```

With a CLI coding agent (generates a bespoke `run.py` tailored to the description):

```python
out = scaffold_from_description(
    description="...",
    model_key="qwen2.5-vl-7b-instruct",
    output_dir="./my_experiment",
    provider="claude_code",   # or "gemini_cli", "codex", "opencode", …
    cli_model="sonnet",
)
```

**CLI:**

```bash
# Template mode (no API key needed for scaffolding)
python -m evalvitals.eval_agent.nl_runner \
    --description "My VLM confuses left and right in spatial questions" \
    --model qwen2.5-vl-7b-instruct \
    --out ./my_experiment

# CLI agent mode (Claude Code writes a customised run.py)
python -m evalvitals.eval_agent.nl_runner \
    --description "My VLM confuses left and right in spatial questions" \
    --model qwen2.5-vl-7b-instruct \
    --out ./my_experiment \
    --provider claude_code \
    --cli-model sonnet

# Then launch:
cd my_experiment && docker compose up
```

The generated scaffold contains:

| File | Purpose |
|---|---|
| `run.py` | Diagnosis script with `ExperimentProtocol` pre-filled from your description |
| `Dockerfile` | Builds the evalvitals image with GPU + local-weights support |
| `docker-compose.yml` | Mounts HF cache, GPU, outputs, and agy binary |
| `.gitignore` | Excludes `outputs/` from version control |

In template mode, edit the `CASES` list in `run.py` to add your own
failure examples before running `docker compose up`.

## Install

```bash
pip install -e .
pip install -e ".[viz]"
pip install -e ".[dev]"
```

## Analyzer zoo

| Analyzer | Key | Capability | Modality | Paper | Status |
|---|---|---|---|---|---|
| Attention summary | `attention` | `ATTENTION` | text + image | — | ✓ |
| Attention rollout | `rollout` | `ATTENTION` | text + image | Abnar & Zuidema, 2020 | ✓ |
| Attention sink | `attention_sink` | `ATTENTION` | text + image | [Gu et al. 2023](https://arxiv.org/abs/2309.17453) | ✓ |
| Relative attention | `relative_attention` | `ATTENTION` | image (VLM) | [arXiv:2502.17422](https://arxiv.org/abs/2502.17422) | ✓ |
| RISE | `rise` | `GENERATE` | text | [Petsiuk et al. 2018](https://arxiv.org/abs/1806.07421) | ✓ |
| MM-SHAP | `mm_shap` | `LOGPROBS` | image (VLM) | [arXiv:2212.08158](https://arxiv.org/abs/2212.08158) | ✓ |
| VL-SHAP | `vl_shap` | `LOGPROBS` | image (VLM) | [arXiv:2212.08158](https://arxiv.org/abs/2212.08158) | ✓ |
| Token entropy | `token_entropy` | `LOGITS` | text + image | — | ✓ |
| Logprob entropy | `logprob_entropy` | `LOGPROBS` | text + image | [Kadavath et al. 2022](https://arxiv.org/abs/2207.05221) | ✓ |
| Self-consistency | `self_consistency` | `GENERATE` | text + image | [Wang et al. 2023](https://arxiv.org/abs/2203.11171) | ✓ |
| Verbalized confidence | `verbalized_confidence` | `GENERATE` | text + image | — | ✓ |
| POPE | `pope` | `GENERATE` | image (VLM) | [arXiv:2305.10355](https://arxiv.org/abs/2305.10355) | ✓ |
| CHAIR | `chair` | `GENERATE` | image (VLM) | [arXiv:1809.02156](https://arxiv.org/abs/1809.02156) | ✓ |
| OPERA | `opera` | `ATTENTION` | image (VLM) | [arXiv:2311.17911](https://arxiv.org/abs/2311.17911) | stub |
| VCD | `vcd` | `LOGITS` | image (VLM) | [arXiv:2311.16922](https://arxiv.org/abs/2311.16922) | stub |
| Logit lens | `logit_lens` | `HIDDEN_STATES` | text + image | [nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) | ✓ |
| Tuned lens | `tuned_lens` | `HIDDEN_STATES` | text + image | [Belrose et al. 2023](https://arxiv.org/abs/2303.08112) | stub |
| Grad-CAM | `gradcam` | `GRADIENTS` | image (VLM) | [Selvaraju et al. 2017](https://arxiv.org/abs/1610.02391) | stub |
| Generic attn explain | `generic_attention` | `ATTENTION + GRADIENTS` | text + image | [Chefer et al. 2021](https://arxiv.org/abs/2103.15679) | stub |
| Linear CKA | `cka` | `HIDDEN_STATES` | text + image | [Kornblith et al. 2019](https://arxiv.org/abs/1905.00414) | ✓ |
| Linear probe | `linear_probe` | `HIDDEN_STATES` | text + image | — | stub |
| Causal trace | `causal_trace` | `HIDDEN_STATES` | text + image | [Meng et al. 2022](https://arxiv.org/abs/2202.05262) | stub |
| Loop detect | `loop_detect` | Trajectory | agent | — | ✓ |
| Ignored obs | `ignored_obs` | Trajectory | agent | — | ✓ |
| First-error judge | `first_error_judge` | Trajectory | agent | [Zhang et al. 2024](https://arxiv.org/abs/2406.14855) | ✓ |
| Counterfactual | `counterfactual` | Trajectory | agent | Pearl 2000 | ✓ |

**Model registry** (14 specs, from `list_specs()`):

| Key | Family | Type | Notes |
|---|---|---|---|
| `qwen2.5-7b-instruct` | Qwen2 | LLM | reference checkpoint |
| `qwen3-4b` | Qwen3 | LLM | reasoning, small smoke-test |
| `qwen3-8b` | Qwen3 | LLM | reasoning |
| `qwen3-30b-a3b` | Qwen3-MoE | LLM | MoE |
| `deepseek-v3` | DeepSeek-V3 | LLM | MoE + MLA |
| `llama-3.1-8b-instruct` | Llama | LLM | — |
| `gemma-3-1b-it` | Gemma3 | LLM | text-only |
| `qwen3-vl-4b-instruct` | Qwen3-VL | VLM | small smoke-test |
| `qwen2.5-vl-7b-instruct` | Qwen2.5-VL | VLM | reference for relative-attention |
| `qwen2-vl-7b-instruct` | Qwen2-VL | VLM | — |
| `qwen3-vl-8b-instruct` | Qwen3-VL | VLM | — |
| `glm-4.5v` | GLM-MoE | VLM | MoE + reasoning, 106B |
| `glm-4.1v-9b-thinking` | GLM | VLM | reasoning |
| `kimi-vl-a3b-thinking` | Kimi-VL | VLM | MoE + MLA + reasoning |
| `llama-4-scout` | Llama4 | VLM | MoE |
| `step-1o-vision` | Step | VLM | API-only |

## Package structure

```
evalvitals/
│
├── core/                          ← Foundational contracts (no deps on other submodules)
│   ├── case.py          FailureCase, CaseBatch  ← central data unit; Inputs, Trajectory, Step
│   ├── result.py        Result    ← findings (light) + artifacts (heavy)
│   ├── model.py         Model (ABC)  ← generate(), forward(capture)->Trace; Trace, CaptureSpec
│   ├── analyzer.py      Analyzer  ← sklearn-style estimator: Analyzer(**params).run(model, data)
│   ├── pipeline.py       Pipeline  ← compose analyzers
│   ├── experiment.py    Experiment, ExperimentRunner  ← content-fingerprint result cache
│   ├── spec.py           ModelSpec  ← model identity facts (no capabilities); ModulePaths, VisionSpec
│   ├── registry.py      Registry, AnalyzerRegistry  ← @register_model / @register_analyzer
│   ├── capability.py    Capability flags + CapabilityError
│   ├── tool.py           Tool, ToolCall, ChatTurn
│   └── tokentype.py     TokenTypeMap  ← image/text token-type masks for VLM analyzers
│
├── config.py            ModelConfig, AnalysisConfig, load_config()  ← YAML-driven run()
├── specs.py             ModelSpec registry (14 specs) — get_spec(), list_specs()
│
├── models/                        ← Model construction (implements core.Model)
│   ├── base.py          BaseAgent
│   ├── compose.py       compose(spec, backend, runtime, want) -> Model  ← the one constructor
│   ├── inference.py     infer_spec()  ← used by wrap() to infer a spec from a live model
│   ├── agent.py         Agent, ToolExecutor, APIToolHandlerExecutor
│   ├── toolcodec.py     ToolCallCodec, OpenAIToolCodec, QwenToolCodec
│   ├── _discover.py     decoder-layer / unembed / final-norm resolution (drift-proof)
│   ├── blackbox/        API-only (no weights)
│   │   ├── base.py      BlackboxModel
│   │   ├── llm_api.py   OpenAI-compat LLM
│   │   ├── vlm_api.py   OpenAI-compat VLM
│   │   ├── gemini.py    Gemini
│   │   └── agent.py     BlackboxAgentModel
│   ├── whitebox/        local weights + internals capture
│   │   ├── base.py      WhiteboxModel
│   │   ├── qwen.py      Qwen-2.5
│   │   ├── qwen_vl.py   Qwen-VL
│   │   ├── qwen_omni.py Qwen-Omni
│   │   └── agent.py     WhiteboxAgentModel
│   └── backends/        inference engines (own the capabilities)
│       ├── base.py       Backend (ABC), RuntimeConfig
│       ├── api.py        HTTP/OpenAI
│       ├── hf_local.py   HuggingFace local — HFLocalModel, HFLocalBackend
│       └── vllm_offline.py vLLM offline (throughput stub)
│
├── analyzers/                     ← Analyzers (implement core.Analyzer)
│   ├── base.py
│   ├── agent/           agentic-trace analyzers       [Trajectory]
│   │   ├── loop_detect.py
│   │   ├── first_error_judge.py
│   │   ├── ignored_obs.py
│   │   └── counterfactual.py
│   ├── attention/        attention-weight analyzers   [ATTENTION]
│   │   ├── rollout.py, sink.py, relative_attn.py, summary.py
│   ├── attribution/      gradient/saliency            [GRADIENTS]
│   │   ├── gradcam.py, generic_attn.py
│   ├── geometry/         representational geometry    [HIDDEN_STATES]
│   │   ├── cka.py, linear_probe.py
│   ├── hallucination/    hallucination detectors      [GENERATE / ATTENTION]
│   │   ├── chair.py, pope.py, opera.py, vcd.py
│   ├── lens/             logit/tuned lens             [HIDDEN_STATES]
│   │   ├── logit_lens.py, tuned_lens.py
│   ├── patching/         causal tracing               [HIDDEN_STATES]
│   │   └── causal_trace.py
│   ├── perturbation/     input-perturbation           [GENERATE / LOGPROBS]
│   │   ├── mm_shap.py, vl_shap.py, rise.py, prompt_contrast.py, _shapley.py
│   └── uncertainty/      confidence / consistency     [LOGITS / GENERATE]
│       ├── logprob_entropy.py, entropy.py
│       ├── verbalized_conf.py, self_consistency.py
│
├── datasets/                      ← Dataset loaders, produce FailureCase / CaseBatch
│   ├── base.py
│   ├── pure_qa.py, llm_qa.py, vlm_qa.py
│   ├── gui_os.py
│   └── web_search_qa.py
│
├── stats/                         ← Statistical tests (standalone, no evalvitals deps)
│   ├── api.py           compare() / compare_multiple() entry points
│   ├── mcnemar.py
│   ├── bootstrap.py     clustered-bootstrap CI
│   ├── friedman.py      Friedman + Nemenyi (3+ strategies)
│   ├── ebh.py           e-BH procedure
│   ├── evalue.py
│   └── subset_sampling.py
│
└── eval_agent/                    ← Automated failure attribution (M1→M5)
    ├── loop.py          AutoDiagnoseLoop (legacy M1→M4), VLDiagnoseLoop (M1→M5); run_fix()
    ├── run_context.py   RunContext, Trial — single owner of a run's output directory
    ├── nl_runner.py     NL → Docker scaffold (scaffold_from_description)
    ├── cli_agent.py     CLI coding-agent backends
    │                    (claude_code, codex, opencode, gemini_cli, kimi_cli, antigravity)
    ├── run_logger.py    structured JSONL event log + artifact sink
    ├── hypothesis.py    Hypothesis, HypothesisStatus, serialization
    ├── orchestrator.py  EvalOrchestrator — pre-registered A/B comparison
    ├── ab_runner.py     A/B experiment runner
    ├── evolution.py     EvolutionStore — JSONL lesson store (30-day half-life decay)
    ├── store.py         Store / InMemoryStore / JsonlStore
    ├── sandbox.py       ExperimentSandbox, SandboxProtocol
    ├── factory.py       sandbox factory (subprocess / docker backends)
    ├── git_manager.py   git-native experiment versioning (eval/{run_id} branches)
    ├── preregister.py   DataSplit, PreregisteredHypothesis, PreregistrationLog
    ├── report.py        DiagnosticReport
    ├── experiment_harness.py  immutable evaluation harness injected into projects
    ├── _docker_runner.py      Docker worker (reads JSON payload from stdin)
    └── stages/                ← M1–M5 stage implementations
        ├── protocol.py        ExperimentProtocol — NL description anchoring M1 + M5
        ├── probe.py           M1 · StrategyProbe — model-kind detection + analyzer ranking
        ├── probe_agent.py     M1 · ProbeAgent — execute ranked analyzers (direct / Docker)
        ├── probe_generator.py M1 tier(b) · ProbeGenerator — LLM/CLI writes bespoke probe
        ├── whitebox_probe_generator.py  M1 tier(b) · WhiteboxProbeGenerator
        ├── analysis.py        M2 · AnalysisModule — threshold rules → AnalysisReport
        ├── stats_agent.py     M2 · StatsAnalysisAgent — stats tools + FDR + LLM chain
        ├── stats_tools.py     M2 · stats tool catalog (signal_label_assoc, mcnemar, …)
        ├── stats_tool_agent.py        M2 · legacy deterministic stats tools
        ├── stats_tool_generator.py    M2 tier(b) · LLM/CLI writes new stats script
        ├── diagnosis.py       M3 · DiagnosisAgent — hypothesis generation
        ├── surgery.py         M4 · SurgeryAgent — verify / correlate / ExperimentWriter
        ├── experiment_writer.py  M4 · multi-phase LLM/CLI agent writes + runs fix scripts
        ├── fix_agent.py       M4 (post-loop) · FixAgent — tiered, validated repair attempts
        ├── fix_tiers.py       FixTier ladder (L1 prompt → L4 parameter space)
        ├── fix_tools.py       L2 declarative pipeline catalog
        ├── fix_internals.py  L3a/L3b internals read/write primitives
        ├── fix_pipeline.py   sandboxed execution of L2 coded pipelines
        ├── hypothesis_tester.py  M5 · HypothesisTester — stats test + protocol consistency
        └── case_discovery.py  Data · CaseDiscoveryAgent — run model + label PASS/FAIL
```

**Data flow:**

```
Dataset  →  Model (blackbox | whitebox)  →  FailureCase
                                              ↓
                                          Analyzer(s)
                                              ↓
                                            Result
                                              ↓
                                  stats.compare() / eval_agent loop
```

## Testing Principles & Running Tests

We follow a tiered testing strategy modeled after standard practices in scientific computing libraries (like `scikit-learn` and `matplotlib`):

*   **Fast Unit Tests (Default)**: Use simulated, in-memory mocks ([FakeModel](file:///tealab-data/rjin02/evalvitals/tests/conftest.py)) to verify all core logic, APIs, registers, and analysis helpers. These run in **milliseconds** on standard CPUs without any model weight downloads or network dependencies, making them perfect for local development and standard CI commits.
*   **GPU Integration Tests**: Run actual forward passes and analyzers on real model weights (e.g. `Qwen2.5-7B-Instruct`). These are kept separate to prevent network/API flakiness and high compute costs from slowing down iteration.

### Commands

**Run fast unit tests only (CPU, offline-friendly):**
```bash
pytest        # 513 tests (+11 GPU-gated), no GPU required (models are mocked)
```

**Run GPU integration tests (requires CUDA GPU and model checkpoint cache):**
```bash
pytest --run-gpu
```

## Docker examples

Examples are grouped by layer: `examples/analyzer_demos/` for direct analyzer demos,
`examples/m2_statistics/` for standalone M2/statistical demos, and `examples/diagnosis_loops/`
for full diagnosis loop demos. Each concrete example directory has its own
`docker-compose.yml`:

```bash
cd examples/analyzer_demos/qwen_attention       && docker compose up   # attention analysis on a text LLM
cd examples/analyzer_demos/hallucination        && docker compose up   # POPE / CHAIR hallucination
cd examples/analyzer_demos/mm_shap              && docker compose up   # multimodal SHAP attribution
cd examples/analyzer_demos/logprob_entropy      && docker compose up   # logprob uncertainty
cd examples/m2_statistics/stats_compare         && docker compose up   # A/B statistical comparison
cd examples/diagnosis_loops/eval_agent          && docker compose up   # AutoDiagnoseLoop M1→M4
cd examples/diagnosis_loops/qwen_loop_agy       && docker compose up   # VLDiagnoseLoop M1→M5 (VLM)
cd examples/diagnosis_loops/qwen_video_temporal && docker compose up   # video temporal diagnosis
cd examples/diagnosis_loops/vlm_research_topics && docker compose up   # research topic discovery
```
