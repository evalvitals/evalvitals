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

## AutoDiagnoseLoop — automated false-attribution pipeline

`AutoDiagnoseLoop` runs a four-module cycle that selects analyzers, executes
them, asks an LLM to propose hypotheses, and verifies each one through
intervention — looping back if the problem is not yet resolved.

```
M1 StrategyProbe   → select analyzers for this model kind (VLM / agent / LLM)
M2 Execution       → run via ExperimentRunner (content-hash cached)
M3 DiagnosisAgent  → LLM judge reads findings, proposes HYPOTHESIS:/FAILURE_MODE: pairs
M4 SurgeryAgent     → correlate per-case signals with PASS/FAIL labels; refocus data
     ↑_______________________________________________________________|  (repeat)
```

```python
from evalvitals.eval_agent import AutoDiagnoseLoop, DiagnosisAgent

# Any model with Capability.GENERATE as the judge (Claude, GPT-4o, local chat model, …)
judge = compose("qwen3-8b", "api", RuntimeConfig(generate_fn=my_generate))

loop = AutoDiagnoseLoop(
    model=my_vlm,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    max_cycles=3,
    max_analyzers=4,
)
report = loop.run(failure_cases)

print(report.resolved)           # True when an intervention eliminated the failures
print(report.final_hypotheses)   # list[Hypothesis] with status SUPPORTED/REFUTED/INCONCLUSIVE
print(report.final_results)      # {analyzer_name: Result} from the last cycle
```

**`StrategyProbe`** ranks analyzers by diagnostic value for the model kind it
detects:

| Kind detected | First analyzers selected |
|---|---|
| VLM (image modality) | `pope`, `chair`, `attention`, `attention_rollout`, `mm_shap` |
| Agent (`TOOL_CALLS`) | `loop_detect`, `ignored_obs`, `first_error_judge`, `counterfactual` |
| LLM (text-only) | `attention`, `logit_lens`, `token_entropy`, `logprob_entropy` |

**`SurgeryAgent`** has three verification strategies (first match wins):

1. **Injected `verify_fn`** — full caller control.
2. **`analyzer_params`** — re-run named analyzers with modified settings; returns before/after findings.
3. **Default label correlation** — extracts per-case signals (e.g. `has_loop`, `n_ignored`) from findings,
   splits cases into signal vs. control groups, compares FAIL rates with a 10 % gap threshold.
   When `SUPPORTED`, produces `new_data` (the non-signal cases) for the next M1 cycle.

Analyzers that require mandatory constructor arguments (e.g. `CounterfactualReplay`) are
passed via `analyzer_overrides`:

```python
from evalvitals.analyzers.agent.counterfactual import CounterfactualReplay

loop = AutoDiagnoseLoop(
    model=my_agent_model,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    analyzer_overrides={"counterfactual": CounterfactualReplay(rerun_fn=my_rerun, n_replays=5)},
)
```

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
├── core/                          ← Foundational abstractions (no deps on other submodules)
│   ├── case.py          Case      ← single failure record (input + expected + actual)
│   ├── result.py        AnalysisResult
│   ├── model.py         ModelBase
│   ├── analyzer.py      AnalyzerBase
│   ├── pipeline.py      Pipeline  ← Case → [Analyzer...] → AnalysisResult
│   ├── experiment.py    Experiment
│   ├── spec.py          ExperimentSpec
│   ├── registry.py      Registry  ← global analyzer/model lookup
│   ├── capability.py    Capability flags
│   ├── tool.py          Tool
│   └── tokentype.py     TokenType
│
├── config.py            RuntimeConfig   ← API keys, generate_fn injection
├── specs.py             built-in ExperimentSpec presets
│
├── models/                        ← Model wrappers (implement core.ModelBase)
│   ├── base.py          ModelBase (re-export)
│   ├── compose.py       ComposedModel   ← fan-out over multiple models
│   ├── inference.py     run_inference()
│   ├── agent.py         AgentModel
│   ├── toolcodec.py     tool call encode/decode
│   ├── _discover.py     auto-register models
│   ├── blackbox/        API-only (no weights)
│   │   ├── base.py
│   │   ├── llm_api.py   OpenAI-compat LLM
│   │   ├── vlm_api.py   OpenAI-compat VLM
│   │   ├── gemini.py    Gemini
│   │   └── agent.py     BlackboxAgentModel
│   ├── whitebox/        local weights + internals capture
│   │   ├── base.py
│   │   ├── qwen.py      Qwen-2.5
│   │   ├── qwen_vl.py   Qwen-VL
│   │   ├── qwen_omni.py Qwen-Omni
│   │   └── agent.py     WhiteboxAgentModel
│   └── backends/        inference engines
│       ├── base.py
│       ├── api.py        HTTP/OpenAI
│       ├── hf_local.py   HuggingFace local
│       └── vllm_offline.py vLLM offline
│
├── analyzers/                     ← Analyzers (implement core.AnalyzerBase)
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
│   │   ├── mm_shap.py, vl_shap.py, rise.py, _shapley.py
│   └── uncertainty/      confidence / consistency     [LOGITS / GENERATE]
│       ├── logprob_entropy.py, entropy.py
│       ├── verbalized_conf.py, self_consistency.py
│
├── datasets/                      ← Dataset loaders (implement DatasetBase)
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
└── eval_agent/                    ← Experiment automation loop (M1→M4)
    ├── orchestrator.py  top-level driver
    ├── loop.py          run loop (cycles, checkpoint resume)
    ├── experiment_harness.py
    ├── experiment_writer.py  LLM → experiment code
    ├── cli_agent.py     CLI coding-agent backends
    │                    (claude_code, codex, opencode, gemini_cli, kimi_cli)
    ├── ab_runner.py     A/B experiment runner
    ├── evolution.py     EvolutionStore, build_overlay
    ├── store.py         artifact store
    ├── run_logger.py    structured event log (trace_id, span_id)
    ├── hypothesis.py    hypothesis tracking
    ├── probe.py / probe_agent.py
    ├── diagnosis.py
    ├── analysis.py
    ├── report.py
    ├── surgery.py       model weight surgery
    ├── sandbox.py
    ├── preregister.py
    ├── factory.py
    ├── git_manager.py
    ├── _docker_runner.py
    └── _tools.py
```

**Data flow:**

```
Dataset  →  Model (blackbox | whitebox)  →  Case
                                              ↓
                                          Analyzer(s)
                                              ↓
                                         AnalysisResult
                                              ↓
                                  stats.compare() / eval_agent loop
```

## The automated diagnosis loop

`eval_agent/` provides both the concrete `AutoDiagnoseLoop` (M1→M4 implemented)
and `SelfEvolveLoop` (original propose→record skeleton, kept for backward
compatibility).  The agent acts only through `eval_agent/tools.py` (discovery +
run + memory), so the package's public API *is* the agent's action space.

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

Each `examples/` subdirectory has its own `docker-compose.yml`:

```bash
cd examples/qwen_attention  && docker compose up
cd examples/hallucination   && docker compose up
cd examples/mm_shap         && docker compose up
cd examples/logprob_entropy && docker compose up
cd examples/stats_compare   && docker compose up
cd examples/eval_agent      && docker compose up
```
Partial 1 - Close-loop
Partial 2 - self-envolving 


1. M1 agent probing tool (agent, VLM, LLM), launch each analyze in the container.
2. M2 statistical testing (self-envolving agent to provide initial insight).
3. M3 propose hypothesis.
4. M4 perform intervention. (self-envolving)