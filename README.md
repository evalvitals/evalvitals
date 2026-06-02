# EvalVitals

[![CI](https://github.com/evalvitals/evalvitals/actions/workflows/ci.yml/badge.svg)](https://github.com/evalvitals/evalvitals/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalvitals)](https://pypi.org/project/evalvitals/)
[![Python](https://img.shields.io/pypi/pyversions/evalvitals)](https://pypi.org/project/evalvitals/)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://evalvitals.github.io/evalvitals/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

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
├── core/                       # the sklearn-like substrate (torch-free)
│   ├── capability.py           Capability enum (+ TOOL_CALLS, LOGPROBS) + CapabilityError
│   ├── spec.py                 ModelSpec / VisionSpec / ModulePaths / AttnSemantics  ← NEW
│   ├── tool.py                 Tool / ToolCall / ChatTurn (agent value types)  ← NEW
│   ├── model.py                Model ABC, Trace, CaptureSpec, chat(), call_x shim
│   ├── analyzer.py             Analyzer ABC (run/get_params/set_params)
│   ├── case.py                 FailureCase, CaseBatch + Step/Trajectory (agent traces)  ← NEW
│   ├── result.py               Result (findings + artifacts)
│   ├── registry.py             model/analyzer registries + capability matching
│   ├── pipeline.py             Pipeline (compose analyzers)
│   └── experiment.py           Experiment + ExperimentRunner (content-hash cache)
├── specs.py                    ModelSpec REGISTRY: Qwen3(-VL/-Omni)/DeepSeek/GLM/Kimi/Llama/Gemma/Step  ← NEW
├── models/
│   ├── compose.py              compose(spec, backend, want) + capability negotiation  ← NEW
│   ├── agent.py                Agent(wraps=handle) + ToolExecutor → Trajectory  ← NEW
│   ├── toolcodec.py            ToolCallCodec: OpenAI (native) / Qwen (Hermes text)  ← NEW
│   ├── _discover.py            runtime decoder-layer discovery (anti-hardcoding)  ← NEW
│   ├── backends/{api,hf_local,vllm_offline}.py   ModelSpec × Backend runtimes  ← NEW
│   └── whitebox/{qwen,qwen_vl,qwen_omni}.py  per-version factories (qwen3_8b(), qwen3_vl_8b_instruct(), qwen3_omni_30b_a3b_instruct(), …) → compose(spec,'hf_local')  ← NEW
├── analyzers/                  # functional taxonomy by CAPABILITY (not black/white-box)  ← NEW
│   │                           #   each declares required_capabilities + applies_to_modalities
│   ├── perturbation/  rise✓ vl_shap✓ mm_shap✓          # GENERATE / LOGPROBS (Shapley-over-masking)
│   ├── uncertainty/   entropy✓ self_consistency✓ verbalized_conf✓   # LOGITS / GENERATE (black-box-feasible)
│   ├── hallucination/ pope✓ chair✓ opera vcd          # GENERATE (BB) / ATTENTION (VLM)
│   ├── attention/     summary✓ rollout✓ sink✓ relative_attn✓  # ATTENTION
│   ├── attribution/   gradcam generic_attn             # GRADIENTS (white-box)
│   ├── lens/          logit_lens✓ tuned_lens           # HIDDEN_STATES
│   ├── patching/      causal_trace                     # HIDDEN_STATES read+write (nnsight)
│   ├── geometry/      cka✓ linear_probe                # HIDDEN_STATES (CLIP/SigLIP-scoped)
│   └── agent/         loop_detect✓ ignored_obs✓ first_error_judge✓ counterfactual✓   # Trajectory
│                      #  ✓ = implemented + unit-tested; others declare contract, raise (Stage 2)
├── datasets/                   LLMQA✓ / VLMQA✓ + Spatial457✓ (HF 6D-spatial VQA) / WebSearchQA✓ / GUIOS✓ → CaseBatch + verifiers✓
├── stats/                      compare() single entry — never a bare p  ← NEW
│   ├── mcnemar.py✓ bootstrap.py✓ (clustered CI)  evalue.py✓ ebh.py✓  friedman.py✓ (Friedman+Nemenyi, >2 strategies)  subset_sampling.py✓
│   └── api.py✓                 compare() (pairwise) + compare_multiple() (3+ strategies) → StatResult / MultiCompareResult
└── eval_agent/                 closed loop with selective-inference discipline  ← NEW
    ├── preregister.py✓         DataSplit (explore/validate/confirm) + PreregisteredHypothesis + log
    ├── ab_runner.py✓           two strategies → stats.compare
    ├── orchestrator.py✓        define → split → pre-register → test → report
    ├── report.py✓ store.py✓    DiagnosticReport ; InMemoryStore(+query)
    ├── hypothesis.py           Hypothesis + ManualHypothesisGenerator✓ (LLM proposer = Stage 2)
    └── loop.py✓                SelfEvolveLoop (propose→record→until-dry; case-synthesis = Stage 2)
```

## The self-evolving loop (interfaces in place, logic in Stage 2)

`eval_agent/` lays out the closed cycle the package is built to serve:

```
hypothesize → construct cases → experiment → run → attribute → evaluate → record → mutate
     ↑________________________________________________________________________|
```

The agent acts only through `eval_agent/tools.py` (discovery + run + memory),
so the package's public API *is* the agent's action space.

## Testing Principles & Running Tests

We follow a tiered testing strategy modeled after standard practices in scientific computing libraries (like `scikit-learn` and `matplotlib`):

*   **Fast Unit Tests (Default)**: Use simulated, in-memory mocks ([FakeModel](file:///tealab-data/rjin02/evalvitals/tests/conftest.py)) to verify all core logic, APIs, registers, and analysis helpers. These run in **milliseconds** on standard CPUs without any model weight downloads or network dependencies, making them perfect for local development and standard CI commits.
*   **GPU Integration Tests**: Run actual forward passes and analyzers on real model weights (e.g. `Qwen2.5-7B-Instruct`). These are kept separate to prevent network/API flakiness and high compute costs from slowing down iteration.

### Commands

**Run fast unit tests only (CPU, offline-friendly):**
```bash
pytest        # 182 tests (+11 GPU-gated), no GPU required (models are mocked)
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
