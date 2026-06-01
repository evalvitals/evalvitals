# EvalVitals

EvalVitals is an sklearn-like toolkit for failure-case analysis in the era of
LLMs, VLMs, and agentic systems.

The package is organized around a small set of uniform contracts so researchers,
engineers, and automated agents can discover, compose, and run evaluations
programmatically:

| Contract | Role |
|---|---|
| `ModelSpec` | Model identity: family, Hugging Face repo, architecture traits, VLM/MoE/MLA caveats. |
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
from evalvitals import registry

registry.models.list()                           # ['qwen']
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
├── specs.py                    ModelSpec REGISTRY: Qwen3(-VL)/DeepSeek/GLM/Kimi/Llama/Gemma/Step  ← NEW
├── models/
│   ├── compose.py              compose(spec, backend, want) + capability negotiation  ← NEW
│   ├── agent.py                Agent(wraps=handle) + ToolExecutor → Trajectory  ← NEW
│   ├── toolcodec.py            ToolCallCodec: OpenAI (native) / Qwen (Hermes text)  ← NEW
│   ├── _discover.py            runtime decoder-layer discovery (anti-hardcoding)  ← NEW
│   ├── backends/{api,hf_local,vllm_offline}.py   ModelSpec × Backend runtimes  ← NEW
│   └── whitebox/qwen.py        QwenLLM (legacy concrete model; still supported)
├── analyzers/                  # functional taxonomy by CAPABILITY (not black/white-box)  ← NEW
│   │                           #   each declares required_capabilities + applies_to_modalities
│   ├── perturbation/  rise✓ vl_shap mm_shap            # GENERATE / LOGPROBS
│   ├── uncertainty/   entropy✓ self_consistency✓ verbalized_conf✓   # LOGITS / GENERATE (black-box-feasible)
│   ├── hallucination/ pope chair(metric✓) opera vcd    # GENERATE / ATTENTION (VLM)
│   ├── attention/     summary✓ rollout✓ sink✓ relative_attn   # ATTENTION
│   ├── attribution/   gradcam generic_attn             # GRADIENTS (white-box)
│   ├── lens/          logit_lens✓ tuned_lens           # HIDDEN_STATES
│   ├── patching/      causal_trace                     # HIDDEN_STATES read+write (nnsight)
│   ├── geometry/      cka✓ linear_probe                # HIDDEN_STATES (CLIP/SigLIP-scoped)
│   └── agent/         loop_detect✓ ignored_obs✓ first_error_judge✓ counterfactual   # Trajectory
│                      #  ✓ = implemented + unit-tested; others declare contract, raise (Stage 2)
├── datasets/                   loaders → CaseBatch (Stage 2)
├── stats/                      compare() single entry — never a bare p  ← NEW
│   ├── mcnemar.py✓ bootstrap.py✓ (clustered CI)  evalue.py✓ ebh.py✓  subset_sampling.py✓
│   └── api.py✓                 compare() → StatResult(effect, CI, e-value, reject, underpowered)
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

## Running tests

```bash
pytest        # 147 tests (+11 GPU-gated), no GPU required (models are mocked)
```

## Docker

```bash
docker build -f docker/Dockerfile.qwen_attention -t evalvitals-qwen-attention .
docker run --gpus all evalvitals-qwen-attention
```
