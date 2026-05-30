# EvalVitals

Failure case analysis for LLMs and VLMs — built so an agent can drive it.

The package is organised around four uniform, sklearn-like contracts so that
both humans and an automated agent can explore and run it programmatically:

| Contract | What it is |
|---|---|
| **`Capability`** | the vocabulary connecting models and analyses (`GENERATE`, `TOOL_CALLS`, `LOGPROBS`, `ATTENTION`, `HIDDEN_STATES`, `GRADIENTS`, …) |
| **`Model`** | an analyzable model: `generate()` + `forward(capture, spec) -> Trace`; declares the capabilities it provides |
| **`Analyzer`** | a sklearn-style estimator: `Analyzer(**params).run(model, data) -> Result`; declares the capabilities it requires |
| **`FailureCase`** | the central data unit (unit OR multi-step `Trajectory`); datasets produce it, analyzers attribute it, the agent accumulates it |
| **`ModelSpec` × `Backend`** | identity (spec) is orthogonal to runtime (backend); `compose(spec, backend)` builds a Model whose **capabilities come from the backend** |

Models and analyses connect by **capability matching** — an analyzer runs on any
model that provides what it requires, with no per-model wiring. Adding a new
analysis instantly makes it available on every compatible model.

`black-box` vs `white-box` is **not** a type here — it's the capability set a
backend grants. The same `ModelSpec` runs as a remote API (`GENERATE`), an
in-process vLLM batch (`+LOGPROBS`), or an HF-eager capture (`+ATTENTION`,
`+HIDDEN_STATES`), and an analyzer requesting `ATTENTION` against the API backend
fails at `compose()` time, not deep in a hook.

## Quickstart

**1. Canonical (sklearn-style)** — configure an analyzer, run it on a model:

```python
from evalvitals.models.whitebox.qwen import QwenLLM
from evalvitals.analysis.whitebox.attention import AttentionAnalyzer

model = QwenLLM(checkpoint="Qwen/Qwen2.5-7B-Instruct")
result = AttentionAnalyzer(layer=-1, top_k=5).run(model, "The Eiffel Tower is in")

print(result.summary())          # human/LLM-readable
print(result.findings)           # agent-facing dict (JSON-serialisable)
result.plot()                    # heavy artifacts (requires evalvitals[viz])
```

**2. Config-driven** — declare model + analysis in YAML:

```python
from evalvitals import load_config, run

config = load_config("configs/qwen_attention.yaml")
result = run(config, "The Eiffel Tower is in")
```

```yaml
# configs/qwen_attention.yaml
model: qwen
analysis: attention        # registered analyzer name
```

**3. Hybrid shim** — `model.call_<analysis>()`, auto-derived from capabilities:

```python
result = model.call_attention("The Eiffel Tower is in")
```

## Discovery (the agent's planning surface)

```python
from evalvitals import registry

registry.models.list()                          # ['qwen']
registry.analyzers.list()                        # ['attention', 'saliency', ...]
registry.analyzers.names_compatible_with(model)  # analyses runnable on this model
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

## Install

```bash
pip install -e .              # LIGHT core: pure-API failure analysis, no torch
pip install -e ".[local]"     # + torch/transformers — local white-box models
pip install -e ".[api]"       # + openai client (or inject your own call_vision_api)
pip install -e ".[interp]"    # + captum / inseq / nnsight (Stage 2 analyzers)
pip install -e ".[viz]"       # + matplotlib for heatmaps
pip install -e ".[dev]"       # + pytest / ruff / mypy
```

## Package structure

```
evalvitals/
├── core/                       # the sklearn-like substrate (torch-free)
│   ├── capability.py           Capability enum (+ TOOL_CALLS, LOGPROBS) + CapabilityError
│   ├── spec.py                 ModelSpec / VisionSpec / ModulePaths / AttnSemantics  ← NEW
│   ├── model.py                Model ABC, Trace, CaptureSpec, call_x shim
│   ├── analyzer.py             Analyzer ABC (run/get_params/set_params)
│   ├── case.py                 FailureCase, CaseBatch + Step/Trajectory (agent traces)  ← NEW
│   ├── result.py               Result (findings + artifacts)
│   ├── registry.py             model/analyzer registries + capability matching
│   ├── pipeline.py             Pipeline (compose analyzers)
│   └── experiment.py           Experiment + ExperimentRunner (content-hash cache)
├── specs.py                    ModelSpec REGISTRY: Qwen3(-VL)/DeepSeek/GLM/Kimi/Llama/Gemma/Step  ← NEW
├── models/
│   ├── compose.py              compose(spec, backend, want) + capability negotiation  ← NEW
│   ├── _discover.py            runtime decoder-layer discovery (anti-hardcoding)  ← NEW
│   ├── backends/{api,hf_local,vllm_offline}.py   ModelSpec × Backend runtimes  ← NEW
│   └── whitebox/qwen.py        QwenLLM (legacy concrete model; still supported)
├── analysis/                   Analyzers; declare `requires`
│   ├── whitebox/attention.py   AttentionAnalyzer (findings + artifacts)
│   ├── whitebox/uncertainty.py TokenEntropyAnalyzer (free, LOGITS-only)  ← NEW
│   ├── whitebox/{saliency,probing,shapley,activation,embedding_geometry}.py  (Stage 2)
│   ├── blackbox/{rise,vl_shap,transformer_mm}.py   (Stage 2)
│   └── agent/{failure_attribution,trajectory_eval}.py  (Stage 2; consume Trajectory)
├── datasets/                   loaders → CaseBatch (Stage 2)
├── stats/                      consume Results (Stage 2)
└── eval_agent/                 self-evolving loop (interfaces + stubs)
    ├── tools.py                the agent's action space
    ├── hypothesis.py           Hypothesis + generator
    ├── store.py                persistent memory (Store)
    └── loop.py                 SelfEvolveLoop controller
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
pytest        # 89 tests, no GPU required (models are mocked)
```

## Docker

```bash
docker build -f docker/Dockerfile.qwen_attention -t evalvitals-qwen-attention .
docker run --gpus all evalvitals-qwen-attention
```

## Roadmap

| Stage | Scope | Timeline |
|---|---|---|
| 1a | Capability-matched core + Qwen + attention (this release) | 1–2 weeks |
| 1b | More models + analysis strategies | 1 month |
| 1c (parallel) | VLM cases + real data | 1 month |
| 2 | Statistical testing, A/B, self-evolving agent loop | TBD |
```
