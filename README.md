# EvalVitals

Failure case analysis for LLMs and VLMs — built so an agent can drive it.

The package is organised around four uniform, sklearn-like contracts so that
both humans and an automated agent can explore and run it programmatically:

| Contract | What it is |
|---|---|
| **`Capability`** | the vocabulary connecting models and analyses (`ATTENTION`, `HIDDEN_STATES`, `GRADIENTS`, …) |
| **`Model`** | an analyzable model: `generate()` + `forward(capture) -> Trace`; declares the capabilities it provides |
| **`Analyzer`** | a sklearn-style estimator: `Analyzer(**params).run(model, data) -> Result`; declares the capabilities it requires |
| **`FailureCase`** | the central data unit; datasets produce it, analyzers attribute it, the agent accumulates it |

Models and analyses connect by **capability matching** — an analyzer runs on any
model that provides what it requires, with no per-model wiring. Adding a new
analysis instantly makes it available on every compatible model.

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

## Install

```bash
pip install -e .
pip install -e ".[viz]"   # + matplotlib for heatmaps
pip install -e ".[dev]"   # + pytest / ruff / mypy
```

## Package structure

```
evalvitals/
├── core/                       # the sklearn-like substrate
│   ├── capability.py           Capability enum + CapabilityError
│   ├── model.py                Model ABC, Trace, call_x shim
│   ├── analyzer.py             Analyzer ABC (run/get_params/set_params)
│   ├── case.py                 FailureCase, CaseBatch, as_casebatch
│   ├── result.py               Result (findings + artifacts)
│   ├── registry.py             model/analyzer registries + matching
│   ├── pipeline.py             Pipeline (compose analyzers)
│   └── experiment.py           Experiment + ExperimentRunner
├── models/                     one file per model; declare capabilities
│   ├── whitebox/qwen.py        QwenLLM           ← Stage 1 (this release)
│   ├── whitebox/qwen_vl.py     QwenVL            (Stage 2)
│   └── blackbox/               API-based models  (Stage 2)
├── analysis/                   Analyzers; declare `requires`
│   ├── whitebox/attention.py   AttentionAnalyzer ← Stage 1
│   ├── whitebox/{saliency,probing,shapley,activation,embedding_geometry}.py  (Stage 2)
│   ├── blackbox/{rise,vl_shap,transformer_mm}.py   (Stage 2)
│   └── agent/{failure_attribution,trajectory_eval}.py  (Stage 2)
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
pytest        # 58 tests, no GPU required (models are mocked)
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
