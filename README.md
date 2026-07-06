# EvalVitals

[![CI](https://github.com/evalvitals/evalvitals/actions/workflows/ci.yml/badge.svg)](https://github.com/evalvitals/evalvitals/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalvitals)](https://pypi.org/project/evalvitals/)
[![Python](https://img.shields.io/pypi/pyversions/evalvitals)](https://pypi.org/project/evalvitals/)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://evalvitals.github.io/evalvitals/)
[![License: CC0-1.0](https://img.shields.io/badge/license-CC0--1.0-green)](LICENSE)

EvalVitals is an sklearn-like toolkit for failure-case analysis in the era of
LLMs, VLMs, omni (text+image+audio+video) models, and agentic systems. Model
identity (`ModelSpec`) is separate from runtime (`Backend`); analyzers
(`Analyzer`) connect to models by capability matching (`Capability`), so the
same spec runs through a black-box API or a white-box local backend — only
the capability set changes.

| Contract | Role |
|---|---|
| `ModelSpec` | Model identity: family, HF repo, architecture traits, modalities. |
| `Backend` | Runtime: local HF internals, black-box API, or offline batch engines. |
| `Model` | A runnable object with `generate()` and `forward(capture=...) -> Trace`. |
| `Analyzer` | An sklearn-style estimator: `Analyzer(**params).run(model, data) -> Result`. |
| `Capability` | Vocabulary matching analyzers to compatible model runtimes. |
| `FailureCase` | Central data unit for prompts, labels, provenance, agent trajectories. |
| `Result` | Uniform output: human-readable summary + structured findings. |

## Demo

Relative attention on a VLM — ["MLLMs Know Where to Look"](https://arxiv.org/abs/2502.17422) ([code](https://github.com/saccharomycetes/mllms_know)):

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

The same call shape works for any registered model/analyzer pair — a plain
text LLM, a config-driven YAML run, or an explicit backend — see
[Quickstart](docs/quickstart.md) for those and for the automated
failure-attribution loop (`VLDiagnoseLoop`, M1→M2→M3→M5) and no-code
exploratory analysis (`evalvitals explore`).

## Install

```bash
pip install -e .
pip install -e ".[viz]"
pip install -e ".[dev]"
```

## Documentation

- [Docs overview](docs/index.md)
- [Quickstart](docs/quickstart.md) — runnable examples for every entry point, the diagnosis loop, and submitting a run
- [Exploratory Analysis (M2/M3)](docs/m2_analysis.md) — standalone `evalvitals explore`
- [Analyzer Zoo](docs/analyzers.md) — reference tables of analyzers and registered models
- [Architecture](docs/architecture.md) — package structure and design contracts
- [Extending EvalVitals](docs/extending.md) — add analyzers, specs, backends
- [Roadmap](docs/roadmap.md)

## Examples

Each directory under `examples/` is a self-contained, runnable demo:

```bash
cd examples/analyzer_demos/qwen_attention  && docker compose up   # attention analysis on a text LLM
cd examples/m2_statistics/deco_hallu_explore && docker compose up # M2/M3 explore, real M1 data
cd examples/m2_statistics/deco_hallu_explore && bash run_attn.sh  # ... attention-enriched: FAIL/PASS distributions + cross-checkpoint geometry (no GPU; data ships with the repo)
cd examples/m2_statistics/deco_hallu_explore && bash run_attn_pipeline.sh  # ... FULL held-out pipeline: propose → held-out test + LLM judge → L1..L3b fix → one 4-tab report (SKIP_FIX=1 for the no-GPU half)
cd examples/diagnosis_loops/qwen_loop_agy  && docker compose up   # VLDiagnoseLoop M1→M5 (VLM)
```

See [`examples/README.md`](examples/README.md) for the full list, grouped by
layer (single analyzers, standalone M2/M3, full diagnosis loop).
