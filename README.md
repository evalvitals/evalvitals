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

## What's Inside

EvalVitals covers three layers of the analysis workflow, usable independently
or chained into one automated loop:

1. **Analyzer toolkit** — 26 registered analyzers (attention, uncertainty,
   hallucination, Shapley attribution, logit lens, representation geometry,
   agent-trajectory analysis) run against any model/backend pair through the
   same `Analyzer(**params).run(model, data) -> Result` call shape. See the
   [Demo](#demo) below and the [Analyzer Zoo](docs/analyzers.md).
2. **Data analysis agent (M2/M3)** — `evalvitals explore` points a coding
   agent at a raw results directory (any JSON/JSONL shape, no host-side
   parsing) and gets back descriptive takeaways, charts, and 1-3 falsifiable
   hypotheses — no code required. See
   [Exploratory Analysis (M2/M3)](docs/m2_analysis.md).
3. **Intervention (M4/M5)** — `HypothesisTester` verifies a hypothesis
   statistically and against the stated experiment protocol; `FixAgent` then
   proposes and validates candidate repairs (prompt → scaffold → internals →
   parameter space) against the unmodified baseline with paired McNemar +
   e-value. See [Intervention & Verification (M4/M5)](docs/intervention.md).

`VLDiagnoseLoop` chains all three (M1→M2→M3→M5, M4 post-loop) into one
automated failure-attribution pipeline — see [Quickstart](docs/quickstart.md).

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
- [Intervention & Verification (M4/M5)](docs/intervention.md) — `HypothesisTester` verification, `FixAgent` tiered repair
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
cd examples/diagnosis_loops/qwen_loop_agy  && docker compose up   # VLDiagnoseLoop M1→M5 (VLM)
```

See [`examples/README.md`](examples/README.md) for the full list, grouped by
layer (single analyzers, standalone M2/M3, full diagnosis loop).
