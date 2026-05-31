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
from evalvitals.analysis.whitebox.attention import AttentionAnalyzer

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

## Install

```bash
pip install -e .
pip install -e ".[viz]"
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Docker

```bash
docker build -f docker/Dockerfile.qwen_attention -t evalvitals-qwen-attention .
docker run --gpus all evalvitals-qwen-attention
```
