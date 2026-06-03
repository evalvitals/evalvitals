# EvalVitals Documentation

EvalVitals is a package for LLM and VLM evaluation designed around the same
engineering posture that made sklearn useful: small composable contracts,
discoverable estimators, uniform result objects, and predictable behavior across
many model/runtime combinations.

Where sklearn standardizes `fit`, `predict`, and `score` around tabular learning,
EvalVitals standardizes `generate`, `forward(capture=...)`, `Analyzer.run`, and
`Result` around model behavior, internals, failures, and agent trajectories.

## Core Idea

EvalVitals separates three things that are often mixed together:

| Concern | EvalVitals object | Example |
|---|---|---|
| Model identity | `ModelSpec` (curated) or inferred via `wrap()` | `qwen2.5-7b-instruct`, or any loaded HF model |
| Runtime | `Backend` | `hf_local`, `api`, `vllm_offline` |
| Analysis | `Analyzer` | `AttentionAnalyzer`, `TokenEntropyAnalyzer` |

This separation lets an analyzer ask for capabilities instead of asking for a
specific model class. For example, an attention analyzer requires
`Capability.ATTENTION`; any model runtime that provides attention traces can run
it.

## Mental Model

There are two ways to get a `Model`:

```text
# Public on-ramp (captum-style): user brings their own loaded model
wrap(hf_model, tokenizer)  ->  Model

# Curated path: load from the spec registry by key
ModelSpec + Backend  ->  compose(...)  ->  Model
```

Both paths produce the same `Model` object — the same analyzers work on both.

```text
Model + data + Analyzer -> Result
Result + Experiment    -> comparable evidence
FailureCase + Trajectory -> reusable cases for humans and agents
```

The intended workflow is:

1. Get a model: `evalvitals.wrap(your_model, tokenizer)` or `evalvitals.load("key")`.
2. Discover compatible analyzers from the registry.
3. Run analyzers that match the model's capabilities.
4. Store results as structured findings and artifacts.
5. Use those results to refine cases, hypotheses, and experiments.

Or hand the loop to `AutoDiagnoseLoop` and let it drive steps 2–5 automatically:

```python
from evalvitals.eval_agent import AutoDiagnoseLoop, DiagnosisAgent

loop   = AutoDiagnoseLoop(model=my_model, diagnosis_agent=DiagnosisAgent(judge=judge))
report = loop.run(failure_cases)
# → report.resolved, report.final_hypotheses, report.final_results
```

## Documentation Map

- [Quickstart](quickstart.md): runnable examples and common entry points.
- [Architecture](architecture.md): package structure and design contracts.
- [Extending EvalVitals](extending.md): how to add analyzers, specs, and backends.
- [Roadmap](roadmap.md): current implementation status and planned surfaces.

## Current Status

EvalVitals is currently an alpha package. The core contracts, spec/backend
composition, capability matching, public `wrap()` on-ramp, 26 registered
analyzers, statistics layer, and the full automated diagnosis pipeline
(`AutoDiagnoseLoop` — M1 StrategyProbe, M3 DiagnosisAgent, M4 SurgeryAgent) are
implemented and covered by 513 unit tests (no GPU required).  VLM forward capture
(image-token mask + spatial layout) is implemented for all models in the spec
registry. Several analyzers are Stage-2 stubs that intentionally raise
`NotImplementedError`; see the [Roadmap](roadmap.md) for details.
