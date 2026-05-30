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
| Model identity | `ModelSpec` | `qwen2.5-7b-instruct`, `llama-3.1-8b-instruct` |
| Runtime | `Backend` | `hf_local`, `api`, `vllm_offline` |
| Analysis | `Analyzer` | `AttentionAnalyzer`, `TokenEntropyAnalyzer` |

This separation lets an analyzer ask for capabilities instead of asking for a
specific model class. For example, an attention analyzer requires
`Capability.ATTENTION`; any model runtime that provides attention traces can run
it.

## Mental Model

```text
ModelSpec + Backend -> Model
Model + data + Analyzer -> Result
Result + Experiment -> comparable evidence
FailureCase + Trajectory -> reusable cases for humans and agents
```

The intended workflow is:

1. Declare or select model identity with `ModelSpec`.
2. Choose a runtime backend based on what data the analysis needs.
3. Run analyzers that are compatible with that runtime's capabilities.
4. Store results as structured findings and artifacts.
5. Use those results to refine cases, hypotheses, and experiments.

## Documentation Map

- [Quickstart](quickstart.md): runnable examples and common entry points.
- [Architecture](architecture.md): package structure and design contracts.
- [Extending EvalVitals](extending.md): how to add analyzers, specs, and backends.
- [Roadmap](roadmap.md): current implementation status and planned surfaces.

## Current Status

EvalVitals is currently an alpha package. The core contracts, spec/backend
composition, capability matching, attention analysis, token-entropy analysis,
and test scaffolding are in place. Several modules exist as planned Stage 2/3
surfaces and intentionally raise `NotImplementedError`.
