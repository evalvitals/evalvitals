# Roadmap

EvalVitals is currently in alpha. The repository contains both implemented
surfaces and planned interfaces.

## Implemented or Partially Implemented

| Area | Status |
|---|---|
| Core contracts | Implemented: `Model`, `Analyzer`, `Result`, `FailureCase`, registry, pipeline, experiment. |
| Model specs | Implemented as a torch-free registry in `evalvitals.specs`. |
| Backend composition | Implemented through `compose(spec, backend, want=...)`. |
| Capability matching | Implemented for analyzer discovery and early backend negotiation. |
| Attention analysis | Implemented for compatible white-box runtimes. |
| Token entropy analysis | Implemented as a white-box logits analyzer. |
| Tests | Core/model/analysis tests are present and do not require loading large weights. |

## Planned

| Area | Planned Scope |
|---|---|
| More white-box analyzers | Saliency, probing, Shapley-style attribution, activation analysis, embedding geometry. |
| Black-box analyzers | API-only methods for text and vision-language systems. |
| VLM forward capture | Image-token maps, multimodal traces, vision tower/language model bridging. |
| vLLM backend | Offline batch inference and high-throughput logprob collection. |
| Datasets | Loaders that return `FailureCase` and `CaseBatch`. |
| Stats | A/B tests, e-values, subset sampling, hypothesis generation. |
| Agent loop | Hypothesize, construct cases, experiment, attribute, evaluate, record, mutate. |

## Near-Term Priorities

1. Keep the public API narrow and stable.
2. Convert roadmap stubs into implemented modules incrementally.
3. Ensure optional dependency boundaries match runtime choices.
4. Add docs examples for each implemented analyzer.
5. Add a small, CPU-safe demo path that exercises the full user workflow.
