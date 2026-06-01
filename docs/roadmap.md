# Roadmap

EvalVitals is currently in alpha. The repository contains both implemented
surfaces and planned interfaces.

## Implemented or Partially Implemented

| Area | Status |
|---|---|
| Core contracts | Implemented: `Model`, `Analyzer`, `Result`, `FailureCase`, registry, pipeline, experiment. |
| Model specs | Implemented as a torch-free registry in `evalvitals.specs`. |
| Backend composition | Implemented through `compose(spec, backend, want=...)`. |
| Public `wrap()` on-ramp | Implemented: wrap any loaded HF causal LM + tokenizer; capabilities auto-inferred; attention fix-up applied. |
| Capability matching | Implemented for analyzer discovery and early backend negotiation. |
| Attention analysis | Implemented for compatible white-box runtimes. |
| Relative attention (VLM) | Implemented: `RelativeAttentionAnalyzer` — "MLLMs Know Where to Look" ([arXiv 2502.17422](https://arxiv.org/abs/2502.17422), [code](https://github.com/saccharomycetes/mllms_know)). Requires `hf_local` VLM with `ATTENTION` capability. |
| VLM forward capture | Implemented: `HFLocalModel._vlm_forward` handles image + text inputs; populates `image_token_mask` and `image_spatial_shape` in `Trace.extras`. Supported for all VLMs in the spec registry. |
| Token entropy analysis | Implemented as a white-box logits analyzer. |
| Tests | Core/model/analysis/wrap tests present; run without downloading model weights. |

## Planned

| Area | Planned Scope |
|---|---|
| More white-box analyzers | Saliency, probing, Shapley-style attribution, activation analysis, embedding geometry. |
| Black-box analyzers | API-only methods for text and vision-language systems. |
| vLLM backend | Offline batch inference and high-throughput logprob collection. |
| vLLM backend | Offline batch inference and high-throughput logprob collection. |
| VLM forward capture (vision tower internals) | Vision-tower attention maps, cross-modal hidden states, modality-level attribution. |
| Datasets | Loaders that return `FailureCase` and `CaseBatch`. |
| Stats | A/B tests, e-values, subset sampling, hypothesis generation. |
| Agent loop | Hypothesize, construct cases, experiment, attribute, evaluate, record, mutate. |

## Near-Term Priorities

1. Keep the public API narrow and stable.
2. Implement black-box model API (`BlackboxLLM`, `BlackboxVLM`) — tests first, then implementation.
3. Convert roadmap stubs into implemented modules incrementally.
4. Ensure optional dependency boundaries match runtime choices.
5. Add docs examples for each implemented analyzer.
