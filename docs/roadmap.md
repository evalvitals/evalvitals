# Roadmap

EvalVitals is currently in alpha. The repository contains both implemented
surfaces and planned interfaces.

## Implemented

| Area | Status |
|---|---|
| Core contracts | `Model`, `Analyzer`, `Result`, `FailureCase`, registry, pipeline, experiment. |
| Model specs | Torch-free registry in `evalvitals.specs` (16 specs: Qwen3/VL/Omni, DeepSeek, GLM, Kimi, Llama, Gemma, Step). |
| Backend composition | `compose(spec, backend, want=...)` with early capability negotiation. |
| Public `wrap()` on-ramp | Wrap any loaded HF causal LM + tokenizer; capabilities auto-inferred; attention fix-up applied. |
| Capability matching | Analyzer discovery and early backend negotiation. |
| Attention analysis | `AttentionAnalyzer`, `AttentionRolloutAnalyzer`, `AttentionSinkAnalyzer`, `RelativeAttentionAnalyzer` (VLM). |
| VLM forward capture | `HFLocalModel._vlm_forward`: image + text; `image_token_mask` + `image_spatial_shape` in `Trace.extras`. |
| Uncertainty analysis | `TokenEntropyAnalyzer` (LOGITS), `LogprobEntropyAnalyzer` (LOGPROBS), `SelfConsistencyAnalyzer`, `VerbalizedConfidenceAnalyzer`. |
| Hallucination analysis | `POPEAnalyzer`, `CHAIRAnalyzer` (black-box, GENERATE-only). |
| Shapley attribution | `MMShapAnalyzer` (text + image), `VLShapAnalyzer` (image regions). |
| Lens analysis | `LogitLensAnalyzer` (hidden states → vocabulary projections). |
| Geometry | `CKAAnalyzer` (linear CKA over residual stream). |
| Agent analyzers | `LoopDetector`, `IgnoredObservationDetector`, `FirstErrorJudge`, `CounterfactualReplay`. |
| Statistics | `compare()` (McNemar + e-value + clustered bootstrap CI), `compare_multiple()` (Friedman + Nemenyi), `ebh()`, `kendall_tau`, `stratified_subset`. |
| Eval agent — pre-registered A/B | `EvalOrchestrator`, `DataSplit`, `PreregisteredHypothesis`, `PreregistrationLog`. |
| **AutoDiagnoseLoop** | **M1 `StrategyProbe`, M3 `DiagnosisAgent` (LLM judge), M4 `SurveyAgent` (label correlation / param sweep / verify_fn), `AutoDiagnoseLoop` controller.** |
| Contract tests | Parametrized pyod-style suite covering all 26 registered analyzers (513 unit tests). |

## Stage 2 stubs (registered, `_run` raises `NotImplementedError`)

| Analyzer | Key | Blocker |
|---|---|---|
| Tuned lens | `tuned_lens` | requires per-model trained translators |
| Causal trace | `causal_trace` | needs nnsight read+write hooks |
| Linear probe | `linear_probe` | needs labeled hidden-state pairs |
| GradCAM | `gradcam` | needs backward pass through vision tower |
| Generic attention explainability | `generic_attention` | needs Chefer relevance propagation |
| OPERA | `opera` | needs decode-loop control |
| VCD | `vcd` | needs contrastive decoding over distorted image |

## Planned (Stage 2+)

| Area | Planned Scope |
|---|---|
| vLLM backend | Offline batch inference, high-throughput logprob collection. |
| VLM vision-tower internals | Cross-modal hidden states, vision-tower attention maps. |
| LLM-backed hypothesis generation | `HypothesisGenerator.propose()` / `.mutate()` backed by the judge model. |
| Case synthesis | `make_cases(spec)` — agent-driven adversarial case construction. |
| Durable store | JSONL / SQLite backend for `Store` (currently in-memory only). |
| RISE | Requires user-supplied `score_fn`; improve discoverability. |
| Omni model forward capture | Audio + video tower internals (needs `transformers≥5.2.0`). |

## Near-Term Priorities

1. Keep the public API narrow and stable.
2. Implement Stage-2 stubs incrementally (causal trace, tuned lens).
3. Add a durable `Store` backend so `AutoDiagnoseLoop` results persist across runs.
4. Add docs examples for the `AutoDiagnoseLoop`.
