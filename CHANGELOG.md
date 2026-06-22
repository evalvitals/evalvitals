# Changelog

All notable changes to EvalVitals will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — run output ownership & tiered repair

- **`RunContext`** (`eval_agent/run_context.py`): single owner of a diagnosis
  run's output directory, replacing the old per-example pattern of
  hand-written report files, `RunLogger` buried under a `logs/` subdir, and
  M4 sandboxes living in ephemeral temp dirs deleted on success. Owns
  `report/`, `figures/`, `artifacts/`, `prompts/`, `experiments/`, `tools/`,
  `workspace/`, `fixes/`, plus `manifest.json` and an auto-generated
  `README.txt`. `write_diagnose_report(report, cases, discovery=...)` writes
  the standard `report/` deliverables, duck-typed across `VLDiagnoseReport`
  and `AutoDiagnoseReport`. Migrated all 11 VL examples + `nl_runner`'s
  codegen template off the old `RunLogger(run_dir / "logs")` convention.

- **Per-trial output folders** (`RunContext.new_trial()` / `Trial`): each
  `FixAgent` candidate and M4 `ExperimentWriter` experiment now gets its own
  lazily-created, numbered folder (`fixes/03_widen_crop/`,
  `experiments/01_...`) holding its generated code, the sandbox it ran in,
  judge prompt/output, `record.md`, and `result.json` — instead of scattering
  those across `tools/` / `workspace/` / `fixes/` and re-correlating by
  filename slug. A discarded/deduped candidate leaves no folder on disk; a
  gap in the numbering means "proposed, then discarded."

- **`relative_attention` overlay heatmaps** (`analyzers/attention/relative_attn.py`):
  `RelativeAttentionResult.overlay()` / `.save_overlay()` / `.image_overlays()`
  alpha-blend each spatial map onto its representative case image (CAM-style),
  resolving lazy `Inputs.image` paths/URLs the same way the model's forward
  pass did. `RunLogger` picks this up via a duck-typed `image_overlays()` hook
  on `Result`, so overlay PNGs land in `figures/` alongside the bare heatmaps
  and reach the multimodal judge the same way.

- **`run_log.jsonl` schema_version**: every event now carries a `schema_version`
  (int), bumped only when an existing event's fields are renamed, removed, or
  change meaning, so downstream parsers can detect breaking changes without
  guessing from `evalvitals_version`.

- **`run_log.jsonl` schema_version 2 — M2 stats payloads externalized above
  4 KB**: `analysis`'s `stats_tool_results`/`stats_results`/`stats_plan`/
  `corrected_rejections` no longer inline unboundedly — once the serialized
  value exceeds 4 KB it's written to `artifacts/c{cycle}_m2_{field}.json` and
  the JSONL line carries `{"path", "n_items", "bytes"}` instead, matching how
  every other heavy field (judge I/O, M1 artifacts) is already handled.
  Typical small runs are unaffected. Also: `RunLogger._codegen_seq` (the
  per-cycle codegen filename counter) now increments under a lock.

- **`FixAgent.max_repair_rounds`** (`eval_agent/stages/fix_agent.py`, from the
  `jiaqiliu` merge): feedback-driven multi-round propose→validate within one
  fix tier. After a round where nothing validates, per-candidate results are
  summarised and fed back to the judge/coder, which proposes a *different*
  strategy — never re-running an identical candidate (candidate dedup via
  `FixAgent._signature`), never raising the tier automatically.

- **`examples/deco_hallu/`** (from the `jiaqiliu` merge): POPE hallucination
  slice example with a no-free-lunch guard.

### Fixed

- **Path-doubling in coded fix/M4 sandboxes**: `RunContext.root`,
  `ExperimentSandbox.workdir`, and `run_coded_pipeline`'s workdir are now
  resolved to absolute paths. A relative `run_dir` (the common case for
  examples) previously made every coded fix/M4 subprocess resolve its own
  script path a second time relative to its new cwd and fail with
  `FileNotFoundError`.
- **`cli_agent.py` venv-PATH fix** (from the `jiaqiliu` merge): spawned CLI
  coding agents now use the same Python interpreter as the loop.

### Changed

- Default `max_cases` raised (32→128) on several white-box analyzers, sized
  for enriched/stratified batches (from the `jiaqiliu` merge).
- Removed the canned `attention_guided_crop` L3a primitive — superseded by
  the L2 coded pipeline's `model_attend()` bridge, since reads need no
  privileged model handle and now go through the agent-authored sandboxed
  path instead.

### Added — experiment infrastructure (ported from AutoResearchClaw)

- **`ExperimentGitManager`** (`eval_agent/git_manager.py`): git-native run versioning.
  Each resolved diagnosis run is committed on branch `eval/{run_id}`; unresolved runs
  are discarded with `git reset --hard HEAD`.  Auto-detected when `run_dir` is inside
  a git repository.

- **`EvolutionStore`** (`eval_agent/evolution.py`): append-only JSONL store for
  cross-run lessons.  Lessons are weighted by a 30-day half-life exponential decay.
  `extract_lessons(report)` derives lessons from `AutoDiagnoseReport` automatically.
  `build_overlay(category)` formats the top-k lessons as a prompt injection string.

- **`JsonlStore`** (`eval_agent/store.py`): durable JSONL-backed implementation of the
  `Store` interface.  Hypotheses are fully round-tripped via `hypothesis_to_dict` /
  `hypothesis_from_dict` and survive process restarts.

- **`create_sandbox` factory** (`eval_agent/factory.py`): `SandboxFactoryConfig(mode=...)`
  dispatches to `ExperimentSandbox` (subprocess, default) or `DockerSandbox` (with
  graceful fallback when Docker is unavailable).

- **`ExperimentSandbox.run_project()`** (`eval_agent/sandbox.py`): multi-file project
  execution.  Path traversal protection (pre-copy syntax check + post-copy symlink
  resolve).  Immutable `experiment_harness.py` injected before execution; projects
  cannot overwrite it.  Numbered `_project_{N}` dirs (thread-safe).  Cleanup-on-success
  policy (failure artefacts preserved for debugging).  `SandboxProtocol` structural type
  for transparent backend substitution.

- **`experiment_harness.py`**: immutable evaluation harness (time budget, NaN-guarded
  metric reporting, `results.json` persistence) injected into every sandbox project.

- **`Hypothesis.to_dict` / `from_dict`** (`eval_agent/hypothesis.py`): serialization
  helpers used by `JsonlStore` and loop checkpointing.

- **Multi-phase `ExperimentWriter`** (`eval_agent/experiment_writer.py`): full port of
  AutoResearchClaw's `CodeAgent`.  Six opt-in phases:
  1. Blueprint — YAML spec with file list, per-file pseudocode, and dependency order.
  2. Sequential generation — files generated in dependency order; each prior file
     summarised by AST-based CodeMem for context injection.
  3. Hard validation — AST parse; critical issues (SyntaxError, missing `__main__` guard,
     unresolvable cross-file imports) trigger targeted repair.
  4. Exec-fix loop — parse traceback to identify failing file/line; targeted ±30-line
     context repair; falls back to full-file repair.
  5. Tree search (opt-in) — explore multiple blueprint variants, score by metrics.
  6. Review dialog (opt-in) — coder-reviewer LLM exchange; reverts if run degrades.
  `result.code` always equals `result.files["main.py"]` for backward compatibility.
  Case images are saved as JPEG files in the sandbox workdir and referenced in the
  codex prompt via `image_path` in `cases.json`.

- **Run-directory infrastructure** (`eval_agent/loop.py`): `AutoDiagnoseLoop` now
  accepts `run_dir`, `git_manager`, and `evolution_store` parameters.
  - Atomic checkpoint writes (temp+rename) to `run_dir/checkpoint.json` after every cycle.
  - Heartbeat writes to `run_dir/heartbeat.json` (pid, last_cycle, timestamp).
  - `AutoDiagnoseLoop.resume(run_dir, model, data)` classmethod reads the checkpoint
    and skips already-completed cycles.
  - `EvolutionStore` auto-created under `run_dir/evolution/` when `run_dir` is set.
  - `ExperimentGitManager` auto-detected from `run_dir` when inside a git repo.

- **VLM image-attention rule** (`eval_agent/analysis.py`): `AnalysisModule` derives
  `image_token_attention_ratio` from `top_attended_tokens` in attention findings.
  Fires a `medium`-severity finding when the ratio is below 0.05 — indicating the VLM
  is ignoring image tokens in favour of structural text tokens.

- **Diagnosis fallback** (`eval_agent/diagnosis.py`): when the LLM judge returns
  `NO_ISSUE` but M2 findings include medium-or-higher severity anomalies, `DiagnosisAgent`
  automatically generates one hypothesis per finding.  Prevents self-diagnosis bias when
  the judge is the same model under test.

- **38 new infrastructure tests** (`tests/test_eval_agent/test_infrastructure.py`):
  git manager, sandbox entry-point validation, `run_project`, harness injection,
  cleanup policy, `JsonlStore` roundtrip, `EvolutionStore` time-decay, `extract_lessons`,
  sandbox factory, `ExperimentWriterResult` backward compat, `AutoDiagnoseLoop` run_dir
  lifecycle, checkpoint/heartbeat/resume.

- **`examples/qwen_loop/`**: end-to-end `AutoDiagnoseLoop` example on Qwen3-VL-4B
  with a real (or synthetic fallback) image.  `VerboseRunLogger` mirrors each M1/M2/M3/M4
  event to stdout as it happens.  Docker Compose with CUDA 12.4 wheels, GPU selection,
  host codex binary mount, and `./outputs/` volume.

### Added
- `ModelSpec` / `Backend` / `compose()` architecture — identity separate from runtime.
- 14 model specs registered: Qwen3/Qwen2.5/Qwen2 (LLM + VLM), DeepSeek-V3, Llama 3.1, Gemma 3, GLM-4, Kimi-VL, Llama-4-Scout, Step-1o.
- Capability enum extended: `LOGPROBS`, `TOOL_CALLS` (split from `LOGITS`).
- `Agent` — backend-agnostic tool-calling loop over any model with `GENERATE + TOOL_CALLS`.
- `ToolCallCodec` — OpenAI native and Qwen/Hermes text codecs.
- `evalvitals.wrap()` — captum-style on-ramp for any already-loaded HF model.
- Attention analyzers: `AttentionAnalyzer`, `AttentionRolloutAnalyzer`, `AttentionSinkAnalyzer`, `RelativeAttentionAnalyzer` (arXiv:2502.17422).
- Perturbation analyzers: `RISEAnalyzer`, `MMSHAPAnalyzer` (arXiv:2212.08158), `VLSHAPAnalyzer`.
- Uncertainty analyzers: `TokenEntropyAnalyzer`, `LogprobEntropyAnalyzer`, `SelfConsistencyAnalyzer`, `VerbalizedConfidenceAnalyzer`.
- Hallucination analyzers: `POPEAnalyzer` (arXiv:2305.10355), `CHAIRAnalyzer` (arXiv:1809.02156); stubs for OPERA and VCD.
- Lens analyzers: `LogitLensAnalyzer`; stub for `TunedLensAnalyzer`.
- Attribution stubs: `GradCAMAnalyzer`, `GenericAttentionExplainability`.
- Patching stub: `CausalTraceAnalyzer`.
- Geometry analyzers: `LinearCKAAnalyzer`; stub for `LinearProbeAnalyzer`.
- Agent analyzers: `LoopDetectAnalyzer`, `IgnoredObsAnalyzer`, `FirstErrorJudgeAnalyzer`, `CounterfactualAnalyzer`.
- Datasets: `PureQADataset`, `WebSearchQADataset`, `GUIOSDataset` → `CaseBatch`.
- Stats: `compare()` / `compare_multiple()` — effect size, clustered-bootstrap CI, e-value, BH correction.
- `eval_agent`: `EvalOrchestrator`, `PreregisteredHypothesis`, `SelfEvolveLoop` (interfaces in place, LLM proposer in Stage 2).
- CI: GitHub Actions matrix (Python 3.10/3.11/3.12) with ruff + mypy + pytest.
- PyPI trusted publishing (OIDC) release workflow.

## [0.1.0] — unreleased

Initial alpha. Core contracts (`Model`, `Analyzer`, `Result`, `FailureCase`, registry, pipeline, experiment).
