# Changelog

All notable changes to EvalVitals will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added/Changed — signal hygiene, descriptive analysis, tensor-level attention

- **Label-leak isolation (the deferred "leak-1" check)** (`eval_agent/stages/stats_tools.py`):
  `label_leak_score` / `isolate_label_leaks` detect per-case signals that
  *reconstruct* the FAIL label (a probe flag equal to the outcome) and route them
  to a new `StatsInput.sanity` lane, out of the tested family / e-BH multiplicity
  / candidate charts / hypothesis seeding. A leak is a **binary flag that ~equals
  the label** (best-split accuracy ≥ 0.985, ≥ 10 cases) — a *continuous* feature
  that perfectly separates the classes is legitimate discovery and is NOT flagged.
  Wired into `build_stats_input` / `build_stats_input_from_records` and the fused
  pipeline's confirm step, so `generated:probe1.false_detection` /
  `explored.probe1_positive` no longer get tested, charted, or "confirmed."

- **Descriptive analysis phase (validity deferred to confirm)**
  (`StatsAnalysisAgent.analyze(confirmatory=…)`, `VLDiagnoseLoop`,
  `reporting/compiler.py`): `run_analysis()` now runs M2 with the e-BH validity
  verdict DEFERRED (`StatsAnalysisReport.descriptive_only=True`, `corrected_rejections`
  marked deferred) — effect sizes + charts only. `run_confirm()` recomputes e-BH
  (`_finalize_confirmatory_stats`) and logs the confirmatory M2, so the dashboard
  shows supported/not-supported claims ONLY after confirmation. The compiler
  demotes every claim to descriptive in analysis-phase mode with a banner; the
  all-in-one `run()` path stays confirmatory and unchanged.

- **Richer per-case attention features + per-case map stack**
  (`analyzers/attention/relative_attn.py`): each case now also emits
  `attention_entropy`, `top1_share`, `center_offset`, `edge_mass` (diffuse-vs-spike,
  peripheral, positional-sink proxies) alongside max/mean/focus, and the full
  per-case spatial maps are stored (float16) in `artifacts["per_case_maps"]`.
  `prompt_contrast` now surfaces a non-tautological per-case `prompt_sensitivity`
  signal (answer instability across prompts, correctness-independent) — the
  tautological `fixed_by_*`/`broken_by_*` flags stay in artifacts.

- **`attention_decoding` — tensor-level omnibus** (`stats_tools.py`): a new M2 tool
  over the FULL per-case attention map (not a scalar reduction), answering "do
  FAIL and PASS attend differently anywhere?" — feature-agnostic, pure numpy,
  robust at features≫samples. Primary test is a two-sample **energy-distance
  permutation** test (more powerful than linear decoding at low n; sensitive to
  nonlinear / distributional differences), with a cross-validated linear-decoder
  AUC reported alongside as an interpretable companion. Reads
  `StatsInput.per_case_vectors`, runs as a mandatory global/omnibus tool in M5,
  and is added to `default_plan` when map vectors exist. On the deco_hallu slice
  the energy test flips the verdict from inconclusive (CV-AUC 0.42) to a real
  finding (energy-distance D=1.88, permutation p=0.018) — the maps differ
  distributionally even though no linear boundary separates them.

### Fixed — analysis-phase dashboard rendering

- **Stale confirm runs no longer leak verdicts into the analysis view**
  (`analysis/dashboard.py`): `load_loop_story` merged events from *every*
  `logs*/` dir, so a directory holding both a descriptive `logs_analysis/` run
  and a stale all-in-one `logs_m2_5/` run would resurrect the old surgeries +
  supported/not-supported verdicts on top of the descriptive run. It now keeps
  the shared M1 probe log plus only the **single most-recent M2+ arc** (by
  mtime), so an analysis-phase directory renders descriptively without a symlink
  workaround.
- **Analysis phase shows candidate signals descriptively** (`analysis/dashboard_app.py`):
  when the loaded run is analysis-only (`_story_is_descriptive`: every M2
  `descriptive_only`, no surgery), evidence cards render a **"Descriptive"**
  badge ranked by |effect| instead of mapping the explorer's `reject` flags to
  "Supported"/"Not supported", the method card drops the e-BH/"tested signals
  survived" framing, and an "Analysis phase" banner orients the reader. The
  explorer's per-recipe `reject` is not confirmation.
- **Scatter tables with real-named columns no longer crash** (`_resolve_scatter_axes`):
  the explorer now writes scatter CSVs with real signal columns
  (`attention_entropy, center_offset, outcome`); the dashboard assumed legacy
  `x`/`y` columns and fell back to literal `"x"`/`"y"`, raising plotly's
  `Value of 'x' is not the name of a column`. Axis resolution now prefers the
  report's recovered names, then the CSV's own non-outcome columns, and skips
  cleanly when two value axes can't be formed.

### Added — decoupled analysis vs. confirm+fix

- **`VLDiagnoseLoop.run_analysis()` + `VLDiagnoseLoop.run_confirm()`**
  (`eval_agent/loop.py`): split the diagnosis pipeline so the analysis
  dashboard can be produced *before* hypotheses are confirmed. `run_analysis()`
  runs **M1 → M2 → M3** (the same rigorous e-BH stats + charts, then *propose*
  hypotheses) and stops — the returned `VLDiagnoseReport` carries
  `all_hypotheses` (proposed, unconfirmed) and `final_stats_report`, with
  `all_test_results` / `verified_hypotheses` empty (`stopped_by="analysis_complete"`).
  `run_confirm(data, hypotheses, stats_report=...)` runs **M5** on those
  hypotheses — typically reloaded via `hypothesis_from_dict` and confirmed
  against the *exact* M2 report the dashboard showed (regenerated from the
  frozen M1 when omitted) — then feeds `run_m4` / `run_fix` as before. The
  shared per-stage helpers (`_do_m1/_do_m2/_do_m3/_do_m5`) are factored out of
  `run()`, whose behavior is unchanged. The dashboard renders proposed
  hypotheses without M5/M4/Fix verdicts and gains them once the confirm phase's
  log dir is present.

- **deco_hallu decoupled scripts** (`examples/diagnosis_loops/deco_hallu/`):
  `run_analysis.py` (GPU-free: replay M1 → M2 stats/charts → M3 propose →
  dashboard, persisting `outputs/analysis/{proposed_hypotheses.json,
  analysis_state.pkl}`) and `run_confirm_fix.py` (reload those artifacts → M5
  confirm → M4 + tiered Fix), with matching `run_analysis.sh` /
  `run_confirm_fix.sh` wrappers. The shared frozen-M1 `ReplayProbeAgent` and a
  GPU-free `FrozenModel` stub now live in `run.py`. The one-shot `run_m2-5.py`
  path is unchanged.

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

- **`run_start` provenance — dataset + code version**: the first
  `run_log.jsonl` event now also records `data_fingerprint` (an
  order-independent SHA-1 over the case batch, so two runs can be confirmed to
  use the same data) and `label_distribution` (the base PASS/FAIL/UNKNOWN
  counts the whole diagnosis is conditioned on) alongside the existing
  `n_cases`. `git_commit` now falls back to the `EVALVITALS_GIT_COMMIT` env var
  when the `git` CLI is unavailable, so the code-version provenance is no longer
  silently dropped inside the example Docker images (which ship no git). The
  `eval_agent` compose file forwards `EVALVITALS_GIT_COMMIT`.

- **Published JSON Schema for `run_log.jsonl`** (`eval_agent/log_schema.py` +
  shipped `run_log.schema.json`): the log event format is now a machine-readable
  contract (Draft 2020-12), not just a docstring + an opaque `schema_version`
  int. `build_schema()` generates it from the stdlib (no new core dependency —
  the light install is preserved); `load_schema()`, `validate_event()` and
  `iter_log_errors()` validate logs (needing the optional `jsonschema` dev dep).
  The schema is permissive (pins the envelope, per-event required fields and
  core types; allows additive fields). A contract test drives every `RunLogger`
  event type and asserts the real output conforms, so the schema can't silently
  drift from the producer. Opt-in `EVALVITALS_VALIDATE_LOG=1` makes `RunLogger`
  self-check each event and warn (never raise) on a violation.

- **`self_consistency` records its sampling config**: the analyzer's findings
  now include `gen_kwargs` (the kwargs passed to `model.generate`, temperature
  above all). The consistency score is uninterpretable without it — a low score
  at temperature 0 is a real defect, the same score at 1.0 is expected — so the
  parameter the measurement is conditioned on now travels with it into the
  `probe` event. Empty dict means the model's own `generate()` defaults.

- **`FixAgent.max_repair_rounds`** (`eval_agent/stages/fix_agent.py`, from the
  `jiaqiliu` merge): feedback-driven multi-round propose→validate within one
  fix tier. After a round where nothing validates, per-candidate results are
  summarised and fed back to the judge/coder, which proposes a *different*
  strategy — never re-running an identical candidate (candidate dedup via
  `FixAgent._signature`), never raising the tier automatically.

- **`examples/diagnosis_loops/deco_hallu/`** (from the `jiaqiliu` merge): POPE hallucination
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
