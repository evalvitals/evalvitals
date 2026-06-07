# Architecture

EvalVitals is structured as a small framework substrate plus extension points.
The goal is to make LLM/VLM evaluation feel like using sklearn estimators:
objects are composable, parameters are explicit, capabilities are discoverable,
and outputs follow a common shape.

## Package Layout

```text
evalvitals/
+-- core/              # stable contracts and shared substrate
+-- specs.py           # model identity registry
+-- models/            # model composition, runtime backends, compatibility shims
+-- analyzers/         # analyzers grouped by capability (attention, lens, uncertainty, …)
+-- datasets/          # loaders that produce FailureCase / CaseBatch
+-- stats/             # statistical tests: McNemar, e-value, bootstrap CI, Friedman
`-- eval_agent/        # automated diagnosis loop + selective-inference orchestration
```

## Core Contracts

### Two paths to a `Model`

```text
# Public on-ramp — user brings their own already-loaded HF causal LM
evalvitals.wrap(model, tokenizer)  ->  HFLocalModel

# Curated path — load a registered checkpoint by key
evalvitals.load("qwen2.5-7b-instruct")  ->  HFLocalModel
```

Both paths return the same `HFLocalModel`: capabilities are inferred from the
live model in the `wrap()` case, and read off the spec in the `load()` case.
`wrap()` also applies attention fix-ups automatically (eager mode is required to
capture attention weights; sdpa/flash return `None`).

### `ModelSpec`

`ModelSpec` describes what a model is, not how it is run. It stores identity and
architecture facts such as model family, Hugging Face repo, decoder-layer paths,
vision-token handling, MoE flags, reasoning flags, and attention semantics.

Specs live in `evalvitals.specs` and are intentionally torch-free.  When
`wrap()` is used, a minimal spec is inferred at runtime from `model.config` via
`evalvitals.models.inference.infer_spec` — no registry entry is required.

### `Backend`

`Backend` describes how a model is run. Backends declare the capabilities they
can provide and build concrete `Model` objects from a `ModelSpec`.

Current backend categories:

| Backend | Purpose |
|---|---|
| `hf_local` | Local Hugging Face execution with internals capture. |
| `api` | Black-box generation through an injected API function. |
| `vllm_offline` | Planned high-throughput offline inference backend. |

Capabilities belong to the backend because the same model identity can expose
different information under different runtimes.

### `Model`

`Model` is the runtime object analyzers consume. It exposes:

```python
model.generate(inputs, **kwargs) -> str
model.forward(inputs, capture={...}, spec=None) -> Trace
```

`forward` returns a `Trace`, which is the common carrier for captured internals
such as tokens, token ids, attentions, hidden states, logits, and backend-specific
extras.

### `Analyzer`

`Analyzer` is the EvalVitals analogue of an sklearn estimator. It has explicit
constructor parameters, declares required capabilities, and returns a `Result`.

```python
analyzer = SomeAnalyzer(**params)
result = analyzer.run(model, data)
```

Analyzers should not depend on concrete model classes. They should depend on
the `Model` protocol, requested captures, and `Trace` fields.

### `Capability`

`Capability` is the matching vocabulary between analyzers and runtimes.

An analyzer declares:

```python
requires = frozenset({Capability.ATTENTION})
```

A backend/model declares:

```python
capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
```

The registry can then list compatible analyzers for a model, and `compose(...,
want=...)` can fail early before loading weights.

### `FailureCase`

`FailureCase` is the common data unit. It is meant to hold inputs, labels,
provenance, metadata, and agent trajectories. Datasets should produce
`FailureCase` or `CaseBatch`; analyzers should accept those types in addition to
plain strings where appropriate.

### `Result`

`Result` is the common output object. It separates:

- a short human-readable summary,
- structured `findings` for agents and downstream code,
- optional heavy artifacts such as plots, tensors, or tables.

## Why This Shape Works

The design keeps common failure modes contained:

- Adding a new model family should usually mean adding a `ModelSpec`, not
  rewriting analyzers.
- Adding a new runtime should usually mean implementing a `Backend`, not
  changing model identity.
- Adding a new analysis should usually mean implementing an `Analyzer` that
  requests capabilities, not adding methods to every model.
- Agent tooling can discover what is possible from registries instead of reading
  source code or hard-coding model names.

## eval_agent — automated diagnosis pipeline

`eval_agent/` implements a multi-stage automated diagnosis cycle on top of the
core contracts described above.  Two loops are available:

```text
AutoDiagnoseLoop  (legacy, four-stage sweep)
  M1 · ProbeAgent         detect model kind → run ranked analyzers
  M2 · AnalysisModule     threshold rules + derived metrics → AnalysisReport
  M3 · DiagnosisAgent     judge.generate(report) → Hypothesis list
  M4 · SurgeryAgent       correlate / param-sweep / ExperimentWriter → SUPPORTED/REFUTED
       ↑_________________________________________________________________| (refocus or stop)

VLDiagnoseLoop  (Plan A — protocol-guided, stops on verified hypothesis)
  ExperimentProtocol  ← user's NL description of what to investigate
       │
  M1 · ProbeAgent         same as above; protocol.probe_hints() boosts relevant analyzers
  M2 · StatsAnalysisAgent protocol-aware; LLM judge writes conclusion + evidence chain
  M3 · DiagnosisAgent     same as above
  M5 · HypothesisTester   statistical test + protocol consistency check
       │
  loop exits when M5 finds a SUPPORTED + protocol-consistent hypothesis
       │
  M4 · SurgeryAgent       called once post-loop on the best verified hypothesis
```

The agent touches models only through the `Model` protocol and stores all
evidence in a `Store`.

### Package layout

```text
eval_agent/
├── loop.py               AutoDiagnoseLoop, VLDiagnoseLoop, SelfEvolveLoop
├── run_logger.py         RunLogger — per-cycle JSONL log + artifact sink
├── hypothesis.py         Hypothesis, HypothesisStatus — shared across M3/M4/M5
├── cli_agent.py          CliAgentConfig, create_cli_agent — shared CLI coding agent
│                           (agy / codex); any stage can use this to launch experiments
├── store.py              Store / InMemoryStore / JsonlStore
├── evolution.py          EvolutionStore — cross-run lesson accumulation
├── orchestrator.py       EvalOrchestrator — thin A/B facade
├── ab_runner.py          ABRunner — A/B execution
├── preregister.py        DataSplit, PreregisteredHypothesis
├── sandbox.py            ExperimentSandbox, SandboxProtocol
├── factory.py            create_sandbox (subprocess / docker)
├── git_manager.py        ExperimentGitManager
├── report.py             DiagnosticReport
└── stages/               ← M1–M5 stage implementations
    ├── probe.py          M1  StrategyProbe
    ├── probe_agent.py    M1  ProbeAgent
    ├── protocol.py       M1  ExperimentProtocol, ProbingSchema
    ├── analysis.py       M2  AnalysisModule, AnalysisReport
    ├── stats_agent.py    M2  StatsAnalysisAgent, StatsAnalysisReport
    ├── diagnosis.py      M3  DiagnosisAgent, DiagnosisResult
    ├── surgery.py        M4  SurgeryAgent, InterventionResult
    ├── experiment_writer.py  M4  ExperimentWriter
    └── hypothesis_tester.py  M5  HypothesisTester, HypothesisTestResult
```

### Stage contracts

| Stage | Module | Class | Key method |
|---|---|---|---|
| M1 | `stages/probe.py` | `StrategyProbe` | `detect_kind(model) → ModelKind`; `select(model, hints) → list[str]` |
| M1 | `stages/probe_agent.py` | `ProbeAgent` | `probe(model, data, hint_failure_modes) → dict[str, Result]` |
| M1 | `stages/protocol.py` | `ExperimentProtocol` | `probe_hints() → list[str]` — maps NL description to failure-mode tags |
| M2 | `stages/analysis.py` | `AnalysisModule` | `analyze(results, model_name) → AnalysisReport` |
| M2 | `stages/stats_agent.py` | `StatsAnalysisAgent` | `analyze(results, model_name, protocol) → StatsAnalysisReport` |
| M3 | `stages/diagnosis.py` | `DiagnosisAgent` | `diagnose(report, prior_cycles) → DiagnosisResult` |
| M4 | `stages/surgery.py` | `SurgeryAgent` | `operate(hypothesis, model, results, data) → InterventionResult` |
| M5 | `stages/hypothesis_tester.py` | `HypothesisTester` | `test(hypotheses, report, data, protocol) → list[HypothesisTestResult]`; `stopping_criteria_met(results) → bool` |

All stages are injectable:

```python
# AutoDiagnoseLoop
loop = AutoDiagnoseLoop(
    model=model,
    probe_agent=ProbeAgent(...),
    analysis_module=AnalysisModule(...),
    diagnosis_agent=DiagnosisAgent(judge=judge),
    surgery_agent=SurgeryAgent(judge=judge),
    store=JsonlStore(run_dir / "store"),
    run_logger=RunLogger(run_dir / "logs"),
    run_dir=run_dir,
)

# VLDiagnoseLoop
protocol = ExperimentProtocol(
    description="VLM suspected to ignore visual tokens ...",
    failure_patterns="spatial confusion, hallucinated objects",
)
loop = VLDiagnoseLoop(
    model=model,
    protocol=protocol,
    stats_agent=StatsAnalysisAgent(judge=model),
    diagnosis_agent=DiagnosisAgent(judge=model),
    hypothesis_tester=HypothesisTester(judge=model, min_effect=0.05),
    surgery_agent=SurgeryAgent(judge=model),
    max_cycles=5,
    run_logger=RunLogger(run_dir / "logs"),
)
report = loop.run(cases)
fix    = loop.run_m4(report, cases)   # M4 called post-loop on best hypothesis
```

### M4 SurgeryAgent — four strategies

M4 selects the first matching strategy:

1. **`verify_fn`** — caller-supplied callable; full custom override.
2. **`analyzer_params`** — re-run named analyzers with modified parameters; surface before/after findings.
3. **`ExperimentWriter`** (when `judge` is provided) — multi-phase LLM/CLI agent writes and executes a targeted Python diagnostic project:
   - Phase 1: Blueprint (YAML spec: file list, pseudocode, dependency order)
   - Phase 2: Sequential file generation with AST-based CodeMem context
   - Phase 3: Hard validation (AST parse; critical issues trigger repair)
   - Phase 4: Exec-fix loop (parse traceback → targeted single-file repair)
   - Phase 5: Tree search (optional; explore multiple candidates, score by metrics)
   - Phase 6: Review dialog (optional; coder-reviewer LLM exchange)
4. **Label correlation** — passive; correlate per-case signals with PASS/FAIL labels.

CLI coding agents (`codex`, `claude_code`, `opencode`, …) can substitute for the
LLM writer in Phase 1+2 via `ExperimentWriterConfig(cli_agent=CliAgentConfig(provider="codex"))`.
The generated code project is executed by `ExperimentSandbox.run_project()`.  Case images
are saved as JPEG files in the sandbox workdir so the agent can load them.

### Sandbox

`ExperimentSandbox` runs Python code safely in a subprocess.

```python
sandbox = ExperimentSandbox(workdir=Path("tmp/"))

# Single-file execution
result = sandbox.run("print('verdict: 1.0')")

# Multi-file project execution (M4 ExperimentWriter path)
result = sandbox.run_project(
    project_dir,
    entry_point="main.py",
    timeout_sec=60,
)
```

Key properties:
- **Path traversal protection**: `entry_point` is validated syntactically before copy and
  after copy (symlink-resolved) to prevent directory escape.
- **Harness injection**: `experiment_harness.py` is copied into every project directory before
  execution.  It provides time-budget management, metric reporting with NaN guards, and
  `results.json` persistence.  Projects cannot overwrite it.
- **Numbered project dirs**: concurrent calls produce `_project_1/`, `_project_2/`, etc. (thread-safe).
- **Cleanup policy**: project/script directories are deleted only on success (`rc==0` and
  not timed out), preserving failure artefacts for debugging.

`SandboxProtocol` is a structural type allowing transparent substitution of subprocess,
Docker, SSH, or other backends.  `create_sandbox(SandboxFactoryConfig, workdir)` selects
the backend from a `mode` string (`"subprocess"` default, `"docker"` with graceful fallback).

### Run-directory infrastructure

Pass `run_dir` to `AutoDiagnoseLoop` to enable the full operational stack:

```python
loop = AutoDiagnoseLoop(
    model=model,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    run_dir=Path("runs/my_experiment"),   # enables all infrastructure below
)
report = loop.run(cases)

# Resume a previously interrupted run
report = AutoDiagnoseLoop.resume(Path("runs/my_experiment"), model=model, data=cases)
```

When `run_dir` is set, `AutoDiagnoseLoop` creates:

```text
run_dir/
├── checkpoint.json          ← atomic (temp+rename); last_completed_cycle + run_id
├── heartbeat.json           ← pid + last_cycle + timestamp (liveness signal)
├── artifacts/<run_id>/      ← per-run staging area
└── evolution/
    └── lessons.jsonl        ← cross-run lesson accumulation (append-only)
```

**Checkpointing and resume**: after every completed cycle, `checkpoint.json` is written
atomically.  `AutoDiagnoseLoop.resume(run_dir, model, data)` reads the checkpoint and
restarts from `last_completed_cycle + 1`, skipping already-completed work.

**Git integration**: when `run_dir` is inside a git repository, `ExperimentGitManager`
auto-detects it and:
- Creates branch `eval/{run_id}` at the start of the run.
- Commits all staged files with hypothesis statuses on a resolved run.
- Calls `git reset --hard HEAD` on an unresolved run (non-destructive: only uncommitted
  changes are discarded).

### EvolutionStore — cross-run lesson accumulation

`EvolutionStore` accumulates lessons from every diagnosis run in an append-only JSONL
file.  Lessons are weighted by a 30-day half-life exponential decay so recent findings
rank higher.

```python
store = EvolutionStore(Path("runs/my_experiment/evolution"))

# Lessons are appended automatically when run_dir is set.
# Read them back for prompt injection:
overlay = store.build_overlay("surgery", max_lessons=5)
# → "## Lessons from Prior Diagnosis Runs\n1. [WARN] ..."
```

`extract_lessons(report)` auto-derives lessons from an `AutoDiagnoseReport`:
- INCONCLUSIVE hypotheses → `surgery / warning`
- Loop exhausted without resolution → `diagnosis / warning`
- HIGH/CRITICAL analysis severity with no resolution → `analysis / info`

### Persistent store — `JsonlStore`

`JsonlStore` is a durable implementation of the `Store` interface backed by three JSONL files
(`hypotheses.jsonl`, `results.jsonl`, `cases.jsonl`).  Hypotheses survive process restarts
and are fully reconstructed via `hypothesis_to_dict` / `hypothesis_from_dict`.

```python
store = JsonlStore(Path("runs/store"))
loop = AutoDiagnoseLoop(model=model, store=store, ...)
```

### Analysis rules — VLM image-attention

`AnalysisModule` includes a VLM-specific derived metric for the `attention` analyzer.
Before applying threshold rules, it sums the attention weights of image-related tokens
(`<|image_pad|>`, `<|vision_start|>`, `<|vision_end|>`, …) from `top_attended_tokens`
and exposes `image_token_attention_ratio`.  A ratio below 0.05 fires a `medium`-severity
finding:

```
[MEDIUM] attention.image_token_attention_ratio=0.012 < 0.05:
VLM nearly ignores image tokens — attention dominated by text/structural tokens
```

This finding propagates to M3 DiagnosisAgent and M4 SurgeryAgent, closing the loop from
attention measurement to codex-generated diagnostic code.

### Logging and artifact persistence

`RunLogger` captures every M1→M4 event as a JSON line in `run_log.jsonl` and
saves heavy artifacts (attention tensors, CKA matrices, hidden-state arrays) to
an `artifacts/` subdirectory keyed by cycle number.  It is entirely opt-in —
the default `run_logger=None` leaves existing behaviour unchanged.

```text
run_dir/logs/<run_id>/
├── run_log.jsonl                         ← one JSON line per event
└── artifacts/
    ├── c0_attention_attentions.json      ← top-attended tokens, cycle 0
    ├── c0_attention_rollout_rollout.npy  ← rollout tensor
    └── c1_cka_layer_similarities.npy     ← after data refocus, cycle 1
```

Each log line carries `event`, `cycle`, `ts` (ISO-8601), and stage-specific
fields (findings, narrative, raw LLM output, intervention status …).
This makes the full diagnosis trail inspectable offline with standard tools:

```bash
tail -f run_dir/logs/*/run_log.jsonl                    # live stream
jq 'select(.event=="diagnosis")' run_log.jsonl          # all judge outputs
jq 'select(.event=="probe") | .findings' run_log.jsonl  # M1 findings
jq 'select(.event=="surgery") | .evidence' run_log.jsonl
```

## Public Surface Guidance

The intended stable public entry points are:

```python
# Model construction — two paths, same result object
evalvitals.wrap(model, tokenizer, *, want=(), **runtime)  # bring your own model
evalvitals.load(key, *, backend, want, checkpoint, **runtime)  # curated checkpoints

# Config-driven run
evalvitals.run(config, data)
evalvitals.load_config(path)

# Registry / discovery
evalvitals.list_specs()
evalvitals.get_spec(key)
evalvitals.registry

# Core types
evalvitals.Capability
evalvitals.FailureCase
evalvitals.Result

# Automated diagnosis — AutoDiagnoseLoop (legacy M1→M4 sweep)
from evalvitals.eval_agent import AutoDiagnoseLoop, DiagnosisAgent, RunLogger, StrategyProbe, SurgeryAgent

# Protocol-guided diagnosis — VLDiagnoseLoop (M1→M2→M3→M5, M4 post-loop)
from evalvitals.eval_agent import (
    VLDiagnoseLoop, ExperimentProtocol,
    StatsAnalysisAgent, HypothesisTester,
)
```

Lower-level implementation details (`compose`, `HFLocalModel`, `infer_spec`,
`Backend`, `ModelSpec`) should remain under their package namespaces unless they
are meant to become long-term extension APIs.
