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

## AutoDiagnoseLoop — M1→M4 pipeline

`eval_agent/` implements a four-module automated diagnosis cycle on top of the
core contracts described above.

```text
M1 · StrategyProbe   detect model kind (VLM/AGENT/LLM) → ranked analyzer list
M2 · Execution       Experiment + ExperimentRunner (content-hash cache)
M3 · DiagnosisAgent  judge.generate(findings_json) → HYPOTHESIS:/FAILURE_MODE: pairs
M4 · SurgeryAgent     correlate per-case signals with PASS/FAIL → SUPPORTED/REFUTED
     ↑____________________________________________________________| (refocus or stop)
```

The agent touches models only through `eval_agent/tools.py`
(`list_analyses`, `compatible_analyses`, `run_analysis`) and stores all
evidence in a `Store`.  `AutoDiagnoseLoop` is the concrete controller that wires
the four modules; `SelfEvolveLoop` is the original Stage-1 skeleton kept for
backward compatibility.

### Module responsibilities

| Module | Class | Contract |
|---|---|---|
| `probe.py` | `StrategyProbe` | `detect_kind(model) → ModelKind`; `select(model) → list[str]` |
| *(M2 uses core)* | `ExperimentRunner` | `run(Experiment) → Result` (cached by fingerprint) |
| `diagnosis.py` | `DiagnosisAgent` | `diagnose(results, model_name) → DiagnosisResult` |
| `surgery.py` | `SurgeryAgent` | `operate(hypothesis, model, results, data) → InterventionResult` |
| `loop.py` | `AutoDiagnoseLoop` | `run(data) → AutoDiagnoseReport` |

All modules are injectable: pass your own `probe`, `diagnosis_agent`,
`surgery_agent`, `store`, and `runner` to `AutoDiagnoseLoop` to customise any
step without touching the others.

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

# Automated diagnosis
from evalvitals.eval_agent import AutoDiagnoseLoop, DiagnosisAgent, StrategyProbe, SurgeryAgent
```

Lower-level implementation details (`compose`, `HFLocalModel`, `infer_spec`,
`Backend`, `ModelSpec`) should remain under their package namespaces unless they
are meant to become long-term extension APIs.
