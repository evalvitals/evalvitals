"""Self-evolving evaluation agent — automated failure discovery and diagnosis.

Two loops are available:

  AutoDiagnoseLoop   M1 → M2 → M3 → M4 (legacy, four-stage sweep)
  VLDiagnoseLoop     M1 → M2 → M3 → M5 inner loop, M4 called post-loop (Plan A)
                     Stops when M5 finds a statistically supported,
                     protocol-consistent hypothesis.

Stage modules live in ``stages/``; shared infrastructure stays at the top level.

Top-level (shared / orchestration):
  loop.py              AutoDiagnoseLoop, VLDiagnoseLoop, SelfEvolveLoop
  run_logger.py        RunLogger — per-cycle JSONL log + artifact sink
  hypothesis.py        Hypothesis, HypothesisStatus, serialization helpers
  cli_agent.py         CliAgentConfig, create_cli_agent — shared CLI coding agent
                         (agy / codex); any stage can use this to launch experiments
  store.py             Store / InMemoryStore / JsonlStore — persistent memory
  orchestrator.py      thin facade over the loop (pre-registered A/B)
  ab_runner.py         A/B execution across prompting strategies
  report.py            DiagnosticReport — final diagnostic conclusions
  evolution.py         EvolutionStore — JSONL lesson store, 30-day half-life decay
  preregister.py       pre-registration helpers (DataSplit, PreregisteredHypothesis)
  sandbox.py           ExperimentSandbox, SandboxProtocol
  factory.py           sandbox factory (subprocess / docker backends)
  git_manager.py       git-native experiment versioning (eval/{run_id} branches)
  experiment_harness.py immutable evaluation harness injected into projects

stages/ (M1–M5 implementation):
  probe.py             M1 — StrategyProbe: model-kind detection + analyzer ranking
  probe_agent.py       M1 — ProbeAgent: execute ranked analyzers (direct or Docker);
                              protocol-guided via ExperimentProtocol.probe_hints()
  protocol.py          M1 — ExperimentProtocol (NL description → probe hints);
                              ProbingSchema (records M1 selection rationale)
  analysis.py          M2 — AnalysisModule: threshold rules → AnalysisReport
  stats_agent.py       M2 — StatsAnalysisAgent: extends AnalysisModule with LLM-guided
                              conclusion + evidence chain (StatsAnalysisReport)
  diagnosis.py         M3 — DiagnosisAgent: judge reads report → Hypothesis list
  surgery.py           M4 — SurgeryAgent: correlate / param-sweep / ExperimentWriter
                              → InterventionResult (SUPPORTED / REFUTED / INCONCLUSIVE)
  experiment_writer.py M4 — multi-phase LLM/CLI agent writes + executes fix scripts
  hypothesis_tester.py M5 — HypothesisTester: statistical test + protocol consistency;
                              stopping_criteria_met() drives the VLDiagnoseLoop exit
"""

from evalvitals.eval_agent.ab_runner import ABResult, ABRunner
from evalvitals.eval_agent.cli_agent import CliAgentConfig, CliAgentResult, create_cli_agent
from evalvitals.eval_agent.evolution import EvolutionStore, LessonEntry, extract_lessons
from evalvitals.eval_agent.factory import SandboxConfig, SandboxFactoryConfig, create_sandbox
from evalvitals.eval_agent.git_manager import ExperimentGitManager
from evalvitals.eval_agent.hypothesis import (
    Hypothesis,
    HypothesisGenerator,
    HypothesisStatus,
    ManualHypothesisGenerator,
    hypothesis_from_dict,
    hypothesis_to_dict,
)
from evalvitals.eval_agent.loop import (
    AutoDiagnoseLoop,
    AutoDiagnoseReport,
    SelfEvolveLoop,
    VLDiagnoseLoop,
    VLDiagnoseReport,
)
from evalvitals.eval_agent.orchestrator import EvalOrchestrator
from evalvitals.eval_agent.preregister import (
    DataSplit,
    PreregisteredHypothesis,
    PreregistrationLog,
    Split,
)
from evalvitals.eval_agent.report import DiagnosticReport
from evalvitals.eval_agent.run_logger import RunLogger
from evalvitals.eval_agent.sandbox import (
    ExperimentSandbox,
    SandboxProtocol,
    SandboxResult,
    parse_metrics,
    validate_entry_point,
    validate_entry_point_resolved,
)
from evalvitals.eval_agent.stages.analysis import AnalysisModule, AnalysisReport
from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent, DiagnosisResult
from evalvitals.eval_agent.stages.experiment_writer import (
    ExperimentWriter,
    ExperimentWriterConfig,
    ExperimentWriterResult,
    SolutionNode,
    build_model_context,
)
from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTester, HypothesisTestResult
from evalvitals.eval_agent.stages.probe import ModelKind, StrategyProbe
from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
from evalvitals.eval_agent.stages.protocol import ExperimentProtocol, ProbingSchema
from evalvitals.eval_agent.stages.stats_agent import StatsAnalysisAgent, StatsAnalysisReport
from evalvitals.eval_agent.stages.surgery import InterventionResult, SurgeryAgent
from evalvitals.eval_agent.store import InMemoryStore, JsonlStore, Store

__all__ = [
    # M1
    "ProbeAgent",
    "StrategyProbe",
    "ModelKind",
    # M2
    "AnalysisModule",
    "AnalysisReport",
    # M3
    "DiagnosisAgent",
    "DiagnosisResult",
    # M4
    "SurgeryAgent",
    "InterventionResult",
    # Loop
    "AutoDiagnoseLoop",
    "AutoDiagnoseReport",
    "SelfEvolveLoop",
    "VLDiagnoseLoop",
    "VLDiagnoseReport",
    # Protocol
    "ExperimentProtocol",
    "ProbingSchema",
    # M2 stats agent
    "StatsAnalysisAgent",
    "StatsAnalysisReport",
    # M5 hypothesis tester
    "HypothesisTester",
    "HypothesisTestResult",
    # Shared
    "EvalOrchestrator",
    "Hypothesis",
    "HypothesisGenerator",
    "ManualHypothesisGenerator",
    "HypothesisStatus",
    "hypothesis_to_dict",
    "hypothesis_from_dict",
    "Store",
    "InMemoryStore",
    "JsonlStore",
    "ABRunner",
    "ABResult",
    "DataSplit",
    "Split",
    "PreregisteredHypothesis",
    "PreregistrationLog",
    "DiagnosticReport",
    "RunLogger",
    # Experiment execution
    "ExperimentWriter",
    "ExperimentWriterConfig",
    "ExperimentWriterResult",
    "SolutionNode",
    "build_model_context",
    "ExperimentSandbox",
    "SandboxResult",
    "SandboxProtocol",
    "parse_metrics",
    "validate_entry_point",
    "validate_entry_point_resolved",
    # Sandbox factory
    "SandboxConfig",
    "SandboxFactoryConfig",
    "create_sandbox",
    # Git versioning
    "ExperimentGitManager",
    # Evolution store
    "EvolutionStore",
    "LessonEntry",
    "extract_lessons",
    # CLI agents
    "CliAgentConfig",
    "CliAgentResult",
    "create_cli_agent",
]
