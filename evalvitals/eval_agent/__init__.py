"""Self-evolving evaluation agent — automated failure discovery and diagnosis.

Layout:
  probe_agent.py       M1 — ProbeAgent: select + execute analyzers (direct or Docker)
  analysis.py          M2 — AnalysisModule: interpret results → AnalysisReport
  diagnosis.py         M3 — DiagnosisAgent: Gemini reads report → hypotheses
  surgery.py           M4 — SurgeryAgent: operate + verify; correlate or sweep
  loop.py              AutoDiagnoseLoop (M1→M4) + SelfEvolveLoop (original skeleton)
  probe.py             StrategyProbe (tool-selection component used by ProbeAgent)
  hypothesis.py        Hypothesis + HypothesisGenerator + serialization helpers
  store.py             persistent memory (Store / InMemoryStore / JsonlStore)
  orchestrator.py      thin facade over the loop (pre-registered A/B)
  ab_runner.py         A/B execution across prompting strategies
  report.py            diagnostic conclusions
  git_manager.py       git-native experiment versioning (eval/{run_id} branches)
  evolution.py         JSONL lesson store with 30-day half-life time decay
  factory.py           sandbox factory (subprocess / docker backends)
  experiment_harness.py immutable evaluation harness injected into projects
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
