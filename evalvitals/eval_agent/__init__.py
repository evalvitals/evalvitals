"""Self-evolving evaluation agent — automated failure discovery and diagnosis.

Layout:
  probe_agent.py  M1 — ProbeAgent: select + execute analyzers (direct or Docker)
  analysis.py     M2 — AnalysisModule: interpret results → AnalysisReport
  diagnosis.py    M3 — DiagnosisAgent: Gemini reads report → hypotheses
  surgery.py      M4 — SurgeryAgent: operate + verify; correlate or sweep
  loop.py         AutoDiagnoseLoop (M1→M4) + SelfEvolveLoop (original skeleton)
  probe.py        StrategyProbe (tool-selection component used by ProbeAgent)
  hypothesis.py   Hypothesis + HypothesisGenerator
  store.py        persistent memory (Store / InMemoryStore)
  orchestrator.py thin facade over the loop (pre-registered A/B)
  ab_runner.py    A/B execution across prompting strategies
  report.py       diagnostic conclusions
"""

from evalvitals.eval_agent.ab_runner import ABResult, ABRunner
from evalvitals.eval_agent.analysis import AnalysisModule, AnalysisReport
from evalvitals.eval_agent.cli_agent import CliAgentConfig, CliAgentResult, create_cli_agent
from evalvitals.eval_agent.diagnosis import DiagnosisAgent, DiagnosisResult
from evalvitals.eval_agent.experiment_writer import (
    ExperimentWriter,
    ExperimentWriterConfig,
    ExperimentWriterResult,
    build_model_context,
)
from evalvitals.eval_agent.hypothesis import (
    Hypothesis,
    HypothesisGenerator,
    HypothesisStatus,
    ManualHypothesisGenerator,
)
from evalvitals.eval_agent.loop import AutoDiagnoseLoop, AutoDiagnoseReport, SelfEvolveLoop
from evalvitals.eval_agent.orchestrator import EvalOrchestrator
from evalvitals.eval_agent.preregister import (
    DataSplit,
    PreregisteredHypothesis,
    PreregistrationLog,
    Split,
)
from evalvitals.eval_agent.probe import ModelKind, StrategyProbe
from evalvitals.eval_agent.probe_agent import ProbeAgent
from evalvitals.eval_agent.report import DiagnosticReport
from evalvitals.eval_agent.run_logger import RunLogger
from evalvitals.eval_agent.sandbox import ExperimentSandbox, SandboxResult, parse_metrics
from evalvitals.eval_agent.store import InMemoryStore, Store
from evalvitals.eval_agent.surgery import InterventionResult, SurgeryAgent

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
    # Shared
    "EvalOrchestrator",
    "Hypothesis",
    "HypothesisGenerator",
    "ManualHypothesisGenerator",
    "HypothesisStatus",
    "Store",
    "InMemoryStore",
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
    "build_model_context",
    "ExperimentSandbox",
    "SandboxResult",
    "parse_metrics",
    # CLI agents
    "CliAgentConfig",
    "CliAgentResult",
    "create_cli_agent",
]
