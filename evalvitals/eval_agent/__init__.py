"""Self-evolving evaluation agent — automated failure discovery and diagnosis.

Layout:
  probe.py        M1 — StrategyProbe: select analyzers for a given model kind
  tools.py        M2 — action space: run_analysis, list_analyses, compatible_analyses
  diagnosis.py    M3 — DiagnosisAgent: LLM judge proposes hypotheses from findings
  survey.py       M4 — SurveyAgent: intervene + verify; correlate or sweep
  loop.py         AutoDiagnoseLoop (M1→M4) + SelfEvolveLoop (original skeleton)
  hypothesis.py   Hypothesis + HypothesisGenerator
  store.py        persistent memory (Store / InMemoryStore)
  orchestrator.py thin facade over the loop (pre-registered A/B)
  ab_runner.py    A/B execution across prompting strategies
  report.py       diagnostic conclusions
"""

from evalvitals.eval_agent.ab_runner import ABResult, ABRunner
from evalvitals.eval_agent.diagnosis import DiagnosisAgent, DiagnosisResult
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
from evalvitals.eval_agent.report import DiagnosticReport
from evalvitals.eval_agent.store import InMemoryStore, Store
from evalvitals.eval_agent.survey import InterventionResult, SurveyAgent

__all__ = [
    # M1
    "StrategyProbe",
    "ModelKind",
    # M3
    "DiagnosisAgent",
    "DiagnosisResult",
    # M4
    "SurveyAgent",
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
]
