"""Self-evolving evaluation agent (interfaces + stubs for Stage 2).

Layout:
  tools.py        the agent's action space over the package
  hypothesis.py   Hypothesis + HypothesisGenerator
  store.py        persistent memory (Store / InMemoryStore)
  loop.py         SelfEvolveLoop — the closed-loop controller
  orchestrator.py thin facade over the loop
  ab_runner.py    A/B execution across prompting strategies
  report.py       diagnostic conclusions
"""

from evalvitals.eval_agent.ab_runner import ABResult, ABRunner
from evalvitals.eval_agent.hypothesis import (
    Hypothesis,
    HypothesisGenerator,
    HypothesisStatus,
    ManualHypothesisGenerator,
)
from evalvitals.eval_agent.loop import SelfEvolveLoop
from evalvitals.eval_agent.orchestrator import EvalOrchestrator
from evalvitals.eval_agent.preregister import (
    DataSplit,
    PreregisteredHypothesis,
    PreregistrationLog,
    Split,
)
from evalvitals.eval_agent.report import DiagnosticReport
from evalvitals.eval_agent.store import InMemoryStore, Store

__all__ = [
    "SelfEvolveLoop",
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
