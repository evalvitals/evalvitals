"""Report dataclasses returned by the diagnosis loops.

Split out of ``loop.py`` so the loop/legacy modules can share them without a
circular import (``legacy.AutoDiagnoseLoop`` and ``loop.VLDiagnoseLoop`` both
import from here; neither imports the other).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.store import InMemoryStore, Store

if TYPE_CHECKING:
    from evalvitals.analysis.analysis_module import AnalysisReport
    from evalvitals.analysis.stats_agent import StatsAnalysisReport
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.hypothesis import Hypothesis
    from evalvitals.eval_agent.stages.hypothesis_tester import HypothesisTestResult


@dataclass
class AutoDiagnoseReport:
    """Summary returned by :class:`~evalvitals.eval_agent.legacy.AutoDiagnoseLoop.run`.

    Attributes:
        cycles:           Number of M1→M4 cycles executed.
        resolved:         ``True`` when M4 surgery confirmed the problem is fixed.
        final_hypotheses: All hypotheses proposed across every cycle.
        final_results:    Raw analyzer results from the last M1 probe.
        final_analysis:   Structured report from the last M2 pass.
        store:            Accumulated results and hypotheses.
    """

    cycles: int
    resolved: bool
    final_hypotheses: list["Hypothesis"] = field(default_factory=list)
    final_results: dict[str, "Result"] = field(default_factory=dict)
    final_analysis: "AnalysisReport | None" = None
    store: Store = field(default_factory=InMemoryStore)
    # Internal — set by AutoDiagnoseLoop for evolution/git integration
    _run_id: str = field(default="", repr=False)


@dataclass
class VLDiagnoseReport:
    """Summary returned by :class:`~evalvitals.eval_agent.loop.VLDiagnoseLoop.run`
    (also the report shape :class:`~evalvitals.eval_agent.agentic.AgenticDiagnoseLoop.run`
    returns).

    Attributes:
        cycles:               Number of M1→M5 cycles executed (or agentic
                              decision steps taken).
        stopped_by:           Why the loop stopped: ``"criteria_met"``,
                              ``"max_cycles"``, ``"budget"``,
                              ``"no_hypotheses"``, ``"no_probe_results"``,
                              ``"analysis_complete"`` (from ``run_analysis``,
                              which proposes hypotheses without confirming
                              them) — or, for the agentic loop, ``"agent_stop"``
                              / ``"max_actions"`` / ``"time_budget"`` /
                              ``"invalid_actions"``.
        verified_hypotheses:  Statistically supported, protocol-consistent
                              test results from M5 — sorted highest confidence
                              first.  Feed into ``run_m4``.
        all_hypotheses:       All M3 proposals across every cycle.
        all_test_results:     All M5 test results across every cycle.
        final_stats_report:   M2 report from the last cycle.
        fix_proposal:         Populated by ``run_m4`` when called after ``run``.
        fix_outcome:          Populated by ``run_fix`` — tiered fix attempts +
                              escalation recommendation.
        store:                Accumulated results and hypotheses.
    """

    cycles: int
    stopped_by: str
    verified_hypotheses: "list[HypothesisTestResult]" = field(default_factory=list)
    all_hypotheses: "list[Any]" = field(default_factory=list)
    all_test_results: "list[HypothesisTestResult]" = field(default_factory=list)
    final_stats_report: "StatsAnalysisReport | None" = None
    fix_proposal: "Any | None" = None
    fix_outcome: "Any | None" = None
    store: Store = field(default_factory=InMemoryStore)
    _run_id: str = field(default="", repr=False)
