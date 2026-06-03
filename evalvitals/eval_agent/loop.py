"""SelfEvolveLoop and AutoDiagnoseLoop — closed-loop failure-discovery controllers.

``SelfEvolveLoop`` is the original Stage-1 skeleton (kept for backward compat).
``AutoDiagnoseLoop`` is the concrete M1→M2→M3→M4 implementation:

    ┌───────────────────────────────────────────────────────────────────┐
    │ M1 · ProbeAgent      select analyzers + execute (direct / Docker) │
    │ M2 · AnalysisModule  interpret results → structured report        │
    │ M3 · DiagnosisAgent  Gemini reads report → hypotheses             │
    │ M4 · SurgeryAgent    operate + verify; stop or refocus data       │
    └───────────────────────────────────────────────────────────────────┘
                            ↑__________________________│  (repeat)

Usage::

    loop   = AutoDiagnoseLoop(model=my_model)   # Gemini default for M3
    report = loop.run(failure_cases)
    print(report.resolved, report.final_hypotheses)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.eval_agent.store import InMemoryStore, Store

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.analysis import AnalysisModule, AnalysisReport
    from evalvitals.eval_agent.diagnosis import DiagnosisAgent
    from evalvitals.eval_agent.hypothesis import Hypothesis, HypothesisGenerator
    from evalvitals.eval_agent.probe_agent import ProbeAgent
    from evalvitals.eval_agent.run_logger import RunLogger
    from evalvitals.eval_agent.surgery import SurgeryAgent


class SelfEvolveLoop:
    """Original Stage-1 skeleton — kept for backward compatibility.

    Args:
        generator: Proposes and mutates hypotheses.
        store:     Persistent memory; defaults to an in-memory store.
    """

    def __init__(
        self,
        generator: "HypothesisGenerator | None" = None,
        store: Store | None = None,
        runner: Any = None,  # kept for signature compat
    ) -> None:
        from evalvitals.core.experiment import ExperimentRunner

        self.generator = generator
        self.store = store or InMemoryStore()
        self.runner = runner if runner is not None else ExperimentRunner()

    def step(self) -> list:
        if self.generator is None:
            raise ValueError("SelfEvolveLoop needs a generator (e.g. ManualHypothesisGenerator).")
        proposed = self.generator.propose(self.store.summarize())
        for h in proposed:
            self.store.add_hypothesis(h)
        return list(proposed)

    def run(self, max_cycles: int = 10) -> list:
        history: list = []
        for _ in range(max_cycles):
            proposed = self.step()
            history.append(proposed)
            if not proposed:
                break
        return history


# ──────────────────────────────────────────────────────────────────────────────
# AutoDiagnoseLoop — M1 → M2 → M3 → M4
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AutoDiagnoseReport:
    """Summary returned by :class:`AutoDiagnoseLoop.run`.

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


class AutoDiagnoseLoop:
    """Automated M1→M2→M3→M4 diagnosis loop.

    Args:
        model:            The model under evaluation.
        probe_agent:      M1 — selects and executes analyzers (direct or Docker).
                          Defaults to ``ProbeAgent()``.
        analysis_module:  M2 — interprets raw Results into an AnalysisReport.
                          Defaults to ``AnalysisModule()``.
        diagnosis_agent:  M3 — reads the AnalysisReport and proposes hypotheses via
                          Gemini (default when ``GEMINI_API_KEY`` is set).
                          Pass ``None`` to run in *analysis-only* mode (M1+M2 only).
        surgery_agent:    M4 — operates on each hypothesis to verify or refute it.
                          Defaults to ``SurgeryAgent()``.
        store:            Persistent memory.  Defaults to ``InMemoryStore()``.
        max_cycles:       Hard cap on M1→M4 iterations.
        run_logger:       Optional :class:`~evalvitals.eval_agent.run_logger.RunLogger`
                          that writes a JSONL event log and saves analyzer artifacts
                          (tensors, arrays) to disk after each cycle.
    """

    def __init__(
        self,
        model: "Model",
        probe_agent: "ProbeAgent | None" = None,
        analysis_module: "AnalysisModule | None" = None,
        diagnosis_agent: "DiagnosisAgent | None" = None,
        surgery_agent: "SurgeryAgent | None" = None,
        store: Store | None = None,
        max_cycles: int = 5,
        run_logger: "RunLogger | None" = None,
    ) -> None:
        from evalvitals.eval_agent.analysis import AnalysisModule
        from evalvitals.eval_agent.probe_agent import ProbeAgent
        from evalvitals.eval_agent.surgery import SurgeryAgent

        self.model = model
        self.probe_agent = probe_agent or ProbeAgent()
        self.analysis_module = analysis_module or AnalysisModule()
        self.diagnosis_agent = diagnosis_agent  # None = analysis-only mode
        self.surgery_agent = surgery_agent or SurgeryAgent()
        self.store = store or InMemoryStore()
        self.max_cycles = max_cycles
        self.run_logger = run_logger

    def run(self, data: "CaseBatch") -> AutoDiagnoseReport:
        """Drive the M1→M2→M3→M4 loop until resolved or *max_cycles* reached.

        Args:
            data: Cases to analyse.  Refocused to unexplained cases when M4
                  returns a SUPPORTED finding with ``new_data``.

        Returns:
            :class:`AutoDiagnoseReport` with the final state.
        """
        all_hypotheses: list[Any] = []
        final_results: dict[str, Any] = {}
        final_analysis = None

        for cycle in range(self.max_cycles):
            # ── M1: probe — select analyzers and execute them ──────────────
            probe_results = self.probe_agent.probe(self.model, data)
            if not probe_results:
                break
            final_results = probe_results
            for r in probe_results.values():
                self.store.add_result(r)
            if self.run_logger:
                self.run_logger.log_probe(cycle, probe_results)

            # ── M2: analyze — interpret raw results into a report ──────────
            analysis = self.analysis_module.analyze(probe_results, repr(self.model))
            final_analysis = analysis
            if self.run_logger:
                self.run_logger.log_analysis(cycle, analysis)

            if self.diagnosis_agent is None:
                break  # analysis-only mode: stop after first M1+M2 pass

            # ── M3: diagnose — LLM proposes hypotheses from the report ─────
            diag = self.diagnosis_agent.diagnose(analysis)
            if self.run_logger:
                self.run_logger.log_diagnosis(cycle, diag)
            if not diag.hypotheses:
                break
            for h in diag.hypotheses:
                self.store.add_hypothesis(h)
            all_hypotheses.extend(diag.hypotheses)

            # ── M4: surgery — intervene and verify each hypothesis ─────────
            for h in diag.hypotheses:
                iv = self.surgery_agent.operate(h, self.model, probe_results, data)
                h.status = iv.status
                if self.run_logger:
                    self.run_logger.log_surgery(cycle, h, iv)
                if iv.fixed:
                    report = AutoDiagnoseReport(
                        cycles=cycle + 1,
                        resolved=True,
                        final_hypotheses=all_hypotheses,
                        final_results=final_results,
                        final_analysis=final_analysis,
                        store=self.store,
                    )
                    if self.run_logger:
                        self.run_logger.log_loop_end(report)
                    return report
                if iv.new_data is not None and len(iv.new_data) > 0:
                    data = iv.new_data  # refocus on unexplained cases

        report = AutoDiagnoseReport(
            cycles=self.max_cycles,
            resolved=False,
            final_hypotheses=all_hypotheses,
            final_results=final_results,
            final_analysis=final_analysis,
            store=self.store,
        )
        if self.run_logger:
            self.run_logger.log_loop_end(report)
        return report
