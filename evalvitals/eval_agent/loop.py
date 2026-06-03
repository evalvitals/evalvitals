"""SelfEvolveLoop and AutoDiagnoseLoop — closed-loop failure-discovery controllers.

``SelfEvolveLoop`` is the original Stage-1 skeleton (kept for backward compat).
``AutoDiagnoseLoop`` is the concrete M1→M2→M3→M4 implementation:

    ┌──────────────────────────────────────────────────────────────────┐
    │ M1 · StrategyProbe   select analyzers for this model kind        │
    │ M2 · Execution       run analyzers via ExperimentRunner          │
    │ M3 · DiagnosisAgent  LLM judge proposes hypotheses               │
    │ M4 · SurveyAgent     intervene + verify; stop or refocus data    │
    └──────────────────────────────────────────────────────────────────┘
                            ↑_________________________│  (repeat)

Usage::

    loop   = AutoDiagnoseLoop(model=my_model, diagnosis_agent=DiagnosisAgent(judge))
    report = loop.run(data)
    print(report.resolved, report.final_hypotheses)

``SelfEvolveLoop`` is the original Stage-1 skeleton (kept for backward compat).

This is the top-level orchestrator the whole package is designed to serve. It
ties the pieces together into one cycle:

    ┌─────────────────────────────────────────────────────────────┐
    │  1. HYPOTHESIZE   generator.propose(store.summarize())        │
    │  2. CONSTRUCT     tools.make_cases(hypothesis) -> CaseBatch    │
    │  3. EXPERIMENT    Experiment(model, analyzer, cases)           │
    │  4. RUN           ExperimentRunner.run(exp) -> Result          │
    │  5. ATTRIBUTE     read result.findings; localise the failure   │
    │  6. EVALUATE      stats over results (significance)            │
    │  7. RECORD        store.add_result / add_case / add_hypothesis │
    │  8. MUTATE        generator.mutate(hypothesis, feedback)       │
    └─────────────────────────────────────────────────────────────┘
                         ↑___________________________│  (repeat)

"Self-evolving" = step 8 feeding step 1: each cycle's findings reshape the next
cycle's hypotheses, and the store accumulates so the agent builds on itself.

Stage 1 ships this as a documented skeleton: the cycle and wiring points are
fixed, the intelligence (hypothesis generation, attribution, case synthesis)
arrives in Stage 2.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.core.experiment import Experiment, ExperimentRunner
from evalvitals.core.registry import registry
from evalvitals.eval_agent.store import InMemoryStore, Store

if TYPE_CHECKING:
    from evalvitals.core.analyzer import Analyzer
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.diagnosis import DiagnosisAgent
    from evalvitals.eval_agent.hypothesis import Hypothesis, HypothesisGenerator
    from evalvitals.eval_agent.probe import StrategyProbe
    from evalvitals.eval_agent.survey import SurveyAgent


class SelfEvolveLoop:
    """Controller that drives the hypothesize→run→attribute→record→mutate cycle.

    Args:
        generator: Proposes and mutates hypotheses (Stage-2 LLM-backed).
        store:     Persistent memory; defaults to an in-memory store.
        runner:    Executes experiments; defaults to a fresh runner.
    """

    def __init__(
        self,
        generator: "HypothesisGenerator | None" = None,
        store: Store | None = None,
        runner: ExperimentRunner | None = None,
    ) -> None:
        self.generator = generator
        self.store = store or InMemoryStore()
        self.runner = runner or ExperimentRunner()

    def step(self) -> list:
        """One cycle: propose hypotheses (from the store's summary) and record them.

        Returns the hypotheses proposed this cycle (empty => nothing new to try).
        The test/attribute/mutate intelligence (case synthesis, LLM judging) plugs
        in here in Stage 2; the propose→record→repeat skeleton is functional now.
        """
        if self.generator is None:
            raise ValueError("SelfEvolveLoop needs a generator (e.g. ManualHypothesisGenerator).")
        proposed = self.generator.propose(self.store.summarize())
        for h in proposed:
            self.store.add_hypothesis(h)
        return list(proposed)

    def run(self, max_cycles: int = 10) -> list:
        """Run until the generator stops proposing (converged) or *max_cycles*."""
        history: list = []
        for _ in range(max_cycles):
            proposed = self.step()
            history.append(proposed)
            if not proposed:  # dry — nothing new
                break
        return history


# ──────────────────────────────────────────────────────────────────────────────
# AutoDiagnoseLoop — concrete M1→M2→M3→M4 implementation
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AutoDiagnoseReport:
    """Summary returned by :class:`AutoDiagnoseLoop.run`.

    Attributes:
        cycles:             Number of M1→M4 cycles executed.
        resolved:           ``True`` when a survey confirmed the problem is fixed.
        final_hypotheses:   All hypotheses proposed across every cycle.
        final_results:      Analyzer results from the last cycle.
        store:              The store instance, containing all accumulated results
                            and hypotheses for further inspection.
    """

    cycles: int
    resolved: bool
    final_hypotheses: list["Hypothesis"] = field(default_factory=list)
    final_results: dict[str, "Result"] = field(default_factory=dict)
    store: Store = field(default_factory=InMemoryStore)


class AutoDiagnoseLoop:
    """Automated M1→M2→M3→M4 diagnosis loop.

    Args:
        model:              The model under evaluation.
        probe:              M1 — selects which analyzers to run.  Defaults to a
                            fresh :class:`~evalvitals.eval_agent.probe.StrategyProbe`.
        diagnosis_agent:    M3 — proposes hypotheses from findings.  When ``None``
                            the loop runs in *analysis-only* mode (M1+M2 only).
        survey_agent:       M4 — verifies each hypothesis.  Defaults to a fresh
                            :class:`~evalvitals.eval_agent.survey.SurveyAgent`.
        store:              Persistent memory; defaults to an in-memory store.
        runner:             Executes experiments; defaults to a fresh runner.
        max_cycles:         Hard cap on the number of M1→M4 iterations.
        max_analyzers:      Passed to :meth:`StrategyProbe.select` to limit how
                            many analyzers run per cycle.
        analyzer_overrides: Pre-instantiated analyzers for those that require
                            mandatory constructor arguments (e.g.
                            ``{"counterfactual": CounterfactualReplay(rerun_fn=...)}``).
                            Used by ``_make_analyzer``; analyzers not overridden are
                            instantiated with default args, or skipped with a warning.
    """

    def __init__(
        self,
        model: "Model",
        probe: "StrategyProbe | None" = None,
        diagnosis_agent: "DiagnosisAgent | None" = None,
        survey_agent: "SurveyAgent | None" = None,
        store: Store | None = None,
        runner: ExperimentRunner | None = None,
        max_cycles: int = 5,
        max_analyzers: int | None = None,
        analyzer_overrides: "dict[str, Analyzer] | None" = None,
    ) -> None:
        from evalvitals.eval_agent.probe import StrategyProbe
        from evalvitals.eval_agent.survey import SurveyAgent

        self.model = model
        self.probe = probe or StrategyProbe()
        self.diagnosis_agent = diagnosis_agent
        self.survey_agent = survey_agent or SurveyAgent()
        self.store = store or InMemoryStore()
        self.runner = runner or ExperimentRunner()
        self.max_cycles = max_cycles
        self.max_analyzers = max_analyzers
        self._overrides: dict[str, "Analyzer"] = analyzer_overrides or {}

    def _make_analyzer(self, name: str) -> "Analyzer | None":
        """Instantiate *name* via override dict or default constructor."""
        if name in self._overrides:
            return self._overrides[name]
        cls = registry.analyzers.get(name)
        try:
            return cls()
        except TypeError as exc:
            warnings.warn(
                f"Skipping analyzer '{name}': cannot instantiate with default args "
                f"({exc}). Pass an instance via analyzer_overrides.",
                stacklevel=3,
            )
            return None

    def run(self, data: "CaseBatch") -> AutoDiagnoseReport:
        """Drive the M1→M2→M3→M4 loop until resolved or *max_cycles* reached.

        Args:
            data: The :class:`~evalvitals.core.case.CaseBatch` to analyse.
                  Refocused to unexplained cases after a SUPPORTED survey step.

        Returns:
            :class:`AutoDiagnoseReport` with the final state.
        """
        all_hypotheses: list[Any] = []
        final_results: dict[str, Any] = {}

        for cycle in range(self.max_cycles):
            # M1: select analyzers
            names = self.probe.select(self.model, max_analyzers=self.max_analyzers)
            if not names:
                break

            # M2: run each analyzer, skip failures gracefully
            cycle_results: dict[str, Any] = {}
            for name in names:
                analyzer = self._make_analyzer(name)
                if analyzer is None:
                    continue
                exp = Experiment(model=self.model, analyzer=analyzer, data=data)
                try:
                    result = self.runner.run(exp)
                    cycle_results[name] = result
                    self.store.add_result(result)
                except Exception as exc:
                    warnings.warn(f"Analyzer '{name}' raised during run: {exc}", stacklevel=2)
            final_results = cycle_results

            if self.diagnosis_agent is None:
                break  # analysis-only mode: stop after first M1+M2 pass

            # M3: propose hypotheses
            diag = self.diagnosis_agent.diagnose(cycle_results, repr(self.model))
            if not diag.hypotheses:
                break

            for h in diag.hypotheses:
                self.store.add_hypothesis(h)
            all_hypotheses.extend(diag.hypotheses)

            # M4: survey each hypothesis; stop on first fix, refocus otherwise
            for h in diag.hypotheses:
                iv = self.survey_agent.survey(h, self.model, cycle_results, data)
                h.status = iv.status
                if iv.fixed:
                    return AutoDiagnoseReport(
                        cycles=cycle + 1,
                        resolved=True,
                        final_hypotheses=all_hypotheses,
                        final_results=final_results,
                        store=self.store,
                    )
                if iv.new_data is not None and len(iv.new_data) > 0:
                    data = iv.new_data  # refocus on unexplained cases

        return AutoDiagnoseReport(
            cycles=min(self.max_cycles, self.max_cycles),
            resolved=False,
            final_hypotheses=all_hypotheses,
            final_results=final_results,
            store=self.store,
        )
