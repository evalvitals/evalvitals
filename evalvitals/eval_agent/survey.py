"""M4 — SurveyAgent: verify hypotheses via targeted intervention.

Given a hypothesis from the diagnosis agent, the survey agent runs an
intervention to check whether the hypothesized cause actually predicts the
observed failures.  Three strategies are available (in priority order):

1. **verify_fn** (injected) — caller supplies the full verification logic.
2. **analyzer_params** (param sweep) — re-run named analyzers with modified
   parameters and surface before/after findings for comparison.
3. **Default** (label correlation) — look for per-case signals already present
   in the analysis results (e.g. ``has_loop``, ``n_ignored``), split cases into
   "signal" vs "no-signal" groups, and compare FAIL rates using a 10 % gap
   threshold.  Cases in the signal group can be filtered out to produce
   ``new_data`` for the next M1 cycle.

Usage::

    agent = SurveyAgent()
    iv = agent.survey(hypothesis, model, results, data)
    print(iv.status, iv.fixed, iv.evidence)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from evalvitals.core.case import CaseBatch, Label
from evalvitals.eval_agent.hypothesis import Hypothesis, HypothesisStatus

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result


@dataclass
class InterventionResult:
    """Output of :class:`SurveyAgent.survey`.

    Attributes:
        hypothesis: The hypothesis under test.
        status:     Updated :class:`~evalvitals.eval_agent.hypothesis.HypothesisStatus`.
        fixed:      ``True`` when the intervention completely separates failing from
                    passing cases (signal group always fails, control group never
                    fails).  Signals the loop that the problem is resolved.
        evidence:   Supporting statistics or sweep findings.
        new_data:   When *status* is SUPPORTED, the cases **not** in the signal
                    group — the refined subset for the next M1 cycle.
    """

    hypothesis: Hypothesis
    status: HypothesisStatus
    fixed: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    new_data: CaseBatch | None = None


# Keys that carry diagnostic meaning in per-case finding entries.
# Excluded from the signal scan because they are indices, not boolean flags.
_NON_SIGNAL_KEYS = frozenset(
    {"sample_id", "id", "step", "first_error_step", "action", "judge_raw"}
)


def _extract_per_case_signals(results: dict[str, "Result"]) -> dict[str, bool]:
    """Collect a per-case boolean signal from all per_case finding entries.

    A case is marked as having a signal if any finding entry for it contains a
    truthy numeric or boolean value (e.g. ``has_loop=True``, ``n_ignored=2``).
    """
    signal: dict[str, bool] = {}
    for result in results.values():
        for entry in result.findings.get("per_case", []):
            cid = entry.get("sample_id") or entry.get("id", "")
            if not cid:
                continue
            hit = any(
                isinstance(v, (int, float, bool)) and bool(v)
                for k, v in entry.items()
                if k not in _NON_SIGNAL_KEYS
            )
            signal[cid] = signal.get(cid, False) or hit
    return signal


class SurveyAgent:
    """Verify hypotheses through targeted intervention.

    Args:
        verify_fn:      Optional callable ``(hypothesis, model, results, data)
                        -> InterventionResult`` that fully overrides the default
                        logic.  Use when you need domain-specific verification.
        analyzer_params: ``{analyzer_name: {param: value}}`` dict triggering the
                        param-sweep path.  Runs each named analyzer with the
                        given params and surfaces before/after findings.
    """

    def __init__(
        self,
        verify_fn: Callable[
            [Hypothesis, "Model", dict[str, "Result"], CaseBatch],
            InterventionResult,
        ]
        | None = None,
        analyzer_params: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.verify_fn = verify_fn
        self.analyzer_params = analyzer_params or {}

    def survey(
        self,
        hypothesis: Hypothesis,
        model: "Model",
        results: dict[str, "Result"],
        data: CaseBatch,
    ) -> InterventionResult:
        """Run the appropriate intervention for *hypothesis*.

        Strategy selection (first match wins):

        1. ``verify_fn`` injected → delegate entirely.
        2. ``analyzer_params`` provided → param sweep.
        3. Default → label-correlation analysis.
        """
        if self.verify_fn is not None:
            return self.verify_fn(hypothesis, model, results, data)
        if self.analyzer_params:
            return self._param_sweep(hypothesis, model, data)
        return self._correlate_with_labels(hypothesis, results, data)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _correlate_with_labels(
        self,
        hypothesis: Hypothesis,
        results: dict[str, "Result"],
        data: CaseBatch,
    ) -> InterventionResult:
        """Default: correlate per-case signals with PASS/FAIL labels."""
        signal = _extract_per_case_signals(results)

        if not signal:
            return InterventionResult(
                hypothesis=hypothesis,
                status=HypothesisStatus.INCONCLUSIVE,
                fixed=False,
                evidence={"reason": "no per-case findings available to correlate"},
            )

        # Index by both the case UUID and the trajectory sample_id so that
        # per-case entries (which use sample_id) can be matched.
        labeled: dict[str, bool] = {}
        for c in data:
            if getattr(c, "label", None) is None:
                continue
            is_fail = c.label == Label.FAIL
            labeled[c.id] = is_fail
            traj = getattr(c, "trajectory", None)
            if traj is not None:
                labeled[getattr(traj, "sample_id", "")] = is_fail

        if not labeled:
            return InterventionResult(
                hypothesis=hypothesis,
                status=HypothesisStatus.INCONCLUSIVE,
                fixed=False,
                evidence={"reason": "no labeled cases to correlate with"},
            )

        with_signal    = [labeled[cid] for cid, hit in signal.items() if hit     and cid in labeled]
        without_signal = [labeled[cid] for cid, hit in signal.items() if not hit and cid in labeled]

        if not with_signal:
            return InterventionResult(
                hypothesis=hypothesis,
                status=HypothesisStatus.REFUTED,
                fixed=False,
                evidence={"reason": "no cases match the hypothesis signal"},
            )

        fail_signal  = sum(with_signal)    / len(with_signal)
        fail_control = sum(without_signal) / len(without_signal) if without_signal else 0.0

        evidence = {
            "n_with_signal":    len(with_signal),
            "n_without_signal": len(without_signal),
            "fail_rate_signal":  round(fail_signal, 3),
            "fail_rate_control": round(fail_control, 3),
        }

        if fail_signal > fail_control + 0.10:
            status = HypothesisStatus.SUPPORTED
            fixed  = fail_signal >= 1.0 and fail_control == 0.0
        elif fail_signal < fail_control - 0.05:
            status = HypothesisStatus.REFUTED
            fixed  = False
        else:
            status = HypothesisStatus.INCONCLUSIVE
            fixed  = False

        new_data: CaseBatch | None = None
        if status == HypothesisStatus.SUPPORTED:
            new_data = CaseBatch([c for c in data if not signal.get(c.id, False)])

        return InterventionResult(
            hypothesis=hypothesis,
            status=status,
            fixed=fixed,
            evidence=evidence,
            new_data=new_data,
        )

    def _param_sweep(
        self,
        hypothesis: Hypothesis,
        model: "Model",
        data: CaseBatch,
    ) -> InterventionResult:
        """Re-run specified analyzers with modified parameters."""
        from evalvitals.eval_agent.tools import run_analysis

        sweep: dict[str, Any] = {}
        for analyzer_name, params in self.analyzer_params.items():
            try:
                result = run_analysis(model, analyzer_name, data, **params)
                sweep[analyzer_name] = result.findings
            except Exception as exc:
                sweep[analyzer_name] = {"error": str(exc)}

        return InterventionResult(
            hypothesis=hypothesis,
            status=HypothesisStatus.INCONCLUSIVE,
            fixed=False,
            evidence={"param_sweep": sweep},
        )
