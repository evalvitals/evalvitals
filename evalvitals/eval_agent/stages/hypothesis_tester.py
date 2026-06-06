"""M5 — HypothesisTester: verify hypotheses via statistical tests + protocol consistency.

M5 is the *gatekeeper* between M3 (hypothesis generation) and the stopping
decision.  It asks two questions for each hypothesis:

1. **Statistical support** — do cases that exhibit the hypothesised signal fail
   at a higher rate than cases that do not?  Uses the same per-case signal
   extraction as M4 (:func:`~evalvitals.eval_agent.surgery._extract_per_case_signals`),
   but reports a richer picture: effect size, confidence, and whether the
   result clears a minimum-effect threshold.

2. **Protocol consistency** — is the hypothesis consistent with what the user
   described in their experiment protocol?  Uses keyword-based heuristics by
   default; an optional LLM ``judge=`` runs a critic call for a richer check.

**Stopping criteria** (Plan A from the 2026-06-05 meeting):
:meth:`stopping_criteria_met` returns ``True`` when at least one hypothesis is
*both* statistically supported *and* protocol-consistent.  The loop calls this
after each M5 pass and breaks when it returns ``True``.

Usage::

    tester = HypothesisTester()
    results = tester.test(hypotheses, stats_report, data, protocol=protocol)

    if tester.stopping_criteria_met(results, protocol):
        best = tester.best_hypotheses(results)
        # pass best[0].hypothesis to M4

    # With LLM judge for richer protocol consistency:
    tester = HypothesisTester(judge=gemini)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalvitals.core.case import CaseBatch, Label
from evalvitals.eval_agent.hypothesis import Hypothesis, HypothesisStatus
from evalvitals.eval_agent.stages.surgery import _compute_confidence, _extract_per_case_signals

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol
    from evalvitals.eval_agent.stages.stats_agent import StatsAnalysisReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template (LLM consistency check)
# ---------------------------------------------------------------------------

_CONSISTENCY_PROMPT = """\
Experiment protocol:
{protocol_text}

Hypothesis under review:
Statement: {statement}
Predicted failure mode: {failure_mode}

Does this hypothesis address a failure mode that is relevant to the experiment protocol above?
Answer YES or NO on the first line, then give a one-sentence reason.

Answer YES if the hypothesis explains a failure that the protocol would care about.
Answer NO if the hypothesis is about something the protocol does not mention or test."""


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class HypothesisTestResult:
    """Result of M5 testing one :class:`~evalvitals.eval_agent.hypothesis.Hypothesis`.

    Attributes:
        hypothesis:                  The hypothesis under test.
        status:                      Statistical verdict
                                     (:class:`~evalvitals.eval_agent.hypothesis.HypothesisStatus`).
        test_name:                   Which test was used (e.g. ``"fail_rate_comparison"``).
        effect_size:                 Observed fail-rate difference
                                     (signal group minus control group).
        is_consistent_with_protocol: ``True`` when the hypothesis is relevant
                                     to the user's experiment protocol.
        confidence:                  Combined score in [0, 1] — geometric mean
                                     of evidence gap, sample adequacy, and
                                     control cleanliness.
        verdict:                     NL one-liner for human consumption.
        evidence:                    Supporting statistics and group sizes.
    """

    hypothesis: Hypothesis
    status: HypothesisStatus
    test_name: str
    effect_size: float | None
    is_consistent_with_protocol: bool
    confidence: float
    verdict: str
    evidence: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------

class HypothesisTester:
    """M5: test hypotheses using statistical methods and protocol consistency.

    Args:
        judge:       Optional LLM for protocol consistency checks.
                     Any :class:`~evalvitals.core.model.Model` with
                     ``Capability.GENERATE``.  When ``None``, uses a
                     keyword-based heuristic instead.
        alpha:       Statistical significance level (unused in default
                     fail-rate comparison; reserved for future integration
                     with :func:`~evalvitals.stats.compare`).
        min_effect:  Minimum fail-rate difference to consider a finding
                     meaningful (default 0.10 = 10 pp).
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        alpha: float = 0.05,
        min_effect: float = 0.10,
    ) -> None:
        self._judge = judge
        self.alpha = alpha
        self.min_effect = min_effect

    def test(
        self,
        hypotheses: list[Hypothesis],
        stats_report: "StatsAnalysisReport",
        data: CaseBatch,
        protocol: "ExperimentProtocol | None" = None,
    ) -> list[HypothesisTestResult]:
        """Test each hypothesis against *data* and *protocol*.

        Args:
            hypotheses:   Hypotheses produced by M3 (DiagnosisAgent).
            stats_report: M2's report — provides ``raw_results`` for
                          per-case signal extraction.
            data:         Cases to test against (must carry labels for
                          fail-rate comparison).
            protocol:     Experiment protocol for consistency checks.
                          ``None`` → all hypotheses are assumed consistent.

        Returns:
            One :class:`HypothesisTestResult` per hypothesis, in the
            same order as *hypotheses*.
        """
        results: list[HypothesisTestResult] = []
        for h in hypotheses:
            result = self._test_one(h, stats_report, data, protocol)
            results.append(result)
        return results

    def stopping_criteria_met(
        self,
        test_results: list[HypothesisTestResult],
        protocol: "ExperimentProtocol | None" = None,
    ) -> bool:
        """Return ``True`` when at least one verified, protocol-consistent hypothesis exists.

        This is the **Plan A stopping criterion** from the 2026-06-05
        architecture meeting: the loop should stop once we have a
        statistically supported hypothesis that addresses what the
        user's protocol was testing — further cycling would only
        rediscover the same root cause.
        """
        return any(
            r.status == HypothesisStatus.SUPPORTED and r.is_consistent_with_protocol
            for r in test_results
        )

    def best_hypotheses(
        self,
        test_results: list[HypothesisTestResult],
    ) -> list[HypothesisTestResult]:
        """Return verified, protocol-consistent hypotheses sorted by confidence.

        The first element is the highest-confidence candidate to hand to M4.
        """
        supported = [
            r for r in test_results
            if r.status == HypothesisStatus.SUPPORTED and r.is_consistent_with_protocol
        ]
        return sorted(supported, key=lambda r: r.confidence, reverse=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _test_one(
        self,
        hypothesis: Hypothesis,
        stats_report: "StatsAnalysisReport",
        data: CaseBatch,
        protocol: "ExperimentProtocol | None",
    ) -> HypothesisTestResult:
        """Test a single hypothesis."""
        # ── Statistical test ──────────────────────────────────────────
        signal = _extract_per_case_signals(stats_report.raw_results)

        # Split cases into signal / no-signal groups with fail indicators.
        fail_signal: list[int] = []
        fail_control: list[int] = []
        for case in data:
            if getattr(case, "label", None) is None:
                continue
            is_fail = int(case.label == Label.FAIL)
            if signal.get(case.id, False):
                fail_signal.append(is_fail)
            else:
                fail_control.append(is_fail)

        if not fail_signal or not fail_control:
            status = HypothesisStatus.INCONCLUSIVE
            effect_size = None
            confidence = 0.0
            verdict = "Insufficient labeled data to test statistically."
            evidence: dict[str, Any] = {"reason": verdict}
        else:
            rate_signal = sum(fail_signal) / len(fail_signal)
            rate_control = sum(fail_control) / len(fail_control)
            effect_size = round(rate_signal - rate_control, 4)

            confidence, dims = _compute_confidence(
                rate_signal, rate_control, len(fail_signal), len(fail_control)
            )

            if effect_size > self.min_effect:
                status = HypothesisStatus.SUPPORTED
                verdict = (
                    f"Signal group fails {effect_size:.0%} more than control "
                    f"(confidence={confidence:.2f})."
                )
            elif effect_size < -self.min_effect:
                status = HypothesisStatus.REFUTED
                verdict = (
                    f"Signal group fails less than control "
                    f"(effect={effect_size:.0%}); hypothesis refuted."
                )
            else:
                status = HypothesisStatus.INCONCLUSIVE
                verdict = (
                    f"Effect size {effect_size:.0%} below minimum {self.min_effect:.0%}; "
                    f"inconclusive."
                )

            evidence = {
                "n_signal": len(fail_signal),
                "n_control": len(fail_control),
                "fail_rate_signal": round(rate_signal, 4),
                "fail_rate_control": round(rate_control, 4),
                "effect_size": effect_size,
                "confidence_dims": dims,
            }

        # ── Protocol consistency check ────────────────────────────────
        is_consistent = self._check_protocol_consistency(
            hypothesis, stats_report, protocol
        )
        if not is_consistent and status == HypothesisStatus.SUPPORTED:
            verdict += " (note: not protocol-consistent — excluded from best candidates)"

        return HypothesisTestResult(
            hypothesis=hypothesis,
            status=status,
            test_name="fail_rate_comparison",
            effect_size=effect_size,
            is_consistent_with_protocol=is_consistent,
            confidence=confidence,
            verdict=verdict,
            evidence=evidence,
        )

    def _check_protocol_consistency(
        self,
        hypothesis: Hypothesis,
        stats_report: "StatsAnalysisReport",
        protocol: "ExperimentProtocol | None",
    ) -> bool:
        """Return True when the hypothesis is relevant to the protocol.

        Uses LLM critic when a judge is available; falls back to keyword
        overlap between the hypothesis failure mode and the protocol hints.
        """
        if protocol is None:
            return True

        if self._judge is not None:
            try:
                return self._llm_consistency_check(hypothesis, protocol)
            except Exception as exc:
                logger.debug("LLM consistency check failed, falling back: %s", exc)

        return self._heuristic_consistency_check(hypothesis, protocol)

    def _heuristic_consistency_check(
        self,
        hypothesis: Hypothesis,
        protocol: "ExperimentProtocol",
    ) -> bool:
        """Keyword overlap between protocol hints and hypothesis failure mode."""
        hints = set(protocol.probe_hints())
        if not hints:
            # Protocol has no extractable hints → treat as consistent.
            return True

        mode = hypothesis.predicted_failure_mode.lower().replace(" ", "_")
        statement_lower = hypothesis.statement.lower()

        for hint in hints:
            hint_lower = hint.lower().replace(" ", "_")
            # Direct match or prefix/substring of the failure mode tag.
            if hint_lower in mode or mode.startswith(hint_lower):
                return True
            # Keyword present anywhere in the hypothesis statement.
            if hint_lower.replace("_", " ") in statement_lower:
                return True

        # No overlap found — check protocol text itself against statement.
        proto_text = (protocol.description + " " + protocol.failure_patterns).lower()
        for word in mode.replace("_", " ").split():
            if len(word) >= 4 and word in proto_text:
                return True

        return False

    def _llm_consistency_check(
        self,
        hypothesis: Hypothesis,
        protocol: "ExperimentProtocol",
    ) -> bool:
        """LLM critic: ask judge if the hypothesis is relevant to the protocol."""
        import inspect

        prompt = _CONSISTENCY_PROMPT.format(
            protocol_text=protocol.description,
            statement=hypothesis.statement,
            failure_mode=hypothesis.predicted_failure_mode,
        )
        sig = inspect.signature(self._judge.generate)  # type: ignore[union-attr]
        if "temperature" in sig.parameters:
            raw = self._judge.generate(prompt, temperature=0)  # type: ignore[union-attr]
        else:
            raw = self._judge.generate(prompt)  # type: ignore[union-attr]

        first_line = str(raw).strip().splitlines()[0].upper()
        return first_line.startswith("YES")
