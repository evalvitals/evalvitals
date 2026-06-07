"""Case discovery and labeling for automated VL diagnosis loops.

This stage turns candidate prompts into labeled cases by running the target
model, storing its observed answer, and scoring the result.  It is intentionally
small: dataset loading and prompt generation can live outside this class, while
M5 receives the PASS/FAIL labels it needs for statistical testing.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable

from evalvitals.core.case import CaseBatch, FailureCase, Label

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol


ScoreFn = Callable[[FailureCase, str], Label | bool | str]


_JUDGE_PROMPT = """\
You are scoring one model answer for an evaluation case.

Experiment protocol:
{protocol}

Success criteria:
{success_criteria}

Prompt:
{prompt}

Expected answer or rubric:
{expected}

Observed model answer:
{observed}

Return a JSON object:
{{"label": "PASS|FAIL|UNKNOWN", "reason": "one concise sentence"}}

Use PASS only when the observed answer satisfies the expected answer/rubric under
the success criteria."""


@dataclass
class CaseDiscoveryReport:
    """Summary of one discovery pass."""

    cases: CaseBatch
    n_pass: int = 0
    n_fail: int = 0
    n_unknown: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def has_m5_groups(self) -> bool:
        """True when M5 has both failure and control examples."""
        return self.n_fail > 0 and self.n_pass > 0


class CaseDiscoveryAgent:
    """Run candidate cases through a model and label PASS/FAIL outcomes.

    Args:
        scorer:            Deterministic scoring function.  It receives the
                           case and observed answer and returns ``Label``, bool,
                           or a label string.
        judge:             Optional LLM judge used when no scorer is supplied.
        generation_kwargs: Extra kwargs forwarded to ``model.generate``.
        include_unknown:   Keep UNKNOWN cases in the returned batch.
    """

    def __init__(
        self,
        scorer: ScoreFn | None = None,
        judge: "Model | None" = None,
        generation_kwargs: dict[str, Any] | None = None,
        include_unknown: bool = True,
    ) -> None:
        self.scorer = scorer
        self.judge = judge
        self.generation_kwargs = generation_kwargs or {}
        self.include_unknown = include_unknown

    def discover(
        self,
        model: "Model",
        candidates: Iterable[FailureCase] | CaseBatch,
        *,
        protocol: "ExperimentProtocol | None" = None,
    ) -> CaseDiscoveryReport:
        """Return a labeled batch created from candidate cases."""
        labeled: list[FailureCase] = []
        errors: list[str] = []
        counts = {Label.PASS: 0, Label.FAIL: 0, Label.UNKNOWN: 0}

        for case in candidates:
            try:
                observed = model.generate(case.inputs, **self.generation_kwargs)
                case.observed = observed
                label, reason = self._score_with_reason(case, str(observed), protocol)
            except Exception as exc:  # noqa: BLE001 - discovery should continue
                observed = ""
                label = Label.UNKNOWN
                reason = "model generation or scoring raised an exception"
                errors.append(f"{case.id}: {exc}")
                case.metadata["discovery_error"] = str(exc)

            case.label = label
            case.metadata["discovery_observed"] = str(observed)
            case.metadata["discovery_label"] = label.value
            case.metadata["discovery_reason"] = reason
            counts[label] += 1

            if label != Label.UNKNOWN or self.include_unknown:
                labeled.append(case)

        return CaseDiscoveryReport(
            cases=CaseBatch(labeled),
            n_pass=counts[Label.PASS],
            n_fail=counts[Label.FAIL],
            n_unknown=counts[Label.UNKNOWN],
            errors=errors,
        )

    def _score_with_reason(
        self,
        case: FailureCase,
        observed: str,
        protocol: "ExperimentProtocol | None",
    ) -> tuple[Label, str]:
        if self.scorer is not None:
            label = _coerce_label(self.scorer(case, observed))
            return label, "scored by injected scorer"
        if self.judge is not None:
            judged, reason = self._judge_score(case, observed, protocol)
            if judged != Label.UNKNOWN:
                return judged, reason
            fallback = _heuristic_score(case.expected, observed)
            return fallback, f"judge returned UNKNOWN; heuristic fallback produced {fallback.value}"
        label = _heuristic_score(case.expected, observed)
        return label, "scored by expected-answer heuristic"

    def _judge_score(
        self,
        case: FailureCase,
        observed: str,
        protocol: "ExperimentProtocol | None",
    ) -> tuple[Label, str]:
        protocol_text = protocol.description if protocol is not None else ""
        success_criteria = protocol.success_criteria if protocol is not None else ""
        prompt = _JUDGE_PROMPT.format(
            protocol=protocol_text,
            success_criteria=success_criteria,
            prompt=case.inputs.prompt,
            expected=case.expected,
            observed=observed,
        )
        raw = self.judge.generate(prompt)  # type: ignore[union-attr]
        text = str(raw).strip()
        parsed = _parse_judge_json(text)
        if parsed is not None:
            return parsed
        first_lines = "\n".join(text.splitlines()[:5]).upper()
        if "PASS" in first_lines and "FAIL" not in first_lines:
            return Label.PASS, "judge text contained PASS"
        if "FAIL" in first_lines and "PASS" not in first_lines:
            return Label.FAIL, "judge text contained FAIL"
        first = text.splitlines()[0].strip().upper() if text else ""
        if first.startswith("PASS"):
            return Label.PASS, "judge first line was PASS"
        if first.startswith("FAIL"):
            return Label.FAIL, "judge first line was FAIL"
        return Label.UNKNOWN, "judge output did not contain a parseable label"


def _coerce_label(value: Label | bool | str) -> Label:
    if isinstance(value, Label):
        return value
    if isinstance(value, bool):
        return Label.PASS if value else Label.FAIL
    text = str(value).strip().lower()
    if text in {"pass", "passed", "true", "correct", "ok"}:
        return Label.PASS
    if text in {"fail", "failed", "false", "incorrect", "wrong"}:
        return Label.FAIL
    return Label.UNKNOWN


def _parse_judge_json(text: str) -> tuple[Label, str] | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    label = _coerce_label(payload.get("label", "unknown"))
    reason = str(payload.get("reason", "judge returned structured label"))
    return label, reason


def _heuristic_score(expected: Any, observed: str) -> Label:
    """Score simple expected-answer rubrics without an LLM judge.

    Supported expected formats:
      - string: substring match after normalization
      - list/tuple/set: every item must appear
      - dict: ``all_of``, ``any_of``, and ``none_of`` substring constraints
    """
    if expected is None:
        return Label.UNKNOWN

    obs = _normalize(observed)
    if isinstance(expected, str):
        return Label.PASS if _normalize(expected) in obs else Label.FAIL

    if isinstance(expected, (list, tuple, set)):
        terms = [_normalize(x) for x in expected]
        return Label.PASS if all(t in obs for t in terms if t) else Label.FAIL

    if isinstance(expected, dict):
        all_of = [_normalize(x) for x in expected.get("all_of", [])]
        any_of = [_normalize(x) for x in expected.get("any_of", [])]
        none_of = [_normalize(x) for x in expected.get("none_of", [])]
        if any(t and t in obs for t in none_of):
            return Label.FAIL
        if all_of and not all(t in obs for t in all_of if t):
            return Label.FAIL
        if any_of and not any(t in obs for t in any_of if t):
            return Label.FAIL
        return Label.PASS

    return Label.UNKNOWN


def _normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()
