"""Prompt-contrast — intervention-grade evidence via paired prompt strategies.

Black-box (``GENERATE``): re-asks every case under k prompt strategies (the
unchanged baseline + targeted rewrites) and scores each answer against the
case's ``expected`` rubric.  The output feeds the paired statistics machinery:

- ``findings["by_strategy"] = {strategy: {case_id: 0/1}}`` → M2's
  ``mcnemar_evalue`` (paired, anytime-valid) and ``friedman_nemenyi`` tools;
- per-case repair flags (``fixed_by_<strategy>``) → ``signal_label_assoc``.

This is the harness's *causal* probe for decision-layer failure hypotheses:
if "describe the region first, then answer" repairs failures, the deficit is in
answer generation, not perception ("language prior bias"); if an explicit
"report even subtle findings" instruction repairs them, the model's internal
yes-threshold is miscalibrated; if nothing helps, the failure sits in visual
feature discrimination itself.

The default strategies are designed to discriminate exactly those mechanisms;
pass ``strategies={name: template}`` for custom interventions (templates use
``{prompt}`` for the original question).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model

# Default interventions — chosen to discriminate decision-layer mechanisms.
_DEFAULT_STRATEGIES: dict[str, str] = {
    "baseline": "{prompt}",
    "describe_first": (
        "First describe what you see in the relevant region of the image in "
        "one sentence, then answer the question. {prompt}"
    ),
    "sensitive": (
        "{prompt} Report a finding as present even if it is subtle — do not "
        "default to a negative answer when uncertain."
    ),
}


def _word_in(term: str, text: str) -> bool:
    """Word-boundary match for plain alphanumeric terms, substring otherwise."""
    term = term.lower().strip()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9]+", term):
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


def _default_score(case: "FailureCase", observed: str) -> Optional[bool]:
    """Score *observed* against ``case.expected`` (dict rubric or string).

    Returns ``True``/``False``, or ``None`` when the case has no usable rubric.
    """
    expected = getattr(case, "expected", None)
    if expected is None:
        return None
    text = re.sub(r"\s+", " ", str(observed).lower())
    if isinstance(expected, dict):
        if any(_word_in(t, text) for t in expected.get("none_of", [])):
            return False
        if not all(_word_in(t, text) for t in expected.get("all_of", [])):
            return False
        any_of = expected.get("any_of", [])
        if any_of and not any(_word_in(t, text) for t in any_of):
            return False
        return True
    if isinstance(expected, str):
        return _word_in(expected, text)
    return None


@register_analyzer("prompt_contrast")
class PromptContrastAnalyzer(Analyzer):
    """Paired prompt-strategy contrast over the whole batch.

    Hyper-parameters:
        strategies: ``{name: template}`` prompt rewrites; templates receive the
                    original question via ``{prompt}``.  Must include a
                    ``"baseline"`` entry (added automatically if missing).
        score_fn:   ``(case, answer) -> bool | None`` answer scorer.  Defaults
                    to a word-boundary rubric scorer over ``case.expected``
                    (dict ``all_of/any_of/none_of`` or plain string).
        max_cases:  Cap on cases (cost = ``len(strategies)`` generations each).
    """

    name = "prompt_contrast"
    description = (
        "Re-asks each question under paired prompt interventions (baseline / "
        "describe-the-image-first / answer-sensitively) and scores each variant "
        "— causal evidence for whether failures are repairable at the prompt "
        "level (language prior / threshold miscalibration) or persist "
        "regardless (perception failure)."
    )
    requires = frozenset({Capability.GENERATE})
    applies_to_modalities = frozenset({"text", "image"})

    def __init__(
        self,
        strategies: dict[str, str] | None = None,
        score_fn: Callable[[Any, str], Optional[bool]] | None = None,
        max_cases: int = 32,
    ) -> None:
        strategies = dict(strategies) if strategies else dict(_DEFAULT_STRATEGIES)
        if "baseline" not in strategies:
            strategies = {"baseline": "{prompt}", **strategies}
        super().__init__(strategies=strategies, score_fn=score_fn, max_cases=max_cases)

    def _run(self, model: "Model", cases: "CaseBatch") -> Result:
        from evalvitals.core.case import Inputs

        score = self.score_fn or _default_score
        by_strategy: dict[str, dict[str, float]] = {s: {} for s in self.strategies}
        answers: dict[str, dict[str, str]] = {s: {} for s in self.strategies}
        n_unscored = 0

        selected = list(cases)[: self.max_cases]
        for case in selected:
            prompt = str(getattr(case.inputs, "prompt", ""))
            image = getattr(case.inputs, "image", None)
            for strat, template in self.strategies.items():
                rewritten = template.format(prompt=prompt)
                out = str(model.generate(Inputs(prompt=rewritten, image=image)))
                verdict = score(case, out)
                answers[strat][case.id] = out[:200]
                if verdict is None:
                    n_unscored += 1
                    continue
                by_strategy[strat][case.id] = float(bool(verdict))

        # Per-case repair/breakage flags relative to baseline.  Deliberately NO
        # baseline_correct flag here: baseline correctness is (inversely) the
        # PASS/FAIL label itself, so as a stats signal it produces tautological
        # verdicts and hijacks evidence routing — the intervention information
        # lives in the fixed_by_*/broken_by_* flags.
        base = by_strategy.get("baseline", {})
        variants = [s for s in self.strategies if s != "baseline"]
        per_case: list[dict[str, Any]] = []
        for case in selected:
            cid = case.id
            if cid not in base:
                continue
            entry: dict[str, Any] = {"sample_id": cid}
            for strat in variants:
                if cid in by_strategy[strat]:
                    entry[f"fixed_by_{strat}"] = bool(
                        base[cid] == 0.0 and by_strategy[strat][cid] == 1.0
                    )
                    entry[f"broken_by_{strat}"] = bool(
                        base[cid] == 1.0 and by_strategy[strat][cid] == 0.0
                    )
            per_case.append(entry)

        findings: dict[str, Any] = {
            "n_cases": len(selected),
            "n_strategies": len(self.strategies),
            "n_unscored": n_unscored,
            "by_strategy": by_strategy,
            "per_case": per_case,
        }
        for strat in self.strategies:
            scored = by_strategy[strat]
            if scored:
                findings[f"success_rate_{strat}"] = round(
                    sum(scored.values()) / len(scored), 4
                )
        for strat in variants:
            n_fixed = sum(1 for e in per_case if e.get(f"fixed_by_{strat}"))
            n_broken = sum(1 for e in per_case if e.get(f"broken_by_{strat}"))
            findings[f"n_fixed_by_{strat}"] = n_fixed
            findings[f"n_broken_by_{strat}"] = n_broken

        return Result(
            analyzer=self.name,
            model=repr(model),
            cases=cases,
            findings=findings,
            artifacts={"answers": answers},
        )
