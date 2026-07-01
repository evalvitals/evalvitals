"""Generic planning utilities for M2 analysis.

The planner consumes a profile of the available data and emits a deterministic
tool plan. It is intentionally conservative: discovery may inspect everything,
but confirmatory plans should make tested families explicit and avoid silently
dropping columns because of their original order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalvitals.analysis.profile import DatasetProfile, profile_stats_input


@dataclass
class AnalysisPlanItem:
    """One planned statistical analysis."""

    tool: str
    config: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    estimand: str = ""
    family: str = "confirmatory"
    priority: float = 0.0

    def as_legacy_tuple(self) -> tuple[str, dict[str, Any], str]:
        return self.tool, self.config, self.rationale

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "config": self.config,
            "rationale": self.rationale,
            "estimand": self.estimand,
            "family": self.family,
            "priority": self.priority,
        }


def _signal_priority(inp: Any, signal: str, profile: DatasetProfile) -> float:
    col = profile.columns.get(signal)
    sigmap = (getattr(inp, "per_case", {}) or {}).get(signal, {})
    labels = getattr(inp, "labels", {}) or {}
    if not sigmap:
        return 0.0
    n_labeled = max(1, len(labels))
    coverage = min(1.0, len(sigmap) / n_labeled)
    values = [float(v) for v in sigmap.values()]
    unique = len(set(values))
    variance = 0.0
    if values:
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
    kind_bonus = 0.2 if col and col.is_binary else 0.1 if col and col.dtype == "numeric" else 0.0
    sparse_binary_bonus = 0.15 if col and col.is_binary and coverage < 0.95 else 0.0
    constant_penalty = -1.0 if unique <= 1 else 0.0
    return coverage + min(1.0, variance) + kind_bonus + sparse_binary_bonus + constant_penalty


def ranked_signal_names(inp: Any, *, max_signals: int | None = None) -> list[str]:
    """Rank per-case signals by testability rather than insertion order."""
    profile = profile_stats_input(inp)
    scored = [
        (_signal_priority(inp, signal, profile), signal)
        for signal in (getattr(inp, "per_case", {}) or {})
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    names = [name for score, name in scored if score > 0]
    if max_signals is not None:
        names = names[:max(0, int(max_signals))]
    return names


def plan_stats_input(
    inp: Any,
    *,
    max_signals: int | None = None,
) -> list[AnalysisPlanItem]:
    """Create a deterministic confirmatory plan for a ``StatsInput`` object."""
    labels = getattr(inp, "labels", {}) or {}
    groups = getattr(inp, "groups", None) or {}
    vectors = getattr(inp, "per_case_vectors", {}) or {}
    n_fail = sum(1 for value in labels.values() if value)
    n_pass = len(labels) - n_fail
    plan: list[AnalysisPlanItem] = []

    if n_pass > 0 and n_fail > 0 and getattr(inp, "per_case", None):
        ranked = ranked_signal_names(inp, max_signals=max_signals)
        for key in ranked:
            plan.append(AnalysisPlanItem(
                tool="signal_label_assoc",
                config={"signal": key},
                rationale=f"test whether per-case signal '{key}' predicts FAIL",
                estimand=f"P(FAIL | {key}=present/high) - P(FAIL | {key}=absent/low)",
                priority=1.0,
            ))
        continuous = [
            key for key in ranked
            if key in (getattr(inp, "per_case", {}) or {})
            and len({float(v) for v in inp.per_case[key].values()}) > 2
        ]
        for key in continuous:
            plan.append(AnalysisPlanItem(
                tool="rank_corr",
                config={"signal": key},
                rationale=f"monotonic association between continuous '{key}' and FAIL",
                estimand=f"Kendall tau({key}, FAIL)",
                family="descriptive",
                priority=0.5,
            ))

    if n_pass > 0 and n_fail > 0 and vectors:
        vector_names = list(vectors)
        if max_signals is not None:
            vector_names = vector_names[:max(0, int(max_signals))]
        for key in vector_names:
            plan.append(AnalysisPlanItem(
                tool="attention_decoding",
                config={"signal": key},
                rationale=f"omnibus: do FAIL/PASS per-case maps '{key}' differ?",
                estimand=f"distributional difference in vector signal {key}",
                priority=0.8,
            ))

    if labels:
        plan.append(AnalysisPlanItem(
            tool="single_rate_evalue",
            config={},
            rationale="describe the overall FAIL rate against an explicit baseline if provided",
            estimand="P(FAIL) - p0",
            family="descriptive",
            priority=0.1,
        ))

    n_groups = len(groups)
    if n_groups >= 3:
        plan.append(AnalysisPlanItem(
            tool="friedman_nemenyi",
            config={},
            rationale="rank 3+ strategies on shared cases",
            estimand="strategy rank differences",
            priority=0.9,
        ))
        names = list(groups)
        base = names[0]
        for variant in names[1:]:
            plan.append(AnalysisPlanItem(
                tool="mcnemar_evalue",
                config={"strategies": [base, variant]},
                rationale=f"paired contrast: does '{variant}' repair '{base}' failures?",
                estimand=f"{variant} success - {base} success on paired cases",
                priority=1.0,
            ))
    elif n_groups == 2:
        plan.append(AnalysisPlanItem(
            tool="mcnemar_evalue",
            config={},
            rationale="paired two-strategy comparison",
            estimand="strategy success difference on paired cases",
            priority=1.0,
        ))

    return plan
