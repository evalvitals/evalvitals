"""M2 — statistical tool catalog: thin wrappers around :mod:`evalvitals.stats`.

This module turns the rigorous-but-low-level :mod:`evalvitals.stats` building
blocks (McNemar + e-value, clustered bootstrap, Friedman/Nemenyi, e-BH) into a
small **catalog of named tools** that
:class:`~evalvitals.eval_agent.stages.stats_agent.StatsAnalysisAgent` can
*select* and call with a config dict — the "select stats tools" half of the M2
plan (Plan A, 2026-06-05).

The flow is:

1. :func:`build_stats_input` normalises ``{analyzer: Result}`` + a labeled
   ``CaseBatch`` into a single :class:`StatsInput` (per-case signals, labels,
   scalar metrics, optional strategy groups).
2. The agent picks tool names from :data:`STATS_TOOL_CATALOG` (LLM-guided) or
   falls back to :func:`default_plan` (deterministic).
3. Each ``(tool, config)`` runs via :func:`run_stats_tool`, returning a uniform
   :class:`StatsToolResult` (effect, CI, e-value, reject, underpowered).
4. :func:`fdr_correct` applies e-BH across all tools that produced an e-value so
   multiple-metric testing is FDR-controlled, not naive.
5. :func:`plot_effects` draws an optional forest plot of effect ± CI.

No tool ever returns a bare p-value: every verdict carries an effect size and a
corrected reject decision, inherited from :func:`evalvitals.stats.compare`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from evalvitals.core.case import Label
from evalvitals.stats import (
    compare,
    compare_multiple,
    e_value_test,
    ebh,
    kendall_tau,
)

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.result import Result

logger = logging.getLogger(__name__)

# Keys in a per_case finding entry that identify the case, not a signal.
_ID_KEYS = ("sample_id", "case_id", "id")


# ---------------------------------------------------------------------------
# Normalised input + uniform result
# ---------------------------------------------------------------------------

@dataclass
class StatsInput:
    """Normalised view of M1 results + labels, ready for statistical tests.

    Attributes:
        labels:    ``{case_id -> is_fail}`` for PASS/FAIL cases (UNKNOWN dropped).
        per_case:  ``{"analyzer.metric" -> {case_id -> value}}`` — per-case
                   numeric/boolean signals harvested from ``findings["per_case"]``.
        scalars:   ``{"analyzer.metric" -> value}`` — aggregate numeric findings.
        groups:    Optional ``{strategy -> {case_id -> success}}`` for
                   paired/omnibus strategy comparisons (from ``findings["by_strategy"]``).
    """

    labels: dict[str, bool] = field(default_factory=dict)
    per_case: dict[str, dict[str, float]] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    groups: dict[str, dict[str, float]] | None = None

    @classmethod
    def from_results(
        cls,
        results: "dict[str, Result]",
        data: "CaseBatch | None" = None,
    ) -> "StatsInput":
        """Build statistical input from EvalVitals analyzer results."""
        return build_stats_input(results, data)

    @classmethod
    def from_records(
        cls,
        records: "Any",
        *,
        id_col: str = "case_id",
        label_col: str = "label",
        signal_cols: "list[str] | tuple[str, ...] | None" = None,
        scalar_cols: "list[str] | tuple[str, ...] | None" = None,
        signal_prefix: str = "",
    ) -> "StatsInput":
        """Build statistical input from plain row dictionaries.

        ``label_col`` accepts booleans, ``Label`` values, common strings
        (``"pass"``, ``"fail"``, ``"success"``, ``"error"``), or 0/1 values
        where 1 means FAIL.  Signal and scalar columns must be numeric/bool.
        """
        return build_stats_input_from_records(
            records,
            id_col=id_col,
            label_col=label_col,
            signal_cols=signal_cols,
            scalar_cols=scalar_cols,
            signal_prefix=signal_prefix,
        )


@dataclass
class StatsToolResult:
    """Uniform output of one statistical tool.

    Attributes:
        tool:         Catalog name of the tool that produced this.
        config:       The (resolved) config the tool ran with.
        ok:           ``True`` when the test ran; ``False`` when skipped/failed.
        summary:      Human-readable one-liner.
        effect:       Effect size (e.g. fail-rate difference, τ) when applicable.
        ci:           Confidence interval on the effect, when applicable.
        reject:       Corrected reject decision (e-value / CI), never a bare p.
        e_value:      Anytime-valid e-value, when the test produces one.
        p_value:      Raw p (diagnostic only — never the decision basis).
        underpowered: Inconclusive *and* CI too wide to rule out a real effect.
        details:      Tool-specific extras (group sizes, ranks, contingency …).
        figure_path:  Path to a per-tool figure, if any.
        error:        Reason string when ``ok`` is ``False``.
    """

    tool: str
    config: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    summary: str = ""
    effect: float | None = None
    ci: tuple[float, float] | None = None
    reject: bool | None = None
    e_value: float | None = None
    p_value: float | None = None
    underpowered: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    figure_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "config": self.config,
            "ok": self.ok,
            "summary": self.summary,
            "effect": self.effect,
            "ci": list(self.ci) if self.ci is not None else None,
            "reject": self.reject,
            "e_value": self.e_value,
            "p_value": self.p_value,
            "underpowered": self.underpowered,
            "details": self.details,
            "figure_path": self.figure_path,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def _entry_id(entry: dict) -> str:
    for k in _ID_KEYS:
        v = entry.get(k)
        if v:
            return str(v)
    return ""


def build_stats_input(
    results: "dict[str, Result]",
    data: "CaseBatch | None" = None,
) -> StatsInput:
    """Normalise ``{analyzer: Result}`` + optional labeled *data* into a :class:`StatsInput`."""
    labels: dict[str, bool] = {}
    if data is not None:
        for c in data:
            lab = getattr(c, "label", None)
            if lab is None or lab == Label.UNKNOWN:
                continue
            is_fail = lab == Label.FAIL
            labels[c.id] = is_fail
            traj = getattr(c, "trajectory", None)
            sid = getattr(traj, "sample_id", "") if traj is not None else ""
            if sid:
                labels[sid] = is_fail

    per_case: dict[str, dict[str, float]] = {}
    scalars: dict[str, float] = {}
    groups: dict[str, dict[str, float]] = {}

    for aname, res in results.items():
        findings = res.findings or {}
        for k, v in findings.items():
            if isinstance(v, (int, float, bool)):
                scalars[f"{aname}.{k}"] = float(v)
        for entry in findings.get("per_case", []) or []:
            if not isinstance(entry, dict):
                continue
            cid = _entry_id(entry)
            if not cid:
                continue
            for k, v in entry.items():
                if k in _ID_KEYS:
                    continue
                if isinstance(v, (int, float, bool)):
                    per_case.setdefault(f"{aname}.{k}", {})[cid] = float(v)
        by_strategy = findings.get("by_strategy")
        if isinstance(by_strategy, dict):
            for sname, vec in by_strategy.items():
                slot = groups.setdefault(str(sname), {})
                if isinstance(vec, dict):
                    for cid, val in vec.items():
                        if isinstance(val, (int, float, bool)):
                            slot[str(cid)] = float(bool(val))

    return StatsInput(
        labels=labels,
        per_case=per_case,
        scalars=scalars,
        groups=groups or None,
    )


def build_stats_input_from_records(
    records: "Any",
    *,
    id_col: str = "case_id",
    label_col: str = "label",
    signal_cols: "list[str] | tuple[str, ...] | None" = None,
    scalar_cols: "list[str] | tuple[str, ...] | None" = None,
    signal_prefix: str = "",
) -> StatsInput:
    """Normalise plain records into :class:`StatsInput`.

    This is the standalone on-ramp for users who have a table of cases and
    signals rather than EvalVitals ``Result`` objects.
    """
    labels: dict[str, bool] = {}
    per_case: dict[str, dict[str, float]] = {}
    scalars: dict[str, float] = {}

    rows = list(records or [])
    if signal_cols is None and rows:
        excluded = {id_col, label_col, *(scalar_cols or ())}
        signal_cols = [
            str(k) for k, v in _row_items(rows[0])
            if k not in excluded and isinstance(v, (int, float, bool))
        ]
    signal_cols = tuple(signal_cols or ())
    scalar_cols = tuple(scalar_cols or ())

    for i, row in enumerate(rows):
        cid = _row_get(row, id_col, None)
        if cid in (None, ""):
            cid = str(i)
        cid = str(cid)

        label = _parse_label(_row_get(row, label_col, None))
        if label is not None:
            labels[cid] = label

        for col in signal_cols:
            val = _row_get(row, col, None)
            if isinstance(val, (int, float, bool)):
                key = f"{signal_prefix}.{col}" if signal_prefix else str(col)
                per_case.setdefault(key, {})[cid] = float(val)

        for col in scalar_cols:
            val = _row_get(row, col, None)
            if isinstance(val, (int, float, bool)):
                scalars[str(col)] = float(val)

    return StatsInput(labels=labels, per_case=per_case, scalars=scalars)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _row_items(row: Any) -> list[tuple[str, Any]]:
    if isinstance(row, dict):
        return list(row.items())
    if hasattr(row, "_asdict"):
        return list(row._asdict().items())
    if hasattr(row, "__dict__"):
        return list(vars(row).items())
    return []


def _parse_label(value: Any) -> bool | None:
    if value is None or value == Label.UNKNOWN:
        return None
    if value == Label.FAIL:
        return True
    if value == Label.PASS:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"fail", "failed", "failure", "false", "incorrect", "error", "bad", "1"}:
        return True
    if text in {"pass", "passed", "success", "true", "correct", "ok", "0"}:
        return False
    return None


def _is_binary(values: "Any") -> bool:
    return all(float(v) in (0.0, 1.0) for v in values)


def _binarize(
    signal_map: dict[str, float],
    mode: str = "median",
    threshold: float | None = None,
) -> dict[str, bool]:
    """Binarise a continuous per-case signal (median split by default)."""
    vals = list(signal_map.values())
    if not vals or _is_binary(vals):
        return {cid: bool(v) for cid, v in signal_map.items()}
    if threshold is None:
        if mode == "median":
            s = sorted(vals)
            n = len(s)
            threshold = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
        else:
            threshold = 0.0
    return {cid: (v > threshold) for cid, v in signal_map.items()}


def describe_data(inp: StatsInput) -> dict[str, Any]:
    """Compact, LLM-friendly summary of what statistical tests are feasible."""
    n_fail = sum(1 for v in inp.labels.values() if v)
    n_labeled = len(inp.labels)
    continuous = [k for k, m in inp.per_case.items() if not _is_binary(m.values())]
    return {
        "n_labeled": n_labeled,
        "n_fail": n_fail,
        "n_pass": n_labeled - n_fail,
        "per_case_signals": list(inp.per_case),
        "continuous_signals": continuous,
        "scalar_metrics": list(inp.scalars),
        "n_strategy_groups": len(inp.groups) if inp.groups else 0,
    }


def has_testable_data(inp: StatsInput) -> bool:
    """True when at least one tool could run (labels or strategy groups present)."""
    return bool(inp.labels) or bool(inp.groups)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _split_signal_groups(
    inp: StatsInput,
    signal_key: str,
    binarize: str = "median",
    threshold: float | None = None,
) -> tuple[list[int], list[int]] | None:
    """Split labeled cases into (signal-present, signal-absent) fail indicators.

    A sparse binary flag only lists the cases where it fired, so a labeled case
    missing from the signal map means the signal was *absent* (control group).
    For a continuous signal we cannot assume a value, so missing cases are
    skipped rather than defaulted.
    """
    sigmap = inp.per_case.get(signal_key)
    if not sigmap:
        return None
    binar = _binarize(sigmap, binarize, threshold)
    treat_missing_as_absent = _is_binary(sigmap.values())
    signal_fail: list[int] = []
    control_fail: list[int] = []
    for cid, is_fail in inp.labels.items():
        if cid in binar:
            present = binar[cid]
        elif treat_missing_as_absent:
            present = False
        else:
            continue
        (signal_fail if present else control_fail).append(int(is_fail))
    return signal_fail, control_fail


def _mean(xs: list[int]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _tool_signal_label_assoc(inp: StatsInput, config: dict) -> StatsToolResult:
    """Unpaired fail-rate difference between cases with/without a per-case signal."""
    key = config.get("signal") or next(iter(inp.per_case), None)
    cfg = {**config, "signal": key}
    if not key or key not in inp.per_case:
        return StatsToolResult(
            tool="signal_label_assoc", config=cfg, ok=False,
            error="no per-case signal available", summary="signal_label_assoc: no signal",
        )
    split = _split_signal_groups(
        inp, key, config.get("binarize", "median"), config.get("threshold")
    )
    assert split is not None
    signal_fail, control_fail = split
    if not signal_fail or not control_fail:
        return StatsToolResult(
            tool="signal_label_assoc", config=cfg, ok=False,
            error="one group empty (need both signal-present and signal-absent cases)",
            summary="signal_label_assoc: degenerate split",
            details={"n_signal": len(signal_fail), "n_control": len(control_fail)},
        )
    sr = compare(
        control_fail, signal_fail, paired=False,
        alpha=config.get("alpha", 0.05),
        min_effect=config.get("min_effect", 0.0),
        n_boot=config.get("n_boot", 2000),
    )
    return StatsToolResult(
        tool="signal_label_assoc", config=cfg, ok=True,
        effect=sr.effect, ci=sr.ci, reject=sr.reject, underpowered=sr.underpowered,
        summary=f"signal '{key}' vs FAIL: {sr.summary()}",
        details={
            "n_signal": len(signal_fail), "n_control": len(control_fail),
            "fail_rate_signal": round(_mean(signal_fail), 4),
            "fail_rate_control": round(_mean(control_fail), 4),
            **sr.details,
        },
    )


def _aligned_groups(inp: StatsInput, names: list[str]) -> tuple[list[str], list[list[float]]]:
    sets = [set(inp.groups[n]) for n in names]  # type: ignore[index]
    common = sorted(set.intersection(*sets)) if sets else []
    vecs = [[inp.groups[n][cid] for cid in common] for n in names]  # type: ignore[index]
    return common, vecs


def _tool_bootstrap_diff(inp: StatsInput, config: dict) -> StatsToolResult:
    """Unpaired fail-rate difference between two strategy groups (bootstrap CI)."""
    if not inp.groups or len(inp.groups) < 2:
        return StatsToolResult(
            tool="bootstrap_diff", config=config, ok=False,
            error="need >=2 strategy groups", summary="bootstrap_diff: <2 groups",
        )
    names = config.get("strategies") or list(inp.groups)[:2]
    names = list(names)[:2]
    common, vecs = _aligned_groups(inp, names)
    if not common:
        return StatsToolResult(
            tool="bootstrap_diff", config={**config, "strategies": names}, ok=False,
            error="no shared cases between groups", summary="bootstrap_diff: no overlap",
        )
    a, b = vecs
    sr = compare(
        a, b, paired=False,
        alpha=config.get("alpha", 0.05), min_effect=config.get("min_effect", 0.0),
        n_boot=config.get("n_boot", 2000),
    )
    return StatsToolResult(
        tool="bootstrap_diff", config={**config, "strategies": names}, ok=True,
        effect=sr.effect, ci=sr.ci, reject=sr.reject, underpowered=sr.underpowered,
        summary=f"{names[1]} vs {names[0]}: {sr.summary()}",
        details={"n": len(common), "strategies": names, **sr.details},
    )


def _tool_mcnemar_evalue(inp: StatsInput, config: dict) -> StatsToolResult:
    """Paired binary comparison of two strategies (McNemar + anytime-valid e-value)."""
    if not inp.groups or len(inp.groups) < 2:
        return StatsToolResult(
            tool="mcnemar_evalue", config=config, ok=False,
            error="need >=2 strategy groups", summary="mcnemar_evalue: <2 groups",
        )
    names = config.get("strategies") or list(inp.groups)[:2]
    names = list(names)[:2]
    common, vecs = _aligned_groups(inp, names)
    if not common:
        return StatsToolResult(
            tool="mcnemar_evalue", config={**config, "strategies": names}, ok=False,
            error="no shared cases between groups", summary="mcnemar_evalue: no overlap",
        )
    a, b = vecs
    sr = compare(
        a, b, paired=True,
        alpha=config.get("alpha", 0.05), min_effect=config.get("min_effect", 0.0),
    )
    return StatsToolResult(
        tool="mcnemar_evalue", config={**config, "strategies": names}, ok=True,
        effect=sr.effect, ci=sr.ci, reject=sr.reject, e_value=sr.e_value,
        p_value=sr.details.get("p_value"), underpowered=sr.underpowered,
        summary=f"{names[1]} vs {names[0]} (paired): {sr.summary()}",
        details={"n": len(common), "strategies": names, **sr.details},
    )


def _tool_friedman_nemenyi(inp: StatsInput, config: dict) -> StatsToolResult:
    """Rank 3+ strategies across shared cases (Friedman omnibus + Nemenyi post-hoc)."""
    if not inp.groups or len(inp.groups) < 3:
        return StatsToolResult(
            tool="friedman_nemenyi", config=config, ok=False,
            error="need >=3 strategy groups", summary="friedman_nemenyi: <3 groups",
        )
    names = list(inp.groups)
    common, vecs = _aligned_groups(inp, names)
    if not common:
        return StatsToolResult(
            tool="friedman_nemenyi", config=config, ok=False,
            error="no shared cases across all groups", summary="friedman_nemenyi: no overlap",
        )
    by_strategy = dict(zip(names, vecs))
    mc = compare_multiple(by_strategy, alpha=config.get("alpha", 0.05))
    return StatsToolResult(
        tool="friedman_nemenyi", config=config, ok=True,
        reject=mc.reject_global, p_value=mc.p_value,
        summary=mc.summary(),
        details={
            "avg_ranks": mc.avg_ranks,
            "critical_difference": mc.critical_difference,
            "significant_pairs": mc.significant_pairs,
            "n": mc.n,
        },
    )


def _tool_single_rate_evalue(inp: StatsInput, config: dict) -> StatsToolResult:
    """Anytime-valid test that the overall FAIL rate differs from a baseline p0.

    DESCRIPTIVE ONLY. The result is meaningful only when the case batch is a
    REPRESENTATIVE sample and ``p0`` is a justified baseline (the model's
    natural fail rate on this task). On a curated/enriched batch — the norm for
    diagnosis, where failures are over-sampled so a mechanism can be tested —
    the rate is a sampling artifact and ``p0=0.5`` tests nothing. The verdict
    layer treats this tool as descriptive and never makes it a hypothesis
    headline; pass ``config["p0"]`` = the manifest's recorded base rate to make
    it interpretable.
    """
    if not inp.labels:
        return StatsToolResult(
            tool="single_rate_evalue", config=config, ok=False,
            error="no labeled cases", summary="single_rate_evalue: no labels",
        )
    p0 = config.get("p0", 0.5)
    p0_justified = "p0" in config  # explicit baseline vs the meaningless default
    alpha = config.get("alpha", 0.05)
    fails = sum(1 for v in inp.labels.values() if v)
    n = len(inp.labels)
    res = e_value_test(fails, n, p0=p0, alpha=alpha)
    rate = fails / n
    caveat = "" if p0_justified else " (descriptive only: p0=0.5 is not a justified baseline)"
    return StatsToolResult(
        tool="single_rate_evalue", config={**config, "p0": p0}, ok=True,
        # No reported effect when p0 is the unjustified default — its rate − 0.5
        # would otherwise pollute any |effect|-based ranking downstream.
        effect=round(rate - p0, 4) if p0_justified else None,
        e_value=res["e_value"], reject=res["reject"] if p0_justified else False,
        summary=(
            f"FAIL rate {rate:.1%} ({fails}/{n}) vs p0={p0:.2f}: "
            f"e={res['e_value']:.2f} -> "
            f"{'reject' if (res['reject'] and p0_justified) else 'inconclusive'}{caveat}"
        ),
        details={"fails": fails, "n": n, "rate": round(rate, 4),
                 "p0_justified": p0_justified, **res},
    )


def _tool_rank_corr(inp: StatsInput, config: dict) -> StatsToolResult:
    """Kendall τ between a continuous per-case signal and FAIL (monotonic association)."""
    key = config.get("signal") or next(iter(inp.per_case), None)
    cfg = {**config, "signal": key}
    if not key or key not in inp.per_case:
        return StatsToolResult(
            tool="rank_corr", config=cfg, ok=False,
            error="no per-case signal available", summary="rank_corr: no signal",
        )
    sigmap = inp.per_case[key]
    xs: list[float] = []
    ys: list[float] = []
    for cid, is_fail in inp.labels.items():
        if cid in sigmap:
            xs.append(sigmap[cid])
            ys.append(float(is_fail))
    if len(xs) < 3:
        return StatsToolResult(
            tool="rank_corr", config=cfg, ok=False,
            error="need >=3 paired (signal, label) points",
            summary="rank_corr: too few points", details={"n": len(xs)},
        )
    tau = kendall_tau(xs, ys)
    return StatsToolResult(
        tool="rank_corr", config=cfg, ok=True, effect=round(tau, 4),
        summary=f"Kendall τ between '{key}' and FAIL = {tau:+.3f} (n={len(xs)})",
        details={"n": len(xs), "tau": round(tau, 4), "signal": key},
    )


# Registry: name -> callable. Edit STATS_TOOL_CATALOG in lockstep.
STATS_TOOLS: dict[str, Callable[[StatsInput, dict], StatsToolResult]] = {
    "signal_label_assoc": _tool_signal_label_assoc,
    "bootstrap_diff": _tool_bootstrap_diff,
    "mcnemar_evalue": _tool_mcnemar_evalue,
    "friedman_nemenyi": _tool_friedman_nemenyi,
    "single_rate_evalue": _tool_single_rate_evalue,
    "rank_corr": _tool_rank_corr,
}

# Catalog text shown to the LLM selector (name -> when to use it).
STATS_TOOL_CATALOG: dict[str, str] = {
    "signal_label_assoc": (
        "Unpaired fail-rate difference between cases that exhibit a per-case "
        "analyzer signal and those that don't (bootstrap CI). Use when you have "
        "per-case signals AND PASS/FAIL labels — this is the main M2 test."
    ),
    "bootstrap_diff": (
        "Unpaired fail-rate difference between two strategy groups (bootstrap CI). "
        "Needs >=2 strategy groups (findings['by_strategy'])."
    ),
    "mcnemar_evalue": (
        "Paired binary comparison of two strategies on the same cases "
        "(McNemar + anytime-valid e-value). Needs exactly 2 paired strategy groups."
    ),
    "friedman_nemenyi": (
        "Rank 3+ strategies across shared cases (Friedman omnibus + Nemenyi "
        "post-hoc). Needs >=3 strategy groups."
    ),
    "single_rate_evalue": (
        "DESCRIPTIVE context only: tests whether the overall FAIL rate differs "
        "from a baseline p0. Meaningful ONLY on a representative sample with a "
        "justified p0 (pass config['p0'] = the natural base rate); on a curated/"
        "enriched diagnosis batch it tests nothing. Never decides a hypothesis."
    ),
    "rank_corr": (
        "Kendall tau between a continuous per-case signal and FAIL (monotonic "
        "association). Needs a continuous per-case signal."
    ),
}


def run_stats_tool(name: str, inp: StatsInput, config: dict | None = None) -> StatsToolResult:
    """Run a single catalog tool by name. Raises KeyError for unknown names."""
    tool = STATS_TOOLS[name]
    return tool(inp, config or {})


# ---------------------------------------------------------------------------
# Deterministic planner (fallback when no judge / LLM selection fails)
# ---------------------------------------------------------------------------

def default_plan(inp: StatsInput, max_signals: int = 4) -> list[tuple[str, dict, str]]:
    """Deterministic ``[(tool, config, rationale)]`` plan from the data shape."""
    d = describe_data(inp)
    plan: list[tuple[str, dict, str]] = []

    if d["n_pass"] > 0 and d["n_fail"] > 0 and inp.per_case:
        for key in list(inp.per_case)[:max_signals]:
            plan.append((
                "signal_label_assoc", {"signal": key},
                f"test whether per-case signal '{key}' predicts FAIL",
            ))
        for key in d["continuous_signals"][:max_signals]:
            plan.append((
                "rank_corr", {"signal": key},
                f"monotonic association between continuous '{key}' and FAIL",
            ))

    if d["n_labeled"] > 0:
        plan.append((
            "single_rate_evalue", {},
            "is the overall FAIL rate worse than chance (p0=0.5)?",
        ))

    n_groups = d["n_strategy_groups"]
    if n_groups >= 3:
        plan.append(("friedman_nemenyi", {}, "rank 3+ strategies"))
        # Pairwise paired tests against the first (baseline) strategy — the
        # informative contrasts for intervention experiments (prompt variants).
        names = list(inp.groups or {})
        base = names[0]
        for variant in names[1:4]:
            plan.append((
                "mcnemar_evalue", {"strategies": [base, variant]},
                f"paired contrast: does '{variant}' repair '{base}' failures?",
            ))
    elif n_groups == 2:
        plan.append(("mcnemar_evalue", {}, "paired two-strategy comparison"))

    return plan


# ---------------------------------------------------------------------------
# Multiple-testing correction + visualization
# ---------------------------------------------------------------------------

def fdr_correct(results: list[StatsToolResult], alpha: float = 0.05) -> dict[str, Any]:
    """Apply e-BH across every tool that produced an e-value (FDR under dependence)."""
    indexed = [(i, r) for i, r in enumerate(results) if r.e_value is not None]
    if not indexed:
        return {"method": "e-BH", "alpha": alpha, "n_tested": 0,
                "rejected_tools": [], "note": "no e-values to correct"}
    evalues = [r.e_value for _, r in indexed]
    keep = ebh(evalues, alpha)  # indices into `evalues`
    rejected = [results[indexed[j][0]].tool for j in keep]
    return {
        "method": "e-BH", "alpha": alpha, "n_tested": len(evalues),
        "rejected_tools": rejected,
    }


def plot_effects(results: list[StatsToolResult], out_path: str) -> str | None:
    """Forest plot of effect ± CI for tools that produced both. Returns path or None."""
    items = [
        (r.tool, r.effect, r.ci)
        for r in results
        if r.ok and r.effect is not None and r.ci is not None
    ]
    if not items:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    labels = [t for t, _, _ in items]
    effects = [e for _, e, _ in items]
    lows = [e - ci[0] for _, e, ci in items]
    highs = [ci[1] - e for _, e, ci in items]
    ys = list(range(len(items)))

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(items) + 1.5))
    ax.errorbar(effects, ys, xerr=[lows, highs], fmt="o", capsize=4, color="#2b6cb0")
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("effect size (fail-rate difference)")
    ax.set_title("M2 statistical effects (± CI)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
