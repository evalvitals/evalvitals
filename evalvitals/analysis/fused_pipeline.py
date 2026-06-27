"""Fused explore -> confirm pipeline (DESIGN §4).

One pass that DISCOVERS freely and CONFIRMS rigorously, with the two split apart:

1. ``_split_records`` — deterministic, stratified (by label) EXPLORE / CONFIRM
   partition (mirrors ``VLDiagnoseLoop._split_explore_confirm``). The explorer and
   catalog planner see ONLY explore rows; every e-value/CI is computed only on the
   disjoint confirm rows. That is the double-dip firewall.
2. Discovery on EXPLORE — TWO sources, blind to each other:
   - the LAMBDA explorer (free codegen EDA) -> observations / charts /
     candidate_signals, each optionally carrying a deterministic ``recipe``;
   - the catalog planner -> the per-case signal columns already present.
   Their union is deduped by signal name (estimand proxy; value-level estimand
   identity is an open question, see DESIGN §10).
3. Operationalization bridge — each explorer recipe is compiled on the CONFIRM rows
   into a frozen per-case column (``compile_recipe``) and merged into the confirm
   ``StatsInput.per_case`` so it competes like any analyzer signal.
4. Confirmation on CONFIRM — the real M2 firewall (``StatsAnalysisAgent``) runs the
   validated catalog over the EXPANDED signal family and produces effect/CI/e-value
   verdicts + e-BH ``corrected_rejections`` + conclusion. The explorer never decides.
5. Output assembly — per-signal host verdicts (provenance-tagged), and a graceful
   ``recommended_confirmatory_tests`` channel for candidates that could not be
   operationalized / were underpowered / had no confirm data. Nothing is silently
   dropped and nothing is over-claimed.

HONEST LIMITATION (encoded, not hidden): the catalog's marginal signal test
(``signal_label_assoc``) is an unpaired bootstrap-CI test with NO e-value, so
marginal per-case signals get a CI reject at ``alpha`` but sit OUTSIDE the e-BH
family (``fdr_corrected=False``). Genuinely FDR-controlling marginal signals needs
an e-value two-group test (DESIGN §10, "catalog 表达力"); this pipeline uses what
the validated core actually provides and flags the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalvitals.analysis.operationalize import (
    RecipeError,
    SignalRecipe,
    compile_recipe,
)
from evalvitals.eval_agent.stages.stats_tools import build_stats_input_from_records


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class FusedSignal:
    """One candidate signal confirmed (or attempted) on the held-out CONFIRM split."""

    name: str
    source: str                 # "explorer" | "catalog" | "both"
    description: str = ""
    suggested_test: str = ""
    operationalized: bool = True   # became a per-case column on CONFIRM
    # host verdict (from the M2 firewall on CONFIRM) — authoritative
    effect: float | None = None
    ci: tuple[float, float] | None = None
    e_value: float | None = None
    reject: bool | None = None
    underpowered: bool = False
    host_adjudicated: bool = False
    fdr_corrected: bool = False
    confirmed_on: str = ""      # "held_out" | "in_sample"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "suggested_test": self.suggested_test,
            "operationalized": self.operationalized,
            "effect": self.effect,
            "ci": list(self.ci) if self.ci is not None else None,
            "e_value": self.e_value,
            "reject": self.reject,
            "underpowered": self.underpowered,
            "host_adjudicated": self.host_adjudicated,
            "fdr_corrected": self.fdr_corrected,
            "confirmed_on": self.confirmed_on,
        }


@dataclass
class FusedReport:
    """Output of :func:`run_fused_analysis` — explore (descriptive) + confirm (verdicts)."""

    question: str = ""
    ok: bool = False
    observations: list[str] = field(default_factory=list)
    candidate_signals: list[FusedSignal] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    recommended_confirmatory_tests: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    adjudication: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    conclusion: str = ""
    explore_report: dict[str, Any] = field(default_factory=dict)
    confirm_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def confirmed_signal_names(self) -> list[str]:
        return [s.name for s in self.candidate_signals if s.reject]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ok": self.ok,
            "observations": self.observations,
            "candidate_signals": [s.to_dict() for s in self.candidate_signals],
            "charts": self.charts,
            "recommended_confirmatory_tests": self.recommended_confirmatory_tests,
            "caveats": self.caveats,
            "adjudication": self.adjudication,
            "split": self.split,
            "conclusion": self.conclusion,
        }


# ---------------------------------------------------------------------------
# Deterministic stratified EXPLORE / CONFIRM split (records level)
# ---------------------------------------------------------------------------

def _split_records(
    records: list[dict[str, Any]],
    *,
    frac: float,
    seed: int,
    label_col: str,
) -> "tuple[list[dict], list[dict] | None]":
    """Deterministic, label-stratified ``(explore, confirm)`` split.

    Returns ``(explore, confirm)``. Like the in-loop split, returns
    ``(records, None)`` when ``frac <= 0`` or the batch is too small to hold out —
    a no-op, so the caller falls back to in-sample confirmation with a caveat.
    """
    from evalvitals.stats.subset_sampling import stratified_subset

    rows = list(records)
    if frac <= 0.0 or len(rows) < 4:
        return rows, None
    n_confirm = round(len(rows) * frac)
    if n_confirm <= 0 or n_confirm >= len(rows):
        return rows, None

    def _key(row: dict[str, Any]) -> Any:
        return str(row.get(label_col))

    confirm = stratified_subset(rows, _key, n_confirm, seed=seed)
    confirm_ids = {id(r) for r in confirm}
    explore = [r for r in rows if id(r) not in confirm_ids]
    if not explore or not confirm:
        return rows, None
    return explore, confirm


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_fused_analysis(
    records: list[dict[str, Any]],
    *,
    question: str = "What distinguishes failures from passes?",
    explorer: Any,
    stats_agent: Any | None = None,
    confirm_split: float = 0.3,
    seed: int = 0,
    id_col: str = "case_id",
    label_col: str = "label",
) -> FusedReport:
    """Run discovery on EXPLORE, confirmation on the held-out CONFIRM split.

    Args:
        records:        list of row dicts (id/label/signal columns).
        explorer:       an object with ``explore_records(rows, question=...) ->
                        ExploratoryAnalysisReport`` (e.g. :class:`M2ExplorerAgent`).
        stats_agent:    optional :class:`StatsAnalysisAgent`; one is built with a
                        high signal cap when omitted so the whole family is tested.
        confirm_split:  fraction held out for confirmation (0 = in-sample fallback).
    """
    rows = list(records or [])
    explore_rows, confirm_rows = _split_records(
        rows, frac=confirm_split, seed=seed, label_col=label_col
    )
    held_out = confirm_rows is not None
    confirm_label = "held_out" if held_out else "in_sample"
    if not held_out:
        confirm_rows = explore_rows  # in-sample fallback (small batch)

    report = FusedReport(question=question)
    report.split = {
        "mode": confirm_label,
        "n_total": len(rows),
        "n_explore": len(explore_rows),
        "n_confirm": len(confirm_rows),
        "frac": confirm_split,
        "seed": seed,
    }

    # ── 1. Discovery source A: the LAMBDA explorer (EXPLORE only) ──
    explore_report = explorer.explore_records(explore_rows, question=question)
    report.ok = bool(getattr(explore_report, "ok", False))
    report.observations = list(getattr(explore_report, "observations", []) or [])
    report.charts = list(getattr(explore_report, "charts", []) or [])
    report.caveats = list(getattr(explore_report, "caveats", []) or [])
    report.explore_report = (
        explore_report.to_dict() if hasattr(explore_report, "to_dict") else {}
    )
    explorer_candidates = list(getattr(explore_report, "candidate_signals", []) or [])
    explorer_named = {c.name: c for c in explorer_candidates if getattr(c, "name", "")}

    # ── 2. Discovery source B: catalog columns present on EXPLORE ──
    explore_inp = build_stats_input_from_records(
        explore_rows, id_col=id_col, label_col=label_col
    )
    catalog_names = set(explore_inp.per_case)

    # ── 3. Operationalization bridge: compile explorer recipes on CONFIRM ──
    bridged: dict[str, dict[str, float]] = {}
    bridge_failed: list[Any] = []
    for c in explorer_candidates:
        recipe_data = getattr(c, "recipe", None)
        if not isinstance(recipe_data, dict):
            continue
        recipe = SignalRecipe.from_dict(recipe_data)
        if not recipe.name:
            recipe.name = c.name
        try:
            values = compile_recipe(recipe, confirm_rows, id_col=id_col)
        except (RecipeError, NotImplementedError):
            values = {}
        if values:
            bridged[recipe.name] = values
        else:
            bridge_failed.append(c)

    # ── 4. Confirmation: build confirm StatsInput, inject bridged signals, run M2 ──
    confirm_inp = build_stats_input_from_records(
        confirm_rows, id_col=id_col, label_col=label_col
    )
    catalog_confirm = set(confirm_inp.per_case)  # real catalog columns on CONFIRM
    bridged_keys: set[str] = set()               # final per_case keys for bridged signals
    bridged_origin: dict[str, str] = {}          # final key -> originating recipe name
    collisions: list[str] = []
    for name, values in bridged.items():
        key = name
        if key in catalog_confirm:
            # A bridged recipe must NEVER silently overwrite a real catalog column —
            # that would test a DIFFERENT estimand under the same name. Namespace it.
            key = f"bridged.{name}"
            while key in confirm_inp.per_case:
                key = f"_{key}"
            collisions.append(name)
        confirm_inp.per_case[key] = values
        bridged_keys.add(key)
        bridged_origin[key] = name
    if collisions:
        report.caveats.append(
            "bridged recipe name(s) collided with catalog column(s) and were "
            f"namespaced to avoid overwriting a real signal: {sorted(collisions)}"
        )

    if stats_agent is None:
        from evalvitals.analysis.stats_agent import StatsAnalysisAgent

        stats_agent = StatsAnalysisAgent(
            max_signal_tools=max(8, len(confirm_inp.per_case))
        )

    confirm_report = stats_agent.analyze_input(confirm_inp)
    report.confirm_stats = (
        confirm_report.to_dict() if hasattr(confirm_report, "to_dict") else {}
    )
    report.conclusion = getattr(confirm_report, "conclusion", "")
    corrected = getattr(confirm_report, "corrected_rejections", {}) or {}
    rejected_tools = set(corrected.get("rejected_tools", []))

    # ── 5. Assemble per-signal verdicts (provenance-tagged) ──
    verdict_by_signal = _verdicts_by_signal(getattr(confirm_report, "stats_results", []))
    for name in confirm_inp.per_case:
        result = verdict_by_signal.get(name)
        origin = bridged_origin.get(name, name)  # recipe name for a (possibly renamed) bridged key
        report.candidate_signals.append(
            _fused_signal(
                name=name,
                source=_source_of(name, catalog_names, bridged_keys, explorer_named),
                description=_describe(name, origin, explorer_named, bridged_keys),
                suggested_test=_suggested_test(origin, explorer_named),
                result=result,
                rejected_tools=rejected_tools,
                confirmed_on=confirm_label,
            )
        )

    # ── 6. Graceful degradation channel: nothing silently dropped ──
    rec = list(getattr(explore_report, "recommended_confirmatory_tests", []) or [])
    for c in bridge_failed:
        rec.append(
            f"{c.name}: {getattr(c, 'rationale', '')} "
            "(recipe could not be operationalized on the confirm split)"
        )
    for c in explorer_candidates:
        if (
            not getattr(c, "recipe", None)
            and c.name not in catalog_names
            and c.name not in bridged
        ):
            rec.append(
                f"{c.name}: {getattr(c, 'rationale', '')} "
                "(descriptive only — no host-testable recipe)"
            )
    for fs in report.candidate_signals:
        if fs.underpowered:
            rec.append(f"{fs.name}: underpowered on the confirm split")
    report.recommended_confirmatory_tests = rec

    # ── 7. Family metadata + honest caveats ──
    report.adjudication = {
        "method": corrected.get("method", "e-BH"),
        "alpha": corrected.get("alpha"),
        "n_in_family": corrected.get("n_tested", 0),  # e-value-bearing tests only
        "rejected_tools": sorted(rejected_tools),
        "n_signals_tested": len(confirm_inp.per_case),
        "n_signals_rejected": sum(1 for s in report.candidate_signals if s.reject),
        "split": confirm_label,
    }
    if not held_out:
        report.caveats.append(
            "CONFIRM split is IN-SAMPLE (batch too small to hold out); verdicts are "
            "not double-dip-protected — enlarge the batch or lower confirm_split"
        )
    if any(s.host_adjudicated and not s.fdr_corrected for s in report.candidate_signals):
        report.caveats.append(
            "marginal per-case signal verdicts are CI-based at alpha and NOT e-BH "
            "FDR-corrected across the signal family (unpaired test has no e-value)"
        )
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verdicts_by_signal(stats_results: list[Any]) -> dict[str, Any]:
    """Map ``signal name -> StatsToolResult`` from the per-case catalog tools.

    Prefers ``signal_label_assoc`` (carries a reject) over ``rank_corr``.
    """
    out: dict[str, Any] = {}
    for r in stats_results or []:
        if not getattr(r, "ok", False):
            continue
        signal = (getattr(r, "config", {}) or {}).get("signal")
        if not signal:
            continue
        if getattr(r, "tool", "") == "signal_label_assoc":
            out[signal] = r
        elif signal not in out:
            out[signal] = r
    return out


def _fused_signal(
    *,
    name: str,
    source: str,
    description: str,
    suggested_test: str,
    result: Any,
    rejected_tools: set[str],
    confirmed_on: str,
) -> FusedSignal:
    fs = FusedSignal(
        name=name,
        source=source,
        description=description,
        suggested_test=suggested_test,
        confirmed_on=confirmed_on,
    )
    if result is None:
        # Still a per-case column, but no usable test ran on CONFIRM (e.g. a
        # degenerate one-sided split). Left host_adjudicated=False, reject=None.
        return fs
    fs.host_adjudicated = True
    fs.effect = getattr(result, "effect", None)
    fs.ci = getattr(result, "ci", None)
    fs.e_value = getattr(result, "e_value", None)
    fs.underpowered = bool(getattr(result, "underpowered", False))
    fs.reject = bool(getattr(result, "reject", False))
    # e-BH only governs e-value-bearing tools; a marginal CI test is not FDR-corrected.
    fs.fdr_corrected = fs.e_value is not None and getattr(result, "tool", "") in rejected_tools
    if fs.fdr_corrected:
        fs.reject = True
    return fs


def _source_of(
    name: str,
    catalog_names: set[str],
    bridged_keys: set[str],
    explorer_named: dict[str, Any],
) -> str:
    in_bridged = name in bridged_keys
    in_catalog = name in catalog_names
    pointed = name in explorer_named
    # A bridged key is namespaced away from catalog columns, so a bridged signal is
    # purely explorer-sourced; "both" is reserved for the explorer naming a real column.
    if in_bridged:
        return "both" if in_catalog else "explorer"
    if in_catalog and pointed:
        return "both"
    return "catalog"


def _describe(
    name: str,
    origin: str,
    explorer_named: dict[str, Any],
    bridged_keys: set[str],
) -> str:
    c = explorer_named.get(origin) or explorer_named.get(name)
    if c is not None and getattr(c, "rationale", ""):
        return c.rationale
    if name in bridged_keys:
        return "bridged explorer signal"
    return "existing per-case column"


def _suggested_test(name: str, explorer_named: dict[str, Any]) -> str:
    c = explorer_named.get(name)
    return getattr(c, "suggested_test", "") if c is not None else ""
