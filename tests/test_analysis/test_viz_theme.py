"""The vendored eval-chart-style theme: the host loader + the plotly builders.

Plotly is the dashboard extra; guard on it. The loader itself never needs plotly
(the asset imports it lazily), which `charts.py` relies on under [viz]-only.
"""

from __future__ import annotations

import pytest

from evalvitals.analysis.viz_theme import load_viz_theme


def test_loader_finds_asset_and_palette_without_plotly():
    viz = load_viz_theme()
    assert viz is not None
    # palette + matplotlib rcParams are accessible with no plotly import
    assert viz.PALETTE["FAIL"] == "#C0413B"
    assert viz.PALETTE["PASS"] == "#5B7A99"
    rc = viz.matplotlib_rcparams()
    assert rc["axes.spines.top"] is False
    assert "DejaVu Sans" in rc["font.sans-serif"]  # server-safe fallback


def test_short_name_registry_is_case_agnostic():
    viz = load_viz_theme()
    # ships empty -> deterministic abbreviation, not a hardcoded case alias
    assert viz.short("a_made_up_long_signal_name") == "a.made.up"
    viz.register_short_names({"a_made_up_long_signal_name": "made.up"})
    assert viz.short("a_made_up_long_signal_name") == "made.up"
    viz.SHORT_NAMES.clear()  # don't leak into other tests


def test_outcome_color_normalizes_label_forms():
    viz = load_viz_theme()
    assert viz.outcome_color("fail") == viz.outcome_color("FAIL") == viz.PALETTE["FAIL"]
    assert viz.outcome_color(0) == viz.outcome_color(False) == viz.PALETTE["PASS"]
    assert viz.outcome_color("weird") == viz.PALETTE["ACCENT"]


def test_fmt_and_human_bins():
    viz = load_viz_theme()
    assert viz.fmt(0.4567, "effect") == "0.46"
    assert viz.fmt(0.41, "pct") == "41%"
    assert viz.fmt(None) == "—"
    assert viz.human_bins([10, 50, 200]) == ["10–50", "50–200"]


# -- plotly builders (dashboard extra) --------------------------------------
pytest.importorskip("plotly")
pd = pytest.importorskip("pandas")


def _records():
    # 6 fail / 4 pass, one continuous signal + one binary signal
    rows = []
    for i in range(6):
        rows.append({"case_id": f"f{i}", "label": "fail", "sig": 0.8 + i * 0.05, "bin": 1})
    for i in range(4):
        rows.append({"case_id": f"p{i}", "label": "pass", "sig": 0.1 + i * 0.05, "bin": 0})
    return pd.DataFrame(rows)


def test_counts_bar_per_case_and_preaggregated():
    viz = load_viz_theme()
    df = _records()
    fig = viz.counts_bar(df, outcome="label")
    ys = list(fig.data[0].y)
    assert ys == [6.0, 4.0]                      # FAIL, PASS (value_counts)
    # pre-aggregated table with an explicit count column must NOT be re-counted
    agg = pd.DataFrame({"label": ["fail", "pass"], "count": [25, 96]})
    ys2 = list(viz.counts_bar(agg, outcome="label").data[0].y)
    assert ys2 == [25.0, 96.0]


def test_violin_logistic_and_missing_column_empty_state():
    viz = load_viz_theme()
    df = _records()
    v = viz.violin_by_outcome(df, "sig", outcome="label")
    assert len(v.data) == 2                       # FAIL + PASS violins
    lg = viz.logistic_failrate(df, "sig", outcome="label")
    assert len(lg.data) == 2                       # curve + density rug
    empty = viz.violin_by_outcome(df, "nope", outcome="label")
    assert len(empty.data) == 0                    # missing col -> empty state


def test_forest_unknown_significance_is_grey_not_green():
    viz = load_viz_theme()
    # significance absent -> inconclusive (grey), never asserted as a rejection
    fig = viz.forest_effects([{"signal": "s1", "effect": 0.5}])
    marker_colors = {tr.marker.color for tr in fig.data if tr.mode and "markers" in tr.mode}
    assert viz.PALETTE["INCONCLUSIVE"] in marker_colors
    assert viz.PALETTE["SIGNIFICANT"] not in marker_colors
    # explicit significant -> green
    fig2 = viz.forest_effects([{"signal": "s1", "effect": 0.5, "significant": True}])
    assert any(tr.marker.color == viz.PALETTE["SIGNIFICANT"]
               for tr in fig2.data if tr.mode and "markers" in tr.mode)


def test_logistic_insufficient_data_returns_empty_state():
    viz = load_viz_theme()
    one_class = pd.DataFrame({"label": ["fail"] * 5, "sig": [0.1, 0.2, 0.3, 0.4, 0.5]})
    fig = viz.logistic_failrate(one_class, "sig", outcome="label")
    assert len(fig.data) == 0                       # single-class y -> no fit


def test_logistic_non_numeric_column_degrades_gracefully():
    # builders promise "never raise" — a present-but-string column -> empty state
    viz = load_viz_theme()
    df = pd.DataFrame({"label": ["fail", "pass", "fail", "pass"],
                       "sig": ["a", "b", "c", "d"]})
    fig = viz.logistic_failrate(df, "sig", outcome="label")
    assert len(fig.data) == 0


def test_forest_effects_tolerates_row_without_label():
    # a numeric-effect row missing its label_key must not KeyError
    viz = load_viz_theme()
    fig = viz.forest_effects([{"effect": 0.5}])
    assert len(fig.data) >= 1                        # renders with a '?' label
