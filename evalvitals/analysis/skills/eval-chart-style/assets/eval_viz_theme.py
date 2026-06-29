"""
eval_viz_theme.py
=================
Drop-in styling + chart layer for FAIL-vs-PASS LLM eval analysis.

Goal: one import fixes the three recurring problems —
  (1) inconsistent color / font / truncated names  -> global template + semantic
      palette + short-name registry
  (2) overflow clipping + misallocated space        -> sizing helpers (full-row vs
      compact) and chart builders that pick distribution plots over two-bar plots
  (3) inconsistent number / bin formatting           -> fmt() and human_bins()

Targets plotly (matches a Streamlit ``st.plotly_chart`` pipeline). A matplotlib
rcParams equivalent is at the bottom for non-plotly callers.

This module is CASE-AGNOSTIC: it ships no domain-specific signal names. Callers
register display aliases for their own columns at runtime with
``register_short_names({...})``; everything else falls back to a deterministic
abbreviation. Heavy deps (plotly, numpy) are imported lazily inside the builders,
so importing this module to read its palette / rcParams never requires plotly.

IMPORTANT (data shape): the distribution builders (``violin_by_outcome``,
``logistic_failrate``, ``joint_scatter``, ``counts_bar``) need a PER-CASE table —
one row per case with a raw signal column and an outcome label. Do NOT feed them
pre-aggregated tables (mean-by-outcome, fail-rate-by-bin, class-count); those
have no per-case column and the builders will return an empty-state figure.

Usage
-----
    import eval_viz_theme as viz
    viz.apply()                       # register + set default template once, at startup
    viz.register_short_names({"my_long_metric_name": "my.metric"})  # optional

    fig = viz.violin_by_outcome(df, signal="my_long_metric_name", outcome="label")
    st.plotly_chart(fig, use_container_width=True)

Every builder returns a plotly Figure. Nothing renders on its own.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# 1. SEMANTIC PALETTE  — color encodes ROLE, never decoration.
#    Reuse these names everywhere; never hardcode a hex in a chart.
# ---------------------------------------------------------------------------
PALETTE = {
    "FAIL":          "#C0413B",   # warning hue — the failing group, every chart
    "PASS":          "#5B7A99",   # neutral slate — the passing group, every chart
    "SIGNIFICANT":   "#2E8B6F",   # green — survived FDR / REJECT H0
    "INCONCLUSIVE":  "#B8BCC2",   # grey — did not survive
    "ACCENT":        "#3A6EA5",   # single-series / neutral measurement
    "LEAKY":         "#9AA0A6",   # greyed-out: target-leaking signal, do not elevate
    "GRID":          "#E6E8EB",
    "AXIS":          "#5A5F66",
    "TEXT":          "#2B2F33",
    "BAND":          "rgba(91,122,153,0.12)",  # CI / reference band fill
}

FONT = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def outcome_color(v) -> str:
    """Role color for an outcome value. Normalizes via str().upper() so 'fail',
    'FAIL', 1, True all map to the FAIL hue (and pass/0/False to PASS)."""
    key = str(v).strip().upper()
    if key in ("FAIL", "1", "TRUE"):
        return PALETTE["FAIL"]
    if key in ("PASS", "0", "FALSE"):
        return PALETTE["PASS"]
    return PALETTE["ACCENT"]


# ---------------------------------------------------------------------------
# 2. SHORT-NAME REGISTRY  — never print a raw signal column on an axis/header.
#    short() returns the display alias; full() / tooltip kept for hover.
#    Ships EMPTY (case-agnostic); register aliases for your columns at runtime.
# ---------------------------------------------------------------------------
SHORT_NAMES: dict[str, str] = {}


def register_short_names(mapping: dict[str, str]) -> None:
    """Register display aliases for raw column names (e.g. {'long_col':'short'})."""
    SHORT_NAMES.update({str(k): str(v) for k, v in (mapping or {}).items()})


def short(name: str) -> str:
    """Display alias. Falls back to a deterministic abbreviation."""
    if name in SHORT_NAMES:
        return SHORT_NAMES[name]
    parts = str(name).replace("-", "_").split("_")
    if len(parts) <= 2:
        return str(name)
    return ".".join(p[:4] for p in parts[:3])


def full(name: str) -> str:
    return str(name)


# ---------------------------------------------------------------------------
# 3. NUMBER + BIN FORMATTING  — one global precision policy, by quantity kind.
# ---------------------------------------------------------------------------
def fmt(x, kind: str = "effect") -> str:
    """
    kind: 'effect'|'corr'  -> 2 dp     (e.g. 0.87)
          'stat'|'p'       -> 3 dp     (e.g. 0.082)
          'pct'            -> integer % (e.g. 41%)
          'count'          -> integer
          'val'            -> adaptive: 2 sig-ish, trims long floats
    Small-n data cannot support 4 dp — never invent precision the sample lacks.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if kind in ("effect", "corr"):
        return f"{x:.2f}"
    if kind in ("stat", "p"):
        return f"{x:.3f}"
    if kind == "pct":
        # values in [0,1] are fractions; values >1 are already percents
        return f"{round(x * 100)}%" if -1.0 <= x <= 1.0 else f"{round(x)}%"
    if kind == "count":
        return f"{int(round(x))}"
    # 'val'
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def human_bins(edges) -> list[str]:
    """
    Turn pandas.cut float edges into readable labels.
    [10, 50, 200] -> ['10–50', '50–200']
    Picks decimals from the smallest inter-edge gap so close edges stay distinct.
    """
    edges = list(edges)
    if len(edges) < 2:
        return []
    gaps = [b - a for a, b in zip(edges[:-1], edges[1:]) if b > a]
    gap = min(gaps) if gaps else (max(edges) - min(edges))
    dec = 0 if gap >= 10 else (1 if gap >= 1 else 2)
    f = lambda v: f"{v:.{dec}f}"  # noqa: E731
    return [f"{f(a)}–{f(b)}" for a, b in zip(edges[:-1], edges[1:])]


# ---------------------------------------------------------------------------
# 4. GLOBAL TEMPLATE  — register once; every chart inherits font, grid, margins.
# ---------------------------------------------------------------------------
def apply(set_default: bool = True) -> str:
    import plotly.graph_objects as go
    import plotly.io as pio

    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        font=dict(family=FONT, size=13, color=PALETTE["TEXT"]),
        title=dict(font=dict(size=15), x=0.0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[PALETTE["ACCENT"], PALETTE["FAIL"], PALETTE["PASS"],
                  PALETTE["SIGNIFICANT"], PALETTE["INCONCLUSIVE"]],
        margin=dict(l=64, r=24, t=44, b=56),
        xaxis=dict(showgrid=False, zeroline=False, ticks="outside",
                   linecolor=PALETTE["AXIS"], tickcolor=PALETTE["AXIS"],
                   tickfont=dict(size=12), automargin=True),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["GRID"], zeroline=False,
                   linecolor="rgba(0,0,0,0)", tickfont=dict(size=12),
                   automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0.0, font=dict(size=12)),
    )
    pio.templates["evalvitals"] = tmpl
    if set_default:
        pio.templates.default = "evalvitals"
    return "evalvitals"


# ---------------------------------------------------------------------------
# 5. SIZING  — kills overflow clipping and the value/space inversion.
#    Distribution + scatter get a full row; counts get compact.
# ---------------------------------------------------------------------------
SIZES = {
    "full":    dict(height=420, width=None),   # use_container_width=True
    "wide":    dict(height=380, width=None),   # scatter / joint
    "compact": dict(height=300, width=None),   # counts, single comparison
    "strip":   dict(height=300, width=None),   # small-multiples cell
}


def _size(fig, kind: str):
    s = SIZES.get(kind, SIZES["full"])
    fig.update_layout(height=s["height"], autosize=True)
    return fig


def _empty_fig(message: str):
    """A graceful empty-state figure (used when a column is missing or the data
    is too thin to plot) — never raises, never renders a misleading chart."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(size=12, color=PALETTE["AXIS"]))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return _size(fig, "compact")


def _has_cols(df, *cols) -> bool:
    return all(c is not None and c in df.columns for c in cols)


# ---------------------------------------------------------------------------
# 6. CHART BUILDERS  — each one encodes a "use this, not a bar" decision.
# ---------------------------------------------------------------------------

def violin_by_outcome(df, signal, outcome="label", points="all"):
    """
    REPLACES the two-bar 'mean by outcome'.
    Shows the full distribution + every point — so bimodality and
    outlier-driven means are visible instead of hidden. Use for any
    continuous signal compared across FAIL/PASS. Needs PER-CASE rows.
    """
    import plotly.graph_objects as go

    if not _has_cols(df, signal, outcome):
        return _empty_fig(f"no per-case column '{signal}'")
    up = df[outcome].astype(str).str.upper()
    fig = go.Figure()
    for grp in ["FAIL", "PASS"]:
        sub = df[up == grp]
        if sub.empty or sub[signal].dropna().empty:
            continue
        fig.add_trace(go.Violin(
            y=sub[signal], name=grp, line_color=outcome_color(grp),
            fillcolor=outcome_color(grp), opacity=0.55, meanline_visible=True,
            points=points, jitter=0.35, marker=dict(size=4, opacity=0.6),
            box_visible=True, spanmode="hard",
        ))
    if not fig.data:
        return _empty_fig(f"no non-null '{short(signal)}' values by outcome")
    fig.update_layout(title=f"{short(signal)} by outcome  ·  full {full(signal)}",
                      yaxis_title=short(signal), xaxis_title="", showlegend=False,
                      violingap=0.25)
    return _size(fig, "full")


def forest_effects(rows, label_key="signal", effect_key="effect",
                   lo_key="ci_lo", hi_key="ci_hi", sig_key="significant",
                   leaky_key="leaky"):
    """
    REPLACES the green/grey effect-size bar chart.
    Horizontal dot + CI (forest style): keeps the REJECT-H0 color coding AND
    shows the confidence interval the bar chart dropped. Greys out any signal
    flagged leaky so a target-leaking signal is never visually #1.
    rows: list of dicts. ci_lo/ci_hi/leaky optional. Unknown significance is
    treated as INCONCLUSIVE (grey), not asserted as a rejection.
    """
    import plotly.graph_objects as go

    rows = [r for r in (rows or []) if isinstance(r.get(effect_key), (int, float))]
    if not rows:
        return _empty_fig("no numeric effect sizes to plot")
    rows = sorted(rows, key=lambda r: r.get(effect_key, 0))
    fig = go.Figure()
    for r in rows:
        leaky = r.get(leaky_key, False)
        if leaky:
            color = PALETTE["LEAKY"]
        elif r.get(sig_key, False):          # unknown -> inconclusive (honest default)
            color = PALETTE["SIGNIFICANT"]
        else:
            color = PALETTE["INCONCLUSIVE"]
        y = short(r.get(label_key, "?")) + ("  (leaky)" if leaky else "")
        lo, hi = r.get(lo_key), r.get(hi_key)
        if lo is not None and hi is not None:
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[y, y], mode="lines",
                line=dict(color=color, width=3), showlegend=False,
                hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[r[effect_key]], y=[y], mode="markers+text",
            marker=dict(color=color, size=11),
            text=[fmt(r[effect_key], "effect")], textposition="middle right",
            textfont=dict(size=11), showlegend=False,
            hovertext=full(r.get(label_key, "?")), hoverinfo="text"))
    fig.add_vline(x=0, line=dict(color=PALETTE["AXIS"], width=1, dash="dot"))
    fig.update_layout(title="Effect size (failure association) — green = REJECT H₀",
                      xaxis_title="effect size", yaxis_title="")
    return _size(fig, "full")


def logistic_failrate(df, signal, outcome="label", n_grid=80, band=True):
    """
    REPLACES 'Fail rate by <binned signal>' line charts.
    Keeps the x-axis continuous and fits a logistic fail-rate curve with a
    sample-density rug — no arbitrary bin edges, no fake monotone line.
    Needs PER-CASE rows. Returns an empty-state figure when the fit is undefined.
    """
    import numpy as np
    import plotly.graph_objects as go

    if not _has_cols(df, signal, outcome):
        return _empty_fig(f"no per-case column '{signal}'")
    try:
        x = df[signal].to_numpy(dtype=float)
    except (ValueError, TypeError):
        return _empty_fig(f"'{short(signal)}' is not numeric")
    y = (df[outcome].astype(str).str.upper() == "FAIL").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.unique(y[finite]).size < 2 or np.unique(x[finite]).size < 2:
        return _empty_fig(f"insufficient data for a logistic fit of {short(signal)}")
    b0, b1 = _fit_logistic(x[finite], y[finite])
    gx = np.linspace(np.nanmin(x[finite]), np.nanmax(x[finite]), n_grid)
    gy = 1 / (1 + np.exp(-(b0 + b1 * gx)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines",
                             line=dict(color=PALETTE["FAIL"], width=2.5),
                             name="P(FAIL)"))
    fig.add_trace(go.Scatter(x=x[finite], y=np.full(int(finite.sum()), -0.04),
                             mode="markers",
                             marker=dict(color=PALETTE["AXIS"], size=4,
                                         opacity=0.4, symbol="line-ns-open"),
                             name="samples", hoverinfo="x"))
    fig.update_layout(title=f"Fail rate vs {short(signal)} (logistic fit)",
                      xaxis_title=short(signal), yaxis_title="P(FAIL)",
                      yaxis=dict(range=[-0.08, 1.05]), showlegend=False)
    return _size(fig, "full")


def joint_scatter(df, x_sig, y_sig, outcome="label"):
    """
    For two continuous signals: scatter colored by outcome WITH marginal
    distributions. Give it a full row, not a narrow column. Needs PER-CASE rows.
    """
    import plotly.express as px

    if not _has_cols(df, x_sig, y_sig, outcome):
        return _empty_fig(f"missing column(s) for scatter of {short(x_sig)} vs {short(y_sig)}")
    d = df[[x_sig, y_sig, outcome]].dropna().copy()
    if d.empty:
        return _empty_fig(f"no rows with both {short(x_sig)} and {short(y_sig)}")
    d["_grp"] = d[outcome].astype(str).str.upper()
    fig = px.scatter(d, x=x_sig, y=y_sig, color="_grp",
                     color_discrete_map={"FAIL": PALETTE["FAIL"], "PASS": PALETTE["PASS"]},
                     marginal_x="histogram", marginal_y="histogram",
                     labels={x_sig: short(x_sig), y_sig: short(y_sig), "_grp": "outcome"})
    fig.update_traces(marker=dict(size=7, opacity=0.75), selector=dict(type="scatter"))
    fig.update_traces(marker=dict(size=7, opacity=0.75), selector=dict(type="scattergl"))
    fig.update_layout(title=f"{short(x_sig)} vs {short(y_sig)} by outcome",
                      legend_title_text="")
    return _size(fig, "wide")


def counts_bar(df, outcome="label"):
    """
    The ONE place a bar chart is correct: discrete class counts.
    Bars encode counts (the 'from zero' baseline is meaningful here). Accepts a
    per-case table (counts via value_counts) OR a pre-aggregated one-row-per-
    outcome table with a count/n column.
    """
    import plotly.graph_objects as go

    if outcome not in df.columns:
        return _empty_fig(f"no '{outcome}' column for class balance")
    up = df[outcome].astype(str).str.upper()
    count_col = next((c for c in ("count", "n", "freq") if c in df.columns), None)
    if count_col is not None and up.nunique() == len(df):
        # already aggregated: one row per outcome with an explicit count column
        counts = {k: float(v) for k, v in zip(up, df[count_col])}
    else:
        counts = {str(k): float(v) for k, v in up.value_counts().items()}
    order = [g for g in ("FAIL", "PASS") if g in counts] or list(counts)
    fig = go.Figure(go.Bar(
        x=order, y=[counts[g] for g in order],
        marker_color=[outcome_color(g) for g in order],
        text=[fmt(counts[g], "count") for g in order], textposition="outside"))
    fig.update_layout(title="Class balance", xaxis_title="", yaxis_title="count")
    return _size(fig, "compact")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fit_logistic(x, y, iters=200, lr=0.3, l2=1e-3):
    """Tiny L2-regularized gradient logistic fit; standardizes x for stability.
    The small L2 term tames the (near-)separable small-n regime so the curve is
    not an artifact of iteration count."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sd = x.mean(), (x.std() or 1.0)
    xs = (x - mu) / sd
    b0, b1 = 0.0, 0.0
    for _ in range(iters):
        z = np.clip(b0 + b1 * xs, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g0 = np.mean(p - y)
        g1 = np.mean((p - y) * xs) + l2 * b1
        b0 -= lr * g0
        b1 -= lr * g1
    # unstandardize back to raw-x coefficients
    return b0 - b1 * mu / sd, b1 / sd


# ---------------------------------------------------------------------------
# matplotlib equivalent (used by the host static-PNG renderer; no plotly needed)
# ---------------------------------------------------------------------------
def matplotlib_rcparams() -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.edgecolor": PALETTE["AXIS"],
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": PALETTE["GRID"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.titlesize": 14, "axes.titlelocation": "left",
        "legend.frameon": False,
    }
