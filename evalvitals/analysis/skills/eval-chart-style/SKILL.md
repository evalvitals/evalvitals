---
name: eval-chart-style
version: 0.1.0
description: >
  Plotly chart-type + house-style standard for FAIL-vs-PASS LLM evaluation
  analysis rendered in an interactive Streamlit/plotly EDA dashboard. Use when
  building or styling the EvalVitals diagnostic dashboard, exploratory
  FAIL-vs-PASS plots, effect-size / fail-rate / signal-distribution charts, or
  any "plot this eval signal" request where the point is to NOT default to a bar
  chart. Scope: interactive plotly + Streamlit EDA only. For static
  publication / manuscript figures (SVG/PDF/TIFF, journal style) defer to the
  nature-figure skill instead — the two do not both apply to one request.
---

# Eval Chart Style

A house style + chart-type policy for FAIL-vs-PASS eval analysis in an
**interactive plotly / Streamlit** dashboard. It exists to stop three recurring
failures: inconsistent color/font/truncated names, charts clipped or starved of
space, and ragged number/bin formatting — and to stop the default reflex of
plotting a **mean as a bar** when the analysis depends on the *distribution*.

A drop-in implementation lives in `assets/eval_viz_theme.py`. Prefer calling its
builders over hand-rolling figures: they bake every rule below in. To import it,
add the skill's `assets/` dir to `sys.path` (it is not an installed package):

```python
import sys; sys.path.insert(0, "<this-skill>/assets")
import eval_viz_theme as viz
viz.apply()                 # once at startup — registers template + palette
viz.register_short_names({"my_long_metric_name": "my.metric"})   # optional aliases
fig = viz.violin_by_outcome(df, signal="my_long_metric_name", outcome="label")
st.plotly_chart(fig, use_container_width=True)
```

**Scope / when NOT to use.** This skill is for interactive plotly charts in a
Streamlit EDA/diagnostic dashboard. It is NOT for static manuscript/journal
figures (SVG/PDF/TIFF, Nature-style) — use the **nature-figure** skill for those.
The two cover disjoint domains; do not apply both to the same chart.

**Data shape (important).** The distribution builders (`violin_by_outcome`,
`logistic_failrate`, `joint_scatter`, `counts_bar`) need a **per-case** table —
one row per case with a raw signal column and an outcome label. Do NOT feed them
pre-aggregated tables (mean-by-outcome, fail-rate-by-bin, class-count); those have
no per-case column and the builder returns a graceful empty-state figure.

---

## 0. Chart-type policy (the most important rule)

**A bar's filled area means "amount accumulated from zero." Only use it when that
is literally true — i.e. counts.** For a mean, a proportion, a measurement, or an
effect size, a bar hides the distribution and fakes certainty. Pick the chart by
*what the reader must see*, using this table:

| What you're showing | DON'T | DO | Builder |
|---|---|---|---|
| Discrete class counts (FAIL/PASS n) | — | grouped/colored bars | `counts_bar` |
| A continuous signal compared across outcomes | two-bar "mean by outcome" | violin + box + jittered points | `violin_by_outcome` |
| Multiple signals' effect sizes ranked | green/grey bars | horizontal **dot + CI** (forest) | `forest_effects` |
| Fail rate vs a continuous signal | bin → connect with a line | logistic curve + density rug | `logistic_failrate` |
| Two continuous signals jointly | scatter squeezed in a column | scatter colored by outcome + marginals | `joint_scatter` |

Why distributions, specifically: with small FAIL n, a mean is often
outlier-driven, and claims about multiple sub-populations are *bimodality claims*.
A bar can show neither. A violin or strip makes both visible — it turns an
assertion in the text into something the reader can see.

**Suppress degenerate charts.** If a "signal" is binary or constant, a fail-rate-
by-bin plot collapses to one dot. Don't render it; emit a one-line note instead.
Treat any signal whose fail rate splits 1.00 / 0.00 as **target leakage** — §4.

---

## 1. Theme + semantic palette + short names  (fixes color/font/truncation)

Call `viz.apply()` once. Never set a font or hardcode a hex in an individual
chart — pull from the palette so the same role looks the same in every chart.

**Color encodes role, not decoration.** Lock these and reuse everywhere:

- FAIL → warning red `#C0413B`; PASS → neutral slate `#5B7A99` — in *every* chart,
  same side, same hue. The reader must never re-learn who is who.
- REJECT H₀ / survived FDR → green `#2E8B6F`; inconclusive → grey `#B8BCC2`.
- Leaky signal → greyed `#9AA0A6` (never a "winner" color).
- One neutral accent for single-series measurements. Nothing else.

**Never print a raw column name on an axis, header, or tick.** Long names get
truncated and become ambiguous. Register aliases for your columns with
`viz.register_short_names({...})` and use `viz.short(name)`; put the full name in
the title suffix or hover, never the axis. Keep aliases ≤10 chars. The registry
ships empty (case-agnostic) and falls back to a deterministic abbreviation.

---

## 2. Sizing + layout  (fixes overflow clipping and the value–space inversion)

The space a chart gets must match its information value.

- **Distribution plots and scatters get a full row.** Use
  `st.plotly_chart(fig, use_container_width=True)` and let the builder set height
  (violin/scatter → 380–420px). Never put a scatter in a narrow column.
- **Compress low-information comparisons.** A FAIL-vs-PASS comparison on one
  signal does not deserve a full panel ×8. Collapse into one standardized strip,
  or give each the compact 300px size.
- **Fix container overflow at the source.** Always `use_container_width=True`;
  the template sets `automargin=True` on both axes; never hardcode a width wider
  than the column.
- **One title per chart, not two.** Don't repeat the section header inside the
  figure.

---

## 3. Number + bin formatting  (one global precision policy)

Use `viz.fmt(x, kind)` everywhere — tables, labels, hovers, cards:

- `effect` / `corr` → 2 dp (`0.87`)
- `stat` / `p` → 3 dp (`0.082`)
- `pct` → integer percent (`41%`)
- `count` → integer
- `val` → adaptive (`196.6 → 197`, `2.135 → 2.13`)

**Never display raw `pandas.cut` edges.** `(113.844, 233.066]` is machine output.
Use `viz.human_bins(edges) → "114–233"`. Always annotate the zero / reference line
on effect-size axes (the builders do) so the reader can judge magnitude.

---

## 4. Statistical honesty (do not skip — it's what makes the report trusted)

- **Show the statistic the method controls on.** If the method is e-BH, the
  e-value column must contain numbers, not `None`.
- **Detect and demote target leakage.** A signal that *is* the outcome re-measured
  (fail rate 1.00 / 0.00; effect ≈ perfect) must not rank #1. Pass `leaky=True` to
  `forest_effects` so it greys out and drops to a "confirmatory, not causal" slot.
- **Unknown significance defaults to inconclusive (grey), never green.** Don't
  assert a rejection on faith.
- **Pick one primary effect metric** for the headline charts; relegate the rest to
  a details table and label every axis with which metric it is.

---

## Default workflow for any eval-result plotting task

1. `import eval_viz_theme as viz; viz.apply()` once; register short names for your
   columns.
2. For each thing to show, consult the §0 table — never default to a bar.
3. Drop binary/constant signals; flag leaky ones (§4).
4. Call the matching builder on a PER-CASE table; render full-width for
   distributions/scatters, compact for counts.
5. Format every number with `viz.fmt`, every bin with `viz.human_bins`, every name
   with `viz.short`.
6. One title per chart; one home per fact.
