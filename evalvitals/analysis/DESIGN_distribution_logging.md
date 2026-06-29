# DESIGN: log & analyze per-case distributions, not aggregations

**Status:** proposed (plan only — no code changed yet)
**Scope:** the explore / fused pipeline data contract, the host chart renderer,
the explorer prompt, the dashboard, and (optionally) the M2 statistics.

---

## 1. The problem (an expert's critique, confirmed)

> "The problem isn't the visualization itself — it's the **data you log** (i.e.
> *what* you visualize). Statistical analysis looks at the **distribution**, but
> you only have the **post-aggregation** result."

This is correct, and it is broader than visualization. **Aggregation happens at
the most upstream step and is destructive**: each per-case signal is collapsed to
a mean / a binned fail-rate *before it is ever logged*. By the time anything looks
at it — a chart, the M3 diagnosis agent, even the M2 statistics — the distribution
is already gone; only aggregated scalars survive.

### Evidence (from `examples/diagnosis_loops/deco_hallu/outputs/fused/`)

Every logged chart points at an **already-aggregated** CSV:

| logged chart data | shape | what's lost |
|---|---|---|
| `groupstats_*.csv` | one row per outcome: `mean, median` (2 rows) | the whole distribution — only 1–2 numbers remain |
| `failrate_by_*.csv` | `fail_rate` per bin | raw values + arbitrary bin edges (and `n=3` per bin → `fail_rate=0.667` is pure noise) |
| `class_balance.csv` | `count` per outcome | (correctly aggregated — fine) |

- **Smoking gun:** `groupstats_relative_attention_max_relative_weight.csv` has
  FAIL `mean=196` but `median=157` — heavily right-skewed / outlier-driven. A
  "mean bar" shows neither the skew nor the FAIL/PASS overlap, yet that bar is the
  evidence M3 diagnoses from.
- The only file that actually holds the distribution, `sandbox/records.json`
  (121 per-case rows), is referenced by **zero** charts — it is the explorer's
  *input*, never a logged chart/analysis artifact.
- **It is designed this way.** The explorer prompt (`explorer.py:66`) literally
  says *"PRE-AGGREGATE distributions into the CSV … never rely on a raw dump"*,
  and (`:54`) asks for *"group mean/median per outcome (grouped bar)"* as the
  headline per-signal view.
- **The statistics share the flaw.** M2 binarizes each continuous signal at its
  **median** (`stats_tools.py:323`, `binarize="median"`) and compares the two
  groups' fail-rates — itself a lossy aggregation that discards distribution shape
  before any test is run.

### Why the earlier dashboard work doesn't fix it

The `eval-chart-style` integration added live plotly violins to the dashboard by
reading `records.json` per-case. That was a **dashboard-only patch**: the *logged*
artifacts, the PNGs M3 reasons over, and the M2 statistics are all still
aggregated. The fix must move upstream, into the data layer.

---

## 2. Principle

**The per-case distribution is the source of truth. Visualization *and*
statistics both derive from it. Aggregations (means, binned fail-rates) are
derived views computed from the retained distribution — never the only thing that
survives.**

Corollary (the user's framing): *don't aggregate, then bolt a chart on afterward;
retain the distribution first, then aggregate as a presentation step.* "Visualize
before aggregation" is a consequence of this, not the root rule.

---

## 3. Changes

### §1 — Persist a per-case "tidy" table as the canonical source (keystone)

- For each numeric signal, write `tables/dist_<signal>.csv` with columns
  `case_id, <signal>, label` — **one row per case** (this *is* the distribution).
- **Write it on the host, deterministically** — do **not** rely on the agent.
  `run_fused.py` already builds `records = per_case_to_records(...)` and drops
  `records.json` into the sandbox, so the host has the per-case data in hand.
  Host-authored tidy tables are reproducible, correct, and immune to prompt drift.
- New host helper (suggested home: `operationalize.py`, or beside `charts.py`):
  ```
  write_distribution_tables(records, labels, out_dir) -> list[chart_spec]
  ```
  For each numeric signal: `dropna`, write the tidy CSV, and return a
  `kind="violin"` spec (plus optionally `kind="ecdf"`). Skip degenerate signals
  (`<3` distinct values) — reuse the existing empty-state guard rationale.

### §2 — Extend the chart-spec schema

- New distribution kinds: `violin`, `box`, `strip`, `ecdf` (keep `bar`/`line`/`scatter`).
- Distribution kinds use `value` (numeric column) + `group` (outcome column):
  ```json
  {"name": "...", "kind": "violin", "data": "tables/dist_x.csv",
   "value": "x", "group": "label", "title": "..."}
  ```
- Backward compatible: `bar`/`line`/`scatter` keep using `x`/`y`; the renderer
  dispatches on `kind`.

### §3 — Host renderer (`charts.py`)

Today `_KINDS = {bar, line, scatter, timeseries}`, all single-series.

- Add `violin`/`box`/`strip`/`ecdf` to `_KINDS`.
- `_can_render`: for distribution kinds require `value` + `group` columns (instead
  of `x`/`y`).
- `_render_one`: new branches that group rows by `group` and render the grouped
  distribution with matplotlib (`ax.violinplot` / `boxplot` / ECDF step), overlay
  jittered per-case points + a median line, colored by the **eval-chart-style
  semantic palette** (FAIL red / PASS slate — already wired via `_load_chart_style`).
- **Determinism:** jitter must be derived from the **row index** (a fixed offset),
  never an RNG, to keep byte-identical PNGs and the existing determinism test.
- Result: the **logged PNG that M3 sees becomes a real distribution**, from the
  same tidy CSV the dashboard's plotly violin reads.

### §4 — Explorer prompt (`explorer.py` `_GENERATE_PROMPT`, ~lines 54-67)

- **Remove** the *"PRE-AGGREGATE distributions into the CSV … never rely on a raw
  dump"* instruction and the *"group mean/median per outcome (grouped bar)"* as
  the headline per-signal view.
- New guidance: the **distribution is the primary per-signal view** (violin / box
  / ECDF from a per-case tidy table); mean/median bars are demoted to optional
  secondary; class-balance stays a count; binned fail-rate becomes optional (or is
  replaced by a logistic view).
- Because the host now guarantees the distribution battery (§1), the prompt's job
  narrows to observations / candidate_signals / recipes — it no longer needs to
  (and should not) emit misleading mean bars as the headline.

### §5 — Merge & dashboard wiring

- `run_fused.py` / `explore_run.py`: merge the host distribution specs into
  `report.charts` **before** `render_chart_specs`, so the logged report carries
  distribution charts sourced from per-case data regardless of agent behavior.
- `dashboard_app.py`: `_render_chart_card` already prefers `figure_path` (now a
  distribution PNG). Keep the plotly per-case section but point it at the same
  `dist_<signal>.csv`, and de-dup against the logged PNGs (suggestion: plotly =
  interactive on-screen view; logged PNG = portable, what M3 saw).

### §6 — Distribution-aware statistics (only in the "full chain" scope)

The median-split (`stats_tools.py:323`, `binarize="median"`) discards shape.

- Add a **non-binarizing** effect computed on the raw per-case values:
  **Cliff's δ / rank-biserial** (preferred — nonparametric, skew-robust), optionally
  Mann–Whitney U and/or the KS statistic.
- Feed M3 a **quantile summary** (`min / Q1 / median / Q3 / max / n`) instead of a
  single mean (the `summary` built at `diagnosis.py:517`).
- **Impact:** changes M2's effect numbers and the e-BH family inputs → must update
  `log_schema.py`, re-render `run_log.schema.json` (the builder is the
  source-of-truth, jsonschema-gated test), update tests, and **re-run M2**.

---

## 4. What needs re-running

- After §1–§5: re-run **`run_fused.py`** (Step 1; no GPU, reuses frozen M1) so the
  current outputs regenerate the tidy tables + distribution PNGs in
  `fused_report.json`. The dashboard then shows distributions from the logged source.
- After §6: also re-run **`run_m2-5.py`** (Step 2) so M2's distribution-aware
  statistics and the M3 inputs regenerate.

## 5. Tests

- `charts.py`: render `violin`/`box`/`ecdf` from a tidy CSV; determinism with
  index-derived jitter; `_can_render` honoring `value`/`group`.
- `write_distribution_tables`: tidy shape, `dropna`, sparse signals (attention is
  present on only 20/121 cases here).
- `test_explorer_battery`: update assertions (no longer expecting pre-aggregated
  mean tables as the headline).
- §6: tests for Cliff's δ / MWU / quantile summaries; update the e-BH family tests;
  re-render `run_log.schema.json` + the jsonschema gate.

## 6. Risks / boundaries

- **Determinism:** violin/strip jitter must be deterministic (index-derived) or it
  breaks byte-identical PNGs and the determinism test.
- **Sparse signals:** attention present on 20/121 cases; tidy tables `dropna` →
  small `n` per chart; reuse the `<3`-distinct empty-state guard so degenerate
  signals are skipped, not drawn as misleading spikes.
- **No leakage:** distribution charts remain descriptive observations → still
  **M3-only / UNCONFIRMED**, same double-blind guardrail; never name a mechanism.
- **Relationship to `eval-chart-style`:** complementary. Its plotly distribution
  builders + semantic palette are exactly the §3/§5 renderers; §6's Cliff's δ aligns
  with its "one primary effect metric / statistical honesty" guidance. See
  `skills/eval-chart-style/SKILL.md` and `DESIGN_m3_charts.md`.

## 7. Scope options

- **Data + visualization layer** (§1–§5): per-case tidy tables become the logged
  source; the explorer stops pre-aggregating; the host renderer + dashboard draw
  real distributions. M3 then reasons over distribution charts. M2 untouched.
  Lower risk, self-contained.
- **Full chain** (§1–§6): the above **plus** distribution-aware M2 statistics and
  quantile summaries to M3. Most faithful to "statistics looks at the
  distribution," but touches e-BH / schema / tests and needs a Step-2 re-run.
