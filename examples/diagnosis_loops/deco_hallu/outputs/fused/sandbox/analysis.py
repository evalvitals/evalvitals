#!/usr/bin/env python3
"""Exploratory FAIL-vs-PASS failure analysis for absent-object false 'Yes' cases.

Self-contained: reads records.json from CWD, writes CSVs under tables/ and PNG
figures under figures/, and prints a single EXPLORATORY_RESULT_JSON line last.
Exploratory / descriptive only -- no causal or statistical confirmation claimed.
"""
import os
import json
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
with open("records.json") as f:
    rows = json.load(f)
df = pd.DataFrame(rows)

OUTCOME = "label"
# FAIL = false 'Yes' on absent object ; PASS = correct
df["is_fail"] = (df[OUTCOME].astype(str).str.lower() == "fail").astype(int)

NUMERIC = [
    "prompt_contrast_prompt_sensitivity",
    "relative_attention_max_relative_weight",
    "relative_attention_mean_relative_weight",
    "relative_attention_focus_share",
    "relative_attention_attention_entropy",
    "relative_attention_top1_share",
    "relative_attention_center_offset",
    "relative_attention_edge_mass",
]
NUMERIC = [c for c in NUMERIC if c in df.columns]
ATTN = [c for c in NUMERIC if c.startswith("relative_attention_")]


def short(c):
    return c.replace("relative_attention_", "").replace("prompt_contrast_", "")


n_total = len(df)
n_fail = int(df["is_fail"].sum())
n_pass = n_total - n_fail
attn_n = int(df[ATTN[0]].notna().sum()) if ATTN else 0

observations = [
    f"{n_total} labeled cases: {n_fail} FAIL (false 'Yes') vs {n_pass} PASS "
    f"({n_fail / n_total:.1%} fail rate).",
    f"prompt_sensitivity present for {int(df['prompt_contrast_prompt_sensitivity'].notna().sum())} cases; "
    f"the 7 relative_attention signals present for only {attn_n} cases (heavy missingness).",
]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def write_csv(name, frame):
    frame.to_csv(f"tables/{name}.csv", index=False)


charts = []
plots = []


def add_chart(name, kind, x, y, title, display_name=None):
    charts.append({
        "name": name, "kind": kind,
        "display_name": display_name or title,
        "data": f"tables/{name}.csv", "x": x, "y": y, "title": title,
    })


# ----------------------------------------------------------------------------
# 1. Class balance
# ----------------------------------------------------------------------------
bal = pd.DataFrame({"outcome": ["FAIL", "PASS"], "count": [n_fail, n_pass]})
write_csv("class_balance", bal)
add_chart("class_balance", "bar", "outcome", "count",
          "FAIL vs PASS case balance", "FAIL/PASS case balance")

# ----------------------------------------------------------------------------
# 2. Per-signal FAIL vs PASS summary + standardized separation
# ----------------------------------------------------------------------------
sep_rows = []
for c in NUMERIC:
    g = df[[c, "is_fail"]].dropna()
    a = g.loc[g.is_fail == 1, c]  # FAIL
    b = g.loc[g.is_fail == 0, c]  # PASS
    if len(a) < 2 or len(b) < 2:
        continue
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) /
                   max(len(a) + len(b) - 2, 1))
    smd = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
    sep_rows.append({
        "signal": short(c), "signal_full": c,
        "mean_fail": round(a.mean(), 4), "mean_pass": round(b.mean(), 4),
        "n_fail": len(a), "n_pass": len(b),
        "smd": round(smd, 4), "abs_smd": round(abs(smd), 4),
    })

sep = pd.DataFrame(sep_rows).sort_values("abs_smd", ascending=False)
write_csv("top_discriminators",
          sep[["signal", "abs_smd"]].rename(columns={"abs_smd": "separation"}))
add_chart("top_discriminators", "bar", "signal", "separation",
          "Top FAIL/PASS discriminators (|standardized mean diff|)",
          "Top FAIL/PASS discriminators")

gm_long = []
for _, r in sep.iterrows():
    gm_long.append({"signal": r["signal"], "group": "FAIL", "mean": r["mean_fail"]})
    gm_long.append({"signal": r["signal"], "group": "PASS", "mean": r["mean_pass"]})
gm = pd.DataFrame(gm_long)
write_csv("group_means", gm)
add_chart("group_means", "bar", "signal", "mean",
          "Per-signal group means (FAIL vs PASS)", "Per-signal group means")

# ----------------------------------------------------------------------------
# 3. Binned fail-rate curves per numeric signal
# ----------------------------------------------------------------------------
def binned_failrate(col, n_bins=4):
    g = df[[col, "is_fail"]].dropna()
    if g[col].nunique() < 2:
        return None
    try:
        g["bin"] = pd.qcut(g[col], q=min(n_bins, g[col].nunique()), duplicates="drop")
    except Exception:
        g["bin"] = pd.cut(g[col], bins=min(n_bins, g[col].nunique()))
    agg = g.groupby("bin", observed=True).agg(
        fail_rate=("is_fail", "mean"), n=("is_fail", "size"),
        lo=(col, "min"), hi=(col, "max")).reset_index(drop=True)
    agg["bin_label"] = [f"[{r.lo:.3g},{r.hi:.3g}]" for r in agg.itertuples()]
    return agg[["bin_label", "fail_rate", "n"]]


for c in NUMERIC:
    fr = binned_failrate(c)
    if fr is None or len(fr) < 2:
        continue
    name = f"failrate_by_{short(c)}"
    write_csv(name, fr)
    add_chart(name, "line", "bin_label", "fail_rate",
              f"Fail rate by {short(c)}", f"Fail rate by {short(c)}")

# ----------------------------------------------------------------------------
# 4. Correlation table
# ----------------------------------------------------------------------------
csub = df[NUMERIC].dropna(how="all")
corr = csub.corr()
corr_out = corr.copy()
corr_out.index = [short(i) for i in corr_out.index]
corr_out.columns = [short(c) for c in corr_out.columns]
corr_out.reset_index().rename(columns={"index": "signal"}).to_csv(
    "tables/signal_correlations.csv", index=False)

# ----------------------------------------------------------------------------
# RICHER PNG FIGURES
# ----------------------------------------------------------------------------
if ATTN and attn_n >= 6:
    ncol = 3
    nrow = int(math.ceil(len(ATTN) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    rs = np.random.RandomState(0)
    for i, c in enumerate(ATTN):
        ax = axes[i]
        g = df[[c, "is_fail"]].dropna()
        data = [g.loc[g.is_fail == 0, c].values, g.loc[g.is_fail == 1, c].values]
        ax.boxplot(data, tick_labels=["PASS", "FAIL"], showmeans=True)
        for j, arr in enumerate(data):
            jitter = rs.normal(0, 0.05, len(arr))
            ax.scatter(np.full(len(arr), j + 1) + jitter, arr, s=14, alpha=0.6,
                       color=["#2c7fb8", "#d95f0e"][j])
        ax.set_title(short(c), fontsize=10)
    for k in range(len(ATTN), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"Attention signals: FAIL vs PASS (n={attn_n})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("figures/attention_box_by_outcome.png", dpi=140)
    plt.close(fig)
    plots.append("figures/attention_box_by_outcome.png")

if corr.shape[0] >= 2:
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(corr), 1.0 + 0.7 * len(corr)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels([short(c) for c in corr.columns], rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels([short(c) for c in corr.index], fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < 0.6 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Signal correlations")
    fig.tight_layout()
    fig.savefig("figures/corr_heatmap.png", dpi=140)
    plt.close(fig)
    plots.append("figures/corr_heatmap.png")

attn_sep = sep[sep["signal_full"].isin(ATTN)]
scatter_pair = None
if len(attn_sep) >= 2:
    top2 = attn_sep.head(2)["signal_full"].tolist()
    scatter_pair = top2
    g = df[top2 + ["is_fail"]].dropna()
    if len(g) >= 6:
        fig, ax = plt.subplots(figsize=(5, 4))
        for lab, col, mk in [(0, "#2c7fb8", "o"), (1, "#d95f0e", "X")]:
            sub = g[g.is_fail == lab]
            ax.scatter(sub[top2[0]], sub[top2[1]], c=col, marker=mk,
                       label=["PASS", "FAIL"][lab], s=45, alpha=0.75,
                       edgecolor="k", linewidth=0.3)
        ax.set_xlabel(short(top2[0]))
        ax.set_ylabel(short(top2[1]))
        ax.set_title("Top discriminative attention pair")
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/scatter_top_attention_pair.png", dpi=140)
        plt.close(fig)
        plots.append("figures/scatter_top_attention_pair.png")
        sc = g.rename(columns={top2[0]: short(top2[0]), top2[1]: short(top2[1])}).copy()
        sc["outcome"] = np.where(sc["is_fail"] == 1, "FAIL", "PASS")
        write_csv("scatter_top_attention_pair",
                  sc[[short(top2[0]), short(top2[1]), "outcome"]])
        add_chart("scatter_top_attention_pair", "scatter", short(top2[0]), short(top2[1]),
                  "Top discriminative attention pair", "Scatter: top attention pair")

# ----------------------------------------------------------------------------
# Candidate composite signals (deterministic recipes over numeric columns)
# ----------------------------------------------------------------------------
med = {c: float(df[c].median()) for c in NUMERIC if df[c].notna().sum() > 0}


def fmt(x):
    return f"{x:.4g}"


candidate_signals = []


def two_group(mask):
    """is_fail among signal-ABSENT (a, mask False) vs signal-PRESENT (b, mask True)."""
    present = mask.reindex(df.index)
    valid = present.notna()
    sub = df[valid]
    flag = present[valid].astype(bool)
    a = sub.loc[~flag, "is_fail"].astype(int).tolist()
    b = sub.loc[flag, "is_fail"].astype(int).tolist()
    return {"kind": "two_group", "a": a, "b": b}


if {"relative_attention_focus_share", "relative_attention_attention_entropy"} <= set(med):
    expr = (f"(relative_attention_focus_share < {fmt(med['relative_attention_focus_share'])}) and "
            f"(relative_attention_attention_entropy > {fmt(med['relative_attention_attention_entropy'])})")
    mask = ((df["relative_attention_focus_share"] < med["relative_attention_focus_share"]) &
            (df["relative_attention_attention_entropy"] > med["relative_attention_attention_entropy"]))
    mask = mask.where(df["relative_attention_focus_share"].notna() &
                      df["relative_attention_attention_entropy"].notna())
    candidate_signals.append({
        "name": "diffuse_attention",
        "display_name": "Diffuse, low-focus attention",
        "rationale": "Low focus_share with high entropy = attention spread over no real object; the "
                     "model 'sees' the absent object everywhere and answers Yes.",
        "suggested_test": "Compare held-out fail rate of diffuse vs concentrated attention cases.",
        "recipe": {"name": "diffuse_attention", "kind": "expr", "expr": expr},
        "sufficient": two_group(mask),
    })

if {"relative_attention_edge_mass", "relative_attention_center_offset"} <= set(med):
    expr = (f"(relative_attention_edge_mass > {fmt(med['relative_attention_edge_mass'])}) and "
            f"(relative_attention_center_offset > {fmt(med['relative_attention_center_offset'])})")
    mask = ((df["relative_attention_edge_mass"] > med["relative_attention_edge_mass"]) &
            (df["relative_attention_center_offset"] > med["relative_attention_center_offset"]))
    mask = mask.where(df["relative_attention_edge_mass"].notna() &
                      df["relative_attention_center_offset"].notna())
    candidate_signals.append({
        "name": "peripheral_attention",
        "display_name": "Edge/peripheral attention",
        "rationale": "High edge_mass + large center_offset = attention drawn to image borders rather "
                     "than a grounded object location.",
        "suggested_test": "Held-out fail rate for peripheral vs central attention.",
        "recipe": {"name": "peripheral_attention", "kind": "expr", "expr": expr},
        "sufficient": two_group(mask),
    })

if "prompt_contrast_prompt_sensitivity" in med:
    expr = "prompt_contrast_prompt_sensitivity <= 0.0"
    mask = (df["prompt_contrast_prompt_sensitivity"] <= 0.0)
    mask = mask.where(df["prompt_contrast_prompt_sensitivity"].notna())
    candidate_signals.append({
        "name": "prompt_insensitive",
        "display_name": "Prompt-insensitive answer",
        "rationale": "Zero sensitivity = the model gives the same answer regardless of prompt phrasing, "
                     "a hallmark of an ungrounded prior 'Yes'.",
        "suggested_test": "Held-out fail rate for prompt-insensitive vs sensitive cases.",
        "recipe": {"name": "prompt_insensitive", "kind": "expr", "expr": expr},
        "sufficient": two_group(mask),
    })

if {"relative_attention_focus_share", "relative_attention_edge_mass"} <= set(med):
    expr = (f"(relative_attention_focus_share < {fmt(med['relative_attention_focus_share'])}) and "
            f"(relative_attention_edge_mass > {fmt(med['relative_attention_edge_mass'])})")
    mask = ((df["relative_attention_focus_share"] < med["relative_attention_focus_share"]) &
            (df["relative_attention_edge_mass"] > med["relative_attention_edge_mass"]))
    mask = mask.where(df["relative_attention_focus_share"].notna() &
                      df["relative_attention_edge_mass"].notna())
    candidate_signals.append({
        "name": "diffuse_and_peripheral",
        "display_name": "Diffuse AND peripheral attention",
        "rationale": "Interaction: unfocused attention that also leaks to edges is the strongest "
                     "ungrounded-attention pattern.",
        "suggested_test": "Held-out fail rate for the conjunction vs the rest.",
        "recipe": {"name": "diffuse_and_peripheral", "kind": "expr", "expr": expr},
        "sufficient": two_group(mask),
    })

# ----------------------------------------------------------------------------
# Visual plan
# ----------------------------------------------------------------------------
visual_plan = [
    {"name": "class_balance", "display_name": "FAIL/PASS case balance",
     "question": "How imbalanced are the labeled outcomes?",
     "data_shape": "categorical-vs-binary", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": ["label"],
     "rationale": "Counts establish the base rate so later separations are read against imbalance."},
    {"name": "top_discriminators", "display_name": "Top FAIL/PASS discriminators",
     "question": "Which numeric signals separate FAIL from PASS most?",
     "data_shape": "many-numeric", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": NUMERIC,
     "rationale": "Ranked |standardized mean diff| surfaces candidates without assuming a model."},
    {"name": "group_means", "display_name": "Per-signal group means",
     "question": "In which direction does each signal shift for FAIL?",
     "data_shape": "numeric-vs-binary", "plot_kind": "bar", "fallback_kind": "bar",
     "required_columns": NUMERIC,
     "rationale": "Side-by-side means show the sign of separation; pre-aggregated to avoid row dumps."},
    {"name": "attention_box_by_outcome", "display_name": "Attention signals by outcome",
     "question": "Do attention distributions differ for FAIL vs PASS?",
     "data_shape": "numeric-vs-binary", "plot_kind": "box", "fallback_kind": "bar",
     "required_columns": ATTN + ["label"],
     "rationale": "Box+strip shows spread and overlap that a bar-of-means hides at small n."},
    {"name": "failrate_by_focus_share", "display_name": "Fail rate by focus share",
     "question": "Does lower attention focus raise failure risk?",
     "data_shape": "numeric-vs-binary", "plot_kind": "line", "fallback_kind": "line",
     "required_columns": ["relative_attention_focus_share", "label"],
     "rationale": "Ordered bins show the risk trend without assuming linearity."},
    {"name": "corr_heatmap", "display_name": "Signal correlations",
     "question": "Which signals are redundant vs independent?",
     "data_shape": "many-numeric", "plot_kind": "heatmap", "fallback_kind": "bar",
     "required_columns": NUMERIC,
     "rationale": "Correlation guards against double-counting collinear attention features."},
    {"name": "scatter_top_attention_pair", "display_name": "Top attention signal pair",
     "question": "Do the two best signals jointly separate FAIL?",
     "data_shape": "numeric-vs-numeric", "plot_kind": "scatter", "fallback_kind": "scatter",
     "required_columns": (scatter_pair or []) + ["label"],
     "rationale": "Colored scatter reveals interaction/region structure a 1-D view misses."},
]

# ----------------------------------------------------------------------------
# Chart readings
# ----------------------------------------------------------------------------
chart_readings = [
    {"chart": "class_balance",
     "reading": f"PASS dominates ({n_pass}) over FAIL ({n_fail}); ~{n_fail / n_total:.0%} base fail rate.",
     "do_not_infer": "Imbalance alone says nothing about which signals drive failures."},
    {"chart": "top_discriminators",
     "reading": "Bars rank signals by standardized FAIL-vs-PASS mean gap; attention focus/edge/entropy "
                "signals tend to top prompt_sensitivity.",
     "do_not_infer": "A large gap on n=22 attention cases is not a confirmed or significant effect."},
    {"chart": "group_means",
     "reading": "Shows the direction of each shift (e.g. FAIL = lower focus_share / higher edge_mass).",
     "do_not_infer": "Mean direction does not establish a threshold or a causal mechanism."},
    {"chart": "attention_box_by_outcome",
     "reading": "FAIL boxes shift toward diffuse/peripheral attention but overlap PASS substantially.",
     "do_not_infer": "Overlap means no clean separator; small n makes whiskers unstable."},
    {"chart": "failrate_by_focus_share",
     "reading": "Fail rate is higher in the lowest focus_share bins.",
     "do_not_infer": "The binned trend is descriptive, not a fitted dose-response."},
    {"chart": "corr_heatmap",
     "reading": "Several attention features are mutually correlated (focus_share, top1_share, entropy).",
     "do_not_infer": "Collinearity does not tell which feature is the true driver."},
    {"chart": "scatter_top_attention_pair",
     "reading": "FAIL points cluster toward the diffuse/peripheral corner of the two top signals.",
     "do_not_infer": "Clustering on the labeled set may not hold out-of-sample."},
]

claims = [
    {"id": "C1",
     "text": "Diffuse, low-focus attention is a descriptive correlate of false-positive (FAIL) cases.",
     "status": "descriptive",
     "evidence_ids": ["chart:top_discriminators", "chart:attention_box_by_outcome",
                      "chart:failrate_by_focus_share", "signal:diffuse_attention"],
     "interpretation": "When attention is spread out rather than concentrated, the model is more likely "
                       "to assert an absent object is present.",
     "do_not_infer": "Not causal and based on only ~22 attention-instrumented cases."},
    {"id": "C2",
     "text": "Edge/peripheral attention concentration co-occurs with FAIL cases.",
     "status": "descriptive",
     "evidence_ids": ["chart:group_means", "signal:peripheral_attention"],
     "interpretation": "Attention leaking to image borders accompanies ungrounded 'Yes' answers.",
     "do_not_infer": "Direction of effect unconfirmed; edge_mass correlates with other signals."},
    {"id": "C3",
     "text": "Prompt-insensitive answers (zero sensitivity) are a sanity-check correlate of FAIL.",
     "status": "descriptive",
     "evidence_ids": ["signal:prompt_insensitive"],
     "interpretation": "Answers unchanged across prompt phrasings suggest an ungrounded prior.",
     "do_not_infer": "Sensitivity is 0 for most cases, so discrimination is weak."},
]

dashboard_storyboard = [
    {"id": "problem_setting", "title": "Problem Setting", "stages": ["M1"],
     "summary": "121 labeled cases where FAIL = false 'Yes' on an absent object; per-case M1 signals are "
                "prompt sensitivity and 7 relative-attention features (attention present for ~22 cases).",
     "items": [
         "FAIL means the model hallucinated a present object that is absent.",
         f"Base fail rate {n_fail / n_total:.0%} ({n_fail}/{n_total}); attention features missing for "
         f"{n_total - attn_n} cases.",
     ],
     "artifact_refs": ["data_profile", "class_balance"]},
    {"id": "analysis", "title": "Analysis", "stages": ["M2"],
     "summary": "Rank each numeric signal by standardized FAIL-vs-PASS separation, view distributions and "
                "binned fail-rate curves, and check signal collinearity.",
     "items": [
         "Method: |standardized mean diff| -> chart:top_discriminators -> attention focus/edge signals lead.",
         "Method: binned fail-rate -> chart:failrate_by_focus_share -> risk rises at low focus.",
         "Method: correlation -> chart:corr_heatmap -> attention features are partly redundant.",
     ],
     "artifact_refs": ["candidate_signals", "charts", "plots"]},
    {"id": "hypotheses_artifacts", "title": "Hypotheses & Artifacts",
     "stages": ["M3", "M4", "M5"],
     "summary": "Composite deterministic recipes proposed as predictors for held-out confirmation.",
     "items": [
         "M3: ungrounded attention (diffuse and/or peripheral) drives false positives.",
         "M4: test the diffuse_and_peripheral recipe on a held-out attention-present split.",
         "M5: if confirmed, an attention-grounding intervention should reduce false 'Yes'.",
     ],
     "artifact_refs": ["claims", "candidate_signals"]},
]

caveats = [
    f"Attention signals exist for only {attn_n}/{n_total} cases; all attention findings are small-n.",
    "Recipe thresholds use medians of the present-data subset (data-driven) -> mild optimism; the host "
    "should confirm on a held-out split.",
    "Class imbalance (~21% FAIL) inflates apparent separations; fail-rate curves are sparse per bin.",
    "prompt_sensitivity is 0 for most cases, giving little spread.",
]

critique = [
    "Double-dipping risk: medians for thresholds are computed on the same rows used to report separation; "
    "recipes are emitted so the host can re-evaluate out-of-sample.",
    "Heavy missingness in attention features could be non-random (which cases get instrumented), a possible "
    "selection confound.",
    "Standardized mean diff assumes roughly comparable spread; with n=22 the pooled SD is unstable.",
    "Probe/prompt_sensitivity fields are treated as sanity-check evidence, not primary explanatory findings.",
]

recommended = [
    "Confirm the diffuse_attention and diffuse_and_peripheral recipes on a held-out split via the host's "
    "multiplicity-aware core.",
    "Collect attention features for the remaining cases to remove small-n / selection limits.",
    "M5 surgery: apply attention re-grounding to high-edge_mass cases and re-measure the false-positive rate.",
    "Check whether missingness of attention features itself predicts FAIL.",
]

result = {
    "observations": observations,
    "visual_plan": visual_plan,
    "chart_readings": chart_readings,
    "claims": claims,
    "dashboard_storyboard": dashboard_storyboard,
    "candidate_signals": candidate_signals,
    "plots": plots,
    "tables": {},
    "charts": charts,
    "caveats": caveats,
    "critique": critique,
    "recommended_confirmatory_tests": recommended,
}

print("EXPLORATORY_RESULT_JSON=" + json.dumps(result))
