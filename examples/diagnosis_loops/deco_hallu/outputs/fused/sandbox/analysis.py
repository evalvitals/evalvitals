#!/usr/bin/env python3
"""Exploratory FAIL-vs-PASS failure analysis for false-detection (hallucination) cases.

Self-contained: reads records.json from CWD, writes CSVs under tables/, PNGs under
figures/, and prints EXPLORATORY_RESULT_JSON=... as the last stdout line.
"""
import os
import json
import math

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------- setup
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("tables", exist_ok=True)
os.makedirs("figures", exist_ok=True)

with open("records.json") as fh:
    rows = json.load(fh)
df = pd.DataFrame(rows)

observations = []
candidate_signals = []
charts = []
plots = []
caveats = []


def save_table(name, frame):
    path = os.path.join("tables", name + ".csv")
    frame.to_csv(path, index=False)
    return path


def add_chart(name, kind, frame, x, y, title):
    save_table(name, frame)
    charts.append({"name": name, "kind": kind, "data": "tables/%s.csv" % name,
                   "x": x, "y": y, "title": title})


# ------------------------------------------------------------------- outcome column
# Identify the pass/fail outcome column.
outcome_col = None
for c in df.columns:
    low = c.lower()
    if low in ("label", "is_correct", "is_fail", "outcome", "result", "verdict"):
        outcome_col = c
        break
if outcome_col is None:
    # fall back to any string column whose values look like pass/fail
    for c in df.columns:
        vals = set(str(v).strip().lower() for v in df[c].dropna().unique())
        if vals & {"pass", "fail"} or vals <= {"0", "1", "true", "false"}:
            outcome_col = c
            break

raw = df[outcome_col].astype(str).str.strip().str.lower()
# Map to is_fail indicator. "fail"/false-correct -> 1.
def to_is_fail(v):
    if v in ("fail", "failed", "1", "true", "incorrect", "wrong"):
        return 1
    if v in ("pass", "passed", "0", "false", "correct", "right"):
        return 0
    return np.nan

df["is_fail"] = raw.map(to_is_fail)
# If still ambiguous, treat the minority-ish "fail" string explicitly
df = df.dropna(subset=["is_fail"]).copy()
df["is_fail"] = df["is_fail"].astype(int)
df["outcome"] = np.where(df["is_fail"] == 1, "FAIL", "PASS")

observations.append(
    "Outcome column '%s' mapped to %d FAIL and %d PASS cases (n=%d)."
    % (outcome_col, int(df["is_fail"].sum()), int((1 - df["is_fail"]).sum()), len(df))
)

# --------------------------------------------------------------- numeric / cat cols
exclude = {outcome_col, "is_fail", "outcome", "case_id"}
numeric_cols = [c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns
            if c not in exclude and c not in numeric_cols and c != "case_id"
            and df[c].nunique() <= 12]

observations.append("Numeric signal columns: %s." % (numeric_cols or "none"))
if cat_cols:
    observations.append("Categorical group columns: %s." % cat_cols)


# ============================================================= 1. class balance
cb = df["outcome"].value_counts().rename_axis("outcome").reset_index(name="count")
add_chart("class_balance", "bar", cb, "outcome", "count", "FAIL vs PASS")

# per-categorical class balance / fail rate
for c in cat_cols:
    g = df.groupby(c)["is_fail"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"mean": "fail_rate", "count": "n"})
    add_chart("failrate_by_%s" % c, "bar", g, c, "fail_rate",
              "Fail rate by %s" % c)


# ============================================================= 2./3. per-signal sep
disc_rows = []
for col in numeric_cols:
    sub = df[[col, "is_fail", "outcome"]].dropna()
    if sub[col].nunique() < 2 or len(sub) < 4:
        continue
    fail_vals = sub.loc[sub["is_fail"] == 1, col]
    pass_vals = sub.loc[sub["is_fail"] == 0, col]
    if len(fail_vals) == 0 or len(pass_vals) == 0:
        continue

    # group mean/median per outcome (grouped bar -> long form)
    gm = (sub.groupby("outcome")[col]
          .agg(mean="mean", median="median").reset_index())
    add_chart("groupstats_%s" % col, "bar", gm, "outcome", "mean",
              "%s: mean by outcome" % col)

    # standardized mean difference (pooled sd)
    n1, n0 = len(fail_vals), len(pass_vals)
    s1, s0 = fail_vals.std(ddof=1), pass_vals.std(ddof=1)
    pooled = math.sqrt(((n1 - 1) * (s1 ** 2 if not math.isnan(s1) else 0) +
                        (n0 - 1) * (s0 ** 2 if not math.isnan(s0) else 0)) /
                       max(n1 + n0 - 2, 1))
    smd = (fail_vals.mean() - pass_vals.mean()) / pooled if pooled > 0 else 0.0
    disc_rows.append({"signal": col, "mean_FAIL": fail_vals.mean(),
                      "mean_PASS": pass_vals.mean(), "abs_diff": abs(fail_vals.mean() - pass_vals.mean()),
                      "separation": abs(smd), "n": len(sub)})

    # binned fail-rate curve
    nb = min(6, sub[col].nunique())
    try:
        bins = pd.qcut(sub[col], q=nb, duplicates="drop")
    except Exception:
        bins = pd.cut(sub[col], bins=nb)
    fr = sub.groupby(bins, observed=True)["is_fail"].agg(["mean", "count"]).reset_index()
    fr.columns = ["%s_bin" % col, "fail_rate", "n"]
    fr["%s_bin" % col] = fr["%s_bin" % col].astype(str)
    add_chart("failrate_by_%s" % col, "line", fr, "%s_bin" % col, "fail_rate",
              "Fail rate by %s" % col)

if disc_rows:
    disc = pd.DataFrame(disc_rows).sort_values("separation", ascending=False)
    add_chart("top_discriminators", "bar", disc[["signal", "separation"]],
              "signal", "separation", "Top FAIL/PASS discriminators")
    best_signal = disc.iloc[0]["signal"]
    observations.append(
        "Strongest single-signal separation: '%s' (|SMD|=%.2f)."
        % (best_signal, disc.iloc[0]["separation"]))
else:
    disc = pd.DataFrame()
    best_signal = numeric_cols[0] if numeric_cols else None


# ============================================================= 5. correlations
present_num = [c for c in numeric_cols if df[c].notna().sum() >= 4 and df[c].nunique() > 1]
corr_cols = present_num + ["is_fail"]
if len(present_num) >= 1:
    corr = df[corr_cols].corr(method="pearson")
    corr_long = corr.reset_index().rename(columns={"index": "row"})
    save_table("correlations", corr_long)
    # correlation of each signal with is_fail (bar)
    cwf = (corr["is_fail"].drop(labels=["is_fail"], errors="ignore")
           .rename("corr_with_fail").reset_index()
           .rename(columns={"index": "signal"}))
    cwf["abs_corr"] = cwf["corr_with_fail"].abs()
    cwf = cwf.sort_values("abs_corr", ascending=False)
    add_chart("corr_with_fail", "bar", cwf[["signal", "corr_with_fail"]],
              "signal", "corr_with_fail", "Signal correlation with FAIL")

    # heatmap PNG
    if len(present_num) >= 2:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=7)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, "%.2f" % corr.values[i, j], ha="center", va="center",
                        fontsize=6, color="black")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Signal correlation matrix")
        fig.tight_layout()
        fig.savefig("figures/corr_heatmap.png", dpi=150)
        plt.close(fig)
        plots.append("figures/corr_heatmap.png")


# ============================================================= 6. scatter pairs
if len(present_num) >= 2:
    # most discriminative pair = top two by separation if available else first two
    if not disc.empty:
        ranked = [s for s in disc["signal"].tolist() if s in present_num]
    else:
        ranked = present_num
    pair = (ranked + present_num)[:2]
    sub = df[[pair[0], pair[1], "outcome", "is_fail"]].dropna()
    if len(sub) >= 4:
        sc = sub.rename(columns={pair[0]: "x", pair[1]: "y"})[["x", "y", "outcome"]]
        add_chart("scatter_top_pair", "scatter", sc, "x", "y",
                  "%s vs %s by outcome" % (pair[0], pair[1]))
        fig, ax = plt.subplots(figsize=(5, 4))
        for oc, color in (("FAIL", "#d62728"), ("PASS", "#1f77b4")):
            m = sub["outcome"] == oc
            ax.scatter(sub.loc[m, pair[0]], sub.loc[m, pair[1]], label=oc,
                       alpha=0.7, color=color)
        ax.set_xlabel(pair[0]); ax.set_ylabel(pair[1]); ax.legend()
        ax.set_title("%s vs %s" % (pair[0], pair[1]))
        fig.tight_layout()
        fig.savefig("figures/scatter_top_pair.png", dpi=150)
        plt.close(fig)
        plots.append("figures/scatter_top_pair.png")


# ============================================================= candidate signals
# Primary deterministic recipe: probe1 false-detection flag.
pf = "generated_probe1_false_detection"
if pf in df.columns:
    sub = df[[pf, "is_fail"]].dropna()
    a = sub.loc[sub[pf] <= 0.5, "is_fail"].astype(int).tolist()   # signal-absent
    b = sub.loc[sub[pf] > 0.5, "is_fail"].astype(int).tolist()    # signal-present
    candidate_signals.append({
        "name": "probe1_flags_false_detection",
        "rationale": ("The probe1 false-detection flag is ~1 exactly on cases the model "
                      "wrongly answers 'Yes' to an absent object; it is a near-perfect "
                      "per-case predictor of FAIL."),
        "suggested_test": "two_group fail-rate among probe-positive vs probe-negative cases",
        "recipe": {"name": "probe1_positive", "kind": "expr",
                   "expr": "generated_probe1_false_detection > 0.5"},
        "sufficient": {"kind": "two_group", "a": a, "b": b},
    })
    fr_pos = (np.mean(b) if b else float("nan"))
    fr_neg = (np.mean(a) if a else float("nan"))
    observations.append(
        "Fail rate is %.2f among probe1-positive vs %.2f among probe1-negative cases."
        % (fr_pos, fr_neg))

# Attention-based composite (only ~20 rows have attention columns).
att = ["relative_attention_max_relative_weight",
       "relative_attention_mean_relative_weight",
       "relative_attention_focus_share"]
if all(c in df.columns for c in att):
    sub = df[att + ["is_fail"]].dropna()
    if len(sub) >= 6:
        # On the labeled subset, compare attention focus between FAIL and PASS.
        focus = "relative_attention_focus_share"
        # Choose threshold at the PASS/FAIL midpoint of medians for the focus share.
        med_fail = sub.loc[sub["is_fail"] == 1, focus].median()
        med_pass = sub.loc[sub["is_fail"] == 0, focus].median()
        thr = float(np.nanmean([med_fail, med_pass]))
        direction = "<" if med_fail < med_pass else ">"
        candidate_signals.append({
            "name": "low_attention_focus_share",
            "rationale": ("On the attention-instrumented subset, FAIL cases tend to show "
                          "%s attention focus_share than PASS (median FAIL=%.3f vs PASS=%.3f); "
                          "diffuse/peripheral attention may accompany false 'Yes' answers."
                          % ("lower" if direction == "<" else "higher", med_fail, med_pass)),
            "suggested_test": "two_group fail-rate split at focus_share threshold",
            "recipe": {"name": "low_focus_share", "kind": "expr",
                       "expr": "relative_attention_focus_share %s %.6f" % (direction, thr)},
        })
        caveats.append(
            "Attention columns are populated for only %d of %d rows; attention-based "
            "signals are exploratory on that subset only." % (len(sub), len(df)))

if not candidate_signals:
    caveats.append("No deterministic separating signal found among available columns.")

caveats.append("Exploratory only — no causal or multiplicity-adjusted claim; host confirms on held-out split.")

recommended = [
    "Confirm probe1_positive recipe on a held-out split (two-group fail-rate contrast).",
    "Collect attention features for all rows, then re-test focus_share threshold.",
]

result = {
    "observations": observations,
    "candidate_signals": candidate_signals,
    "plots": plots,
    "tables": {},
    "charts": charts,
    "caveats": caveats,
    "recommended_confirmatory_tests": recommended,
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(result))
