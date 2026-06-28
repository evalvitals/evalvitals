#!/usr/bin/env python3
"""Exploratory analysis: what per-case signals separate FAIL (false 'Yes' on an
absent object) from PASS cases? Exploratory only — no causal/statistical claims."""
import json
import os
import math

CWD = os.path.dirname(os.path.abspath(__file__))
os.chdir(CWD)


def load():
    with open("records.json") as f:
        return json.load(f)


def is_fail(row):
    return str(row.get("label", "")).strip().lower() == "fail"


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return ""
        return f"{x:.6g}"
    return str(x)


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


def safe_plot(fn):
    try:
        import matplotlib
        matplotlib.use("Agg")
        return fn(matplotlib)
    except Exception as e:
        return ("__error__", str(e))


def main():
    rows = load()
    n = len(rows)

    numeric_cols = [
        "generated_probe1_false_detection",
        "relative_attention_max_relative_weight",
        "relative_attention_mean_relative_weight",
        "relative_attention_focus_share",
    ]

    observations = []
    candidate_signals = []
    plots = []
    tables = {}
    charts = []
    caveats = []

    n_fail = sum(1 for r in rows if is_fail(r))
    n_pass = n - n_fail
    observations.append(
        f"{n} labeled cases: {n_fail} fail, {n_pass} pass "
        f"(base fail rate {n_fail / n:.3f})."
    )

    # ------------------------------------------------------------------
    # 1) generated_probe1_false_detection vs label (covers all 121 rows)
    # ------------------------------------------------------------------
    cross = {}
    for r in rows:
        v = r.get("generated_probe1_false_detection")
        key = ("nan" if v is None or (isinstance(v, float) and math.isnan(v))
               else fmt(v))
        cross.setdefault(key, [0, 0])
        if is_fail(r):
            cross[key][1] += 1
        else:
            cross[key][0] += 1

    ct_rows = []
    for key in sorted(cross):
        p, fl = cross[key]
        tot = p + fl
        ct_rows.append([key, p, fl, tot, (fl / tot if tot else 0.0)])
    write_csv("tables/probe1_vs_label.csv",
              ["probe1_value", "n_pass", "n_fail", "n_total", "fail_rate"],
              ct_rows)
    tables["probe1_vs_label"] = "tables/probe1_vs_label.csv"
    charts.append({
        "name": "probe1_fail_rate",
        "kind": "bar",
        "data": "tables/probe1_vs_label.csv",
        "x": "probe1_value",
        "y": "fail_rate",
        "title": "Fail rate by generated_probe1_false_detection",
    })

    # Concordance of probe1==1 with fail
    tp = sum(1 for r in rows if r.get("generated_probe1_false_detection") == 1.0 and is_fail(r))
    fp = sum(1 for r in rows if r.get("generated_probe1_false_detection") == 1.0 and not is_fail(r))
    fn = sum(1 for r in rows if r.get("generated_probe1_false_detection") != 1.0 and is_fail(r))
    tn = sum(1 for r in rows if r.get("generated_probe1_false_detection") != 1.0 and not is_fail(r))
    observations.append(
        f"probe1_false_detection==1 vs fail: TP={tp} FP={fp} FN={fn} TN={tn}. "
        "This probe appears to be a near-direct indicator of the fail condition."
    )

    # two_group sufficient stat: is_fail among probe1-absent (a) vs probe1-present (b)
    a_group = [1 if is_fail(r) else 0 for r in rows
               if r.get("generated_probe1_false_detection") != 1.0]
    b_group = [1 if is_fail(r) else 0 for r in rows
               if r.get("generated_probe1_false_detection") == 1.0]

    candidate_signals.append({
        "name": "probe1_false_detection_flag",
        "rationale": (
            f"generated_probe1_false_detection==1 co-occurs with fail "
            f"(TP={tp}, FP={fp}; absent->fail FN={fn}, TN={tn}). It is defined "
            "on all 121 rows and is the strongest single per-case separator."
        ),
        "suggested_test": "Confirm probe1==1 predicts fail on a held-out split.",
        "recipe": {
            "name": "probe1_flag",
            "kind": "expr",
            "expr": "generated_probe1_false_detection >= 0.5",
        },
        "sufficient": {"kind": "two_group", "a": a_group, "b": b_group},
    })

    # ------------------------------------------------------------------
    # 2) relative_attention_* (only 20 rows non-null) vs label
    # ------------------------------------------------------------------
    att_cols = [
        "relative_attention_max_relative_weight",
        "relative_attention_mean_relative_weight",
        "relative_attention_focus_share",
    ]
    att_rows = [r for r in rows
                if r.get("relative_attention_focus_share") is not None
                and not (isinstance(r.get("relative_attention_focus_share"), float)
                         and math.isnan(r.get("relative_attention_focus_share")))]

    if att_rows:
        summ = []
        for col in att_cols:
            for grp_name, grp in (("pass", [r for r in att_rows if not is_fail(r)]),
                                  ("fail", [r for r in att_rows if is_fail(r)])):
                vals = [r[col] for r in grp
                        if isinstance(r.get(col), (int, float))
                        and not math.isnan(r[col])]
                if vals:
                    vals_s = sorted(vals)
                    mean = sum(vals) / len(vals)
                    med = vals_s[len(vals_s) // 2]
                    summ.append([col, grp_name, len(vals), mean,
                                 min(vals), med, max(vals)])
        write_csv("tables/attention_by_label.csv",
                  ["column", "label", "n", "mean", "min", "median", "max"],
                  summ)
        tables["attention_by_label"] = "tables/attention_by_label.csv"

        observations.append(
            f"Only {len(att_rows)}/{n} rows carry relative_attention_* features; "
            "any attention-based signal is descriptive on a small subset."
        )

        # Plot focus_share distribution by label
        def plot_focus(matplotlib):
            import matplotlib.pyplot as plt
            fpass = [r["relative_attention_focus_share"] for r in att_rows
                     if not is_fail(r) and isinstance(r.get("relative_attention_focus_share"), (int, float))]
            ffail = [r["relative_attention_focus_share"] for r in att_rows
                     if is_fail(r) and isinstance(r.get("relative_attention_focus_share"), (int, float))]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist([fpass, ffail], bins=8, label=["pass", "fail"])
            ax.set_xlabel("relative_attention_focus_share")
            ax.set_ylabel("count")
            ax.set_title("Focus share by label (n=%d subset)" % len(att_rows))
            ax.legend()
            os.makedirs("figures", exist_ok=True)
            fig.savefig("figures/focus_share_by_label.png", bbox_inches="tight")
            plt.close(fig)
            return "figures/focus_share_by_label.png"

        res = safe_plot(plot_focus)
        if isinstance(res, str):
            plots.append(res)
        else:
            caveats.append("focus_share plot skipped: " + str(res[1]))

        # Candidate: low focus_share -> attention diffuse -> hallucination prone
        focus_vals = sorted(r["relative_attention_focus_share"] for r in att_rows
                            if isinstance(r.get("relative_attention_focus_share"), (int, float)))
        if focus_vals:
            median_focus = focus_vals[len(focus_vals) // 2]
            candidate_signals.append({
                "name": "diffuse_attention_low_focus",
                "rationale": (
                    "Hypothesis: false detections on absent objects coincide with "
                    "diffuse attention (low focus_share / low max relative weight). "
                    f"Median focus_share on the {len(att_rows)}-row subset is "
                    f"{median_focus:.4g}; treat below-median focus as a risk flag."
                ),
                "suggested_test": (
                    "On held-out rows with attention features, test whether "
                    "low focus_share / low max weight associates with fail."
                ),
                "recipe": {
                    "name": "diffuse_attention",
                    "kind": "expr",
                    "expr": (
                        f"(relative_attention_focus_share < {median_focus:.6g}) and "
                        f"(relative_attention_max_relative_weight < "
                        f"relative_attention_mean_relative_weight * 3)"
                    ),
                },
            })

        # Candidate composite combining probe1 with attention sharpness
        candidate_signals.append({
            "name": "probe1_or_diffuse_attention",
            "rationale": (
                "Union signal: flag a case as fail-prone if probe1 fired OR "
                "attention is unusually diffuse. Probe1 carries most signal; "
                "the attention term is an exploratory add-on for the subset that "
                "has it (rows without it leave the term effectively neutral)."
            ),
            "suggested_test": "Compare against probe1-only on held-out split.",
            "recipe": {
                "name": "probe1_or_diffuse",
                "kind": "expr",
                "expr": (
                    "(generated_probe1_false_detection >= 0.5) or "
                    "(relative_attention_focus_share < 0.2)"
                ),
            },
        })

    caveats.append("Exploratory only; no causal or statistical confirmation claimed.")
    caveats.append(
        "relative_attention_* present on only 20/121 rows — attention signals are "
        "low-coverage and may not generalize."
    )
    caveats.append(
        "probe1_false_detection may be (near-)definitional of the fail label; "
        "verify it is an independent predictor and not a leaked outcome."
    )

    recommended = [
        "Confirm probe1_flag predicts fail on a held-out split (two_group already attached).",
        "Check whether generated_probe1_false_detection leaks the label definition.",
        "On the attention subset, test focus_share / max-weight separation between pass and fail.",
    ]

    result = {
        "observations": observations,
        "candidate_signals": candidate_signals,
        "plots": plots,
        "tables": tables,
        "charts": charts,
        "caveats": caveats,
        "recommended_confirmatory_tests": recommended,
    }
    print("EXPLORATORY_RESULT_JSON=" + json.dumps(result))


if __name__ == "__main__":
    main()
