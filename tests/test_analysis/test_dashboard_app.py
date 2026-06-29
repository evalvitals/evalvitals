"""The loop dashboard renders a data-rich Analysis tab (signal effects, e-BH
adjudication, explorer tables), not just the M2→M3→M5→Fix flow narrative.

Guarded on streamlit/pandas (the optional dashboard extras).
"""

from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("pandas")


def _build_loop_run(root):
    fused = root / "fused"
    (fused / "sandbox" / "tables").mkdir(parents=True)
    logs = root / "logs_m2_5"
    logs.mkdir(parents=True)

    (fused / "sandbox" / "tables" / "fail_by_signal.csv").write_text(
        "signal_value,fail_rate\nabsent,0.62\npresent,0.11\n", encoding="utf-8"
    )
    (fused / "fused_report.json").write_text(json.dumps({
        "observations": ["FAIL skews to small objects"],
        "visual_plan": [{
            "name": "fail_by_signal",
            "question": "Does the signal alter failure rate?",
            "data_shape": "categorical-vs-binary",
            "plot_kind": "bar",
            "fallback_kind": "bar",
            "required_columns": ["signal_value", "label"],
            "rationale": "A fail-rate bar is readable for two signal states.",
        }],
        "chart_readings": [{
            "chart": "Fail rate by signal",
            "reading": "The present state has a lower failure rate in this toy fixture.",
            "do_not_infer": "This chart alone is not causal.",
        }],
        "claims": [{
            "id": "A1",
            "text": "The signal is lower risk when present.",
            "status": "descriptive",
            "evidence_ids": ["chart:fail_rate_by_signal"],
            "interpretation": "Use as an exploratory reading.",
            "do_not_infer": "No causal claim.",
        }],
        "critique": ["toy fixture; no causal interpretation"],
        "caveats": ["probe1 ~ label (circular)"],
        "charts": [{"name": "c", "kind": "bar", "data": "tables/fail_by_signal.csv",
                    "x": "signal_value", "y": "fail_rate", "title": "Fail rate by signal"}],
        "candidate_signals": [
            {"name": "probe1_flag", "source": "explorer", "effect": 0.86, "reject": True,
             "recipe": {"expr": "probe1 > 0"}},
            {"name": "diffuse_attention", "source": "explorer", "effect": None, "reject": None},
            {"name": "probe1_or_diffuse", "source": "explorer", "effect": 0.40, "reject": True},
        ],
        "adjudication": {"method": "e-BH", "alpha": 0.05, "n_signals_tested": 3,
                         "n_signals_rejected": 2, "split": "held_out"},
    }), encoding="utf-8")

    events = [
        {"event": "analysis", "cycle": 1, "severity": "none",
         "conclusion": "Specific over-detection failure mode.",
         "evidence_chain": ["attention near-uniform"],
         "stats_results": [{"summary": "signal vs FAIL: effect=+0.86 -> REJECT H0"}]},
        {"event": "diagnosis", "cycle": 1, "n_hypotheses": 1,
         "hypotheses": [{"statement": "language-prior hallucination", "failure_mode": "lp"}],
         "referenced_charts": ["Fail rate by signal"], "explore_context_used": True},
        # surgery.hypothesis matches the diagnosis statement -> the M5 verdict joins.
        {"event": "surgery", "cycle": 1, "module": "m5", "status": "supported",
         "hypothesis": "language-prior hallucination"},
        {"event": "fix", "cycle": 1},
    ]
    (logs / "run_log.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return root


def _run_app(run_dir):
    from streamlit.testing.v1 import AppTest

    sys.argv = ["dashboard_app.py", str(run_dir)]
    at = AppTest.from_file("evalvitals/analysis/dashboard_app.py", default_timeout=30)
    at.run()
    return at


def test_loop_dashboard_renders_analysis_panel_without_error(tmp_path):
    _build_loop_run(tmp_path)
    at = _run_app(tmp_path)

    assert not at.exception
    assert [t.label for t in at.tabs] == ["📊 Analysis", "🔬 Diagnosis flow", "🗂 Tables"]

    # e-BH adjudication metrics surfaced as a metric row.
    metric_labels = {m.label for m in at.metric}
    assert {"Method", "Signals tested", "Rejected (real)", "Split"} <= metric_labels
    vals = {m.label: m.value for m in at.metric}
    assert vals["Rejected (real)"] in ("2", 2)

    # The candidate-signals table rendered (a dataframe).
    assert len(at.dataframe) >= 1
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Diagnostic report" in blob
    assert "Claims and evidence" in blob
    assert any(e.label == "How to interpret this page" for e in at.expander)
    assert any(e.label == "Chart readings written by the agent" for e in at.expander)
    assert any(e.label == "Critique and limits" for e in at.expander)


def test_loop_dashboard_warns_when_no_explore_report(tmp_path):
    logs = tmp_path / "logs_m2_5"
    logs.mkdir(parents=True)
    (logs / "run_log.jsonl").write_text(
        json.dumps({"event": "diagnosis", "cycle": 1, "n_hypotheses": 0, "hypotheses": []}) + "\n",
        encoding="utf-8",
    )
    at = _run_app(tmp_path)
    assert not at.exception
    # No measured signals → an actionable warning, not a silent empty page.
    assert any("explore report" in str(w.value).lower() for w in at.warning)


def test_analysis_tab_tells_connected_story(tmp_path):
    _build_loop_run(tmp_path)
    at = _run_app(tmp_path)
    assert not at.exception
    heads = [m.value for m in at.markdown if isinstance(m.value, str) and m.value.startswith("###")]
    # the three narrative sections are present
    assert any("What we analysed" in h for h in heads)
    assert any("What we found" in h for h in heads)
    assert any("Hypotheses formed" in h for h in heads)
    # the hypothesis statement + its M5 verdict appear somewhere in the page
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "language-prior hallucination" in blob
    assert "supported" in blob.lower()
    assert any(e.label == "Why these charts were chosen" for e in at.expander)
    assert len(at.dataframe) >= 2  # signal table + visual-plan table


def test_hypotheses_join_with_m5_outcomes():
    from evalvitals.analysis.dashboard_app import _hypotheses_with_outcomes

    story = {
        "diagnoses": [{"cycle": 1, "referenced_charts": ["C"],
                       "hypotheses": [{"statement": "H one", "failure_mode": "fm"},
                                      {"statement": "H two", "failure_mode": "fm2"}]}],
        "surgeries": [{"module": "m5", "status": "supported", "hypothesis": "H one"},
                      {"module": "m4", "status": "fixed", "fixed": True, "hypothesis": "H one"}],
    }
    out = _hypotheses_with_outcomes(story)
    assert [h["statement"] for h in out] == ["H one", "H two"]
    assert len(out[0]["tests"]) == 2          # joined both tests by statement
    assert out[1]["tests"] == []              # H two was never tested


def test_signals_dataframe_and_effect_figure_helpers():
    from evalvitals.analysis.dashboard_app import _signal_effect_figure, _signals_dataframe

    signals = [
        {"name": "a", "source": "explorer", "effect": 0.8, "reject": True},
        {"name": "b", "source": "catalog", "effect": 0.3, "reject": False},
        {"name": "c", "source": "explorer", "effect": None, "reject": None},
    ]
    df = _signals_dataframe(signals)
    assert list(df["verdict"]) == ["REJECT H0", "inconclusive", "descriptive"]

    fig = _signal_effect_figure(signals)  # only the 2 numeric ones plotted
    assert fig is not None
