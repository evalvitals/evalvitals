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
        "dashboard_storyboard": [{
            "id": "analysis",
            "title": "Analysis",
            "stages": ["M2"],
            "summary": "Agent-authored dashboard narrative for this run.",
            "items": ["Method: held-out comparison", "Takeaway: inspect the signal"],
            "artifact_refs": ["candidate_signals", "charts"],
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
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting",
        "2 Analysis",
        "3 Hypotheses & Artifacts",
    ]

    # The analysis tab renders claim-first evidence panels with nearby support.
    assert len(at.dataframe) >= 1
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Problem Setting" in blob
    assert "Bottom line" in blob
    assert "Evidence you can use" in blob
    assert "How to read this evidence" in blob
    assert "Takeaway" in blob
    assert "Supporting experiment" in blob
    assert "Method:" in blob
    assert "Takeaway:" in blob
    assert "Agent-authored dashboard narrative for this run." in blob
    assert "M1" in blob and "M2" in blob and "M3" in blob and "M5" in blob


def test_standalone_m2_dashboard_does_not_show_hypothesis_tab(tmp_path):
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "summary": "M2-only exploratory analysis.",
        "observations": ["Candidate pattern surfaced for review."],
        "candidate_signals": [{"name": "candidate_signal", "effect": 0.4}],
        "recommended_confirmatory_tests": ["Formulate an explicit hypothesis later."],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting",
        "2 Exploratory Analysis",
        "3 Artifacts",
    ]
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "Suggested next steps" in blob
    assert "Hypotheses & Artifacts" not in blob


def test_standalone_dashboard_pairs_takeaway_with_its_chart_and_analysis(tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "yield_by_temp.csv").write_text(
        "temp_bin,mean_yield\nlow,70.1\nhigh,88.4\n", encoding="utf-8"
    )
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "question": "What predicts yield?",
        "observations": ["30 batches, no missing values."],
        "takeaways": [{
            "title": "Higher temperature batches yield more (70.1% vs 88.4%).",
            "chart_names": ["yield_by_temp"],
            "table_names": [],
            "analysis": "Mean yield rises from 70.1% in low-temperature batches to 88.4% in high-temperature ones.",
            "caveat": "Descriptive only; temperature and pressure are correlated.",
        }],
        "charts": [{
            "name": "yield_by_temp", "kind": "line",
            "data": "tables/yield_by_temp.csv", "x": "temp_bin", "y": "mean_yield",
            "title": "Mean yield by temperature bin",
        }],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(m.value) for m in at.markdown)
    # the takeaway's numbered badge, headline, analysis, and caveat all render together
    assert 'ev-takeaway-badge">1<' in blob
    assert "Higher temperature batches yield more" in blob
    assert "Mean yield rises from 70.1%" in blob
    assert "Caveat" in blob and "Descriptive only" in blob
    # its supporting chart was found and rendered, not left orphaned
    assert "referenced evidence not found" not in blob
    assert "Mean yield by temperature bin" in blob  # the chart's own title rendered
    # no leftover M2/hypothesis-generation branding in the standalone view
    assert "Standalone M2" not in blob
    assert "M3 hypotheses" not in blob
    assert "hypothesis formation" not in blob


def test_standalone_dashboard_falls_back_gracefully_with_no_takeaways(tmp_path):
    (tmp_path / "fused_report.json").write_text(json.dumps({
        "observations": ["No structured takeaways in this older report."],
        "charts": [],
    }), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    blob = " ".join(str(i.value) for i in at.info)
    assert "no structured takeaways" in blob.lower()


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
    blob = " ".join(str(m.value) for m in at.markdown)
    # the three-panel narrative sections are present
    assert "ev-section-title\">Problem Setting<" in blob
    assert any("Analysis" in h for h in heads)
    assert any("Hypotheses & Decision" in h for h in heads)
    assert "Measurement" in blob
    assert "Confirmatory analysis" in blob
    assert "Hypothesis generation" in blob
    # the hypothesis statement + its M5 verdict appear somewhere in the page
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


def _build_analysis_phase_run(root):
    """Same fused report as the confirmatory fixture, but the loop log is an
    analysis-phase pass (descriptive_only, no surgery) under logs_analysis/."""
    _build_loop_run(root)
    # Replace the confirmatory M2_5 log with an analysis-phase logs_analysis/ log.
    import shutil
    shutil.rmtree(root / "logs_m2_5")
    logs = root / "logs_analysis"
    logs.mkdir(parents=True)
    events = [
        {"event": "analysis", "cycle": 0, "descriptive_only": True,
         "conclusion": "Candidate signals described; verdict deferred."},
        {"event": "diagnosis", "cycle": 0, "n_hypotheses": 1,
         "hypotheses": [{"statement": "language-prior hallucination", "failure_mode": "lp"}],
         "referenced_charts": ["Fail rate by signal"]},
        {"event": "loop_end", "stopped_by": "analysis_complete"},
    ]
    (logs / "run_log.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return root


def test_analysis_phase_dashboard_shows_no_supported_verdicts(tmp_path):
    # WS2 / decoupling: an analysis-phase run must present candidate signals
    # DESCRIPTIVELY — no "Supported" / "Not supported" verdict pills, since the
    # explorer's reject flags are not confirmation.
    _build_analysis_phase_run(tmp_path)
    at = _run_app(tmp_path)
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting",
        "2 Analysis",
        "3 Proposed Hypotheses",
    ]
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "language-prior hallucination" in blob
    assert "Proposed Hypotheses" in blob
    # the descriptive evidence pills carry "Descriptive", never a verdict
    assert "Descriptive</span>" in blob
    assert ">Supported</span>" not in blob
    assert ">Not supported</span>" not in blob
    # the confirmatory method card ("tested signals survived") is replaced
    assert "tested signals survived" not in blob
    # an explicit analysis-phase banner orients the reader
    assert any("Analysis phase" in str(i.value) for i in at.info)


def test_analysis_phase_dashboard_recovers_proposed_hypotheses_file(tmp_path):
    _build_analysis_phase_run(tmp_path)
    logs = tmp_path / "logs_analysis"
    (logs / "run_log.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "analysis", "cycle": 0, "descriptive_only": True,
                        "conclusion": "Candidate signals described; verdict deferred."}),
            json.dumps({"event": "loop_end", "stopped_by": "analysis_complete"}),
        ]) + "\n",
        encoding="utf-8",
    )
    analysis = tmp_path / "analysis"
    analysis.mkdir(exist_ok=True)
    (analysis / "proposed_hypotheses.json").write_text(json.dumps([
        {
            "hypothesis": "fallback hypothesis from proposed_hypotheses.json",
            "predicted_failure_mode": "fallback_mode",
        }
    ]), encoding="utf-8")

    at = _run_app(tmp_path)

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "1 Problem Setting",
        "2 Analysis",
        "3 Proposed Hypotheses",
    ]
    blob = " ".join(str(m.value) for m in at.markdown)
    assert "fallback hypothesis from proposed_hypotheses.json" in blob
    assert "fallback_mode" in blob


def test_resolve_scatter_axes_handles_real_named_and_legacy_columns():
    import pandas as pd

    from evalvitals.analysis.dashboard_app import _resolve_scatter_axes

    # (a) newer schema: real signal names as columns, no report hint available.
    d = pd.DataFrame({"attention_entropy": [0.8, 0.6], "center_offset": [0.2, 0.4],
                      "outcome": ["PASS", "FAIL"]})
    _d, xs, ys, oc = _resolve_scatter_axes(d, {}, "scatter_top_attention_pair.csv")
    assert (xs, ys, oc) == ("attention_entropy", "center_offset", "outcome")

    # (b) legacy schema: literal x/y renamed from the report's chart title.
    d2 = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "outcome": ["FAIL", "PASS"]})
    report = {"charts": [{"name": "scatter_pair",
                          "title": "focus_share vs edge_mass by outcome"}]}
    d2r, xs, ys, oc = _resolve_scatter_axes(d2, report, "scatter_pair.csv")
    assert (xs, ys, oc) == ("focus_share", "edge_mass", "outcome")
    assert {"focus_share", "edge_mass"} <= set(d2r.columns)  # columns were renamed

    # (c) unresolvable (no two value axes) → all-None sentinel, no crash.
    bad = pd.DataFrame({"outcome": ["FAIL", "PASS"]})
    _b, xs, ys, oc = _resolve_scatter_axes(bad, {}, "scatter_bad.csv")
    assert (xs, ys, oc) == (None, None, None)


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
