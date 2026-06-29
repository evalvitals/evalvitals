from __future__ import annotations

import json

from evalvitals.analysis.dashboard import load_run
from evalvitals.reporting.compiler import compile_diagnostic_report


def _explore_report():
    return {
        "question": "q",
        "conclusion": "Supported signal found.",
        "observations": ["obs"],
        "visual_plan": [{"name": "v", "plot_kind": "bar"}],
        "chart_readings": [{"chart": "C", "reading": "read"}],
        "charts": [{"name": "c", "title": "C", "data": "tables/c.csv"}],
        "candidate_signals": [
            {"name": "low_focus", "effect": 0.4, "ci": [0.1, 0.7], "reject": True},
            {"name": "probe1_false_detection", "effect": 1.0, "ci": [1.0, 1.0],
             "reject": True},
        ],
        "critique": ["watch leakage"],
    }


def test_compile_diagnostic_report_is_claim_first():
    story = {
        "analyses": [{"cycle": 1, "conclusion": "M2 supported low_focus"}],
        "diagnoses": [{"cycle": 1, "hypotheses": [{"statement": "low_focus cause"}]}],
    }
    report = compile_diagnostic_report(story, _explore_report())

    assert report.answer.startswith("Low focus is associated with FAIL cases")
    assert "label-like" in report.answer
    by_text = {c.text: c for c in report.claims}
    assert any("Low focus is associated" in text for text in by_text)
    leaky = next(c for c in report.claims if "Label audit" in c.text)
    assert leaky.status == "descriptive"
    assert report.chart_readings == [{"chart": "C", "reading": "read"}]
    assert report.critique == ["watch leakage"]


def test_load_run_attaches_diagnostic_report(tmp_path):
    fused = tmp_path / "fused"
    logs = tmp_path / "logs_m2_5"
    fused.mkdir()
    logs.mkdir()
    (fused / "fused_report.json").write_text(json.dumps(_explore_report()), encoding="utf-8")
    (logs / "run_log.jsonl").write_text(
        json.dumps({"event": "analysis", "cycle": 1, "conclusion": "ok"}) + "\n",
        encoding="utf-8",
    )

    loaded = load_run(tmp_path)

    assert loaded["kind"] == "loop"
    diag = loaded["story"]["diagnostic_report"]
    assert diag["answer"].startswith("Low focus is associated with FAIL cases")
    assert diag["claims"]
