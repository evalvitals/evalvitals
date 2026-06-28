from __future__ import annotations

import json

from evalvitals.analysis.dashboard import load_loop_story, load_run


def test_load_run_reads_single_explore_report(tmp_path):
    (tmp_path / "exploratory_report.json").write_text(
        json.dumps({
            "ok": True,
            "question": "compare models",
            "observations": ["a"],
            "candidate_signals": [{"name": "trace_steps"}],
            "charts": [{"title": "C", "figure_path": "figures/00_c.png"}],
        }),
        encoding="utf-8",
    )

    run = load_run(tmp_path)

    assert run["root"] == str(tmp_path.resolve())
    assert run["kind"] == "explore"
    assert run["story"] is None
    assert len(run["runs"]) == 1
    assert run["runs"][0]["report"]["ok"] is True


def test_load_run_reads_fused_report(tmp_path):
    (tmp_path / "fused_report.json").write_text(
        json.dumps({"observations": ["x"], "charts": []}), encoding="utf-8"
    )
    run = load_run(tmp_path)
    assert run["kind"] == "explore"
    assert any(r["name"] == "fused_report" for r in run["runs"])


def test_load_run_detects_loop_run_and_parses_story(tmp_path):
    logs = tmp_path / "logs_m2_5"
    logs.mkdir()
    events = [
        {"event": "analysis", "cycle": 1},
        {"event": "diagnosis", "cycle": 1, "n_hypotheses": 2,
         "referenced_charts": ["ObjSize by label"], "explore_context_used": True,
         "hypotheses": [{"statement": "h1", "failure_mode": "fm"}]},
        {"event": "surgery", "cycle": 1, "module": "m5", "status": "supported"},
        {"event": "fix", "cycle": 1},
    ]
    (logs / "run_log.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8"
    )

    run = load_run(tmp_path)
    assert run["kind"] == "loop"
    story = run["story"]
    assert story is not None
    assert len(story["diagnoses"]) == 1
    assert story["diagnoses"][0]["referenced_charts"] == ["ObjSize by label"]
    assert len(story["surgeries"]) == 1 and len(story["fixes"]) == 1


def test_load_run_empty_dir():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        run = load_run(d)
        assert run["kind"] == "empty"
        assert run["runs"] == []


def test_load_loop_story_returns_none_for_explore_output(tmp_path):
    (tmp_path / "exploratory_report.json").write_text("{}", encoding="utf-8")
    assert load_loop_story(tmp_path) is None
