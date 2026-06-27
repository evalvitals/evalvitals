from __future__ import annotations

import json

from evalvitals.analysis.dashboard import load_session


def test_load_session_reads_turn_reports(tmp_path):
    turn = tmp_path / "turn_001"
    turn.mkdir()
    (tmp_path / "chat_history.json").write_text(
        json.dumps([{"turn": 1, "question": "compare models"}]),
        encoding="utf-8",
    )
    (turn / "exploratory_report.json").write_text(
        json.dumps({
            "ok": True,
            "observations": ["a"],
            "candidate_signals": [{"name": "trace_steps"}],
        }),
        encoding="utf-8",
    )

    session = load_session(tmp_path)

    assert session["root"] == str(tmp_path.resolve())
    assert session["history"][0]["question"] == "compare models"
    assert len(session["turns"]) == 1
    assert session["turns"][0]["report"]["ok"] is True
