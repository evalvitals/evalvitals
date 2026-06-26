"""Streamlit dashboard for EvalVitals chat/session artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def launch_dashboard(session_dir: str | Path, *, port: int | None = None) -> int:
    """Launch the Streamlit dashboard app for an EvalVitals chat session."""
    try:
        import streamlit  # noqa: F401
    except Exception:
        print(
            "Streamlit is not installed. Install dashboard extras with:\n"
            "  pip install -e '.[dashboard]'",
            file=sys.stderr,
        )
        return 1

    app_path = Path(__file__).with_name("dashboard_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--", str(session_dir)]
    if port is not None:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(port),
            "--",
            str(session_dir),
        ]
    return subprocess.call(cmd)


def load_session(session_dir: str | Path) -> dict[str, Any]:
    """Load chat turns and reports from a session directory."""
    root = Path(session_dir).resolve()
    turns: list[dict[str, Any]] = []
    for turn_dir in sorted(root.glob("turn_*")):
        report_path = turn_dir / "exploratory_report.json"
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {"ok": False, "error": f"Could not read {report_path}"}
        turns.append({"name": turn_dir.name, "dir": str(turn_dir), "report": report})

    history_path = root / "chat_history.json"
    history: list[Any] = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []
    return {"root": str(root), "history": history, "turns": turns}
