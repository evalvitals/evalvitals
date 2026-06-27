"""Streamlit dashboard for EvalVitals chat/session artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DashboardHandle:
    """Background Streamlit dashboard process."""

    process: subprocess.Popen[Any]
    url: str


def _dashboard_command(
    session_dir: str | Path,
    *,
    port: int | None = None,
    open_browser: bool | None = None,
) -> list[str]:
    app_path = Path(__file__).with_name("dashboard_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if port is not None:
        cmd.extend(["--server.port", str(port)])
    if open_browser is not None:
        cmd.extend(["--server.headless", "false" if open_browser else "true"])
    cmd.extend(["--", str(session_dir)])
    return cmd


def _ensure_streamlit() -> bool:
    try:
        import streamlit  # noqa: F401
    except Exception:
        print(
            "Streamlit is not installed. Install dashboard extras with:\n"
            "  pip install -e '.[dashboard]'",
            file=sys.stderr,
        )
        return False
    return True


def launch_dashboard(session_dir: str | Path, *, port: int | None = None) -> int:
    """Launch the Streamlit dashboard app for an EvalVitals chat session."""
    if not _ensure_streamlit():
        return 1

    return subprocess.call(_dashboard_command(session_dir, port=port))


def start_dashboard(
    session_dir: str | Path,
    *,
    port: int | None = None,
    open_browser: bool = True,
) -> DashboardHandle | None:
    """Start the Streamlit dashboard in the background for an active chat session."""
    if not _ensure_streamlit():
        return None

    cmd = _dashboard_command(session_dir, port=port, open_browser=open_browser)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    url = f"http://localhost:{port}" if port is not None else "Streamlit local URL"
    return DashboardHandle(process=process, url=url)


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
