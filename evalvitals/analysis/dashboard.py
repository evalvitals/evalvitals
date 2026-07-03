"""Streamlit dashboard for EvalVitals single-run artifacts.

A single-shot pipeline produces one output directory; this loader reads it.
Two product shapes are recognised:

- **explore output**: an ``exploratory_report.json`` (or ``fused_report.json``)
  directly in the directory, with ``figures/`` and ``tables/`` beside it.
- **loop run**: a ``logs_*/run_log.jsonl`` (M2 stats / M3 hypotheses + chart
  references / M5 / Fix) plus an optional ``fused_report.json``.

There is no multi-turn chat session anymore — ``load_run`` replaces the old
``load_session`` that walked ``turn_*`` subdirectories.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from evalvitals.reporting.compiler import compile_diagnostic_report


def launch_dashboard(run_dir: str | Path, *, port: int | None = None) -> int:
    """Launch the Streamlit dashboard app for a single explore/loop run dir."""
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
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if port is not None:
        cmd += ["--server.port", str(port)]
    cmd += ["--", str(run_dir)]
    return subprocess.call(cmd)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load a single explore-output or loop-run directory into a view dict.

    Returns ``{root, kind, runs, story}`` where:
      - ``kind`` is ``"explore"``, ``"loop"`` or ``"empty"``;
      - ``runs`` is a list of ``{name, dir, report}`` (explore reports found,
        most-specific first); the dashboard renders these as report cards;
      - ``story`` is a parsed loop narrative (``None`` for explore output) —
        see :func:`load_loop_story`.
    """
    root = Path(run_dir).resolve()
    runs: list[dict[str, Any]] = []

    for fname in ("exploratory_report.json", "fused_report.json"):
        report = _read_json(root / fname)
        if report is not None:
            runs.append({"name": fname.replace(".json", ""), "dir": str(root), "report": report})

    story = load_loop_story(root)
    if story is not None:
        # A loop run may also carry a fused_report.json (Step 1 explore artifact).
        return {"root": str(root), "kind": "loop", "runs": runs, "story": story}

    if runs:
        return {"root": str(root), "kind": "explore", "runs": runs, "story": None}

    # Legacy / fallback: a directory of turn_* explore reports (pre-retirement).
    for turn_dir in sorted(root.glob("turn_*")):
        report = _read_json(turn_dir / "exploratory_report.json")
        if report is not None:
            runs.append({"name": turn_dir.name, "dir": str(turn_dir), "report": report})
    kind = "explore" if runs else "empty"
    return {"root": str(root), "kind": kind, "runs": runs, "story": None}


def load_loop_story(run_dir: str | Path) -> dict[str, Any] | None:
    """Parse a loop run's ``run_log.jsonl`` into an ordered diagnostic story.

    Looks for ``run_log.jsonl`` directly or under any ``logs*`` subdirectory.
    Returns ``None`` when no loop log is present (i.e. this is explore output).
    """
    root = Path(run_dir).resolve()
    # A single run can be split across several logs (e.g. logs_m1/ for M1 and a
    # logs_m2_5/ or logs_analysis/ for the M2+ arc). We MERGE the shared M1 probe
    # with the M2+ arc — but a directory may hold SEVERAL M2+ arcs that are
    # different runs (e.g. a descriptive analysis-phase pass AND a stale
    # all-in-one confirm pass). Merging both mixes a descriptive run with
    # surgeries/verdicts from another run. So: keep every M1-only log, but among
    # the M2+ logs keep only the single most-recent arc.
    candidate_paths = [
        p for p in [root / "run_log.jsonl", *sorted(root.glob("logs*/run_log.jsonl"))]
        if p.exists()
    ]
    if not candidate_paths:
        return None

    _M2PLUS = {"analysis", "diagnosis", "surgery", "fix"}
    events_by_path: dict[Path, list[dict[str, Any]]] = {}
    for lp in candidate_paths:
        evs: list[dict[str, Any]] = []
        try:
            for line in lp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    evs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
        events_by_path[lp] = evs

    m2plus_logs = [p for p, evs in events_by_path.items()
                   if any(str(e.get("event")) in _M2PLUS for e in evs)]
    if len(m2plus_logs) > 1:
        newest = max(m2plus_logs, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
        # Drop the older/stale M2+ arcs; keep M1-only logs and the newest arc.
        log_paths = [p for p in candidate_paths
                     if p not in m2plus_logs or p == newest]
    else:
        log_paths = candidate_paths

    events: list[dict[str, Any]] = [e for lp in log_paths for e in events_by_path.get(lp, [])]
    if not events:
        return None

    by_event: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        by_event.setdefault(str(ev.get("event", "")), []).append(ev)

    # Display the log that actually carries the diagnostic arc (most events).
    log_path = max(log_paths, key=lambda p: p.stat().st_size if p.exists() else 0)
    explore_report, explore_dir = _find_explore_report(root, log_path)

    story = {
        "log_path": str(log_path),
        "analyses": by_event.get("analysis", []),
        "diagnoses": by_event.get("diagnosis", []),
        "surgeries": by_event.get("surgery", []),
        "fixes": by_event.get("fix", []),
        "n_events": len(events),
        # The Step-1 explorer report (charts/observations M3 was allowed to consult)
        # usually lives in a SIBLING dir (e.g. OUT/fused/) — not next to the log —
        # so resolve it across the run tree rather than requiring co-location.
        "explore_report": explore_report,
        "explore_dir": explore_dir,
    }
    if not story["diagnoses"]:
        proposed = _read_proposed_hypotheses(root)
        if proposed:
            story["diagnoses"] = [{
                "event": "diagnosis",
                "cycle": 0,
                "n_hypotheses": len(proposed),
                "hypotheses": proposed,
                "source": "analysis/proposed_hypotheses.json",
            }]
    story["diagnostic_report"] = compile_diagnostic_report(story, explore_report).to_dict()
    return story


def _read_proposed_hypotheses(root: Path) -> list[dict[str, Any]]:
    """Fallback for analysis-phase dashboards when the M3 log event is absent."""
    for path in (root / "analysis" / "proposed_hypotheses.json", root / "proposed_hypotheses.json"):
        raw = _read_json(path)
        if isinstance(raw, list):
            return [h for h in raw if isinstance(h, dict)]
        if isinstance(raw, dict) and isinstance(raw.get("hypotheses"), list):
            return [h for h in raw["hypotheses"] if isinstance(h, dict)]
    return []


def _find_explore_report(root: Path, log_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Locate the Step-1 explore/fused report for a loop run.

    The loop writes ``run_log.jsonl`` under ``logs*/`` while the fused pipeline
    writes ``fused_report.json`` under a separate ``fused/`` dir. Search the run
    tree and its parents/siblings so the dashboard can show what M3 consulted."""
    bases = [root, log_path.parent, log_path.parent.parent]
    seen: set[Path] = set()
    for base in bases:
        if base in seen or not base.exists():
            continue
        seen.add(base)
        for pattern in (
            "fused_report.json", "exploratory_report.json",
            "fused/fused_report.json", "*/fused_report.json", "*/exploratory_report.json",
        ):
            for cand in sorted(base.glob(pattern)):
                report = _read_json(cand)
                if report is not None and (report.get("charts") or report.get("observations")):
                    return report, str(cand.parent)
    return None, None
