"""EvalVitals web workbench — upload a .zip of results, run M2+M3, read the report.

``evalvitals web [WORKSPACE] [--port N] [--backend B] ...`` serves this
Streamlit app. Unlike ``dashboard_app.py`` (a read-only viewer over one
existing run directory), this app *creates* runs: each uploaded archive
becomes one run directory under the workspace and is analysed by the ordinary
``evalvitals explore`` CLI (M2 exploratory analysis + M3 hypothesis proposal)
running as a detached subprocess — closing the browser tab never kills an
analysis, and a finished run renders with the exact same tabs as
``evalvitals dashboard <out>``.

Run layout::

    <workspace>/<run-name>/
        upload.zip     the archive exactly as received
        data/          extracted payload (what the explorer reads)
        output/        explore artifacts (exploratory_report.json, figures/, ...)
        explore.log    live stdout+stderr of the analysis
        job.sh         the exact command (re-runnable by hand)
        job.json       parameters + pid + start time
        exit_code      written when the subprocess finishes
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

# Backends the explore CLI accepts; the first three are skill-capable and get
# the bundled figure/analysis skills automatically (see agent_runtime.skills).
BACKENDS = ("claude_code", "antigravity", "codex", "opencode", "gemini_cli", "kimi_cli")

DEFAULT_QUESTION = "Explore this dataset and surface the patterns that matter."

_JUNK_DIRS = ("__MACOSX",)
_JUNK_NAMES = (".DS_Store", "Thumbs.db")

_STATE_ICONS = {"running": "🟡", "done": "🟢", "failed": "🔴", "stale": "⚪"}


# ── run bookkeeping (pure helpers, unit-tested without streamlit) ────────────


def stage_zip(payload: bytes, run_dir: Path) -> Path:
    """Extract an uploaded zip into ``run_dir/data`` and return the directory
    the explorer should read.

    Guards against zip-slip (absolute member names or ``..`` escapes raise
    ``ValueError``), skips archive junk (``__MACOSX/``, ``.DS_Store``), and
    unwraps the common single-top-level-folder layout ("zip of a directory"):
    when everything sits under one folder, that folder is returned instead of
    ``data/`` itself.
    """
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_root = data_dir.resolve()

    n_files = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir():
                continue
            parts = Path(name).parts
            if name.startswith("/") or ".." in parts:
                raise ValueError(f"unsafe path in archive: {info.filename!r}")
            if any(p in _JUNK_DIRS for p in parts) or parts[-1] in _JUNK_NAMES:
                continue
            target = (data_dir / name).resolve()
            if not target.is_relative_to(data_root):
                raise ValueError(f"unsafe path in archive: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            n_files += 1
    if n_files == 0:
        raise ValueError("archive contains no files")

    children = [p for p in data_dir.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return data_dir


def build_explore_argv(
    data_dir: Path,
    out_dir: Path,
    *,
    question: str,
    outcome_col: str,
    backend: str,
    model: str,
    timeout_sec: int,
) -> list[str]:
    """The exact ``evalvitals explore`` invocation for one uploaded run."""
    argv = [
        sys.executable, "-m", "evalvitals.cli", "explore", str(data_dir),
        "--backend", backend,
        "--out", str(out_dir),
        "-q", question,
        "--timeout-sec", str(int(timeout_sec)),
        "--max-attempts", "3",
    ]
    if outcome_col:
        argv += ["--outcome-col", outcome_col]
    if model:
        argv += ["--model", model]
    return argv


def launch_explore_job(
    run_dir: Path,
    data_dir: Path,
    *,
    question: str,
    outcome_col: str,
    backend: str,
    model: str,
    timeout_sec: int,
) -> dict[str, Any]:
    """Start the analysis as a detached subprocess and persist its job record.

    The command goes through a tiny ``job.sh`` wrapper so the exit code lands
    in ``exit_code`` even if this server restarts while the run is in flight.
    """
    argv = build_explore_argv(
        data_dir, run_dir / "output", question=question, outcome_col=outcome_col,
        backend=backend, model=model, timeout_sec=timeout_sec,
    )
    job_sh = run_dir / "job.sh"
    job_sh.write_text(
        "#!/usr/bin/env bash\n"
        + " ".join(shlex.quote(a) for a in argv)
        + " > explore.log 2>&1\necho $? > exit_code\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        ["bash", str(job_sh)], cwd=run_dir, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    job = {
        "pid": proc.pid,
        "argv": argv,
        "question": question,
        "outcome_col": outcome_col,
        "backend": backend,
        "model": model,
        "timeout_sec": int(timeout_sec),
        "data_dir": str(data_dir),
        "out_dir": str(run_dir / "output"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
    return job


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def job_status(run_dir: Path) -> dict[str, Any]:
    """Classify one run dir: ``done`` (report exists), ``running`` (pid alive),
    ``failed`` (exited without a report), or ``stale`` (no process, no exit
    code — e.g. the machine restarted mid-run)."""
    report_path = run_dir / "output" / "exploratory_report.json"
    exit_code: int | None = None
    exit_file = run_dir / "exit_code"
    if exit_file.exists():
        try:
            exit_code = int(exit_file.read_text().strip())
        except ValueError:
            exit_code = None

    if report_path.exists():
        state = "done"
    elif exit_code is not None:
        state = "failed"
    else:
        pid = 0
        try:
            pid = int(json.loads((run_dir / "job.json").read_text()).get("pid", 0))
        except Exception:
            pass
        state = "running" if _pid_alive(pid) else "stale"
    return {"state": state, "exit_code": exit_code, "report_path": str(report_path)}


def list_runs(workspace: Path) -> list[Path]:
    """Run dirs under the workspace (anything carrying a job.json), newest first."""
    runs = [d for d in workspace.iterdir() if d.is_dir() and (d / "job.json").exists()]
    return sorted(runs, key=lambda d: (d / "job.json").stat().st_mtime, reverse=True)


def _run_name(upload_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(upload_name).stem).strip("_") or "upload"
    return f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _log_tail(run_dir: Path, n_lines: int = 60) -> str:
    log = run_dir / "explore.log"
    if not log.exists():
        return "(no output yet)"
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:]) or "(no output yet)"


# ── streamlit app ────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", nargs="?", default="evalvitals_web_runs")
    parser.add_argument("--backend", default="claude_code", choices=list(BACKENDS))
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-sec", type=int, default=1200)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def main() -> None:
    import streamlit as st

    from evalvitals.analysis import dashboard_app as dapp

    args = _parse_args()
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    st.set_page_config(
        page_title="EvalVitals — Upload & Explore", layout="wide",
        initial_sidebar_state="expanded",
    )
    dapp._inject_css()

    runs = list_runs(workspace)
    states = {d.name: job_status(d)["state"] for d in runs}

    new_label = "➕ New analysis"
    st.sidebar.markdown('<div class="ev-sidebar-title">EvalVitals</div>',
                        unsafe_allow_html=True)
    st.sidebar.caption(str(workspace))
    choice = st.sidebar.radio(
        "Runs",
        [new_label] + [d.name for d in runs],
        key="ev_run_choice",
        format_func=lambda v: v if v == new_label
        else f"{_STATE_ICONS.get(states.get(v, 'stale'), '⚪')} {v}",
    )
    st.sidebar.markdown("---")
    st.sidebar.metric("Runs", len(runs))

    if choice == new_label:
        _render_new_analysis(st, workspace, args, new_label)
    else:
        _render_run(st, dapp, workspace / choice)


def _render_new_analysis(st: Any, workspace: Path, args: argparse.Namespace,
                         new_label: str) -> None:
    st.markdown(
        '<div class="ev-hero"><h1>Upload &amp; Explore</h1></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Upload a **.zip** of results (JSON / JSONL / CSV files — a zipped "
        "folder works too). The archive is extracted and handed as-is to "
        "`evalvitals explore`: the M2 agent reads the files, figures out their "
        "shape itself, analyses them, and M3 proposes falsifiable hypotheses."
    )

    uploaded = st.file_uploader("Results archive", type=["zip"])
    question = st.text_area("Analysis question", value=DEFAULT_QUESTION, height=90)
    col1, col2, col3, col4 = st.columns(4)
    outcome_col = col1.text_input(
        "Outcome column", value="label",
        help="Name of the pass/fail (or target) column. Leave empty to "
             "auto-detect by name heuristics, or fall back to unsupervised EDA.",
    )
    backend = col2.selectbox("Coding-agent backend", list(BACKENDS),
                             index=list(BACKENDS).index(args.backend))
    model = col3.text_input("Model (optional)", value=args.model,
                            help="Backend-specific model id, e.g. claude-opus-4-8.")
    timeout_sec = col4.number_input("Timeout (sec)", min_value=60, max_value=7200,
                                    value=int(args.timeout_sec), step=60)

    if st.button("Start analysis", type="primary", disabled=uploaded is None):
        payload = uploaded.getvalue()
        run_dir = workspace / _run_name(uploaded.name)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            (run_dir / "upload.zip").write_bytes(payload)
            data_dir = stage_zip(payload, run_dir)
        except (ValueError, zipfile.BadZipFile) as exc:
            shutil.rmtree(run_dir, ignore_errors=True)
            st.error(f"Could not read the archive: {exc}")
            st.stop()
        launch_explore_job(
            run_dir, data_dir, question=question.strip() or DEFAULT_QUESTION,
            outcome_col=outcome_col.strip(), backend=backend,
            model=model.strip(), timeout_sec=int(timeout_sec),
        )
        st.session_state["ev_run_choice"] = run_dir.name
        st.rerun()


def _render_run(st: Any, dapp: Any, run_dir: Path) -> None:
    try:
        job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    except Exception:
        job = {}
    status = job_status(run_dir)
    state = status["state"]

    st.markdown(f"## {_STATE_ICONS.get(state, '⚪')} {run_dir.name}")
    bits = [f"state: **{state}**"]
    if job.get("started_at"):
        bits.append(f"started {job['started_at']}")
    if job.get("backend"):
        bits.append(f"backend `{job['backend']}`" + (f" · `{job['model']}`" if job.get("model") else ""))
    st.caption(" · ".join(bits))

    with st.expander("Run parameters", expanded=False):
        st.markdown(f"**Question:** {job.get('question', '—')}")
        st.markdown(f"**Outcome column:** `{job.get('outcome_col') or '(auto)'}`")
        st.code(" ".join(shlex.quote(a) for a in job.get("argv", [])) or "(unknown)",
                language="bash")

    if state == "running":
        st.info("Analysis in flight — M2 exploratory analysis + M3 hypothesis "
                "proposal. This page only observes the run; closing it changes "
                "nothing.")
        st.code(_log_tail(run_dir), language="text")
        if st.toggle("Auto-refresh every 5 s", value=True):
            time.sleep(5)
            st.rerun()
        return

    if state == "failed":
        st.error(f"The analysis exited with code {status['exit_code']} before "
                 "producing a report.")
        st.code(_log_tail(run_dir, n_lines=120), language="text")
        st.caption("Common causes: the backend CLI is not on PATH, the API key "
                   "is missing, or the timeout is too low for the chosen model.")
        return

    if state == "stale":
        st.warning("No live process and no exit code — the server likely "
                   "restarted while this run was in flight. Re-run it by hand "
                   "with the command under *Run parameters* (`bash job.sh`).")
        st.code(_log_tail(run_dir), language="text")
        return

    # done — render with the exact same tabs as `evalvitals dashboard <out>`.
    from evalvitals.analysis.dashboard import load_run

    out_dir = Path(job.get("out_dir") or (run_dir / "output"))
    session = load_run(out_dir)
    if not session["runs"]:
        st.error("exploratory_report.json exists but could not be parsed.")
        st.code(_log_tail(run_dir, n_lines=120), language="text")
        return
    turn = session["runs"][0]
    if not turn["report"].get("ok", True):
        st.warning("The explorer finished with an error — rendering whatever "
                   f"was produced. Error: {turn['report'].get('error') or 'unknown'}")
    dapp.render_explore_report(Path(session["root"]), turn)


if __name__ == "__main__":
    main()
