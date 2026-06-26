"""Streamlit app for EvalVitals M2 chat outputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from evalvitals.analysis.dashboard import load_session


def main() -> None:
    session_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    session = load_session(session_arg)
    root = Path(session["root"])

    st.set_page_config(page_title="EvalVitals Dashboard", layout="wide")
    st.title("EvalVitals M2 Dashboard")
    st.caption(str(root))

    turns = session["turns"]
    if not turns:
        st.warning("No turn_*/exploratory_report.json files found.")
        return

    labels = [f"{t['name']}  ok={t['report'].get('ok')}" for t in turns]
    selected = st.sidebar.selectbox("Turn", range(len(turns)), format_func=lambda i: labels[i])
    turn = turns[selected]
    turn_dir = Path(turn["dir"])
    report = turn["report"]

    _render_summary(report)
    _render_tables_and_charts(report, turn_dir)
    _render_plots(report, turn_dir)
    _render_artifacts(report, turn_dir)


def _render_summary(report: dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("OK", str(report.get("ok")))
    col2.metric("Attempts", report.get("attempts", 0))
    profile = report.get("data_profile") or {}
    col3.metric("Rows", profile.get("loaded_rows", profile.get("n_rows", "?")))

    if report.get("error"):
        st.error(report["error"])

    st.subheader("Observations")
    observations = report.get("observations") or []
    if observations:
        for obs in observations:
            st.markdown(f"- {obs}")
    else:
        st.caption("No observations.")

    st.subheader("Candidate Signals")
    signals = report.get("candidate_signals") or []
    if signals:
        st.dataframe(pd.DataFrame(signals), use_container_width=True)
    else:
        st.caption("No candidate signals.")

    caveats = report.get("caveats") or []
    if caveats:
        with st.expander("Caveats"):
            for caveat in caveats:
                st.markdown(f"- {caveat}")


def _render_tables_and_charts(report: dict[str, Any], turn_dir: Path) -> None:
    st.subheader("Tables And Charts")
    tables = report.get("tables") or {}
    charts = report.get("charts") or []
    if not tables and not charts:
        st.caption("No structured tables or chart specs were reported.")

    for name, value in tables.items():
        st.markdown(f"**{name}**")
        df = _table_to_dataframe(value, turn_dir)
        if df is not None:
            st.dataframe(df, use_container_width=True)
        else:
            st.json(value)

    for chart in charts:
        if not isinstance(chart, dict):
            continue
        title = str(chart.get("title") or chart.get("name") or "chart")
        st.markdown(f"**{title}**")
        df = _table_to_dataframe(chart.get("data"), turn_dir)
        if df is None:
            st.json(chart)
            continue
        x = chart.get("x")
        y = chart.get("y")
        if x in df.columns and y in df.columns:
            kind = str(chart.get("kind", "bar")).lower()
            if kind in {"line", "timeseries"}:
                st.line_chart(df, x=x, y=y)
            else:
                st.bar_chart(df, x=x, y=y)
        else:
            st.dataframe(df, use_container_width=True)


def _render_plots(report: dict[str, Any], turn_dir: Path) -> None:
    plots = report.get("plots") or []
    if not plots:
        return
    st.subheader("Plots")
    for item in plots:
        path = Path(str(item))
        if not path.is_absolute():
            path = turn_dir / path
        if not path.exists():
            sandbox_path = turn_dir / "sandbox" / str(item)
            path = sandbox_path if sandbox_path.exists() else path
        if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            st.image(str(path), caption=str(path))
        else:
            st.caption(f"Missing plot: {item}")


def _render_artifacts(report: dict[str, Any], turn_dir: Path) -> None:
    st.subheader("Artifacts")
    with st.expander("Generated analysis.py"):
        code = report.get("code") or _read_text(turn_dir / "analysis.py")
        st.code(code or "", language="python")
    with st.expander("stdout"):
        st.text(_read_text(turn_dir / "stdout.txt") or report.get("stdout", ""))
    with st.expander("stderr"):
        st.text(_read_text(turn_dir / "stderr.txt") or report.get("stderr", ""))
    with st.expander("Raw JSON report"):
        st.json(report)


def _table_to_dataframe(value: Any, turn_dir: Path) -> pd.DataFrame | None:
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            candidates = [turn_dir / path, turn_dir / "sandbox" / path]
            path = next((p for p in candidates if p.exists()), candidates[0])
        if path.exists() and path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return None
    if isinstance(value, list):
        try:
            return pd.DataFrame(value)
        except Exception:
            return None
    if isinstance(value, dict):
        try:
            return pd.DataFrame(value)
        except Exception:
            try:
                return pd.DataFrame([value])
            except Exception:
                return None
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


if __name__ == "__main__":
    main()
