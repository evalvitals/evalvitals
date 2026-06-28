"""Streamlit app for EvalVitals M2 chat outputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from evalvitals.analysis.dashboard import load_run


def main() -> None:
    run_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    session = load_run(run_arg)
    root = Path(session["root"])
    runs = session["runs"]

    st.set_page_config(page_title="EvalVitals", layout="wide", initial_sidebar_state="expanded")
    _inject_css()

    selected = _render_sidebar(root, session)

    if session.get("kind") == "loop" and session.get("story"):
        _render_loop_story(root, session["story"], runs)
        return

    if not runs:
        _render_empty(root)
        return

    turn = runs[selected]
    turn_dir = Path(turn["dir"])
    report = turn["report"]

    _render_header(root, turn, report)
    _render_top_metrics(report)

    overview, charts, tables, artifacts = st.tabs(["Overview", "Charts", "Tables", "Artifacts"])
    with overview:
        _render_overview(report)
    with charts:
        _render_charts_and_plots(report, turn_dir)
    with tables:
        _render_tables(report, turn_dir)
    with artifacts:
        _render_artifacts(report, turn_dir)


def _render_sidebar(root: Path, session: dict[str, Any]) -> int:
    runs = session["runs"]
    kind = session.get("kind", "explore")

    st.sidebar.markdown('<div class="ev-sidebar-title">EvalVitals</div>', unsafe_allow_html=True)
    st.sidebar.caption(str(root))
    st.sidebar.markdown(f"**Mode:** {kind}")

    if not runs:
        return 0

    if kind == "loop":
        st.sidebar.markdown("---")
        st.sidebar.caption("Diagnostic loop run — see the story view.")
        return 0

    labels = [_turn_label(t) for t in runs]
    selected = st.sidebar.radio("Reports", range(len(runs)), format_func=lambda i: labels[i])

    st.sidebar.markdown("---")
    st.sidebar.metric("Reports", len(runs))
    return int(selected)


def _render_empty(root: Path) -> None:
    st.markdown('<div class="ev-hero"><h1>EvalVitals Dashboard</h1></div>', unsafe_allow_html=True)
    st.warning("No exploratory_report.json / fused_report.json / run_log.jsonl found.")
    st.caption(str(root))


def _render_loop_story(root: Path, story: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    """Render a diagnostic loop run as an ordered story: explore charts (Step 1)
    → M2 stats → M3 hypotheses (each tagged with the explore charts/observations
    it referenced) → M5 tests → fixes."""
    st.markdown(
        f"""
        <div class="ev-header">
          <div>
            <div class="ev-kicker">Diagnostic Loop Run</div>
            <h1>M1 → M2 → M3 → M5 → Fix</h1>
            <div class="ev-path">{_html_escape(str(root))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step 1 explore artifacts (charts the loop's M3 was allowed to consult). The
    # report usually lives in a sibling dir, so load_loop_story resolves it for us.
    explore_report = story.get("explore_report")
    if not explore_report:
        explore_report = next((r["report"] for r in runs if r["name"] == "fused_report"), None)
    explore_dir = Path(story.get("explore_dir") or (runs[0]["dir"] if runs else root))
    if explore_report:
        with st.expander("Step 1 — exploratory charts & observations (UNCONFIRMED; fed to M3 only)", expanded=True):
            _render_charts_and_plots(explore_report, explore_dir)
            obs = explore_report.get("observations") or []
            for o in obs[:8]:
                st.markdown(f"- {o}")

    analyses = story.get("analyses") or []
    diagnoses = story.get("diagnoses") or []
    surgeries = story.get("surgeries") or []
    fixes = story.get("fixes") or []

    if analyses:
        st.markdown("### M2 — stats analysis")
        for a in analyses:
            sev = a.get("severity") or a.get("analysis_severity") or ""
            concl = a.get("conclusion") or a.get("narrative") or ""
            st.markdown(f"- **Cycle {a.get('cycle')}** {('· ' + str(sev)) if sev else ''}")
            if concl:
                st.caption(_truncate(str(concl), 280))

    st.markdown("### M3 — hypotheses")
    if not diagnoses:
        st.caption("No diagnosis events in the loop log.")
    for diag in diagnoses:
        cycle = diag.get("cycle")
        st.markdown(f"**Cycle {cycle}** · {diag.get('n_hypotheses', 0)} hypotheses")
        refs = diag.get("referenced_charts") or []
        if refs:
            st.caption("M3 referenced explore artifacts: " + ", ".join(str(r) for r in refs))
        for h in diag.get("hypotheses") or []:
            st.markdown(
                f"""
                <div class="ev-signal">
                  <div class="ev-signal-title">{_html_escape(str(h.get('statement', '')))}</div>
                  <div class="ev-signal-test">failure_mode: {_html_escape(str(h.get('failure_mode', '')))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if surgeries:
        st.markdown("### M5 / M4 — tested interventions")
        for s in surgeries:
            tag = str(s.get("module", "")).upper()
            status = str(s.get("status", ""))
            st.markdown(f"- **[{tag}]** {status} · {_truncate(str(s.get('hypothesis', '')), 120)}")

    if fixes:
        st.markdown("### Fix — e-BH adjudicated outcomes")
        for f in fixes:
            st.json(f)


def _render_header(root: Path, turn: dict[str, Any], report: dict[str, Any]) -> None:
    ok = bool(report.get("ok"))
    status = "finished" if ok else "failed"
    status_class = "ev-pill-ok" if ok else "ev-pill-fail"
    question = str(report.get("question") or "Exploratory analysis")

    st.markdown(
        f"""
        <div class="ev-header">
          <div>
            <div class="ev-kicker">Standalone M2 Analysis</div>
            <h1>{_html_escape(question)}</h1>
            <div class="ev-path">{_html_escape(str(root))}</div>
          </div>
          <div class="ev-header-right">
            <span class="ev-pill {status_class}">{status}</span>
            <span class="ev-pill">{_html_escape(str(turn["name"]))}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_top_metrics(report: dict[str, Any]) -> None:
    profile = report.get("data_profile") or {}
    columns = profile.get("columns") or {}
    observations = report.get("observations") or []
    signals = report.get("candidate_signals") or []
    charts = report.get("charts") or []
    plots = report.get("plots") or []

    metrics = [
        ("Rows", _format_int(profile.get("loaded_rows", profile.get("n_rows"))), "records sampled"),
        ("Columns", _format_int(len(columns) if isinstance(columns, dict) else None), "profiled fields"),
        ("Signals", _format_int(len(signals)), "candidate follow-ups"),
        ("Charts", _format_int(len(charts) + len(plots)), "visual artifacts"),
        ("Attempts", _format_int(report.get("attempts", 0)), "agent/code runs"),
        ("Notes", _format_int(len(observations)), "observations"),
    ]

    cols = st.columns(len(metrics))
    for col, (label, value, caption) in zip(cols, metrics, strict=False):
        with col:
            st.markdown(
                f"""
                <div class="ev-metric-card">
                  <div class="ev-metric-label">{label}</div>
                  <div class="ev-metric-value">{value}</div>
                  <div class="ev-metric-caption">{caption}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if report.get("error"):
        st.error(report["error"])


def _render_overview(report: dict[str, Any]) -> None:
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        st.markdown("### Observations")
        observations = report.get("observations") or []
        if observations:
            for i, obs in enumerate(observations, start=1):
                st.markdown(
                    f"""
                    <div class="ev-note">
                      <div class="ev-note-index">{i}</div>
                      <div>{_html_escape(str(obs))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No observations.")

        tests = report.get("recommended_confirmatory_tests") or []
        if tests:
            st.markdown("### Confirmatory Next Steps")
            for item in tests:
                st.markdown(f"- {item}")

    with right:
        st.markdown("### Candidate Signals")
        signals = report.get("candidate_signals") or []
        if signals:
            for signal in signals:
                if not isinstance(signal, dict):
                    st.markdown(f"- {signal}")
                    continue
                name = str(signal.get("name") or "Signal")
                rationale = str(signal.get("rationale") or "")
                suggested = str(signal.get("suggested_test") or "")
                st.markdown(
                    f"""
                    <div class="ev-signal">
                      <div class="ev-signal-title">{_html_escape(name)}</div>
                      <div class="ev-signal-body">{_html_escape(rationale)}</div>
                      <div class="ev-signal-test">{_html_escape(suggested)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No candidate signals.")

        caveats = report.get("caveats") or []
        if caveats:
            with st.expander("Caveats", expanded=True):
                for caveat in caveats:
                    st.markdown(f"- {caveat}")


def _render_charts_and_plots(report: dict[str, Any], turn_dir: Path) -> None:
    charts = [c for c in report.get("charts", []) if isinstance(c, dict)]
    plots = report.get("plots") or []

    if not charts and not plots:
        st.caption("No charts or plots were reported.")
        return

    if charts:
        st.markdown("### Interactive Charts")
        chart_cols = st.columns(2)
        for idx, chart in enumerate(charts):
            with chart_cols[idx % 2]:
                _render_chart_card(chart, turn_dir)

    if plots:
        st.markdown("### Generated Figures")
        plot_cols = st.columns(2)
        for idx, item in enumerate(plots):
            with plot_cols[idx % 2]:
                _render_plot_card(item, turn_dir)


def _render_chart_card(chart: dict[str, Any], turn_dir: Path) -> None:
    title = str(chart.get("title") or chart.get("name") or "Chart")
    df = _table_to_dataframe(chart.get("data"), turn_dir)

    st.markdown(f'<div class="ev-card-title">{_html_escape(title)}</div>', unsafe_allow_html=True)

    # Prefer the host-rendered PNG (deterministic, what M3 saw) when present.
    fig_path = chart.get("figure_path")
    if fig_path:
        p = _resolve_artifact_path(fig_path, turn_dir)
        if p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            st.image(str(p), width="stretch")
            return
    if chart.get("render_skipped"):
        st.caption(f"(render skipped: {chart['render_skipped']})")

    if df is None:
        st.json(chart)
        return

    x = chart.get("x")
    y = chart.get("y")
    if x in df.columns and y in df.columns:
        kind = str(chart.get("kind", "bar")).lower()
        if kind in {"line", "timeseries"}:
            st.line_chart(df, x=x, y=y, height=280)
        else:
            st.bar_chart(df, x=x, y=y, height=280)
    else:
        st.dataframe(df, width="stretch", height=280)


def _render_plot_card(item: Any, turn_dir: Path) -> None:
    path = _resolve_artifact_path(item, turn_dir)
    title = Path(str(item)).name
    st.markdown(f'<div class="ev-card-title">{_html_escape(title)}</div>', unsafe_allow_html=True)
    if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        st.image(str(path), width="stretch")
    else:
        st.caption(f"Missing plot: {item}")


def _render_tables(report: dict[str, Any], turn_dir: Path) -> None:
    tables = report.get("tables") or {}
    charts = [c for c in report.get("charts", []) if isinstance(c, dict)]
    chart_tables = {
        str(c.get("name") or c.get("title") or f"chart_{idx}"): c.get("data")
        for idx, c in enumerate(charts)
        if c.get("data") is not None
    }

    if not tables and not chart_tables:
        st.caption("No structured tables were reported.")
        return

    names = list(tables.keys()) + [n for n in chart_tables if n not in tables]
    selected = st.selectbox("Table", names)
    source = tables.get(selected, chart_tables.get(selected))
    df = _table_to_dataframe(source, turn_dir)
    if df is None:
        st.json(source)
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Missing Cells", int(df.isna().sum().sum()))
    st.dataframe(df, width="stretch", height=520)


def _render_artifacts(report: dict[str, Any], turn_dir: Path) -> None:
    st.markdown("### Run Artifacts")
    with st.expander("Generated analysis.py", expanded=True):
        code = report.get("code") or _read_text(turn_dir / "analysis.py")
        st.code(code or "", language="python")
    with st.expander("stdout"):
        st.text(_read_text(turn_dir / "stdout.txt") or report.get("stdout", ""))
    with st.expander("stderr"):
        st.text(_read_text(turn_dir / "stderr.txt") or report.get("stderr", ""))
    with st.expander("Raw JSON report"):
        st.json(report)


def _turn_label(turn: dict[str, Any]) -> str:
    report = turn["report"]
    status = "ok" if report.get("ok") else "failed"
    question = str(report.get("question") or "analysis")
    return f"{turn['name']} · {status} · {_truncate(question, 46)}"


def _table_to_dataframe(value: Any, turn_dir: Path) -> pd.DataFrame | None:
    if isinstance(value, str):
        path = _resolve_artifact_path(value, turn_dir)
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


def _resolve_artifact_path(value: Any, turn_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute() and path.exists():
        return path
    # Relative path, or an absolute path that no longer exists (run dir moved to
    # another machine): fall back through the local layout by basename.
    candidates = [
        turn_dir / path if not path.is_absolute() else turn_dir / path.name,
        turn_dir / path.name,
        turn_dir / "figures" / path.name,
        turn_dir / "tables" / path.name,
        turn_dir / "sandbox" / path.name,
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _format_int(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _truncate(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ev-bg: #f7f8fa;
          --ev-panel: #ffffff;
          --ev-border: #dfe3ea;
          --ev-text: #17202a;
          --ev-muted: #667085;
          --ev-accent: #1f7a8c;
          --ev-accent-soft: #e7f5f7;
          --ev-ok: #0f8a5f;
          --ev-fail: #b42318;
        }
        .stApp {
          background: var(--ev-bg);
          color: var(--ev-text);
        }
        [data-testid="stSidebar"] {
          background: #ffffff;
          border-right: 1px solid var(--ev-border);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
          color: var(--ev-muted);
        }
        .block-container {
          padding-top: 1.4rem;
          padding-bottom: 2.5rem;
          max-width: 1440px;
        }
        .ev-sidebar-title {
          color: var(--ev-text);
          font-size: 1.15rem;
          font-weight: 750;
          margin: 0.4rem 0 0.2rem;
        }
        .ev-header {
          align-items: flex-start;
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: 8px;
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          padding: 1.15rem 1.25rem;
          margin-bottom: 1rem;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ev-header h1 {
          color: var(--ev-text);
          font-size: 1.45rem;
          line-height: 1.25;
          margin: 0.12rem 0 0.35rem;
          font-weight: 760;
          letter-spacing: 0;
        }
        .ev-kicker {
          color: var(--ev-accent);
          font-size: 0.78rem;
          font-weight: 760;
          text-transform: uppercase;
        }
        .ev-path {
          color: var(--ev-muted);
          font-size: 0.82rem;
          word-break: break-all;
        }
        .ev-header-right {
          display: flex;
          flex-wrap: wrap;
          justify-content: flex-end;
          gap: 0.45rem;
          min-width: 12rem;
        }
        .ev-pill {
          background: #f2f4f7;
          border: 1px solid var(--ev-border);
          border-radius: 999px;
          color: var(--ev-text);
          display: inline-block;
          font-size: 0.76rem;
          font-weight: 680;
          padding: 0.25rem 0.62rem;
          white-space: nowrap;
        }
        .ev-pill-ok {
          background: #eaf7f1;
          border-color: #b7e4cf;
          color: var(--ev-ok);
        }
        .ev-pill-fail {
          background: #fff1f0;
          border-color: #fecdca;
          color: var(--ev-fail);
        }
        .ev-metric-card {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: 8px;
          min-height: 6.8rem;
          padding: 0.8rem 0.85rem;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ev-metric-label {
          color: var(--ev-muted);
          font-size: 0.76rem;
          font-weight: 680;
          text-transform: uppercase;
        }
        .ev-metric-value {
          color: var(--ev-text);
          font-size: 1.8rem;
          font-weight: 780;
          line-height: 1.15;
          margin: 0.3rem 0;
        }
        .ev-metric-caption {
          color: var(--ev-muted);
          font-size: 0.78rem;
        }
        .ev-note {
          align-items: flex-start;
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: 8px;
          display: flex;
          gap: 0.75rem;
          margin-bottom: 0.65rem;
          padding: 0.85rem;
        }
        .ev-note-index {
          align-items: center;
          background: var(--ev-accent-soft);
          border: 1px solid #badfe5;
          border-radius: 999px;
          color: var(--ev-accent);
          display: flex;
          flex: 0 0 1.7rem;
          font-size: 0.78rem;
          font-weight: 760;
          height: 1.7rem;
          justify-content: center;
        }
        .ev-signal {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: 8px;
          margin-bottom: 0.75rem;
          padding: 0.95rem;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ev-signal-title {
          color: var(--ev-text);
          font-size: 0.98rem;
          font-weight: 760;
          margin-bottom: 0.35rem;
        }
        .ev-signal-body {
          color: #344054;
          font-size: 0.88rem;
          margin-bottom: 0.55rem;
        }
        .ev-signal-test {
          background: #f8fafc;
          border-left: 3px solid var(--ev-accent);
          color: #475467;
          font-size: 0.82rem;
          padding: 0.45rem 0.6rem;
        }
        .ev-card-title {
          color: var(--ev-text);
          font-size: 0.95rem;
          font-weight: 740;
          margin: 0.3rem 0 0.45rem;
        }
        div[data-testid="stMetric"] {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: 8px;
          padding: 0.65rem 0.75rem;
        }
        div[data-testid="stTabs"] button {
          font-weight: 680;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stImage"],
        div[data-testid="stVegaLiteChart"] {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: 8px;
          padding: 0.4rem;
        }
        h3 {
          color: var(--ev-text);
          font-size: 1.05rem;
          margin-top: 1rem;
        }
        @media (max-width: 900px) {
          .ev-header {
            flex-direction: column;
          }
          .ev-header-right {
            justify-content: flex-start;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
