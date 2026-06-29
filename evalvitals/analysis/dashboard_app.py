"""Streamlit app for EvalVitals M2 chat outputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from evalvitals.analysis.dashboard import load_run
from evalvitals.reporting.stages import stage_specs_as_dicts
from evalvitals.viz.labels import display_name, raw_hint

# Eval-chart-style house theme (FAIL-red / PASS-slate palette, distribution-first
# chart builders, short names + number/bin formatting). See eval_viz_theme.py and
# the eval-chart-style SKILL for the chart-type policy this enforces.
try:
    from evalvitals.analysis import eval_viz_theme as viz
except Exception:  # plotly missing or import error — degrade to legacy rendering
    viz = None

try:
    from evalvitals.analysis.eval_case_matrix import continuous_signals, load_case_matrix
except Exception:
    load_case_matrix = None
    continuous_signals = None


def _viz_ready() -> bool:
    """Register the plotly template once per session; report whether viz is usable."""
    if viz is None:
        return False
    if not st.session_state.get("_viz_applied"):
        viz.apply()
        st.session_state["_viz_applied"] = True
    return True


def main() -> None:
    run_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    session = load_run(run_arg)
    root = Path(session["root"])
    runs = session["runs"]

    st.set_page_config(page_title="EvalVitals", layout="wide", initial_sidebar_state="collapsed")
    _inject_css()
    _viz_ready()

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

    setting, analysis, hypotheses = st.tabs([
        "1 Problem Setting",
        "2 Analysis",
        "3 Hypotheses & Artifacts",
    ])
    with setting:
        _render_problem_setting(root, report, story=None, artifact_dir=turn_dir)
    with analysis:
        _render_standalone_analysis(report, turn_dir, root)
    with hypotheses:
        _render_standalone_hypotheses(report, turn_dir)


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
    """Render a diagnostic loop run with a data-rich Analysis tab (signal effects,
    e-BH adjudication, explorer charts & tables) plus the M2→M3→M5→Fix flow."""
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

    # The Step-1 explore report (signals/charts/tables M2 confirmed & M3 consulted)
    # usually lives in a sibling dir; load_loop_story resolves it for us.
    explore_report = story.get("explore_report")
    if not explore_report:
        explore_report = next((r["report"] for r in runs if r["name"] == "fused_report"), None)
    explore_dir = Path(story.get("explore_dir") or (runs[0]["dir"] if runs else root))
    if not explore_report:
        st.warning(
            "No explore report was found alongside this loop log, so measured "
            "signals/charts are unavailable. Re-run with `--explore-report`, or "
            "point the dashboard at a directory containing `fused_report.json` / "
            "`exploratory_report.json`."
        )

    setting, analysis, hypotheses = st.tabs([
        "1 Problem Setting",
        "2 Analysis",
        "3 Hypotheses & Artifacts",
    ])
    with setting:
        _render_problem_setting(root, explore_report or {}, story=story, artifact_dir=explore_dir)
    with analysis:
        _render_loop_analysis_panel(story, explore_report, explore_dir, root)
    with hypotheses:
        _render_hypothesis_decision_panel(story, explore_report, explore_dir)


def _candidate_signals(explore_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [s for s in ((explore_report or {}).get("candidate_signals") or []) if isinstance(s, dict)]


def _hypotheses_with_outcomes(story: dict[str, Any]) -> list[dict[str, Any]]:
    """Join each M3 hypothesis with the explore artifacts it cited and the
    M5/M4 tests that later evaluated it (matched by statement)."""
    tests_by_stmt: dict[str, list[dict[str, Any]]] = {}
    for s in story.get("surgeries") or []:
        tests_by_stmt.setdefault(str(s.get("hypothesis", "")).strip(), []).append(s)

    out: list[dict[str, Any]] = []
    for d in story.get("diagnoses") or []:
        for h in d.get("hypotheses") or []:
            stmt = str(h.get("statement", "")).strip()
            out.append({
                "statement": stmt,
                "failure_mode": h.get("failure_mode", ""),
                "cycle": d.get("cycle"),
                "referenced_charts": d.get("referenced_charts") or [],
                "tests": tests_by_stmt.get(stmt, []),
            })
    return out


def _render_problem_setting(
    root: Path,
    report: dict[str, Any] | None,
    *,
    story: dict[str, Any] | None,
    artifact_dir: Path | None = None,
) -> None:
    """Panel 1: orient the user before any statistical claims."""
    report = report or {}
    storyboard = _storyboard_panels(report, story=story)
    question = str(report.get("question") or "What distinguishes failures from passes?")
    adj = report.get("adjudication") or {}
    signals = _candidate_signals(report)
    charts = [c for c in report.get("charts", []) if isinstance(c, dict)]

    matrix = None
    if load_case_matrix is not None:
        try:
            matrix = load_case_matrix(root)
        except Exception:
            matrix = None

    n_cases = None
    n_fail = None
    n_pass = None
    n_features = None
    if matrix is not None and not matrix.empty:
        n_cases = len(matrix)
        if "is_fail" in matrix.columns:
            n_fail = int(pd.to_numeric(matrix["is_fail"], errors="coerce").fillna(0).sum())
            n_pass = n_cases - n_fail
        try:
            n_features = len(continuous_signals(matrix)) if continuous_signals else None
        except Exception:
            n_features = None

    profile = report.get("data_profile") or {}
    if n_cases is None:
        n_cases = profile.get("loaded_rows", profile.get("n_rows"))
    columns = profile.get("columns") or {}
    if n_features is None and isinstance(columns, dict):
        n_features = len(columns)
    split = report.get("split") or {}
    if n_cases is None:
        n_cases = split.get("n_total")
    n_explore = split.get("n_explore")
    n_confirm = split.get("n_confirm")
    if n_fail is None or n_pass is None:
        cb = _class_balance_from_report(report, artifact_dir)
        if cb:
            n_fail = cb.get("FAIL", n_fail)
            n_pass = cb.get("PASS", n_pass)

    st.markdown("### Problem Setting")
    _render_stage_map(active={"M1", "M2"} if not story else {"M1"})
    _render_storyboard_panel(storyboard, "problem_setting")
    if not _has_storyboard_panel(storyboard, "problem_setting"):
        st.markdown(
            f"""
            <div class="ev-report-answer">
              <div class="ev-brief-label">User question</div>
              <div class="ev-report-answer-text">{_html_escape(question)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    cols = st.columns(5)
    metrics = [
        ("Cases", n_cases, "rows/cases loaded"),
        ("Explore", n_explore, "agent discovery split"),
        ("Confirm", n_confirm, "held-out confirmation split"),
        ("Signals", len(signals) or n_features, "candidate/features"),
        ("Charts", len(charts) + len(report.get("plots") or []), "analysis visuals"),
    ]
    for col, (label, value, help_text) in zip(cols, metrics, strict=False):
        col.metric(label, _format_int(value), help=help_text)

    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.markdown("#### What data was provided")
        if n_fail is not None or n_pass is not None:
            st.markdown(
                f"""
                <div class="ev-brief-grid ev-brief-grid-two">
                  <div class="ev-brief-card">
                    <div class="ev-brief-label">FAIL cases</div>
                    <div class="ev-metric-value">{_html_escape(_format_int(n_fail))}</div>
                  </div>
                  <div class="ev-brief-card">
                    <div class="ev-brief-label">PASS cases</div>
                    <div class="ev-metric-value">{_html_escape(_format_int(n_pass))}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if matrix is not None and not matrix.empty:
            fields = [
                c for c in matrix.columns
                if c not in {"case_id", "label", "is_fail", "model_yes", "truth_yes"}
            ]
            st.caption(
                "Dashboard reconstructed the per-case M1 feature matrix. "
                "M2 analyses compare these per-case signals against FAIL/PASS labels."
            )
            preview = pd.DataFrame({
                "field": fields[:12],
                "display": [display_name(f) for f in fields[:12]],
                "coverage": [int(matrix[f].notna().sum()) for f in fields[:12]],
            })
            if not preview.empty:
                st.dataframe(preview, width="stretch", hide_index=True, height=260)
        elif isinstance(columns, dict) and columns:
            st.caption("Explorer data profile from the sampled input records.")
            rows = [
                {"field": k, "display": display_name(k), "profile": str(v)[:180]}
                for k, v in list(columns.items())[:12]
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=260)
        else:
            st.info("No structured data profile was saved with this report.")

    with c2:
        st.markdown("#### Evaluation frame")
        split = adj.get("split") or "not recorded"
        method = adj.get("method") or "exploratory only"
        stages = []
        if story:
            stages = [
                ("M2 analyses", len(story.get("analyses") or [])),
                ("M3 diagnoses", len(story.get("diagnoses") or [])),
                ("M5/M4 tests", len(story.get("surgeries") or [])),
                ("Fix events", len(story.get("fixes") or [])),
            ]
        st.markdown(
            f"""
            <div class="ev-brief-card">
              <div class="ev-brief-label">Confirmatory method</div>
              <div class="ev-brief-value">{_html_escape(str(method))}</div>
              <div class="ev-signal-test">Split: {_html_escape(str(split))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for label, count in stages:
            st.markdown(f"- **{label}:** {count}")
        st.caption(f"Run directory: {root}")


def _render_stage_map(*, active: set[str]) -> None:
    """Small stage strip: context, not content."""
    chips = []
    for spec in stage_specs_as_dicts():
        cls = "ev-stage-chip ev-stage-active" if spec["id"] in active else "ev-stage-chip"
        chips.append(
            f'<span class="{cls}"><b>{_html_escape(spec["id"])}</b> '
            f'{_html_escape(spec["name"])}</span>'
        )
    st.markdown('<div class="ev-stage-strip">' + "".join(chips) + "</div>", unsafe_allow_html=True)


def _class_balance_from_report(
    report: dict[str, Any] | None,
    artifact_dir: Path | None,
) -> dict[str, int]:
    if not report or artifact_dir is None:
        return {}
    for chart in report.get("charts") or []:
        if not isinstance(chart, dict):
            continue
        name = str(chart.get("name") or "").lower()
        if "class_balance" not in name:
            continue
        df = _table_to_dataframe(chart.get("data"), artifact_dir)
        if df is None or df.empty:
            continue
        cols = {c.lower(): c for c in df.columns}
        outcome_col = cols.get("outcome") or df.columns[0]
        count_col = cols.get("count")
        if count_col is None:
            continue
        out: dict[str, int] = {}
        for _, row in df.iterrows():
            key = str(row[outcome_col]).upper()
            try:
                out[key] = int(row[count_col])
            except Exception:
                pass
        return out
    return {}


def _storyboard_panels(
    report: dict[str, Any] | None,
    *,
    story: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Agent-authored dashboard storyboard, with compiled diagnostic fallback."""
    report = report or {}
    raw = report.get("dashboard_storyboard") or report.get("ui_panels")
    if isinstance(raw, list) and all(isinstance(p, dict) for p in raw):
        return [dict(p) for p in raw]
    diag = (story or {}).get("diagnostic_report") or {}
    raw = diag.get("dashboard_storyboard")
    if isinstance(raw, list) and all(isinstance(p, dict) for p in raw):
        return [dict(p) for p in raw]
    return []


def _has_storyboard_panel(panels: list[dict[str, Any]], panel_id: str) -> bool:
    return any(str(p.get("id")) == panel_id for p in panels)


def _render_storyboard_panel(panels: list[dict[str, Any]], panel_id: str) -> None:
    panel = next((p for p in panels if str(p.get("id")) == panel_id), None)
    if not panel:
        return
    stages = ", ".join(str(s) for s in (panel.get("stages") or []))
    items = [_humanize_storyboard_text(str(x)) for x in (panel.get("items") or []) if str(x).strip()]
    refs = [str(x) for x in (panel.get("artifact_refs") or []) if str(x).strip()]
    st.markdown(
        f"""
        <div class="ev-storyboard-card">
          <div class="ev-brief-label">Run-specific narrative {("· " + _html_escape(stages)) if stages else ""}</div>
          <div class="ev-storyboard-title">{_html_escape(str(panel.get('title') or panel_id))}</div>
          <div class="ev-storyboard-summary">{_html_escape(_humanize_storyboard_text(str(panel.get('summary') or '')))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if items:
        st.markdown("**Storyboard takeaways**")
        for item in items[:8]:
            st.markdown(f"- {item}")
    if refs:
        st.caption("Storyboard artifact refs: " + ", ".join(refs[:8]))


def _humanize_storyboard_text(text: str) -> str:
    replacements = {
        "generated_probe1_false_detection": "sanity-check probe false-detection flag",
        "probe1_false_detection": "sanity-check probe false-detection flag",
        "relative_attention_max_relative_weight": "maximum relative attention",
        "relative_attention_mean_relative_weight": "mean relative attention",
        "relative_attention_focus_share": "attention focus share",
        "low_focus_share": "low attention focus",
        "probe1_positive": "probe positive flag",
    }
    out = str(text)
    for raw, label in replacements.items():
        out = out.replace(raw, label)
    return out


def _analysis_takeaway(report: dict[str, Any] | None, story: dict[str, Any] | None = None) -> str:
    diag = (story or {}).get("diagnostic_report") or {}
    answer = str(diag.get("answer") or "").strip()
    if answer:
        return answer
    supported = [
        display_name(s.get("display_name") or s.get("name"))
        for s in _candidate_signals(report)
        if s.get("reject") is True and not _is_leaky_signal(s)
    ]
    if supported:
        return f"{supported[0]} is the leading held-out supported signal."
    conclusion = str((report or {}).get("conclusion") or "").strip()
    return conclusion or "No supported diagnostic claim is available in the loaded report."


def _render_method_card(title: str, method: str, evidence: str, takeaway: str) -> None:
    st.markdown(
        f"""
        <div class="ev-analysis-card">
          <div class="ev-brief-label">{_html_escape(title)}</div>
          <div class="ev-analysis-method">Method: {_html_escape(method)}</div>
          <div class="ev-analysis-evidence">Evidence: {_html_escape(evidence)}</div>
          <div class="ev-analysis-takeaway">Takeaway: {_html_escape(takeaway)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_loop_analysis_panel(story, explore_report, explore_dir, root=None) -> None:
    """Panel 2: analysis methods, evidence, charts, and takeaways."""
    explore_report = explore_report or {}
    if not explore_report:
        st.warning(
            "No explore report was found alongside this loop log, so the Analysis "
            "panel cannot show measured signals or charts. Re-run with "
            "`--explore-report`, or point the dashboard at a directory containing "
            "`fused_report.json` / `exploratory_report.json`."
        )
    signals = _candidate_signals(explore_report)
    adj = explore_report.get("adjudication") or {}
    readings = [r for r in explore_report.get("chart_readings") or [] if isinstance(r, dict)]
    storyboard = _storyboard_panels(explore_report, story=story)

    st.markdown("### Analysis")
    _render_stage_map(active={"M2"})
    st.markdown(
        f"""
        <div class="ev-report-answer">
          <div class="ev-brief-label">Bottom line</div>
          <div class="ev-report-answer-text">{_html_escape(_analysis_takeaway(explore_report, story))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if signals:
        st.markdown("#### Evidence you can use")
        _render_confirmatory_evidence(signals)

    with st.expander("Method details and generated storyboard", expanded=False):
        _render_storyboard_panel(storyboard, "analysis")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            _render_method_card(
                "Confirmatory signal testing",
                str(adj.get("method") or "held-out host adjudication"),
                "Effect sizes, confidence intervals, and e-BH/FDR verdicts.",
                f"{adj.get('n_signals_rejected', adj.get('n_rejected', 0))} of "
                f"{adj.get('n_signals_tested', adj.get('n_in_family', len(signals)))} tested signals survived.",
            )
        with c2:
            _render_method_card(
                "Exploratory visualization",
                "Agent-proposed chart plan; host-rendered deterministic specs.",
                f"{len(readings)} chart reading(s), {len(explore_report.get('charts') or [])} chart spec(s).",
                readings[0].get("reading", "Charts are leads for hypotheses, not causal proof.")
                if readings else "No chart readings were recorded.",
            )
        if adj:
            st.dataframe(pd.DataFrame([{
                "method": adj.get("method", "-"),
                "alpha": adj.get("alpha", "-"),
                "signals tested": adj.get("n_signals_tested", adj.get("n_in_family", "-")),
                "rejected": adj.get("n_signals_rejected", adj.get("n_rejected", "-")),
                "split": adj.get("split", "-"),
            }]), width="stretch", hide_index=True)

    with st.expander("Charts, tables, and extra diagnostics", expanded=False):
        _render_visual_plan(explore_report)
        if readings:
            rows = [{
                "chart": display_name(r.get("chart")),
                "takeaway": r.get("reading", ""),
                "do not infer": r.get("do_not_infer", ""),
            } for r in readings]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        if _viz_ready():
            _render_explore_tables(explore_report, explore_dir, full=False, root=root)
        else:
            _render_charts_and_plots(explore_report, explore_dir)
        _render_stat_panel(root)


def _render_hypothesis_decision_panel(story, explore_report, explore_dir) -> None:
    """Panel 3: hypotheses, downstream tests, and artifacts needed for decisions."""
    hyps = _hypotheses_with_outcomes(story)
    storyboard = _storyboard_panels(explore_report or {}, story=story)
    st.markdown("### Hypotheses & Decision")
    _render_stage_map(active={"M3", "M4", "M5"})
    _render_storyboard_panel(storyboard, "hypotheses_artifacts")
    if hyps:
        st.caption(
            "These are M3 hypotheses formed from the analysis panel. Treat them as "
            "decision candidates until M5/M4 tests or fix outcomes support them."
        )
        for h in hyps:
            _render_hypothesis_card(h)
    else:
        st.warning("No M3 hypotheses were recorded for this run.")

    surgeries = story.get("surgeries") or []
    fixes = story.get("fixes") or []
    if surgeries or fixes:
        st.markdown("#### Downstream decision evidence")
        for s in surgeries:
            tag = str(s.get("module", "")).upper()
            st.markdown(
                f"- **[{tag}]** {s.get('status', '')}: "
                f"{_truncate(str(s.get('hypothesis', '')), 180)}"
            )
        if fixes:
            with st.expander("Fix adjudication records", expanded=False):
                for f in fixes:
                    st.json(f)

    with st.expander("Inspect M1-M4 artifacts and flow", expanded=True):
        _render_loop_flow(story, explore_report, explore_dir)
    with st.expander("Inspect raw tables", expanded=False):
        _render_explore_tables(explore_report, explore_dir, full=True)
    if explore_report:
        with st.expander("Explorer original charts/figures", expanded=False):
            _render_charts_and_plots(explore_report, explore_dir)


def _render_standalone_analysis(report: dict[str, Any], turn_dir: Path, root: Path) -> None:
    signals = _candidate_signals(report)
    storyboard = _storyboard_panels(report, story=None)
    st.markdown("### Analysis")
    _render_stage_map(active={"M2"})
    _render_storyboard_panel(storyboard, "analysis")
    st.markdown(
        f"""
        <div class="ev-report-answer">
          <div class="ev-brief-label">Analysis takeaway</div>
          <div class="ev-report-answer-text">{_html_escape(_analysis_takeaway(report))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_method_card(
        "Standalone M2 exploration",
        "Agent-authored EDA plus host-rendered deterministic chart specs.",
        f"{len(signals)} candidate signal(s), {len(report.get('charts') or [])} chart spec(s).",
        "Use these findings to choose confirmatory follow-up tests; exploratory charts are not causal evidence.",
    )
    _render_visual_plan(report)
    if signals:
        st.markdown("#### Candidate signal table")
        st.dataframe(_signals_dataframe(signals), width="stretch", hide_index=True)
    _render_charts_and_plots(report, turn_dir)
    _render_tables(report, turn_dir)


def _render_standalone_hypotheses(report: dict[str, Any], turn_dir: Path) -> None:
    storyboard = _storyboard_panels(report, story=None)
    st.markdown("### Hypotheses & Artifacts")
    _render_stage_map(active={"M3", "M4", "M5"})
    _render_storyboard_panel(storyboard, "hypotheses_artifacts")
    claims = [c for c in report.get("claims") or [] if isinstance(c, dict)]
    tests = report.get("recommended_confirmatory_tests") or []
    if claims:
        st.markdown("#### Claims to carry forward")
        st.dataframe(pd.DataFrame(claims), width="stretch", hide_index=True)
    if tests:
        st.markdown("#### Recommended confirmatory tests")
        for item in tests:
            st.markdown(f"- {item}")
    if not claims and not tests:
        st.info("No explicit hypotheses or confirmatory tests were recorded.")
    with st.expander("Inspect generated artifacts", expanded=True):
        _render_artifacts(report, turn_dir)


def _render_loop_analysis(story, explore_report, explore_dir, root=None) -> None:
    """A connected diagnostic report: what was analysed -> what was found ->
    therefore which hypotheses were formed (each linked to its evidence + test)."""
    analyses = story.get("analyses") or []
    signals = _candidate_signals(explore_report)
    adj = (explore_report or {}).get("adjudication") or {}
    charts = [c for c in (explore_report or {}).get("charts", []) if isinstance(c, dict)]
    plots = (explore_report or {}).get("plots") or []
    hyps = _hypotheses_with_outcomes(story)

    concl = next((a.get("conclusion") or a.get("narrative") for a in analyses
                  if a.get("conclusion") or a.get("narrative")), "")
    _render_diagnostic_report(story.get("diagnostic_report") or {}, story, explore_report, concl)
    _render_interpretation_guide()

    if concl:
        st.markdown("### Conclusion")
        st.info(str(concl))

    # ── ① What we analysed ────────────────────────────────────────────────
    st.markdown("### ① What we analysed")
    obs = (explore_report or {}).get("observations") or []
    n_signals = len(signals)
    n_charts = len(charts) + len(plots)
    st.caption(
        f"Exploratory failure analysis examined {n_signals} candidate signal(s) "
        f"and produced {n_charts} chart(s) comparing FAIL vs PASS cases."
    )
    if obs:
        for o in obs[:10]:
            st.markdown(f"- {o}")
    _render_visual_plan(explore_report)

    # ── ② What we found ───────────────────────────────────────────────────
    st.markdown("### ② What we found")
    _render_plain_findings(story.get("diagnostic_report") or {})
    if adj:
        st.caption(
            "Each candidate signal was confirmed on a HELD-OUT split with e-BH "
            "(FDR-controlled). In the chart below, green survived; grey did not."
        )
        cols = st.columns(5)
        cells = [
            ("Method", adj.get("method", "-")),
            ("alpha", adj.get("alpha", "-")),
            ("Signals tested", adj.get("n_signals_tested", adj.get("n_in_family", "-"))),
            ("Rejected (real)", adj.get("n_signals_rejected", adj.get("n_rejected", "-"))),
            ("Split", adj.get("split", "-")),
        ]
        for col, (label, value) in zip(cols, cells, strict=False):
            col.metric(label, value)

    if signals:
        st.markdown("**Signal effect sizes (failure association)**")
        n_leaky = sum(1 for s in signals if _is_leaky_signal(s))
        if _viz_ready():
            st.plotly_chart(viz.forest_effects(_forest_rows(signals)),
                            width="stretch")
            st.caption(
                "Dot = effect size, bar = confidence interval, dotted line = 0 (no "
                "association). Green survived e-BH (FDR); grey did not."
                + (" Greyed/leaky signals are the outcome re-measured (perfect "
                   "separation) — confirmatory, not causal, so they are demoted "
                   "from the ranking." if n_leaky else "")
            )
        else:
            fig = _signal_effect_figure(signals)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
        st.dataframe(_signals_dataframe(signals), width="stretch", hide_index=True)

    # Statistical deep-dive built from the per-case matrix (model eval, distribution
    # diagnostics, relationships, decision) — replaces the 2-group descriptive view.
    _render_stat_panel(root)

    # The analyst's reasoning chain + statistical-test summaries, when logged.
    evidence = next((a.get("evidence_chain") for a in analyses if a.get("evidence_chain")), None)
    if evidence:
        st.markdown("**Evidence chain (analyst reasoning)**")
        for step in evidence[:10]:
            st.markdown(f"- {step}")
    stat_summaries = [
        str(r.get("summary")) for a in analyses for r in (a.get("stats_results") or [])
        if isinstance(r, dict) and r.get("summary")
    ]
    if stat_summaries:
        st.markdown("**Statistical tests**")
        for s in stat_summaries[:10]:
            st.markdown(f"- {s}")

    # Primary: themed, distribution-first charts rebuilt from the explorer's data
    # tables (counts→bar, two-signal→scatter, fail-rate→binned curve, group
    # stats→dumbbell). Replaces the explorer's default bar PNGs as the lead view.
    if _viz_ready():
        st.markdown("**Exploratory charts**")
        _render_explore_tables(explore_report, explore_dir, full=False, root=root)
        if charts or plots:
            with st.expander("Explorer's original figures (as rendered during Step 1)",
                             expanded=False):
                _render_charts_and_plots(explore_report, explore_dir)
    else:
        if charts or plots:
            st.markdown("**Exploratory charts**")
            _render_charts_and_plots(explore_report, explore_dir)
        _render_explore_tables(explore_report, explore_dir, full=False)

    # ── ③ Hypotheses formed ───────────────────────────────────────────────
    st.markdown("### ③ Hypotheses formed")
    if hyps:
        st.caption(
            "From the findings above, M3 proposed these falsifiable root-cause "
            "hypotheses. Each was tested downstream (M5/M4) before any fix."
        )
        for h in hyps:
            _render_hypothesis_card(h)
    else:
        st.caption("No hypotheses were recorded for this run.")

    if not (signals or adj or charts or plots or hyps or concl):
        st.warning(
            "This run has no explore report alongside the log, so there are no "
            "measured signals/charts to show. Re-run Step 2 with "
            "`--explore-report outputs/fused/fused_report.json`, or point the "
            "dashboard at an explore-output directory."
        )


def _render_stat_panel(root) -> None:
    """Inferential / distributional panel built from the per-case feature matrix
    reconstructed from m1_state.pkl: model-evaluation, distribution diagnostics,
    variable relationships, and decision analysis. Silent if data unavailable."""
    if root is None or not _viz_ready() or load_case_matrix is None:
        return
    try:
        df = load_case_matrix(root)
    except Exception:
        df = None
    if df is None or df.empty:
        return
    sigs = continuous_signals(df)

    st.markdown("---")
    st.markdown("#### 📈 Statistical analysis (per-case)")
    cov = ", ".join(f"{s}: {int(df[s].notna().sum())}" for s in sigs) if sigs else "—"
    st.caption(
        f"Reconstructed from the frozen M1 state ({len(df)} cases). Continuous-signal "
        f"coverage — {cov}. Charts below go beyond FAIL-vs-PASS means: discrimination "
        f"(ROC/coef), error structure (confusion), distribution shape, and relationships."
    )

    # ── Model evaluation ──────────────────────────────────────────────────
    st.markdown("**Model evaluation**")
    c1, c2 = st.columns(2)
    has_answers = df["model_yes"].notna().any() and df["truth_yes"].notna().any()
    if has_answers:
        with c1:
            st.plotly_chart(
                viz.confusion_matrix(df["truth_yes"], df["model_yes"],
                                     pos_label="Yes (present)", neg_label="No (absent)",
                                     title="Model answer vs ground truth"),
                width="stretch")
            st.caption("FP = hallucination (said Yes, object absent); FN = miss.")
    if sigs:
        with (c2 if has_answers else c1):
            st.plotly_chart(viz.roc_curves(df, sigs, label_col="is_fail"),
                            width="stretch")
            st.caption("How well each signal alone separates FAIL from PASS (AUC).")
        st.plotly_chart(viz.coef_plot(df, sigs, label_col="is_fail"),
                        width="stretch")
        st.caption("Standardized univariate logistic coefficients (bootstrap 95% CI); "
                   "comparable across signals — CI crossing 0 ⇒ not significant.")

    # ── Distribution diagnostics (interactive signal picker) ──────────────
    if sigs:
        st.markdown("**Distribution diagnostics**")
        # default to the most discriminative signal (highest |AUC-0.5|)
        aucs = {s: abs(viz._roc(df[s].to_numpy(float), df["is_fail"].to_numpy(float))[2] - 0.5)
                for s in sigs}
        default = max(aucs, key=aucs.get)
        pick = st.selectbox("signal", sigs, index=sigs.index(default),
                            format_func=lambda s: viz.short(s), key="stat_sig")
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(viz.violin_by_outcome(df, pick), width="stretch")
            st.plotly_chart(viz.ecdf_by_outcome(df, pick), width="stretch")
        with d2:
            st.plotly_chart(viz.kde_by_outcome(df, pick), width="stretch")
            st.plotly_chart(viz.qq_normal(df, pick), width="stretch")

    # ── Variable relationships ────────────────────────────────────────────
    rel_sigs = sigs + (["probe1_fd"] if "probe1_fd" in df.columns else [])
    if len(rel_sigs) >= 2:
        st.markdown("**Variable relationships**")
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(viz.corr_heatmap(df, rel_sigs), width="stretch")
        with r2:
            if len(sigs) >= 2:
                st.plotly_chart(viz.quadrant(df, sigs[0], sigs[1]), width="stretch")

    # ── Decision analysis: which prompt strategies fix vs break cases ─────
    pareto_items = [
        ("describe-first fixes", "pc_fixed_describe"),
        ("sensitive fixes", "pc_fixed_sensitive"),
        ("describe-first breaks", "pc_broken_describe"),
        ("sensitive breaks", "pc_broken_sensitive"),
    ]
    avail = [(lbl, col) for lbl, col in pareto_items if col in df.columns]
    if avail:
        vals = [int(df[col].fillna(False).astype(bool).sum()) for _, col in avail]
        if sum(vals) > 0:
            st.markdown("**Decision analysis**")
            st.plotly_chart(viz.pareto([lbl for lbl, _ in avail], vals,
                                       title="Prompt-strategy repairs vs regressions"),
                            width="stretch")
            st.caption("How many cases each reprompting strategy fixed vs broke — "
                       "ranks whether prompt-level fixes are worth pursuing.")


def _render_diagnostic_report(
    report: dict[str, Any],
    story: dict[str, Any],
    explore_report: dict[str, Any] | None,
    fallback_conclusion: str,
) -> None:
    """Claim-first semantic report compiled from raw artifacts."""
    if not report:
        _render_run_briefing(story, explore_report, fallback_conclusion)
        return

    st.markdown("### Diagnostic report")
    confidence = str(report.get("confidence") or "unknown")
    st.markdown(
        f"""
        <div class="ev-report-answer">
          <div class="ev-brief-label">Answer first · confidence: {_html_escape(confidence)}</div>
          <div class="ev-report-answer-text">{_html_escape(str(report.get('answer') or ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    claims = [c for c in (report.get("claims") or []) if isinstance(c, dict)]
    evidence = {
        str(e.get("id")): e for e in (report.get("evidence") or [])
        if isinstance(e, dict) and e.get("id")
    }
    if claims:
        st.markdown("#### Claims and evidence")
        for claim in claims:
            _render_claim_card(claim, evidence)

    timeline = [s for s in (report.get("timeline") or []) if isinstance(s, dict)]
    if timeline:
        with st.expander("Investigation timeline", expanded=False):
            for step in timeline:
                st.markdown(
                    f"**{step.get('stage', '')}: {step.get('title', '')}**  \n"
                    f"{step.get('summary', '')}"
                )

    readings = [r for r in (report.get("chart_readings") or []) if isinstance(r, dict)]
    if readings:
        with st.expander("Chart readings written by the agent", expanded=False):
            st.dataframe(pd.DataFrame(readings), width="stretch", hide_index=True)

    critique = [str(c) for c in (report.get("critique") or [])]
    if critique:
        with st.expander("Critique and limits", expanded=True):
            for note in critique:
                st.markdown(f"- {note}")

    actions = [str(a) for a in (report.get("next_actions") or [])]
    if actions:
        with st.expander("Recommended next actions", expanded=False):
            for action in actions:
                st.markdown(f"- {action}")


def _render_claim_card(claim: dict[str, Any], evidence: dict[str, dict[str, Any]]) -> None:
    status = str(claim.get("status") or "descriptive")
    cls = {
        "supported": "ev-claim-supported",
        "inconclusive": "ev-claim-inconclusive",
        "refuted": "ev-claim-refuted",
    }.get(status, "ev-claim-descriptive")
    st.markdown(
        f"""
        <div class="ev-claim-card {cls}">
          <div class="ev-claim-top">
            <span class="ev-pill">{_html_escape(str(claim.get('id', 'claim')))}</span>
            <span class="ev-pill">{_html_escape(status)}</span>
          </div>
          <div class="ev-claim-text">{_html_escape(str(claim.get('text') or ''))}</div>
          <div class="ev-signal-body">{_html_escape(str(claim.get('interpretation') or ''))}</div>
          <div class="ev-signal-test">Do not infer: {_html_escape(str(claim.get('do_not_infer') or ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    rows = []
    for ev_id in claim.get("evidence_ids") or []:
        ev = evidence.get(str(ev_id))
        if ev:
            rows.append({
                "id": ev.get("id"),
                "kind": ev.get("kind"),
                "title": ev.get("title"),
                "summary": ev.get("summary"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    downstream = [str(x) for x in (claim.get("downstream") or [])]
    if downstream:
        st.caption("Downstream: " + " · ".join(downstream))


def _render_plain_findings(report: dict[str, Any]) -> None:
    claims = [c for c in (report.get("claims") or []) if isinstance(c, dict)]
    if not claims:
        return
    supported = [c for c in claims if c.get("status") == "supported"]
    descriptive = [c for c in claims if c.get("status") == "descriptive"]
    inconclusive = [c for c in claims if c.get("status") == "inconclusive"]

    cols = st.columns(3)
    groups = [
        ("Main supported findings", supported, "These are the findings to carry forward."),
        ("Sanity checks / descriptive", descriptive, "Useful context, not root-cause evidence."),
        ("Not supported", inconclusive, "Tested but not convincing in this run."),
    ]
    for col, (title, rows, caption) in zip(cols, groups, strict=False):
        with col:
            st.markdown(f"**{title}**")
            if rows:
                for claim in rows[:4]:
                    st.markdown(f"- {_html_escape(str(claim.get('text') or ''))}")
            else:
                st.caption("None")
            st.caption(caption)


def _render_run_briefing(
    story: dict[str, Any],
    explore_report: dict[str, Any] | None,
    conclusion: str,
) -> None:
    """A human-first summary before the dense plots. The dashboard has many
    diagnostics; this pins the reader to what was actually done and what evidence
    is allowed to support a claim."""
    signals = _candidate_signals(explore_report)
    supported = [
        str(s.get("name"))
        for s in signals
        if s.get("reject") is True and not _is_leaky_signal(s)
    ]
    leaky = [str(s.get("name")) for s in signals if _is_leaky_signal(s)]
    diagnoses = story.get("diagnoses") or []
    surgeries = story.get("surgeries") or []
    fixes = story.get("fixes") or []
    charts = [c for c in (explore_report or {}).get("charts", []) if isinstance(c, dict)]
    split = ((explore_report or {}).get("adjudication") or {}).get("split", "unknown")

    takeaway = (
        _truncate(str(conclusion), 220)
        if conclusion else
        (
            f"Supported signals: {', '.join(supported[:3])}"
            if supported else
            "No held-out supported signal was found in the loaded report."
        )
    )
    evidence = (
        f"{len(supported)} non-leaky signal(s) survived held-out e-BH"
        if supported else
        "No non-leaky signal survived held-out e-BH"
    )
    if leaky:
        label = "sanity check" if len(leaky) == 1 else "sanity checks"
        evidence += f"; {len(leaky)} {label} demoted"
    downstream = (
        f"{sum(len(d.get('hypotheses') or []) for d in diagnoses)} hypothesis(es), "
        f"{len(surgeries)} test/intervention event(s), {len(fixes)} fix event(s)"
    )

    st.markdown("### Run briefing")
    st.markdown(
        f"""
        <div class="ev-brief-grid">
          <div class="ev-brief-card">
            <div class="ev-brief-label">Question answered</div>
            <div class="ev-brief-value">{_html_escape(takeaway)}</div>
          </div>
          <div class="ev-brief-card">
            <div class="ev-brief-label">Evidence you can trust</div>
            <div class="ev-brief-value">{_html_escape(evidence)}</div>
          </div>
          <div class="ev-brief-card">
            <div class="ev-brief-label">Pipeline stages loaded</div>
            <div class="ev-brief-value">
              Explore charts: {len(charts)} · confirm split: {_html_escape(str(split))}<br/>
              {_html_escape(downstream)}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_interpretation_guide() -> None:
    with st.expander("How to interpret this page", expanded=True):
        st.markdown(
            """
            - **Exploratory observations and charts** show patterns the agent found.
              Treat them as leads, not proof.
            - **Signal effect sizes** are the confirmatory layer. A signal matters
              only if it survives the held-out e-BH check; grey/inconclusive rows
              are descriptive.
            - **Leaky signals** re-measure the label or a near-label proxy. They can
              validate plumbing, but they should not be read as root causes.
            - **Hypotheses** are M3's explanations formed from the confirmed signals
              plus exploratory context. They are not accepted until M5/M4 tests
              support them.
            - **Fix outcomes** are the final gate. A plausible hypothesis without a
              validated fix is still only a diagnosis candidate.
            """
        )


def _render_visual_plan(explore_report: dict[str, Any] | None) -> None:
    plan = [
        p for p in ((explore_report or {}).get("visual_plan") or [])
        if isinstance(p, dict)
    ]
    if not plan:
        st.caption(
            "Visualization plan: not present in this report. Newer explorer runs "
            "record why each plot type was selected."
        )
        return
    rows = []
    for item in plan:
        cols = item.get("required_columns") or []
        if isinstance(cols, list):
            cols_text = ", ".join(str(c) for c in cols)
        else:
            cols_text = str(cols)
        rows.append({
            "visual": item.get("name", ""),
            "question": item.get("question", ""),
            "data_shape": item.get("data_shape", ""),
            "plot_kind": item.get("plot_kind", ""),
            "why this plot": item.get("rationale", ""),
            "columns": cols_text,
        })
    with st.expander("Why these charts were chosen", expanded=False):
        st.caption(
            "This is the agent's intermediate visualization plan: the chart type "
            "decision, data shape, and rationale before code was written."
        )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_hypothesis_card(h: dict[str, Any]) -> None:
    """One hypothesis with its evidence (cited charts) and M5/M4 test verdicts."""
    badges = ""
    for t in h.get("tests") or []:
        mod = str(t.get("module", "")).upper()
        status = str(t.get("status", "") or ("fixed" if t.get("fixed") else "tested"))
        ok = bool(t.get("fixed")) or status.lower() in {"supported", "confirmed"}
        bad = status.lower() in {"refuted", "rejected", "not_fixed", "failed"}
        color = "#0f8a5f" if ok else ("#b42318" if bad else "#667085")
        badges += (
            f'<span class="ev-pill" style="border-color:{color};color:{color};">'
            f'{mod}: {_html_escape(status)}</span> '
        )
    refs = h.get("referenced_charts") or []
    ref_line = ("based on: " + ", ".join(str(r) for r in refs)) if refs else ""
    st.markdown(
        f"""
        <div class="ev-signal">
          <div class="ev-signal-title">{_html_escape(str(h.get('statement', '')))}</div>
          <div class="ev-signal-body">failure mode: <b>{_html_escape(str(h.get('failure_mode', '')))}</b>
            {('· ' + _html_escape(ref_line)) if ref_line else ''}</div>
          <div>{badges or '<span class="ev-pill">not yet tested</span>'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_loop_flow(story, explore_report, explore_dir) -> None:
    """The ordered narrative: explore notes → M2 → M3 hypotheses → M5 → Fix."""
    if explore_report:
        with st.expander("Step 1 — exploratory observations (UNCONFIRMED; fed to M3 only)", expanded=False):
            for o in (explore_report.get("observations") or [])[:10]:
                st.markdown(f"- {o}")
            for c in (explore_report.get("caveats") or [])[:8]:
                st.caption(f"⚠ {c}")

    analyses = story.get("analyses") or []
    diagnoses = story.get("diagnoses") or []
    surgeries = story.get("surgeries") or []
    fixes = story.get("fixes") or []

    if analyses:
        st.markdown("### M2 — stats analysis")
        for a in analyses:
            sev = a.get("severity") or ""
            st.markdown(f"- **Cycle {a.get('cycle')}** {('· severity ' + str(sev)) if sev else ''}")
            concl = a.get("conclusion") or a.get("narrative") or ""
            if concl:
                st.caption(_truncate(str(concl), 320))

    st.markdown("### M3 — hypotheses")
    if not diagnoses:
        st.caption("No diagnosis events in the loop log.")
    for diag in diagnoses:
        st.markdown(f"**Cycle {diag.get('cycle')}** · {diag.get('n_hypotheses', 0)} hypotheses")
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
            st.markdown(f"- **[{tag}]** {s.get('status', '')} · {_truncate(str(s.get('hypothesis', '')), 120)}")

    if fixes:
        st.markdown("### Fix — e-BH adjudicated outcomes")
        for f in fixes:
            st.json(f)


def _signal_effect_figure(signals: list[dict[str, Any]]):
    """Horizontal bar of each signal's effect size, coloured by host verdict
    (REJECT H0 = real association vs inconclusive). Returns a matplotlib Figure
    or None when matplotlib is missing / there is nothing numeric to plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    rows = [(str(s.get("display_name") or s.get("name", "?")), str(s.get("name", "?")),
             float(s["effect"]), bool(s.get("reject")), str(s.get("source", "")))
            for s in signals if isinstance(s.get("effect"), (int, float))]
    if not rows:
        return None
    rows.sort(key=lambda r: r[2])
    names = [f"{display_name(label or raw, compact=True)}  ({src})" for label, raw, _, _, src in rows]
    effects = [e for _, _, e, _, _ in rows]
    colors = ["#0f8a5f" if rej else "#9aa4b2" for _, _, _, rej, _ in rows]

    fig, ax = plt.subplots(figsize=(7.2, max(1.6, 0.55 * len(rows) + 0.8)))
    ax.barh(range(len(rows)), effects, color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("effect size (failure association)")
    ax.set_title("Green = REJECT H0 (real, e-BH); grey = inconclusive")
    fig.tight_layout()
    return fig


def _is_leaky_signal(s: dict[str, Any]) -> bool:
    """Target leakage = the outcome re-measured. Signature: perfect separation
    (effect ~1.0 with a zero-width CI), or a signal derived from the probe that
    defines the failure label. Such a signal must never be ranked #1 (skill §4)."""
    name = str(s.get("name", "")).lower()
    if name.startswith("probe1") or "false_detection" in name:
        return True
    eff = s.get("effect")
    ci = s.get("ci") or [None, None]
    try:
        lo, hi = float(ci[0]), float(ci[1])
        if eff is not None and abs(float(eff)) >= 0.999 and (hi - lo) <= 1e-6:
            return True
    except (TypeError, ValueError, IndexError):
        pass
    return False


def _forest_rows(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt host signal dicts to eval_viz_theme.forest_effects row schema."""
    rows = []
    for s in signals:
        if not isinstance(s.get("effect"), (int, float)):
            continue
        ci = s.get("ci") or [None, None]
        raw = str(s.get("name", "?"))
        rows.append({
            "signal": str(s.get("display_name") or display_name(raw)),
            "raw_signal": raw,
            "effect": float(s["effect"]),
            "ci_lo": ci[0] if isinstance(ci, (list, tuple)) and len(ci) == 2 else None,
            "ci_hi": ci[1] if isinstance(ci, (list, tuple)) and len(ci) == 2 else None,
            "significant": bool(s.get("reject")),
            "leaky": _is_leaky_signal(s),
        })
    return rows


def _render_confirmatory_evidence(signals: list[dict[str, Any]]) -> None:
    rows = _evidence_rows(signals)
    supported = [r for r in rows if r["decision"] == "Supported" and r["role"] == "diagnostic signal"]
    audits = [r for r in rows if r["role"] == "sanity check"]
    lead = supported[0] if supported else None

    if lead:
        summary = (
            f"{lead['finding']} is the main usable finding. FAIL cases have higher values "
            f"for this signal on the confirmation split (effect {lead['effect']}, CI {lead['ci']})."
        )
    elif audits:
        summary = (
            "Only sanity-check signals were supported. That validates the plumbing, "
            "but it does not explain the model failure."
        )
    else:
        summary = "No non-leaky signal was supported on the confirmation split."

    st.markdown(
        f"""
        <div class="ev-evidence-summary">
          <div class="ev-brief-label">How to read this evidence</div>
          <div class="ev-storyboard-summary">{_html_escape(summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for row in rows:
        kind = "ev-evidence-audit" if row["role"] == "sanity check" else "ev-evidence-signal"
        st.markdown(
            f"""
            <div class="ev-evidence-card {kind}">
              <div class="ev-evidence-card-top">
                <span class="ev-evidence-name">{_html_escape(row['finding'])}</span>
                <span class="ev-evidence-pill">{_html_escape(row['decision'])}</span>
              </div>
              <div class="ev-evidence-meaning">{_html_escape(row['plain meaning'])}</div>
              <div class="ev-evidence-meta">
                <span>{_html_escape(row['role'])}</span>
                <span>effect {_html_escape(row['effect'])}</span>
                <span>CI {_html_escape(row['ci'])}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Cards are computed by the host on the confirmation split. Positive effect means the "
        "signal is more common/larger in FAIL than PASS; CI crossing 0 means weak evidence. "
        "Sanity checks validate the pipeline and are not root-cause explanations."
    )

    with st.expander("Technical forest plot", expanded=False):
        if _viz_ready():
            st.plotly_chart(viz.forest_effects(_forest_rows(signals)), width="stretch")
        else:
            fig = _signal_effect_figure(signals)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
        st.dataframe(_signals_dataframe(signals), width="stretch", hide_index=True)


def _evidence_rows(signals: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for s in signals:
        leaky = _is_leaky_signal(s)
        label = display_name(s.get("name") if leaky else (s.get("display_name") or s.get("name")))
        reject = s.get("reject")
        effect = _fmt_effect(s.get("effect"))
        ci = _fmt_ci(s.get("ci"))
        if leaky:
            role = "sanity check"
            if not label.lower().startswith("sanity check"):
                label = f"Sanity check: {label[:1].lower()}{label[1:]}"
            meaning = (
                "This mostly mirrors the known FAIL/PASS label. It confirms the measurement "
                "pipeline, but it should not be treated as an explanation."
            )
        else:
            role = "diagnostic signal"
            meaning = (
                f"{label} separates FAIL from PASS in this run."
                if reject is True else
                f"{label} was tested but did not clearly separate FAIL from PASS."
            )
        decision = "Supported" if reject is True else ("Not supported" if reject is False else "Descriptive")
        rows.append({
            "finding": label,
            "role": role,
            "decision": decision,
            "effect": effect,
            "ci": ci,
            "plain meaning": meaning,
        })
    rows.sort(key=lambda r: (r["role"] != "diagnostic signal", r["decision"] != "Supported", r["finding"]))
    return rows


def _fmt_effect(value: Any) -> str:
    try:
        return f"{float(value):+.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_ci(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return f"{float(value[0]):+.2f} to {float(value[1]):+.2f}"
        except (TypeError, ValueError):
            return "—"
    return "—"


def _signals_dataframe(signals: list[dict[str, Any]]):
    def _verdict(s):
        if _is_leaky_signal(s):
            return "leaky (re-measured label)"
        if s.get("reject") is True:
            return "REJECT H0"
        if s.get("reject") is False:
            return "inconclusive"
        return "descriptive"

    def _fmt(x, kind):
        return viz.fmt(x, kind) if viz is not None else x

    rows = [{
        "signal": display_name(s.get("display_name") or s.get("name"), compact=True),
        "raw field": s.get("name"),
        "source": s.get("source"),
        "effect": _fmt(s.get("effect"), "effect"),
        "e_value": _fmt(s.get("e_value"), "stat") if s.get("e_value") is not None else "—",
        "verdict": _verdict(s),
        "recipe": (s.get("recipe") or {}).get("expr") if isinstance(s.get("recipe"), dict) else None,
    } for s in signals]
    return pd.DataFrame(rows)


def _explore_table_dirs(explore_dir: Path) -> list[Path]:
    return [d for d in (explore_dir / "tables", explore_dir / "sandbox" / "tables") if d.is_dir()]


def _scatter_axis_names(explore_report, csv_name: str) -> tuple[str | None, str | None]:
    """Recover (x_signal, y_signal) for a scatter CSV from the report chart whose
    title is '<xsig> vs <ysig> by outcome'. Returns (None, None) if not found."""
    stem = csv_name.replace(".csv", "")
    for c in (explore_report or {}).get("charts", []):
        if not isinstance(c, dict):
            continue
        if c.get("name") == stem or str(c.get("data", "")).endswith(csv_name):
            title = str(c.get("title", ""))
            if " vs " in title:
                left, _, right = title.partition(" vs ")
                right = right.split(" by outcome")[0].strip()
                return left.strip() or None, right or None
    return None, None


def _counts_bar_agg(df: pd.DataFrame):
    """Class balance from an aggregated (outcome, count) table → a single slim
    100%-stacked composition strip (not two fat bars for two numbers)."""
    rowmap = {str(r["outcome"]).upper(): int(r["count"]) for _, r in df.iterrows()}
    order = [g for g in ("FAIL", "PASS") if g in rowmap] or list(rowmap)
    return viz.composition_bar(order, [rowmap[g] for g in order])


def _groupstats_dumbbell(df: pd.DataFrame, signal: str):
    """FAIL vs PASS group means as a dumbbell (not a two-bar). Only mean/median are
    persisted, so spread cannot be shown — labelled honestly as means."""
    import plotly.graph_objects as go
    rowmap = {str(r["outcome"]).upper(): r for _, r in df.iterrows()}
    fig = go.Figure()
    present = [g for g in ("FAIL", "PASS") if g in rowmap]
    if {"FAIL", "PASS"} <= set(rowmap):
        fig.add_trace(go.Scatter(
            x=[rowmap["FAIL"]["mean"], rowmap["PASS"]["mean"]], y=[signal, signal],
            mode="lines", line=dict(color=viz.PALETTE["AXIS"], width=2),
            showlegend=False, hoverinfo="skip"))
    for g in present:
        m = rowmap[g]
        fig.add_trace(go.Scatter(
            x=[m["mean"]], y=[signal], mode="markers+text",
            marker=dict(color=viz.outcome_color(g), size=14),
            text=[f"{g} {viz.fmt(m['mean'], 'val')}"],
            textposition="top center", textfont=dict(size=11),
            name=g, hovertext=f"{g}: mean={viz.fmt(m['mean'],'val')} median={viz.fmt(m.get('median'),'val')}",
            hoverinfo="text"))
    fig.update_layout(title=f"{viz.short(signal)} — group means (FAIL vs PASS)",
                      xaxis_title=viz.short(signal), yaxis_title="", showlegend=False,
                      height=180, yaxis=dict(showticklabels=False))
    return fig


def _failrate_scatter(df: pd.DataFrame, signal: str):
    """Fail rate vs a binned signal: markers at bin midpoints, size ∝ n, with
    human-readable bin labels — replaces the bar/line over machine bin edges."""
    import re

    import plotly.graph_objects as go
    bincol = df.columns[0]
    labels, mids, rates, ns = [], [], [], []
    for _, r in df.iterrows():
        nums = re.findall(r"-?\d+\.?\d*", str(r[bincol]))
        if len(nums) >= 2:
            lo, hi = float(nums[0]), float(nums[1])
            labels.append(viz.human_bins([lo, hi])[0])
            mids.append((lo + hi) / 2)
        else:
            labels.append(str(r[bincol]))
            mids.append(len(mids))
        rates.append(float(r.get("fail_rate", 0)))
        ns.append(int(r.get("n", 0)) if pd.notna(r.get("n")) else 0)
    sizes = [8 + 28 * (n / max(ns)) for n in ns] if ns and max(ns) else [10] * len(ns)
    fig = go.Figure(go.Scatter(
        x=mids, y=rates, mode="lines+markers+text",
        line=dict(color=viz.PALETTE["FAIL"], width=2),
        marker=dict(color=viz.PALETTE["FAIL"], size=sizes,
                    line=dict(color="white", width=1)),
        text=[f"n={n}" for n in ns], textposition="top center",
        textfont=dict(size=10), hovertext=labels, hoverinfo="text+y"))
    fig.update_layout(title=f"Fail rate vs {viz.short(signal)} (by bin; marker size ∝ n)",
                      xaxis_title=viz.short(signal), yaxis_title="fail rate",
                      yaxis=dict(range=[-0.05, 1.05]), height=320)
    return fig


def _render_explore_tables(explore_report, explore_dir, *, full: bool, root=None) -> None:
    """Surface the CSV tables the explorer wrote. In the Analysis tab (full=False)
    each table is shown as the chart its content calls for (counts→bar, two-signal
    →scatter, fail-rate→binned curve, group stats→dumbbell), per the eval-chart-
    style policy. In the Tables tab (full=True) they are sortable dataframes only —
    one home per fact, no re-plotting."""
    explore_dir = Path(explore_dir)
    csvs: list[Path] = []
    for tdir in _explore_table_dirs(explore_dir):
        csvs.extend(sorted(tdir.glob("*.csv")))
    seen: set[str] = set()
    uniq = [p for p in csvs if not (p.name in seen or seen.add(p.name))]

    if not uniq:
        if full:
            st.caption("No CSV tables found for this run.")
        return

    # Tables tab: real sortable tables, no charts (don't re-plot Analysis findings).
    if full:
        st.markdown("#### Data tables")
        for path in uniq:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            st.markdown(f"**{path.name}**")
            st.dataframe(df, width="stretch", height=240)
        return

    # Analysis tab: prefer ONE combined chart per family over per-signal small-
    # multiples. When the per-case matrix is available, all signals' group means
    # and fail-rate curves collapse into one chart each (skill §2).
    if not _viz_ready():
        return _render_explore_tables_legacy(uniq)

    st.markdown("#### Data behind the analysis")

    matrix = None
    if root is not None and load_case_matrix is not None:
        try:
            matrix = load_case_matrix(root)
        except Exception:
            matrix = None
    sigs = continuous_signals(matrix) if (matrix is not None and continuous_signals) else []

    # Class balance — compact, unique (kept as-is).
    for path in uniq:
        if path.name.startswith("class_balance"):
            try:
                d = pd.read_csv(path)
                if {"outcome", "count"} <= {c.lower() for c in d.columns}:
                    st.plotly_chart(_counts_bar_agg(d), width="stretch")
            except Exception:
                pass
            break

    combined = False
    if matrix is not None and len(sigs) >= 1:
        combined = True
        st.plotly_chart(viz.groupstats_strip(matrix, sigs), width="stretch")
        st.caption("All signals' FAIL-vs-PASS group means on one standardized axis "
                   "(one row per signal — replaces a panel per signal).")
        st.plotly_chart(viz.failrate_percentile(matrix, sigs), width="stretch")
        st.caption("Each signal's fail-rate curve over its own percentile range, overlaid "
                   "on one axis.")

    # Remaining CSVs: per-signal group-stats / fail-rate are now folded into the
    # combined charts above; the binary scatter is degenerate (skill §0). Show only
    # the genuinely tabular summaries (correlations, discriminators) as tables.
    folded = ("groupstats", "failrate", "class_balance")
    for path in uniq:
        name = path.name
        if name.startswith(folded):
            continue
        if name.startswith("scatter"):
            # joint scatter with a binary axis just stacks points on two lines and
            # overlaps the marginals — suppress (the quadrant/violin views cover it).
            try:
                d = pd.read_csv(path)
            except Exception:
                continue
            xs, ys = _scatter_axis_names(explore_report, name)
            binary_axis = any(d[c].nunique(dropna=True) <= 2 for c in ("x", "y") if c in d.columns)
            if binary_axis:
                lbl = f"{viz.short(xs) if xs else 'x'} vs {viz.short(ys) if ys else 'y'}"
                st.caption(f"↳ scatter {lbl}: one axis is binary — suppressed "
                           f"(see the quadrant / distribution views above).")
            else:
                d2 = d.rename(columns={"x": xs, "y": ys}) if xs and ys else d
                st.plotly_chart(viz.joint_scatter(d2, xs or "x", ys or "y", outcome="outcome"),
                                width="stretch")
            continue
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        st.markdown(f"**{name}**")
        st.dataframe(d, width="stretch", height=200)

    # No matrix → fall back to per-signal panels so nothing is lost.
    if not combined:
        for path in uniq:
            name = path.name
            try:
                d = pd.read_csv(path)
            except Exception:
                continue
            cols = {c.lower() for c in d.columns}
            if name.startswith("groupstats") and {"outcome", "mean"} <= cols:
                st.plotly_chart(_groupstats_dumbbell(d, name.replace("groupstats_", "").replace(".csv", "")),
                                width="stretch")
            elif name.startswith("failrate") and "fail_rate" in cols and len(d) > 1:
                st.plotly_chart(_failrate_scatter(d, name.replace("failrate_by_", "").replace(".csv", "")),
                                width="stretch")


def _render_explore_tables_legacy(uniq: list[Path]) -> None:
    """Fallback when plotly/viz is unavailable: the original bar-chart rendering."""
    st.markdown("#### Data tables")
    for path in uniq:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        st.markdown(f"**{path.name}**")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(df.columns) >= 2 and num_cols:
            x = df.columns[0]
            y = num_cols[0] if num_cols[0] != x else (num_cols[1] if len(num_cols) > 1 else num_cols[0])
            try:
                st.bar_chart(df, x=x, y=y, height=260)
            except Exception:
                st.dataframe(df, width="stretch", height=240)


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
                raw = str(signal.get("name") or "Signal")
                name = str(signal.get("display_name") or display_name(raw))
                rationale = str(signal.get("rationale") or "")
                suggested = str(signal.get("suggested_test") or "")
                raw = raw_hint(raw)
                st.markdown(
                    f"""
                    <div class="ev-signal">
                      <div class="ev-signal-title">{_html_escape(name)}</div>
                      <div class="ev-signal-body">{_html_escape(rationale)}</div>
                      <div class="ev-signal-test">{_html_escape(suggested)}</div>
                      <div class="ev-signal-test">{_html_escape(raw)}</div>
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
    title = str(chart.get("display_name") or chart.get("title") or chart.get("name") or "Chart")
    title = display_name(title)
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
        turn_dir / "sandbox" / "tables" / path.name,
        turn_dir / "sandbox" / "figures" / path.name,
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
          padding-top: 0.75rem;
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
          padding: 0.75rem 1rem;
          margin-bottom: 0.65rem;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ev-header h1 {
          color: var(--ev-text);
          font-size: 1.2rem;
          line-height: 1.25;
          margin: 0.08rem 0 0.25rem;
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
        .ev-brief-grid {
          display: grid;
          gap: 0.75rem;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          margin-bottom: 0.75rem;
        }
        .ev-brief-grid-two {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .ev-brief-card {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: 8px;
          min-height: 5rem;
          padding: 0.75rem 0.85rem;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ev-brief-label {
          color: var(--ev-accent);
          font-size: 0.75rem;
          font-weight: 760;
          margin-bottom: 0.45rem;
          text-transform: uppercase;
        }
        .ev-brief-value {
          color: var(--ev-text);
          font-size: 0.92rem;
          line-height: 1.45;
        }
        .ev-report-answer {
          background: #ffffff;
          border: 1px solid var(--ev-border);
          border-left: 4px solid var(--ev-accent);
          border-radius: 8px;
          margin-bottom: 0.9rem;
          padding: 1rem 1.1rem;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ev-report-answer-text {
          color: var(--ev-text);
          font-size: 1rem;
          line-height: 1.45;
        }
        .ev-stage-map {
          display: grid;
          gap: 0.55rem;
          grid-template-columns: repeat(5, minmax(0, 1fr));
          margin: 0.2rem 0 0.95rem;
        }
        .ev-stage-card {
          background: #ffffff;
          border: 1px solid var(--ev-border);
          border-radius: 8px;
          min-height: 3.4rem;
          padding: 0.55rem 0.65rem;
          margin-bottom: 0.65rem;
        }
        .ev-stage-active {
          border-color: #8ec8d2;
          box-shadow: inset 0 0 0 2px var(--ev-accent-soft);
        }
        .ev-stage-id {
          color: var(--ev-accent);
          font-size: 0.78rem;
          font-weight: 800;
          margin-bottom: 0.25rem;
        }
        .ev-stage-name {
          color: var(--ev-text);
          font-size: 0.82rem;
          font-weight: 760;
          line-height: 1.25;
        }
        .ev-stage-strip {
          display: flex;
          flex-wrap: wrap;
          gap: 0.4rem;
          margin: 0.1rem 0 0.75rem;
        }
        .ev-stage-chip {
          background: #ffffff;
          border: 1px solid var(--ev-border);
          border-radius: 999px;
          color: var(--ev-muted);
          display: inline-flex;
          font-size: 0.78rem;
          gap: 0.25rem;
          line-height: 1.2;
          padding: 0.32rem 0.65rem;
          white-space: nowrap;
        }
        .ev-stage-chip.ev-stage-active {
          background: var(--ev-accent-soft);
          border-color: #8ec8d2;
          color: var(--ev-accent);
        }
        .ev-analysis-card {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: 8px;
          min-height: 9.2rem;
          padding: 0.95rem 1rem;
          margin-bottom: 0.8rem;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ev-analysis-method,
        .ev-analysis-evidence,
        .ev-analysis-takeaway {
          color: #344054;
          font-size: 0.9rem;
          line-height: 1.45;
          margin-top: 0.35rem;
        }
        .ev-analysis-takeaway {
          border-left: 3px solid var(--ev-ok);
          background: #f6fef9;
          padding: 0.45rem 0.6rem;
        }
        .ev-storyboard-card {
          background: #ffffff;
          border: 1px solid var(--ev-border);
          border-left: 4px solid var(--ev-ok);
          border-radius: 8px;
          margin: 0.35rem 0 0.85rem;
          padding: 0.9rem 1rem;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ev-storyboard-title {
          color: var(--ev-text);
          font-size: 1rem;
          font-weight: 760;
          margin-bottom: 0.35rem;
        }
        .ev-storyboard-summary {
          color: #344054;
          font-size: 0.92rem;
          line-height: 1.45;
        }
        .ev-evidence-summary {
          background: #f6fef9;
          border: 1px solid #abefc6;
          border-left: 4px solid var(--ev-ok);
          border-radius: 8px;
          margin: 0.25rem 0 0.75rem;
          padding: 0.85rem 1rem;
        }
        .ev-evidence-card {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-left: 4px solid var(--ev-info);
          border-radius: 8px;
          margin: 0.55rem 0;
          padding: 0.85rem 1rem;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ev-evidence-card.ev-evidence-audit {
          border-left-color: #98a2b3;
          background: #fcfcfd;
        }
        .ev-evidence-card-top {
          align-items: center;
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          justify-content: space-between;
          margin-bottom: 0.35rem;
        }
        .ev-evidence-name {
          color: var(--ev-text);
          font-size: 0.98rem;
          font-weight: 760;
        }
        .ev-evidence-pill {
          background: #ecfdf3;
          border: 1px solid #abefc6;
          border-radius: 999px;
          color: #067647;
          font-size: 0.72rem;
          font-weight: 760;
          padding: 0.15rem 0.5rem;
          text-transform: uppercase;
        }
        .ev-evidence-meaning {
          color: #344054;
          font-size: 0.92rem;
          line-height: 1.45;
          margin-bottom: 0.5rem;
        }
        .ev-evidence-meta {
          color: var(--ev-muted);
          display: flex;
          flex-wrap: wrap;
          font-size: 0.78rem;
          gap: 0.45rem 0.75rem;
        }
        .ev-claim-card {
          background: var(--ev-panel);
          border: 1px solid var(--ev-border);
          border-radius: 8px;
          margin: 0.7rem 0 0.35rem;
          padding: 0.9rem 1rem;
          box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .ev-claim-supported {
          border-left: 4px solid var(--ev-ok);
        }
        .ev-claim-inconclusive,
        .ev-claim-descriptive {
          border-left: 4px solid #98a2b3;
        }
        .ev-claim-refuted {
          border-left: 4px solid var(--ev-fail);
        }
        .ev-claim-top {
          display: flex;
          flex-wrap: wrap;
          gap: 0.4rem;
          margin-bottom: 0.5rem;
        }
        .ev-claim-text {
          color: var(--ev-text);
          font-size: 0.98rem;
          font-weight: 760;
          line-height: 1.35;
          margin-bottom: 0.4rem;
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
        div[data-testid="stMetricValue"] {
          font-size: 1.45rem;
          line-height: 1.15;
        }
        div[data-testid="stMetricLabel"] {
          font-size: 0.78rem;
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
          .ev-brief-grid {
            grid-template-columns: 1fr;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
