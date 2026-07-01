"""Single-shot exploratory analysis run (the ``evalvitals explore`` entry).

One call = one pipeline = one artifact directory. This is the non-interactive
replacement for the retired chat REPL: the explorer writes EDA code, the host
adjudicates any host-checkable candidate statistics, charts are rendered
deterministically, and everything is persisted (optionally a dashboard opens).

Flow::

    M2ExplorerAgent.explore_path        # free-form EDA (a CLI coding agent)
      -> adjudicate_report              # host recomputes verdicts (in-sample)
      -> render_chart_specs             # spec + CSV -> PNG, host-side
      -> write_report_artifacts         # report.json + figures/ + tables/ + code
      -> launch_dashboard (optional)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from evalvitals.agent_assets.skills import SKILL_BACKENDS, bundled_skill_paths
from evalvitals.analysis.adjudicate import adjudicate_report
from evalvitals.analysis.explorer import M2ExplorerAgent
from evalvitals.eval_agent.cli_agent import CliAgentConfig
from evalvitals.viz.renderer import render_chart_specs


def run_explore(
    path: str | Path,
    *,
    question: str = "Explore this dataset and surface the patterns that matter.",
    out: str | Path = "evalvitals_explore_output",
    coder_provider: str = "antigravity",
    coder_model: str = "",
    coder_binary: str = "",
    max_rows: int = 2000,
    max_files: int = 200,
    include_tool_calls: bool = False,
    timeout_sec: int = 120,
    max_attempts: int = 2,
    dashboard: bool = False,
    dashboard_port: int | None = None,
    skills: "list[str] | tuple[str, ...]" = (),
    allow_skills: bool = False,
    use_bundled_skills: bool = True,
    outcome_col: str | None = None,
) -> int:
    """Run one exploratory analysis and persist its artifacts.

    Returns a process exit code: ``0`` if the exploration succeeded, else ``1``.
    When *dashboard* is set, the return value is the dashboard process's code.

    Skills style the agent-authored figures (e.g. the bundled nature-figure
    skill). By default (*use_bundled_skills*) the package's bundled skills are
    applied on the claude/agy backends; *skills* adds more dirs and *allow_skills*
    also enables globally-installed (`~/.claude/skills`) skills.

    *outcome_col* optionally names the target/label column explicitly (M1
    passes ``"label"``); omit it to let the agent auto-detect an outcome by
    name heuristics, or fall back to unsupervised EDA when there is none.
    """
    out_dir = Path(out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    skill_dirs = list(skills or [])
    if use_bundled_skills and coder_provider in SKILL_BACKENDS:
        # Vendored skills first so the agent sees them; de-dup explicit repeats.
        bundled = [p for p in bundled_skill_paths() if p not in skill_dirs]
        skill_dirs = bundled + skill_dirs

    cli_config = CliAgentConfig(
        provider=coder_provider,
        binary_path=coder_binary,
        model=coder_model,
        timeout_sec=timeout_sec,
        skills=tuple(skill_dirs),
        allow_skills=allow_skills or bool(skill_dirs),
    )
    agent = M2ExplorerAgent(
        cli_config=cli_config,
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
    )
    report = agent.explore_path(
        path,
        question=question,
        max_rows=max_rows,
        max_files=max_files,
        include_tool_calls=include_tool_calls,
        outcome_col=outcome_col,
    )

    # Host adjudication: any candidate that attached host-checkable `sufficient`
    # stats gets a verdict recomputed by the validated core (the explorer never
    # decides). A standalone run has no held-out split, so verdicts are IN-SAMPLE.
    adjudicate_report(report, split_label="in_sample")

    write_report_artifacts(report, out_dir)

    print(f"ok: {report.ok}")
    print(f"attempts: {report.attempts}")
    print(f"rows loaded: {report.data_profile.get('loaded_rows', report.data_profile.get('n_rows'))}")
    print(f"output: {out_dir}")
    if report.error:
        print(f"error: {report.error}")
    if report.observations:
        print("\nobservations:")
        for obs in report.observations[:8]:
            print(f"- {obs}")
    if report.candidate_signals:
        print("\ncandidate signals:")
        for signal in report.candidate_signals[:8]:
            print(f"- {signal.name}: {signal.rationale}{_verdict_suffix(signal)}")
    if report.adjudication:
        adj = report.adjudication
        print(
            f"\nhost adjudication ({adj.get('split')}): "
            f"{adj.get('n_rejected', 0)}/{adj.get('n_host_adjudicated', 0)} reject "
            f"(e-BH family n={adj.get('n_in_family', 0)}, alpha={adj.get('alpha')})"
        )

    if dashboard:
        from evalvitals.analysis.dashboard import launch_dashboard

        return launch_dashboard(out_dir, port=dashboard_port)
    return 0 if report.ok else 1


def _verdict_suffix(signal: Any) -> str:
    """One-line host verdict tag for a candidate signal, or '' if not adjudicated."""
    if not getattr(signal, "host_adjudicated", False):
        return "  [descriptive]" if getattr(signal, "sufficient", None) else ""
    verdict = "REJECT H0" if signal.reject else "inconclusive"
    parts = [verdict]
    if signal.e_value is not None:
        # In the e-BH family (has an e-value); reject already reflects survival.
        parts.append(f"e={signal.e_value:.2f}")
        parts.append("e-BH family")
    elif signal.ci is not None:
        parts.append(f"CI={signal.ci[0]:+.3f}..{signal.ci[1]:+.3f} (not FDR-corrected)")
    return "  [host: " + ", ".join(parts) + "]"


def write_report_artifacts(report: Any, out_dir: Path) -> None:
    """Persist one explore run's artifacts: figures/, tables/, rendered charts,
    the report JSON, and the generated code/stdout/stderr.

    Charts are rendered here (host-side, from spec + CSV) so the persisted
    ``exploratory_report.json`` carries each chart's ``figure_path``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_artifact_dirs(report, out_dir)

    # Render the explorer's chart specs to PNG from the copied tables/ CSVs.
    try:
        report.charts = render_chart_specs(
            getattr(report, "charts", None), out_dir / "tables", out_dir
        )
    except Exception:  # rendering is best-effort, never blocks persistence
        pass

    (out_dir / "exploratory_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    if report.code:
        (out_dir / "analysis.py").write_text(report.code, encoding="utf-8")
    if report.stdout:
        (out_dir / "stdout.txt").write_text(report.stdout, encoding="utf-8")
    if report.stderr:
        (out_dir / "stderr.txt").write_text(report.stderr, encoding="utf-8")
    if report.raw_outputs:
        (out_dir / "agent_raw_output.txt").write_text(
            "\n\n--- attempt ---\n\n".join(report.raw_outputs),
            encoding="utf-8",
        )


def _copy_artifact_dirs(report: Any, out_dir: Path) -> None:
    workdir = Path(getattr(report, "workdir", "") or "")
    if not workdir.exists():
        return
    for name in ("figures", "tables"):
        src = workdir / name
        dest = out_dir / name
        if src.exists() and src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
