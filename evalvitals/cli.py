"""Top-level EvalVitals command-line interface."""

from __future__ import annotations

import argparse

from evalvitals.analysis.dashboard import launch_dashboard, launch_upload_app
from evalvitals.analysis.explore_run import run_explore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalvitals",
        description="EvalVitals command-line interface.",
    )
    sub = parser.add_subparsers(dest="command")

    explore = sub.add_parser(
        "explore",
        help="Run a single-shot exploratory analysis over a results directory.",
        description="Run one Lambda-style exploratory analysis (no interactive REPL).",
    )
    explore.add_argument("path", nargs="?", help="File or directory of JSON/JSONL results.")
    explore.add_argument(
        "-q",
        "--question",
        default="Explore this dataset and surface the patterns that matter.",
        help="Natural-language analysis question for the local coding agent.",
    )
    explore.add_argument(
        "--outcome-col",
        default=None,
        help="Name of the target/outcome column, if any (e.g. 'label'). Omit to "
             "auto-detect by name heuristics, or fall back to unsupervised EDA "
             "when the data has no recognizable outcome.",
    )
    explore.add_argument(
        "--out",
        default="evalvitals_explore_output",
        help="Output directory for report/code/figures/tables.",
    )
    explore.add_argument(
        "--backend",
        "--coder-provider",
        dest="coder_provider",
        default="antigravity",
        choices=["antigravity", "codex", "claude_code", "opencode", "gemini_cli", "kimi_cli"],
        help="Local CLI coding-agent backend.",
    )
    explore.add_argument("--model", "--coder-model", dest="coder_model", default="")
    explore.add_argument("--coder-binary", default="")
    explore.add_argument("--max-rows", type=int, default=2000)
    explore.add_argument("--max-files", type=int, default=200)
    explore.add_argument("--include-tool-calls", action="store_true")
    explore.add_argument("--timeout-sec", type=int, default=120)
    explore.add_argument("--max-attempts", type=int, default=2)
    explore.add_argument(
        "--dashboard",
        action="store_true",
        help="Open the Streamlit dashboard on the output directory when done.",
    )
    explore.add_argument("--port", type=int, default=None, help="Optional dashboard port.")
    explore.add_argument(
        "--skill", action="append", default=[], metavar="DIR",
        help="Agent-Skill directory (with SKILL.md) to style agent-authored "
             "figures (e.g. nature-figure). Repeatable. claude/agy/codex backends.",
    )
    explore.add_argument(
        "--allow-skills", action="store_true",
        help="Enable the Skill tool so ~/.claude/skills are usable too "
             "(implied by --skill).",
    )
    explore.add_argument(
        "--no-skills", dest="use_bundled_skills", action="store_false", default=True,
        help="Do not apply the package's bundled skills (e.g. nature-figure).",
    )
    explore.add_argument(
        "--no-hypotheses", dest="propose_hypotheses", action="store_false", default=True,
        help="Skip M3 (falsifiable hypotheses proposed from the M2 takeaways). "
             "Runs by default after a successful explore.",
    )

    dashboard = sub.add_parser(
        "dashboard",
        help="Open a Streamlit dashboard for an explore output or loop-run directory.",
        description="Open a Streamlit dashboard for EvalVitals single-run artifacts.",
    )
    dashboard.add_argument("run_dir", help="An explore output dir or a loop-run dir.")
    dashboard.add_argument("--port", type=int, default=None, help="Optional Streamlit port.")

    web = sub.add_parser(
        "web",
        help="Launch the upload-and-explore web workbench (upload a .zip, run M2+M3).",
        description="Serve a Streamlit page where users upload a .zip of results; "
                    "each upload becomes one `evalvitals explore` run (M2 exploratory "
                    "analysis + M3 hypotheses) and renders like `evalvitals dashboard`.",
    )
    web.add_argument(
        "workspace", nargs="?", default="evalvitals_web_runs",
        help="Directory where uploaded runs accumulate (created if missing).",
    )
    web.add_argument("--port", type=int, default=None, help="Optional Streamlit port.")
    web.add_argument(
        "--backend", "--coder-provider", dest="coder_provider", default="claude_code",
        choices=["antigravity", "codex", "claude_code", "opencode", "gemini_cli", "kimi_cli"],
        help="Default coding-agent backend pre-selected in the upload form.",
    )
    web.add_argument("--model", "--coder-model", dest="coder_model", default="",
                     help="Default model id pre-filled in the upload form.")
    web.add_argument("--timeout-sec", type=int, default=1200,
                     help="Default per-attempt explorer timeout in the upload form.")

    args = parser.parse_args(argv)
    if args.command == "explore":
        if not args.path:
            parser.error("evalvitals explore requires a results path")
        return run_explore(
            args.path,
            question=args.question,
            out=args.out,
            coder_provider=args.coder_provider,
            coder_model=args.coder_model,
            coder_binary=args.coder_binary,
            max_rows=args.max_rows,
            max_files=args.max_files,
            include_tool_calls=args.include_tool_calls,
            timeout_sec=args.timeout_sec,
            max_attempts=args.max_attempts,
            dashboard=args.dashboard,
            dashboard_port=args.port,
            skills=args.skill,
            allow_skills=args.allow_skills,
            use_bundled_skills=args.use_bundled_skills,
            outcome_col=args.outcome_col,
            propose_hypotheses=args.propose_hypotheses,
        )
    if args.command == "dashboard":
        return launch_dashboard(args.run_dir, port=args.port)
    if args.command == "web":
        return launch_upload_app(
            args.workspace,
            port=args.port,
            backend=args.coder_provider,
            model=args.coder_model,
            timeout_sec=args.timeout_sec,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
