"""Command-line entrypoints for standalone analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from evalvitals.analysis.chat import M2ChatConfig, M2ChatShell, write_report_artifacts
from evalvitals.analysis.explorer import M2ExplorerAgent
from evalvitals.eval_agent.cli_agent import CliAgentConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalvitals-m2-explore",
        description="Run Lambda-style local exploratory M2 analysis over JSON/JSONL logs.",
    )
    parser.add_argument("path", help="File or directory of JSON/JSONL results.")
    parser.add_argument(
        "-q",
        "--question",
        default="Explore patterns that distinguish failures from passes.",
        help="Natural-language analysis question for the local coding agent.",
    )
    parser.add_argument(
        "--out",
        default="m2_explore_output",
        help="Output directory for report/code/stdout.",
    )
    parser.add_argument(
        "--coder-provider",
        default="antigravity",
        choices=["antigravity", "codex", "claude_code", "opencode", "gemini_cli", "kimi_cli"],
        help="Local CLI coding agent backend.",
    )
    parser.add_argument("--coder-model", default="", help="Optional model flag for the CLI agent.")
    parser.add_argument("--coder-binary", default="", help="Explicit path to the CLI binary.")
    parser.add_argument("--max-rows", type=int, default=2000, help="Maximum loaded records.")
    parser.add_argument("--max-files", type=int, default=200, help="Maximum JSON files to scan.")
    parser.add_argument(
        "--include-tool-calls",
        action="store_true",
        help="Also load tool_calls_*.json files. Off by default because these can dominate logs.",
    )
    parser.add_argument("--timeout-sec", type=int, default=120, help="Sandbox/agent timeout.")
    parser.add_argument("--max-attempts", type=int, default=2, help="Code repair attempts.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cli_config = CliAgentConfig(
        provider=args.coder_provider,
        binary_path=args.coder_binary,
        model=args.coder_model,
        timeout_sec=args.timeout_sec,
    )
    agent = M2ExplorerAgent(
        cli_config=cli_config,
        timeout_sec=args.timeout_sec,
        max_attempts=args.max_attempts,
    )
    report = agent.explore_path(
        args.path,
        question=args.question,
        max_rows=args.max_rows,
        max_files=args.max_files,
        include_tool_calls=args.include_tool_calls,
    )

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
            print(f"- {signal.name}: {signal.rationale}")
    return 0 if report.ok else 1


def chat_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalvitals-m2-chat",
        description="Interactive Lambda-style M2 chat over a results directory.",
    )
    parser.add_argument("path", help="File or directory of JSON/JSONL results.")
    parser.add_argument(
        "--out",
        default="m2_chat_output",
        help="Output directory. Each chat turn gets a numbered subdirectory.",
    )
    parser.add_argument(
        "--coder-provider",
        default="antigravity",
        choices=["antigravity", "codex", "claude_code", "opencode", "gemini_cli", "kimi_cli"],
        help="Local CLI coding agent backend.",
    )
    parser.add_argument("--coder-model", default="", help="Optional model flag for the CLI agent.")
    parser.add_argument("--coder-binary", default="", help="Explicit path to the CLI binary.")
    parser.add_argument("--max-rows", type=int, default=2000, help="Maximum loaded records.")
    parser.add_argument("--max-files", type=int, default=200, help="Maximum JSON files to scan.")
    parser.add_argument(
        "--include-tool-calls",
        action="store_true",
        help="Also load tool_calls_*.json files.",
    )
    parser.add_argument("--timeout-sec", type=int, default=120, help="Sandbox/agent timeout.")
    parser.add_argument("--max-attempts", type=int, default=2, help="Code repair attempts.")
    args = parser.parse_args(argv)

    return M2ChatShell(
        M2ChatConfig(
            path=args.path,
            out=args.out,
            coder_provider=args.coder_provider,
            coder_model=args.coder_model,
            coder_binary=args.coder_binary,
            max_rows=args.max_rows,
            max_files=args.max_files,
            include_tool_calls=args.include_tool_calls,
            timeout_sec=args.timeout_sec,
            max_attempts=args.max_attempts,
            prompt="m2",
        )
    ).run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
