"""Top-level EvalVitals command-line interface."""

from __future__ import annotations

import argparse

from evalvitals.analysis.chat import M2ChatConfig, M2ChatShell
from evalvitals.analysis.dashboard import launch_dashboard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalvitals",
        description="EvalVitals command-line interface.",
    )
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser(
        "chat",
        help="Start an interactive chat session.",
        description="Start an interactive EvalVitals chat session.",
    )
    chat.add_argument("path", nargs="?", help="File or directory of JSON/JSONL results.")
    chat.add_argument(
        "--mode",
        default="explore",
        choices=["explore"],
        help="Chat mode. Currently only exploratory M2 analysis is supported.",
    )
    chat.add_argument(
        "--out",
        default="evalvitals_chat_output",
        help="Output directory. Each chat turn gets a numbered subdirectory.",
    )
    chat.add_argument(
        "--backend",
        "--coder-provider",
        dest="coder_provider",
        default="antigravity",
        choices=["antigravity", "codex", "claude_code", "opencode", "gemini_cli", "kimi_cli"],
        help="Local CLI coding-agent backend.",
    )
    chat.add_argument("--model", "--coder-model", dest="coder_model", default="")
    chat.add_argument("--coder-binary", default="")
    chat.add_argument("--max-rows", type=int, default=2000)
    chat.add_argument("--max-files", type=int, default=200)
    chat.add_argument("--include-tool-calls", action="store_true")
    chat.add_argument("--timeout-sec", type=int, default=120)
    chat.add_argument("--max-attempts", type=int, default=2)
    chat.add_argument(
        "--dashboard",
        action="store_true",
        help="Open the Streamlit dashboard after the first completed chat turn.",
    )
    chat.add_argument(
        "--dashboard-port",
        type=int,
        default=None,
        help="Optional Streamlit port for --dashboard.",
    )
    chat.add_argument(
        "--no-dashboard-browser",
        action="store_true",
        help="Start --dashboard without asking Streamlit to open a browser.",
    )

    dashboard = sub.add_parser(
        "dashboard",
        help="Open a Streamlit dashboard for a chat/session output directory.",
        description="Open a Streamlit dashboard for EvalVitals chat artifacts.",
    )
    dashboard.add_argument("session_dir", help="Directory containing turn_*/ artifacts.")
    dashboard.add_argument("--port", type=int, default=None, help="Optional Streamlit port.")

    args = parser.parse_args(argv)
    if args.command == "chat":
        if not args.path:
            parser.error("evalvitals chat requires a results path for --mode explore")
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
                prompt="evalvitals",
                dashboard=args.dashboard,
                dashboard_port=args.dashboard_port,
                dashboard_browser=not args.no_dashboard_browser,
            )
        ).run()
    if args.command == "dashboard":
        return launch_dashboard(args.session_dir, port=args.port)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
