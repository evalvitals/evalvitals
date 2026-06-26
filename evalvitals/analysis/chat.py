"""Interactive chat shell for standalone M2 exploration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalvitals.analysis.explorer import M2ExplorerAgent
from evalvitals.eval_agent.cli_agent import CliAgentConfig
from evalvitals.eval_agent.sandbox import ExperimentSandbox


@dataclass
class M2ChatConfig:
    """Configuration for the backend-only M2 chat shell."""

    path: str
    out: str = "m2_chat_output"
    coder_provider: str = "antigravity"
    coder_model: str = ""
    coder_binary: str = ""
    max_rows: int = 2000
    max_files: int = 200
    include_tool_calls: bool = False
    timeout_sec: int = 120
    max_attempts: int = 2
    prompt: str = "evalvitals"


class M2ChatShell:
    """Small REPL that routes natural-language turns to :class:`M2ExplorerAgent`."""

    def __init__(self, config: M2ChatConfig) -> None:
        self.config = config
        self.out_root = Path(config.out).resolve()
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []
        self.turn = 0
        self.cli_config = CliAgentConfig(
            provider=config.coder_provider,
            binary_path=config.coder_binary,
            model=config.coder_model,
            timeout_sec=config.timeout_sec,
        )

    def run(self) -> int:
        print("EvalVitals chat")
        print("mode: explore")
        print(f"data: {Path(self.config.path).resolve()}")
        print(f"out : {self.out_root}")
        print("Type a question, or :help / :quit.")

        while True:
            try:
                user_text = input(f"\n{self.config.prompt}> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_text:
                continue
            if self._handle_command(user_text):
                continue
            self._run_turn(user_text)

        self._write_history()
        print(f"history: {self.out_root / 'chat_history.json'}")
        return 0

    def _handle_command(self, text: str) -> bool:
        if text in {":q", ":quit", "quit", "exit"}:
            self._write_history()
            raise SystemExit(0)
        if text == ":help":
            print("Ask questions like:")
            print("  Which failure patterns distinguish wrong answers from correct answers?")
            print("  Compare models by accuracy and tool usage.")
            print("  Find candidate signals I should confirm with StatsAnalysisAgent.")
            print("Commands: :help, :history, :status, :quit")
            return True
        if text == ":history":
            for item in self.history:
                print(f"- turn {item['turn']}: {item['question']}")
            return True
        if text == ":status":
            print("mode: explore")
            print(f"data: {Path(self.config.path).resolve()}")
            print(f"out : {self.out_root}")
            print(f"turns: {len(self.history)}")
            print(f"backend: {self.config.coder_provider}")
            return True
        return False

    def _run_turn(self, user_text: str) -> None:
        self.turn += 1
        turn_dir = self.out_root / f"turn_{self.turn:03d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        question = self._question_with_history(user_text)
        agent = M2ExplorerAgent(
            cli_config=self.cli_config,
            sandbox=ExperimentSandbox(workdir=turn_dir / "sandbox", cleanup=False),
            timeout_sec=self.config.timeout_sec,
            max_attempts=self.config.max_attempts,
        )
        print(f"running turn {self.turn} ...")
        report = agent.explore_path(
            self.config.path,
            question=question,
            max_rows=self.config.max_rows,
            max_files=self.config.max_files,
            include_tool_calls=self.config.include_tool_calls,
        )
        write_report_artifacts(report, turn_dir)
        self.history.append({
            "turn": self.turn,
            "question": user_text,
            "ok": report.ok,
            "observations": report.observations[:5],
            "candidate_signals": report.candidate_signal_names[:8],
            "out_dir": str(turn_dir),
        })
        self._write_history()
        print(f"ok: {report.ok}  attempts: {report.attempts}  out: {turn_dir}")
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

    def _question_with_history(self, question: str) -> str:
        if not self.history:
            return question
        context = []
        for item in self.history[-3:]:
            context.append({
                "question": item.get("question"),
                "observations": item.get("observations"),
                "candidate_signals": item.get("candidate_signals"),
            })
        return (
            question
            + "\n\nPrevious chat context (use only if relevant):\n"
            + json.dumps(context, indent=2, default=str)
        )

    def _write_history(self) -> None:
        (self.out_root / "chat_history.json").write_text(
            json.dumps(self.history, indent=2, default=str),
            encoding="utf-8",
        )


def write_report_artifacts(report: Any, out_dir: Path) -> None:
    """Persist one chat/explore turn's artifacts."""
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
