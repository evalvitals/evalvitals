"""Claude Code CLI coding-provider adapter."""

from __future__ import annotations

from pathlib import Path

from evalvitals.eval_agent.cli_transcript import render_claude_stream
from evalvitals.eval_agent.providers.base import CliAgentBase


class ClaudeCodeAgent(CliAgentBase):
    """Claude Code CLI backend (``claude -p``)."""

    _provider_name = "claude_code"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        allowed = "Bash Edit Write Read" + (" Skill" if self._allow_skills else "")
        cmd = [
            self._binary,
            "-p",
            prompt,
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--verbose",
            "--allowed-tools",
            allowed,
            "--add-dir",
            str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        if self._max_budget_usd:
            cmd += ["--max-budget-usd", str(self._max_budget_usd)]
        cmd.extend(self._extra_args)
        return cmd

    def _postprocess_output(self, stdout: str) -> tuple[str, dict | None]:
        return render_claude_stream(stdout)
