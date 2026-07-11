"""Antigravity CLI coding-provider adapter."""

from __future__ import annotations

from pathlib import Path

from evalvitals.agent_runtime.providers.base import CliAgentBase


class AntigravityAgent(CliAgentBase):
    """Antigravity CLI backend (``agy -p``)."""

    _provider_name = "antigravity"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "-p",
            prompt,
            "--dangerously-skip-permissions",
            "--add-dir",
            str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        cmd.extend(self._extra_args)
        return cmd
