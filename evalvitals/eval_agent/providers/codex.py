"""OpenAI Codex CLI coding-provider adapter."""

from __future__ import annotations

from pathlib import Path

from evalvitals.eval_agent.cli_skills import CodexSkillInstaller, SkillInstaller
from evalvitals.eval_agent.providers.base import CliAgentBase


class CodexAgent(CliAgentBase):
    """OpenAI Codex CLI backend (``codex exec``)."""

    _provider_name = "codex"

    def _make_skill_installer(self, skills: list[str]) -> SkillInstaller:
        return CodexSkillInstaller(skills)

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary,
            "exec",
            prompt,
            "--sandbox",
            "workspace-write",
            "--json",
            "-C",
            str(workdir),
        ]
        if self._model:
            cmd += ["-m", self._model]
        cmd.extend(self._extra_args)
        return cmd
