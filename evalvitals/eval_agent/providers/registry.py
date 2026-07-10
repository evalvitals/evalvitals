"""Provider registry and factory for CLI coding agents."""

from __future__ import annotations

import shutil

from evalvitals.eval_agent.cli_types import BINARY_DEFAULTS, CliAgentConfig
from evalvitals.eval_agent.providers.antigravity import AntigravityAgent
from evalvitals.eval_agent.providers.base import CliAgentBase
from evalvitals.eval_agent.providers.claude_code import ClaudeCodeAgent
from evalvitals.eval_agent.providers.codex import CodexAgent
from evalvitals.eval_agent.providers.gemini_cli import GeminiCliAgent
from evalvitals.eval_agent.providers.kimi_cli import KimiCliAgent
from evalvitals.eval_agent.providers.opencode import OpenCodeAgent

PROVIDER_CLASSES: dict[str, type[CliAgentBase]] = {
    "claude_code": ClaudeCodeAgent,
    "codex": CodexAgent,
    "opencode": OpenCodeAgent,
    "gemini_cli": GeminiCliAgent,
    "kimi_cli": KimiCliAgent,
    "antigravity": AntigravityAgent,
}


def create_cli_agent(config: CliAgentConfig) -> CliAgentBase:
    """Instantiate the appropriate CLI coding provider for *config*."""
    provider = config.provider

    if provider == "llm":
        raise ValueError(
            "'llm' is not a CLI provider. Use ExperimentWriter directly "
            "(leave cli_agent=None or CliAgentConfig(provider='llm'))."
        )

    cls = PROVIDER_CLASSES.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown CLI provider: {provider!r}. Valid: {sorted(PROVIDER_CLASSES)}"
        )

    binary = config.binary_path or shutil.which(BINARY_DEFAULTS[provider]) or ""
    if not binary:
        raise RuntimeError(
            f"CLI agent binary for {provider!r} not found in PATH. "
            f"Install '{BINARY_DEFAULTS[provider]}' or pass "
            f"CliAgentConfig(binary_path='/path/to/{BINARY_DEFAULTS[provider]}')."
        )

    return cls(
        binary_path=binary,
        model=config.model,
        max_budget_usd=config.max_budget_usd,
        timeout_sec=config.timeout_sec,
        extra_args=list(config.extra_args),
        skills=list(config.skills),
        allow_skills=config.allow_skills,
    )
