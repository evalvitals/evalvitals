"""CLI coding-provider adapters."""

from evalvitals.eval_agent.providers.antigravity import AntigravityAgent
from evalvitals.eval_agent.providers.base import CliAgentBase
from evalvitals.eval_agent.providers.claude_code import ClaudeCodeAgent
from evalvitals.eval_agent.providers.codex import CodexAgent
from evalvitals.eval_agent.providers.gemini_cli import GeminiCliAgent
from evalvitals.eval_agent.providers.kimi_cli import KimiCliAgent
from evalvitals.eval_agent.providers.opencode import OpenCodeAgent
from evalvitals.eval_agent.providers.registry import create_cli_agent

__all__ = [
    "AntigravityAgent",
    "CliAgentBase",
    "ClaudeCodeAgent",
    "CodexAgent",
    "GeminiCliAgent",
    "KimiCliAgent",
    "OpenCodeAgent",
    "create_cli_agent",
]
