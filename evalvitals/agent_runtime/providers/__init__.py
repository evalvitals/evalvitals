"""CLI coding-provider adapters."""

from evalvitals.agent_runtime.providers.antigravity import AntigravityAgent
from evalvitals.agent_runtime.providers.base import CliAgentBase
from evalvitals.agent_runtime.providers.claude_code import ClaudeCodeAgent
from evalvitals.agent_runtime.providers.codex import CodexAgent
from evalvitals.agent_runtime.providers.gemini_cli import GeminiCliAgent
from evalvitals.agent_runtime.providers.kimi_cli import KimiCliAgent
from evalvitals.agent_runtime.providers.opencode import OpenCodeAgent
from evalvitals.agent_runtime.providers.registry import create_cli_agent

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
