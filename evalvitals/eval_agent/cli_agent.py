"""Compatibility facade for CLI coding agents and CLI-backed judge models.

The implementation is split by responsibility:

- ``evalvitals.eval_agent.providers`` contains CLI coding-provider adapters.
- ``evalvitals.eval_agent.models`` contains judge model wrappers.
- ``evalvitals.eval_agent.cli_types`` contains public config/result types.

This module keeps the historical import path stable.
"""

from __future__ import annotations

from evalvitals.eval_agent.cli_types import BINARY_DEFAULTS, CliAgentConfig, CliAgentResult
from evalvitals.eval_agent.models.agy import (
    AgyModel,
)
from evalvitals.eval_agent.models.agy import (
    safe_unlink as _safe_unlink,
)
from evalvitals.eval_agent.models.agy import (
    scan_agy_log as _scan_agy_log,
)
from evalvitals.eval_agent.models.claude import ClaudeModel
from evalvitals.eval_agent.providers.antigravity import AntigravityAgent
from evalvitals.eval_agent.providers.base import CliAgentBase as _CliAgentBase
from evalvitals.eval_agent.providers.claude_code import ClaudeCodeAgent
from evalvitals.eval_agent.providers.codex import CodexAgent
from evalvitals.eval_agent.providers.gemini_cli import GeminiCliAgent
from evalvitals.eval_agent.providers.kimi_cli import KimiCliAgent
from evalvitals.eval_agent.providers.opencode import OpenCodeAgent
from evalvitals.eval_agent.providers.registry import (
    PROVIDER_CLASSES as _PROVIDER_CLASSES,
)
from evalvitals.eval_agent.providers.registry import (
    create_cli_agent,
)

__all__ = [
    "AgyModel",
    "AntigravityAgent",
    "BINARY_DEFAULTS",
    "ClaudeCodeAgent",
    "ClaudeModel",
    "CliAgentConfig",
    "CliAgentResult",
    "CodexAgent",
    "GeminiCliAgent",
    "KimiCliAgent",
    "OpenCodeAgent",
    "_CliAgentBase",
    "_PROVIDER_CLASSES",
    "_safe_unlink",
    "_scan_agy_log",
    "create_cli_agent",
]
