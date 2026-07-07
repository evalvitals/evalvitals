"""CLI-backed judge model wrappers."""

from evalvitals.eval_agent.models.agy import AgyModel, scan_agy_log
from evalvitals.eval_agent.models.claude import ClaudeModel

__all__ = ["AgyModel", "ClaudeModel", "scan_agy_log"]
