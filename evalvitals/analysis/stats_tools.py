"""Standalone public entrypoint for M2 statistical tools."""

from evalvitals.eval_agent.stages.stats_tools import (
    STATS_TOOL_CATALOG,
    StatsInput,
    StatsToolResult,
    build_stats_input,
    build_stats_input_from_records,
    default_plan,
    fdr_correct,
    run_stats_tool,
)

__all__ = [
    "StatsInput",
    "StatsToolResult",
    "STATS_TOOL_CATALOG",
    "build_stats_input",
    "build_stats_input_from_records",
    "default_plan",
    "fdr_correct",
    "run_stats_tool",
]
