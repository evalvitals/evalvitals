"""Public statistical analysis API.

This package exposes the M2 statistical analysis layer as a standalone
capability.  The eval-agent loops still use the same implementation internally,
but callers do not need to import from ``evalvitals.eval_agent.stages``.
"""

from evalvitals.analysis.explorer import (
    CandidateSignal,
    ExploratoryAnalysisReport,
    M2ExplorerAgent,
)
from evalvitals.analysis.stats_agent import StatsAnalysisAgent, StatsAnalysisReport
from evalvitals.analysis.stats_tools import (
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
    "StatsAnalysisAgent",
    "StatsAnalysisReport",
    "M2ExplorerAgent",
    "ExploratoryAnalysisReport",
    "CandidateSignal",
    "StatsInput",
    "StatsToolResult",
    "STATS_TOOL_CATALOG",
    "build_stats_input",
    "build_stats_input_from_records",
    "default_plan",
    "fdr_correct",
    "run_stats_tool",
]
