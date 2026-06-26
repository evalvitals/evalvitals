"""Standalone public entrypoint for M2 statistical analysis."""

from evalvitals.eval_agent.stages.stats_agent import (
    StatsAnalysisAgent,
    StatsAnalysisReport,
)

__all__ = ["StatsAnalysisAgent", "StatsAnalysisReport"]
