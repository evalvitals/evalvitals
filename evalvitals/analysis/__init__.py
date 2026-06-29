"""Public statistical analysis API.

This package exposes the M2 statistical analysis layer as a standalone
capability.  The eval-agent loops still use the same implementation internally,
but callers do not need to import from ``evalvitals.eval_agent.stages``.
"""

from evalvitals.analysis.adjudicate import adjudicate_report, adjudicate_signals
from evalvitals.analysis.explorer import (
    CandidateSignal,
    ExploratoryAnalysisReport,
    M2ExplorerAgent,
    load_records_from_path,
)
from evalvitals.analysis.fused_pipeline import (
    FusedReport,
    FusedSignal,
    run_fused_analysis,
)
from evalvitals.analysis.operationalize import (
    RecipeError,
    SignalRecipe,
    bridge_recipes_to_result,
    compile_recipe,
    compile_recipes,
    per_case_finding,
    per_case_to_records,
    safe_ident,
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
from evalvitals.reporting.compiler import compile_diagnostic_report
from evalvitals.reporting.model import Claim, DiagnosticReport, Evidence, ReportStep

__all__ = [
    "StatsAnalysisAgent",
    "StatsAnalysisReport",
    "M2ExplorerAgent",
    "ExploratoryAnalysisReport",
    "CandidateSignal",
    "adjudicate_report",
    "adjudicate_signals",
    "SignalRecipe",
    "compile_recipe",
    "compile_recipes",
    "per_case_finding",
    "per_case_to_records",
    "bridge_recipes_to_result",
    "safe_ident",
    "RecipeError",
    "run_fused_analysis",
    "FusedReport",
    "FusedSignal",
    "DiagnosticReport",
    "Claim",
    "Evidence",
    "ReportStep",
    "compile_diagnostic_report",
    "load_records_from_path",
    "StatsInput",
    "StatsToolResult",
    "STATS_TOOL_CATALOG",
    "build_stats_input",
    "build_stats_input_from_records",
    "default_plan",
    "fdr_correct",
    "run_stats_tool",
]
