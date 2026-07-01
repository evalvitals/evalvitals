"""Public data-analysis API: exploratory analysis + statistical confirmation.

This package exposes EvalVitals' data-analysis layer as a standalone
capability — ``ExploratoryAnalysisAgent`` for descriptive EDA (no hypothesis
generation or validation) and ``StatsAnalysisAgent`` for confirmatory
effect/CI/e-value/FDR verdicts. The eval-agent loops still use the same
implementation internally, but callers do not need to import from
``evalvitals.eval_agent.stages``.
"""

from evalvitals.analysis.adjudicate import adjudicate_report, adjudicate_signals
from evalvitals.analysis.explorer import (
    CandidateSignal,
    ExploratoryAnalysisAgent,
    ExploratoryAnalysisReport,
    Takeaway,
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
from evalvitals.analysis.planner import AnalysisPlanItem, plan_stats_input, ranked_signal_names
from evalvitals.analysis.profile import (
    ColumnProfile,
    DatasetProfile,
    describe_outcome,
    profile_records,
    profile_stats_input,
)
from evalvitals.analysis.stats_agent import StatsAnalysisAgent, StatsAnalysisReport
from evalvitals.analysis.stats_tools import (
    STATS_TOOL_CATALOG,
    EvidenceResult,
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
    "ExploratoryAnalysisAgent",
    "ExploratoryAnalysisReport",
    "Takeaway",
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
    "ColumnProfile",
    "DatasetProfile",
    "describe_outcome",
    "profile_records",
    "profile_stats_input",
    "AnalysisPlanItem",
    "ranked_signal_names",
    "plan_stats_input",
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
    "EvidenceResult",
    "STATS_TOOL_CATALOG",
    "build_stats_input",
    "build_stats_input_from_records",
    "default_plan",
    "fdr_correct",
    "run_stats_tool",
]
