"""Public data-analysis API: exploratory analysis, hypothesis proposal, and
statistical confirmation.

This package exposes EvalVitals' data-analysis layer as a standalone
capability — ``ExploratoryAnalysisAgent`` (M2) for descriptive EDA (no
hypothesis generation or validation itself), ``HypothesisAgent`` (M3) for
proposing falsifiable hypotheses from M2's takeaways (proposal only, no
validation), and ``StatsAnalysisAgent`` for confirmatory effect/CI/e-value/FDR
verdicts. The eval-agent loops still use the same implementation internally,
but callers do not need to import from ``evalvitals.eval_agent.stages``.
"""

from evalvitals.analysis.adjudicate import adjudicate_report, adjudicate_signals
from evalvitals.analysis.api import ExploreRunResult, explore
from evalvitals.analysis.explorer import (
    CandidateSignal,
    ExploratoryAnalysisAgent,
    ExploratoryAnalysisReport,
    Takeaway,
    load_records_from_path,
    scan_folder,
)
from evalvitals.analysis.failure_modes import FailureMode, FailureModeReport, cluster_failures
from evalvitals.analysis.fused_pipeline import (
    FusedReport,
    FusedSignal,
    run_fused_analysis,
)
from evalvitals.analysis.hypothesis_agent import Hypothesis, HypothesisAgent
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
from evalvitals.analysis.workbench import (
    DatasetBundle,
    EventSink,
    ThreadStore,
    UploadLimits,
    extract_archive,
    ingest_directory,
)
from evalvitals.reporting.compiler import compile_diagnostic_report
from evalvitals.reporting.model import Claim, DiagnosticReport, Evidence, ReportStep

__all__ = [
    "explore",
    "ExploreRunResult",
    "cluster_failures",
    "FailureMode",
    "FailureModeReport",
    "StatsAnalysisAgent",
    "StatsAnalysisReport",
    "ExploratoryAnalysisAgent",
    "ExploratoryAnalysisReport",
    "Takeaway",
    "CandidateSignal",
    "Hypothesis",
    "HypothesisAgent",
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
    "scan_folder",
    "StatsInput",
    "StatsToolResult",
    "EvidenceResult",
    "STATS_TOOL_CATALOG",
    "build_stats_input",
    "build_stats_input_from_records",
    "default_plan",
    "fdr_correct",
    "run_stats_tool",
    "DatasetBundle",
    "EventSink",
    "ThreadStore",
    "UploadLimits",
    "extract_archive",
    "ingest_directory",
]
