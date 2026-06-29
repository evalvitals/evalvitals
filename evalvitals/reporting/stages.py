"""Stage semantics for EvalVitals diagnostic reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    id: str
    name: str
    question: str
    artifacts: str
    dashboard_role: str


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec(
        id="M1",
        name="Measurement",
        question="What per-case signals did the evaluators/analyzers extract?",
        artifacts="Frozen per-case feature matrix, analyzer outputs, attention/probe artifacts.",
        dashboard_role="Problem Setting: defines the dataset and available signals.",
    ),
    StageSpec(
        id="M2",
        name="Confirmatory analysis",
        question="Which signals distinguish FAIL from PASS on a held-out split?",
        artifacts="Effect sizes, confidence intervals, e-values/e-BH decisions, charts/tables.",
        dashboard_role="Analysis: method, evidence, chart, and takeaway.",
    ),
    StageSpec(
        id="M3",
        name="Hypothesis generation",
        question="What falsifiable failure mechanisms explain the confirmed signals?",
        artifacts="Hypotheses, failure modes, cited M2 charts/observations.",
        dashboard_role="Hypotheses: candidate mechanisms linked back to evidence.",
    ),
    StageSpec(
        id="M4",
        name="Mechanism test",
        question="Does an intervention or controlled probe support the mechanism?",
        artifacts="Targeted experiment results and intervention records.",
        dashboard_role="Hypotheses & Artifacts: decision evidence before fixes.",
    ),
    StageSpec(
        id="M5",
        name="Repair / surgery test",
        question="Does a proposed change repair failures without unacceptable regressions?",
        artifacts="Surgery/fix outcomes, adjudication records, regression checks.",
        dashboard_role="Hypotheses & Artifacts: final gate for action.",
    ),
)


def stage_specs_as_dicts() -> list[dict[str, str]]:
    return [spec.__dict__.copy() for spec in STAGE_SPECS]

