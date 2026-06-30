"""Compile raw dashboard artifacts into a claim-first diagnostic report."""

from __future__ import annotations

from typing import Any

from evalvitals.reporting.model import Claim, DiagnosticReport, Evidence, ReportStep
from evalvitals.viz.labels import display_name


def compile_diagnostic_report(
    story: dict[str, Any] | None,
    explore_report: dict[str, Any] | None,
) -> DiagnosticReport:
    """Build a semantic report from a loop story plus the Step-1 explore report.

    This compiler is deterministic.  Agent-authored fields such as ``claims`` or
    ``chart_readings`` are preserved when present, but host-confirmed signal
    verdicts and loop outcomes remain the authoritative backbone.
    """
    story = story or {}
    explore_report = explore_report or {}
    signals = _candidate_signals(explore_report)
    evidence = _compile_evidence(explore_report, story)
    claims = _compile_claims(explore_report, story, signals)
    answer = _answer(explore_report, claims)
    confidence = _confidence(claims)

    return DiagnosticReport(
        question=str(explore_report.get("question") or "What distinguishes failures from passes?"),
        answer=answer,
        confidence=confidence,
        claims=claims,
        evidence=evidence,
        timeline=_compile_timeline(story, explore_report),
        visual_decisions=list(explore_report.get("visual_plan") or []),
        chart_readings=list(explore_report.get("chart_readings") or []),
        dashboard_storyboard=_dashboard_storyboard(explore_report, story, claims),
        critique=_critique(explore_report, signals),
        caveats=[str(c) for c in (explore_report.get("caveats") or [])],
        next_actions=_next_actions(explore_report, claims),
    )


def _dashboard_storyboard(
    explore_report: dict[str, Any],
    story: dict[str, Any],
    claims: list[Claim],
) -> list[dict[str, Any]]:
    raw = explore_report.get("dashboard_storyboard") or explore_report.get("ui_panels")
    if isinstance(raw, list) and all(isinstance(p, dict) for p in raw):
        return [dict(p) for p in raw]

    supported = next((c.text for c in claims if c.status == "supported"), "")
    observations = [str(x) for x in (explore_report.get("observations") or [])[:3]]
    readings = [
        str(r.get("reading"))
        for r in (explore_report.get("chart_readings") or [])
        if isinstance(r, dict) and r.get("reading")
    ][:3]
    hypotheses = []
    for diag in story.get("diagnoses") or []:
        for h in diag.get("hypotheses") or []:
            hypotheses.append(str(h.get("statement") or h))

    return [
        {
            "id": "problem_setting",
            "title": "Problem Setting",
            "stages": ["M1"],
            "summary": str(explore_report.get("question") or "What distinguishes failures from passes?"),
            "items": observations,
            "artifact_refs": ["data_profile", "candidate_signals"],
        },
        {
            "id": "analysis",
            "title": "Analysis",
            "stages": ["M2"],
            "summary": supported or str(explore_report.get("conclusion") or ""),
            "items": readings,
            "artifact_refs": ["candidate_signals", "charts", "chart_readings"],
        },
        {
            "id": "hypotheses_artifacts",
            "title": "Hypotheses & Artifacts",
            "stages": ["M3", "M4", "M5"],
            "summary": "Hypotheses and downstream decisions generated from the analysis.",
            "items": hypotheses[:5],
            "artifact_refs": ["diagnoses", "surgeries", "fixes"],
        },
    ]


def _candidate_signals(explore_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        s for s in (explore_report.get("candidate_signals") or [])
        if isinstance(s, dict)
    ]


def _compile_evidence(explore_report: dict[str, Any], story: dict[str, Any]) -> list[Evidence]:
    evidence: list[Evidence] = []
    for idx, signal in enumerate(_candidate_signals(explore_report), start=1):
        name = str(signal.get("name") or f"signal_{idx}")
        evidence.append(Evidence(
            id=f"signal:{name}",
            kind="confirmed_signal",
            title=str(signal.get("display_name") or display_name(name)),
            summary=_signal_summary(signal),
            artifact=signal,
        ))
    for idx, chart in enumerate(explore_report.get("charts") or [], start=1):
        if not isinstance(chart, dict):
            continue
        raw_title = str(chart.get("display_name") or chart.get("title") or chart.get("name") or f"Chart {idx}")
        title = display_name(raw_title)
        evidence.append(Evidence(
            id=f"chart:{_slug(title)}",
            kind="chart",
            title=title,
            summary=str(chart.get("description") or chart.get("kind") or ""),
            artifact=chart,
        ))
    for idx, analysis in enumerate(story.get("analyses") or [], start=1):
        summary = str(analysis.get("conclusion") or analysis.get("narrative") or "")
        evidence.append(Evidence(
            id=f"analysis:{idx}",
            kind="m2_analysis",
            title=f"M2 analysis cycle {analysis.get('cycle', idx)}",
            summary=summary,
            artifact=analysis,
        ))
    return evidence


def _compile_claims(
    explore_report: dict[str, Any],
    story: dict[str, Any],
    signals: list[dict[str, Any]],
) -> list[Claim]:
    explicit = [
        _claim_from_agent(c, i)
        for i, c in enumerate(explore_report.get("claims") or [], start=1)
        if isinstance(c, dict)
    ]
    if explicit:
        return explicit

    claims: list[Claim] = []
    for idx, signal in enumerate(signals, start=1):
        name = str(signal.get("name") or f"signal_{idx}")
        status = _signal_status(signal)
        downstream = _downstream_for_signal(name, story)
        claims.append(Claim(
            id=f"C{idx}",
            text=_claim_text(signal),
            status=status,
            evidence_ids=[f"signal:{name}", *_chart_evidence_for_signal(name, explore_report)],
            interpretation=_signal_interpretation(signal),
            do_not_infer=_do_not_infer(signal),
            downstream=downstream,
        ))
    if not claims:
        claims.append(Claim(
            id="C0",
            text="No candidate signal was confirmed in the loaded report.",
            status="inconclusive",
            interpretation="The run may still contain exploratory observations, but no supported "
            "claim can be made from the available confirmation layer.",
            do_not_infer="Do not treat exploratory plots as confirmed root causes.",
        ))
    return _sort_claims(claims)


def _claim_from_agent(raw: dict[str, Any], idx: int) -> Claim:
    status = str(raw.get("status") or "descriptive").lower()
    if status not in {"supported", "inconclusive", "refuted", "descriptive"}:
        status = "descriptive"
    return Claim(
        id=str(raw.get("id") or f"C{idx}"),
        text=str(raw.get("text") or raw.get("claim") or ""),
        status=status,  # type: ignore[arg-type]
        evidence_ids=[str(x) for x in (raw.get("evidence_ids") or [])],
        counter_evidence_ids=[str(x) for x in (raw.get("counter_evidence_ids") or [])],
        interpretation=str(raw.get("interpretation") or ""),
        do_not_infer=str(raw.get("do_not_infer") or ""),
        downstream=[str(x) for x in (raw.get("downstream") or [])],
    )


def _compile_timeline(story: dict[str, Any], explore_report: dict[str, Any]) -> list[ReportStep]:
    steps = [
        ReportStep(
            stage="Explore",
            title="Agent explored patterns and proposed visuals/signals",
            summary=f"{len(explore_report.get('observations') or [])} observation(s), "
            f"{len(explore_report.get('charts') or [])} chart(s), "
            f"{len(explore_report.get('candidate_signals') or [])} signal(s).",
        )
    ]
    analyses = story.get("analyses") or []
    if analyses:
        steps.append(ReportStep(
            stage="M2",
            title="Host confirmed signal associations",
            summary=str(analyses[-1].get("conclusion") or analyses[-1].get("narrative") or ""),
            artifact_ids=[f"analysis:{len(analyses)}"],
        ))
    diagnoses = story.get("diagnoses") or []
    if diagnoses:
        n_h = sum(len(d.get("hypotheses") or []) for d in diagnoses)
        steps.append(ReportStep(
            stage="M3",
            title="Agent formed falsifiable hypotheses",
            summary=f"{n_h} hypothesis/hypotheses proposed from confirmed and exploratory context.",
        ))
    surgeries = story.get("surgeries") or []
    if surgeries:
        steps.append(ReportStep(
            stage="M5/M4",
            title="Hypotheses were tested by interventions",
            summary=f"{len(surgeries)} test/intervention event(s) recorded.",
        ))
    fixes = story.get("fixes") or []
    if fixes:
        steps.append(ReportStep(
            stage="Fix",
            title="Fix candidates were adjudicated",
            summary=f"{len(fixes)} fix event(s) recorded.",
        ))
    return steps


def _answer(explore_report: dict[str, Any], claims: list[Claim]) -> str:
    supported = [c.text for c in claims if c.status == "supported"]
    if supported:
        leaky = sum(1 for c in claims if "sanity check" in c.do_not_infer.lower())
        suffix = f" ({leaky} {_plural(leaky, 'sanity check')} demoted.)" if leaky else ""
        return supported[0] + suffix
    conclusion = str(explore_report.get("conclusion") or "").strip()
    if conclusion:
        return conclusion
    return "No supported diagnostic claim is available in the loaded report."


def _confidence(claims: list[Claim]) -> str:
    if any(c.status == "supported" for c in claims):
        return "medium"
    if any(c.status == "descriptive" for c in claims):
        return "low"
    return "unknown"


def _signal_status(signal: dict[str, Any]) -> str:
    if _is_leaky_signal(signal):
        return "descriptive"
    if signal.get("reject") is True:
        return "supported"
    if signal.get("reject") is False:
        return "inconclusive"
    return "descriptive"


def _signal_summary(signal: dict[str, Any]) -> str:
    parts = []
    if signal.get("effect") is not None:
        parts.append(f"effect={_fmt(signal.get('effect'))}")
    ci = signal.get("ci")
    if isinstance(ci, list | tuple) and len(ci) == 2:
        parts.append(f"CI={_fmt(ci[0])}..{_fmt(ci[1])}")
    if signal.get("e_value") is not None:
        parts.append(f"e={_fmt(signal.get('e_value'))}")
    verdict = "supported" if signal.get("reject") is True else "not supported"
    return f"{verdict}" + (f" ({', '.join(parts)})" if parts else "")


def _claim_text(signal: dict[str, Any]) -> str:
    name = str(signal.get("display_name") or display_name(signal.get("name") or "signal"))
    if _is_leaky_signal(signal):
        return f"{name} tracks the failure label and is treated as descriptive plumbing."
    if signal.get("reject") is True:
        return f"{name} is associated with FAIL cases on the confirmation split."
    if signal.get("reject") is False:
        return f"{name} did not produce a supported FAIL/PASS association."
    return f"{name} is an exploratory signal without a host-confirmed verdict."


def _signal_interpretation(signal: dict[str, Any]) -> str:
    if _is_leaky_signal(signal):
        return "This is useful for auditing the pipeline, not for explaining the failure."
    if signal.get("reject") is True:
        return "Use this as a confirmed association that can motivate M3 hypotheses."
    if signal.get("reject") is False:
        return "Treat this as a negative or underpowered result unless new data changes it."
    return "Treat this as descriptive only."


def _do_not_infer(signal: dict[str, Any]) -> str:
    if _is_leaky_signal(signal):
        return "Sanity check only: do not rank this as a root cause."
    return "Do not infer causality from association without M5/M4 support."


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or f"{singular}s")


def _downstream_for_signal(name: str, story: dict[str, Any]) -> list[str]:
    out = []
    needle = name.lower()
    for diag in story.get("diagnoses") or []:
        refs = " ".join(str(x) for x in (diag.get("referenced_charts") or [])).lower()
        hyps = diag.get("hypotheses") or []
        if needle in refs or any(needle in str(h).lower() for h in hyps):
            out.append(f"M3 cycle {diag.get('cycle')}: referenced in diagnosis context")
    return out


def _chart_evidence_for_signal(name: str, explore_report: dict[str, Any]) -> list[str]:
    out = []
    lname = name.lower()
    for chart in explore_report.get("charts") or []:
        if not isinstance(chart, dict):
            continue
        title = str(chart.get("title") or chart.get("name") or "")
        blob = " ".join(str(chart.get(k, "")) for k in ("name", "title", "data")).lower()
        if lname in blob or any(part and part in blob for part in lname.split("_")):
            out.append(f"chart:{_slug(title)}")
    return out[:3]


def _critique(explore_report: dict[str, Any], signals: list[dict[str, Any]]) -> list[str]:
    raw = explore_report.get("critique")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    notes = [str(c) for c in (explore_report.get("caveats") or [])]
    if any(_is_leaky_signal(s) for s in signals):
        notes.append("One or more signals are sanity checks and are demoted to descriptive evidence.")
    if not notes:
        notes.append("No explicit critique was recorded; inspect raw artifacts before over-claiming.")
    return notes


def _next_actions(explore_report: dict[str, Any], claims: list[Claim]) -> list[str]:
    actions = [str(x) for x in (explore_report.get("recommended_confirmatory_tests") or [])]
    if any(c.status == "supported" for c in claims):
        actions.append("Inspect the linked M3/M5 outcomes before treating supported signals as causes.")
    if not actions:
        actions.append("Re-run with a larger held-out split or richer probes if no claim is supported.")
    return actions


def _sort_claims(claims: list[Claim]) -> list[Claim]:
    order = {"supported": 0, "inconclusive": 1, "refuted": 2, "descriptive": 3}
    return sorted(claims, key=lambda c: (order.get(c.status, 9), c.id))


def _is_leaky_signal(signal: dict[str, Any]) -> bool:
    name = str(signal.get("name", "")).lower()
    if name.startswith("probe1") or "false_detection" in name:
        return True
    eff = signal.get("effect")
    ci = signal.get("ci") or [None, None]
    try:
        lo, hi = float(ci[0]), float(ci[1])
        return eff is not None and abs(float(eff)) >= 0.999 and (hi - lo) <= 1e-6
    except (TypeError, ValueError, IndexError):
        return False


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return str(value)


def _slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return text or "artifact"
