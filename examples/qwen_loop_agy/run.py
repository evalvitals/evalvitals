"""VLDiagnoseLoop on Qwen3-VL-4B: protocol-guided VL failure diagnosis.

Pipeline:

    ExperimentProtocol  ← user's NL description of what to investigate
         │
    M1  ProbeAgent           protocol-guided analyzer selection + execute
    M2  StatsAnalysisAgent   select stats tools (evalvitals.stats) + run +
                             e-BH FDR-correct + LLM-written evidence chain
    M3  DiagnosisAgent       Qwen as judge ("AI scientist" hypothesis gen)
    M5  HypothesisTester     statistical test + protocol consistency check
         │
    loop exits when M5 finds a verified, protocol-consistent hypothesis,
    or after --max-cycles cycles
         │
    M4  SurgeryAgent         agy/codex writes + runs targeted fix script
                             (called separately AFTER the loop)

Outputs written to --run-dir (default: ./outputs/):
    logs/run_log.jsonl          ← one JSON line per M1/M2/M3/M5 event
    logs/artifacts/             ← per-cycle analyzer artifacts (.npy / .json)

Usage (via Docker — preferred):
    docker compose up

Usage (direct):
    python run.py
    python run.py --smoke-test     # fast local wiring test, no Qwen/GPU/agy
    python run.py --model qwen2.5-vl-7b-instruct --device cuda:0
    python run.py --analysis-only   # M1+M2 only, skip M3/M5/M4
    python run.py --max-cycles 3 --max-analyzers 3
"""

from __future__ import annotations

import argparse
import io
import json
import textwrap
import urllib.request
from pathlib import Path

# A stable, public sample photo (real objects/colours/layout for the VQA task).
# NOTE: the old Wikimedia thumbnail URL stopped working — it now 403s on the
# default urllib User-Agent, 400s on disallowed thumbnail sizes, and the file
# itself was removed (404). This raw GitHub asset needs no special headers.
_SAMPLE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
_OUTPUTS_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# Verbose logger
# ---------------------------------------------------------------------------

class VerboseRunLogger:
    """Mirror each loop event to stdout as it happens."""

    def __init__(self, run_dir: Path) -> None:
        from evalvitals.eval_agent import RunLogger
        self._rl = RunLogger(run_dir=run_dir)

    def __getattr__(self, name):
        return getattr(self._rl, name)

    def log_probe(self, cycle: int, results: dict, schema=None) -> None:
        print(f"\n[M1] cycle={cycle}  analyzers={list(results.keys())}", flush=True)
        if schema is not None and getattr(schema, "rationale", ""):
            print(f"     rationale  : {schema.rationale}", flush=True)
        for name, r in results.items():
            scalars = {
                k: round(v, 4)
                for k, v in (getattr(r, "findings", {}) or {}).items()
                if isinstance(v, (int, float))
            }
            print(f"     {name}: {dict(list(scalars.items())[:6])}", flush=True)
        self._rl.log_probe(cycle, results, schema=schema)

    def log_analysis(self, cycle: int, analysis) -> None:
        print(f"\n[M2] cycle={cycle}  severity={analysis.severity}", flush=True)
        # StatsAnalysisReport: show conclusion + first two evidence-chain steps
        conclusion = getattr(analysis, "conclusion", None)
        if conclusion:
            print(f"     conclusion : {textwrap.fill(conclusion, 72, subsequent_indent='     ')}",
                  flush=True)
        chain = getattr(analysis, "evidence_chain", [])
        for step in chain[:3]:
            print(f"     evidence   : {step}", flush=True)
        # Stats-tool layer: show which tools ran and the FDR survivors.
        plan = getattr(analysis, "stats_plan", []) or []
        if plan:
            print(f"     stats_tools: {[p['tool'] for p in plan]}", flush=True)
        corrected = getattr(analysis, "corrected_rejections", {}) or {}
        if corrected.get("rejected_tools"):
            print(f"     fdr_survive: {corrected['rejected_tools']}", flush=True)
        tool_results = getattr(analysis, "stats_tool_results", [])
        for tool in tool_results[:2]:
            print(
                f"     stats_tool : {tool.get('name')} - {tool.get('conclusion', '')}",
                flush=True,
            )
        for fig in getattr(analysis, "figures", []) or []:
            print(f"     figure     : {fig}", flush=True)
        if not conclusion:
            print(f"     {textwrap.fill(analysis.narrative, 72, subsequent_indent='     ')}",
                  flush=True)
        self._rl.log_analysis(cycle, analysis)

    def log_diagnosis(self, cycle: int, diag) -> None:
        print(f"\n[M3] cycle={cycle}  {len(diag.hypotheses)} hypothesis/es", flush=True)
        for h in diag.hypotheses:
            print(f"     hypothesis  : {h.statement}", flush=True)
            print(f"     failure_mode: {h.predicted_failure_mode}", flush=True)
        self._rl.log_diagnosis(cycle, diag)

    def log_surgery(self, cycle: int, hypothesis, intervention) -> None:
        # VLDiagnoseLoop fires this for M5 results via _make_intervention_result_from_test.
        # Detect M5 vs M4 by presence of the m5_* evidence keys.
        ev = getattr(intervention, "evidence", {}) or {}
        is_m5 = "m5_test_name" in ev
        tag = "M5" if is_m5 else "M4"
        status = getattr(getattr(intervention, "status", None), "value", "?")
        print(f"\n[{tag}] cycle={cycle}  '{hypothesis.statement[:70]}'", flush=True)

        if is_m5:
            print(
                f"     status={status}"
                f"  effect={ev.get('m5_effect_size', '?')}"
                f"  confidence={ev.get('m5_confidence', '?')}",
                flush=True,
            )
            print(
                f"     protocol_consistent={ev.get('m5_protocol_consistent', '?')}",
                flush=True,
            )
            print(f"     verdict : {ev.get('m5_verdict', '')}", flush=True)
        else:
            print(f"     status={status}  fixed={intervention.fixed}", flush=True)
            if ev:
                print(f"     evidence: {dict(list(ev.items())[:4])}", flush=True)

        self._rl.log_surgery(cycle, hypothesis, intervention)

    def log_loop_end(self, report) -> None:
        # Supports both AutoDiagnoseReport (resolved=) and VLDiagnoseReport (stopped_by=).
        resolved = getattr(report, "resolved", None)
        stopped_by = getattr(report, "stopped_by", None)
        if stopped_by is not None:
            print(f"\n[DONE] cycles={report.cycles}  stopped_by={stopped_by}", flush=True)
        else:
            print(f"\n[DONE] cycles={report.cycles}  resolved={resolved}", flush=True)
        self._rl.log_loop_end(report)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _get_image(*, download: bool = False):
    from PIL import Image
    if not download:
        return _synthetic_image()
    try:
        # Wikimedia rejects the default urllib User-Agent with HTTP 403,
        # so send a descriptive UA as their policy requires.
        req = urllib.request.Request(
            _SAMPLE_URL,
            headers={"User-Agent": "evalvitals-example/1.0 (https://example.com; contact@example.com)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        print(f"  Downloaded image: {img.size} px")
        return img
    except Exception as exc:
        print(f"  Could not download image ({exc}); generating synthetic fallback")
        return _synthetic_image()


def _synthetic_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (224, 224), color=(200, 220, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 100, 100], fill=(220, 80, 60))   # red box left
    draw.rectangle([124, 20, 204, 100], fill=(60, 160, 80))  # green box right
    draw.rectangle([60, 130, 164, 190], fill=(80, 80, 200))  # blue box centre-bottom
    try:
        draw.text((10, 200), "synthetic test image", fill=(0, 0, 0))
    except Exception:
        pass
    return img


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def _build_candidate_cases(image):
    """Human-prior candidate prompts; labels are assigned after model execution."""
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs

    return CaseBatch([
        FailureCase(
            id="q_count",
            inputs=Inputs(
                prompt="How many colored rectangles are in this image? Answer with the number.",
                image=image,
            ),
            expected={"any_of": ["3", "three"]},
        ),
        FailureCase(
            id="q_colour",
            inputs=Inputs(
                prompt="What are the dominant rectangle colors in this image?",
                image=image,
            ),
            expected={"all_of": ["red", "green", "blue"]},
        ),
        FailureCase(
            id="q_location",
            inputs=Inputs(
                prompt="Describe the spatial layout of the colored rectangles.",
                image=image,
            ),
            expected={"all_of": ["left", "right"], "any_of": ["bottom", "below", "lower"]},
        ),
        FailureCase(
            id="q_weather",
            inputs=Inputs(
                prompt="Does this image show a snowy mountain landscape? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["no"], "none_of": ["yes"]},
            metadata={"pope_label": "no", "case_family": "negative_absence"},
        ),
        FailureCase(
            id="q_text",
            inputs=Inputs(
                prompt="Is the phrase 'synthetic test image' visible? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
            metadata={"pope_label": "yes", "case_family": "ocr_presence"},
        ),
        FailureCase(
            id="q_bottom_colour",
            inputs=Inputs(
                prompt="Which colored rectangle is lowest in the image? Answer with one color.",
                image=image,
            ),
            expected={"all_of": ["blue"]},
        ),
        FailureCase(
            id="q_absent_yellow",
            inputs=Inputs(
                prompt="Is there a yellow rectangle in the image? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["no"], "none_of": ["yes"]},
            metadata={"pope_label": "no", "case_family": "negative_absence"},
        ),
        FailureCase(
            id="q_text_word",
            inputs=Inputs(
                prompt="What word appears immediately after 'synthetic' in the visible text?",
                image=image,
            ),
            expected={"all_of": ["test"]},
        ),
        FailureCase(
            id="q_top_left_colour",
            inputs=Inputs(
                prompt="What color is the top-left rectangle? Answer with one color.",
                image=image,
            ),
            expected={"all_of": ["red"], "none_of": ["green", "blue", "purple"]},
            metadata={"case_family": "spatial_colour"},
        ),
        FailureCase(
            id="q_top_right_colour",
            inputs=Inputs(
                prompt="What color is the top-right rectangle? Answer with one color.",
                image=image,
            ),
            expected={"all_of": ["green"], "none_of": ["red", "blue", "purple"]},
            metadata={"case_family": "spatial_colour"},
        ),
        FailureCase(
            id="q_top_row_count",
            inputs=Inputs(
                prompt="How many rectangles are in the top row? Answer with the number.",
                image=image,
            ),
            expected={"any_of": ["2", "two"]},
            metadata={"case_family": "spatial_counting"},
        ),
        FailureCase(
            id="q_bottom_row_count",
            inputs=Inputs(
                prompt="How many rectangles are in the bottom row? Answer with the number.",
                image=image,
            ),
            expected={"any_of": ["1", "one"]},
            metadata={"case_family": "spatial_counting"},
        ),
        FailureCase(
            id="q_blue_position",
            inputs=Inputs(
                prompt="Where is the blue rectangle relative to the red and green rectangles?",
                image=image,
            ),
            expected={
                "all_of": ["below"],
                "any_of": ["red", "green", "rectangles"],
                "none_of": ["above"],
            },
            metadata={"case_family": "spatial_relation"},
        ),
        FailureCase(
            id="q_red_left_of_green",
            inputs=Inputs(
                prompt="Is the red rectangle left of the green rectangle? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
            metadata={"pope_label": "yes", "case_family": "spatial_relation"},
        ),
        FailureCase(
            id="q_green_left_of_red",
            inputs=Inputs(
                prompt="Is the green rectangle left of the red rectangle? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["no"], "none_of": ["yes"]},
            metadata={"pope_label": "no", "case_family": "spatial_relation"},
        ),
        FailureCase(
            id="q_has_circle",
            inputs=Inputs(
                prompt="Is there a circle in the image? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["no"], "none_of": ["yes"]},
            metadata={"pope_label": "no", "case_family": "negative_absence"},
        ),
        FailureCase(
            id="q_has_black_background",
            inputs=Inputs(
                prompt="Does the image have a black background? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["no"], "none_of": ["yes"]},
            metadata={"pope_label": "no", "case_family": "negative_absence"},
        ),
        FailureCase(
            id="q_exact_visible_phrase",
            inputs=Inputs(
                prompt="What exact phrase is written at the bottom of the image?",
                image=image,
            ),
            expected={"all_of": ["synthetic", "test", "image"]},
            metadata={"case_family": "ocr_exact"},
        ),
        FailureCase(
            id="q_visible_text_word_count",
            inputs=Inputs(
                prompt="How many words are in the visible phrase at the bottom? Answer with the number.",
                image=image,
            ),
            expected={"any_of": ["3", "three"]},
            metadata={"case_family": "ocr_counting"},
        ),
        FailureCase(
            id="q_largest_rectangle_colour",
            inputs=Inputs(
                prompt="What color is the largest rectangle? Answer with one color.",
                image=image,
            ),
            expected={"all_of": ["blue"], "none_of": ["red", "green", "purple"]},
            metadata={"case_family": "size_colour"},
        ),
        FailureCase(
            id="q_all_colours_ordered",
            inputs=Inputs(
                prompt=(
                    "List the rectangle colors in reading order: top-left, top-right, "
                    "then bottom. Answer with only the three color names."
                ),
                image=image,
            ),
            expected={
                "all_of": ["red", "green", "blue"],
                "none_of": ["purple", "yellow"],
            },
            metadata={"case_family": "ordered_colour"},
        ),
        FailureCase(
            id="q_has_purple_rectangle",
            inputs=Inputs(
                prompt="Is there a purple rectangle in the image? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["no"], "none_of": ["yes"]},
            metadata={"pope_label": "no", "case_family": "colour_absence"},
        ),
        FailureCase(
            id="q_lowest_is_purple",
            inputs=Inputs(
                prompt="Is the lowest rectangle purple? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["no"], "none_of": ["yes"]},
            metadata={"pope_label": "no", "case_family": "colour_absence"},
        ),
        FailureCase(
            id="q_bottom_not_purple_colour",
            inputs=Inputs(
                prompt="The bottom rectangle is not purple. What color is it?",
                image=image,
            ),
            expected={"all_of": ["blue"], "none_of": ["purple"]},
            metadata={"case_family": "contrastive_colour"},
        ),
        FailureCase(
            id="q_blue_vs_purple_choice",
            inputs=Inputs(
                prompt="Is the bottom rectangle blue or purple? Answer with one word.",
                image=image,
            ),
            expected={"all_of": ["blue"], "none_of": ["purple"]},
            metadata={"case_family": "forced_choice_colour"},
        ),
    ])


# ---------------------------------------------------------------------------
# Smoke-test path
# ---------------------------------------------------------------------------

class _SmokeVLM:
    """Tiny deterministic VLM stand-in for testing this example's wiring."""

    def __init__(self) -> None:
        from evalvitals.core.capability import Capability

        self.capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
        self.modalities = frozenset({"text", "image"})

    def __repr__(self) -> str:
        return "SmokeVLM()"

    def generate(self, inputs, **kwargs) -> str:
        prompt = str(getattr(inputs, "prompt", inputs)).lower()
        if "top-left rectangle" in prompt:
            return "red"
        if "top-right rectangle" in prompt:
            return "green"
        if "top row" in prompt:
            return "2"
        if "bottom row" in prompt:
            return "2"
        if "blue rectangle relative" in prompt:
            return "It is above the red and green rectangles."
        if "red rectangle left of the green" in prompt:
            return "yes"
        if "green rectangle left of the red" in prompt:
            return "yes"
        if "circle" in prompt:
            return "no"
        if "black background" in prompt:
            return "no"
        if "exact phrase" in prompt:
            return "synthetic test image"
        if "how many words" in prompt:
            return "2"
        if "largest rectangle" in prompt:
            return "red"
        if "reading order" in prompt:
            return "red green purple"
        if "purple rectangle" in prompt:
            return "yes"
        if "lowest rectangle purple" in prompt:
            return "yes"
        if "bottom rectangle is not purple" in prompt:
            return "purple"
        if "blue or purple" in prompt:
            return "purple"
        if "snowy mountain" in prompt:
            return "No."
        if "phrase" in prompt:
            return "Yes, the phrase is visible."
        if "how many" in prompt:
            return "I see two rectangles."
        if "dominant rectangle colors" in prompt:
            return "Red and green."
        if "spatial layout" in prompt:
            return "They are arranged in a row."
        return "Unknown."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError("SmokeVLM only supports generate().")


class _SmokeProbe:
    """M1 probe that emits per-case signals on discovered failures."""

    last_schema = None

    def probe(self, model, data, **kwargs):
        from evalvitals.core.case import Label
        from evalvitals.core.result import Result
        from evalvitals.eval_agent import ProbingSchema

        fail_ids = [case.id for case in data if case.label == Label.FAIL]
        self.last_schema = ProbingSchema(
            selected_analyzers=["attention"],
            rationale="Smoke probe marks discovered failures as attention signals.",
            protocol=kwargs.get("protocol"),
        )
        findings = {
            "mean_entropy": 0.2,
            "per_case": [
                {"sample_id": cid, "attention_signal": True}
                for cid in fail_ids
            ],
        }
        return {
            "attention": Result(
                analyzer="attention",
                model=repr(model),
                cases=data,
                findings=findings,
            )
        }


class _SmokeDiagnosisAgent:
    """M3 diagnosis stand-in with one protocol-consistent hypothesis."""

    def diagnose(self, analysis, prior_cycles=None):
        from evalvitals.eval_agent import DiagnosisResult, Hypothesis

        h = Hypothesis(
            statement=(
                "The model counts incorrectly when the attention diagnostic "
                "signal appears."
            ),
            target_model=analysis.model_name,
            predicted_failure_mode="attention",
        )
        return DiagnosisResult(
            model_name=analysis.model_name,
            hypotheses=[h],
            findings_summary={name: r.findings for name, r in analysis.raw_results.items()},
            raw_judge_output=(
                "HYPOTHESIS: The model counts incorrectly when the attention "
                "diagnostic signal appears.\nFAILURE_MODE: attention"
            ),
        )


def _build_protocol():
    from evalvitals.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "We evaluate a vision-language model on basic image understanding: "
            "counting objects, naming their colours, and describing their spatial "
            "arrangement. The model frequently gives wrong answers — it counts "
            "incorrectly, misnames colours, and confuses left/right positions. "
            "We want to know whether the model is actually using the image or "
            "just guessing from language patterns."
        ),
        task_domain="visual question answering",
        success_criteria=(
            "Object counts, colour names, and position descriptions must match "
            "what is visible in the image."
        ),
        target_modalities=frozenset({"text", "image"}),
    )


def _run_smoke_test(args) -> None:
    from evalvitals.eval_agent import (
        CaseDiscoveryAgent,
        HypothesisTester,
        StatsAnalysisAgent,
        StatsToolAgent,
        SurgeryAgent,
        VLDiagnoseLoop,
    )

    model = _SmokeVLM()
    image = _synthetic_image()
    protocol = _build_protocol()
    discovery = CaseDiscoveryAgent(include_unknown=False).discover(
        model, _build_candidate_cases(image), protocol=protocol
    )
    cases = discovery.cases

    print("\nSmoke test data:")
    print(
        f"  discovered {len(cases)} labeled cases "
        f"(PASS={discovery.n_pass}, FAIL={discovery.n_fail}, UNKNOWN={discovery.n_unknown})"
    )
    if not discovery.has_m5_groups:
        raise SystemExit("Smoke test requires both PASS and FAIL cases.")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = VerboseRunLogger(run_dir=run_dir / "logs")

    loop = VLDiagnoseLoop(
        model=model,
        protocol=protocol,
        probe_agent=_SmokeProbe(),
        stats_agent=StatsAnalysisAgent(stats_tool_agent=StatsToolAgent(max_tools=3)),
        diagnosis_agent=_SmokeDiagnosisAgent(),
        hypothesis_tester=HypothesisTester(min_effect=0.05),
        surgery_agent=SurgeryAgent(),
        max_cycles=1,
        run_logger=logger,
    )
    report = loop.run(cases)
    _write_report_artifacts(run_dir, report, cases)

    print("\nSmoke test result:")
    print(f"  stopped_by={report.stopped_by} cycles={report.cycles}")
    print(f"  verified={len(report.verified_hypotheses)}")
    if report.stopped_by != "criteria_met" or not report.verified_hypotheses:
        raise SystemExit("Smoke test failed: no verified hypothesis.")

    fix = loop.run_m4(report, cases)
    if fix is None or fix.status.value != "supported":
        raise SystemExit("Smoke test failed: M4 did not support the verified hypothesis.")

    print("  m4_status=supported")
    print("Smoke test passed.")


def _write_report_artifacts(run_dir: Path, report, cases) -> None:
    """Write human-readable run artifacts alongside the JSONL event log."""
    hypotheses = [
        {
            "statement": h.statement,
            "failure_mode": h.predicted_failure_mode,
            "status": h.status.value if h.status else None,
        }
        for h in getattr(report, "all_hypotheses", [])
    ]
    m5_results = [
        {
            "hypothesis": tr.hypothesis.statement,
            "failure_mode": tr.hypothesis.predicted_failure_mode,
            "status": tr.status.value,
            "effect_size": tr.effect_size,
            "confidence": tr.confidence,
            "protocol_consistent": tr.is_consistent_with_protocol,
            "verdict": tr.verdict,
            "evidence": tr.evidence,
        }
        for tr in getattr(report, "all_test_results", [])
    ]
    summary = {
        "cycles": report.cycles,
        "stopped_by": report.stopped_by,
        "n_cases": len(cases),
        "n_hypotheses": len(hypotheses),
        "n_m5_results": len(m5_results),
        "n_verified": len(getattr(report, "verified_hypotheses", [])),
    }
    (run_dir / "hypotheses.json").write_text(
        json.dumps(hypotheses, indent=2, default=str),
        encoding="utf-8",
    )
    (run_dir / "m5_results.json").write_text(
        json.dumps(m5_results, indent=2, default=str),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "# qwen_loop_agy Run Summary",
        "",
        f"- stopped_by: {report.stopped_by}",
        f"- cycles: {report.cycles}",
        f"- cases: {len(cases)}",
        f"- hypotheses: {len(hypotheses)}",
        f"- verified: {summary['n_verified']}",
        "",
        "## Hypotheses",
    ]
    if hypotheses:
        for h in hypotheses:
            lines.append(f"- [{h['status']}] {h['failure_mode']}: {h['statement']}")
    else:
        lines.append("- none")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLDiagnoseLoop on Qwen VL + image (new M1→M2→M3→M5+M4 pipeline)"
    )
    parser.add_argument("--model", default="qwen3-vl-4b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--max-analyzers", type=int, default=2)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a fast local wiring test without loading Qwen, GPU, or agy.",
    )
    parser.add_argument(
        "--download-image", action="store_true",
        help="Use the demo Wikimedia image instead of the synthetic labeled image.",
    )
    parser.add_argument(
        "--judge-discovery", action="store_true",
        help="Use the judge model to label discovery outputs against expected rubrics.",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Run M1+M2 only (skip M3/M5/M4)",
    )
    parser.add_argument("--run-dir", default=str(_OUTPUTS_DIR))
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test(args)
        return

    import evalvitals
    from evalvitals.eval_agent import (
        AgyModel,
        CaseDiscoveryAgent,
        CliAgentConfig,
        DiagnosisAgent,
        ExperimentProtocol,
        ExperimentWriterConfig,
        HypothesisTester,
        ProbeAgent,
        StatsAnalysisAgent,
        SurgeryAgent,
        VLDiagnoseLoop,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.model!r} on {args.device} ({args.dtype}) …")
    model = evalvitals.load(
        args.model,
        backend="hf_local",
        device=args.device,
        dtype=args.dtype,
        want=["attention"],
    )
    print(f"  capabilities : {sorted(str(c.name) for c in model.capabilities)}")
    print(f"  modalities   : {sorted(model.modalities)}")

    # ── Judge: agy CLI preferred; falls back to the loaded model ─────────────
    # agy requires no API key — it uses the user's existing OAuth token.
    # If the binary is not properly mounted, the loaded VLM acts as judge
    # (a warning is printed and the rationale will reflect this).
    try:
        judge = AgyModel()
        print(f"\n  judge : antigravity CLI ({judge._binary})  [M1–M5, no API key]")
    except RuntimeError as _agy_err:
        import warnings as _w
        _w.warn(
            f"agy not available ({_agy_err}). "
            "Falling back to the loaded model as judge. "
            "To use agy: export AGY_PATH=$(which agy) before docker compose up.",
            stacklevel=2,
        )
        judge = model
        print(f"\n  judge : {args.model} (agy unavailable — using evaluated model as fallback)")

    protocol = _build_protocol()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Image + cases ─────────────────────────────────────────────────────────
    print("\nPreparing image …")
    image = _get_image(download=args.download_image)
    candidate_cases = _build_candidate_cases(image)
    discovery = CaseDiscoveryAgent(
        judge=judge if args.judge_discovery else None,
        include_unknown=True,
    ).discover(model, candidate_cases, protocol=protocol)
    cases = discovery.cases
    print(
        f"  discovered {len(cases)} labeled cases "
        f"(PASS={discovery.n_pass}, FAIL={discovery.n_fail}, UNKNOWN={discovery.n_unknown})"
    )
    discovery_rows = []
    for case in cases:
        observed = str(case.observed)
        discovery_rows.append({
            "id": case.id,
            "prompt": case.inputs.prompt,
            "expected": case.expected,
            "observed": observed,
            "label": case.label.value,
        })
        print(
            f"    [{case.label.value.upper()}] {case.id}: "
            f"{textwrap.shorten(observed, width=110, placeholder='...')}"
        )
    (run_dir / "discovery_cases.json").write_text(
        json.dumps(discovery_rows, indent=2, default=str),
        encoding="utf-8",
    )
    if discovery.errors:
        print(f"  discovery errors: {len(discovery.errors)}")
    if not discovery.has_m5_groups:
        print(
            "  WARNING: M5 needs both PASS and FAIL cases for fail-rate tests; "
            "this run may stop without verified hypotheses."
        )

    # ── Experiment protocol (the human prior) ─────────────────────────────────
    print("\nExperimentProtocol:")
    print(f"  task_domain : {protocol.task_domain}")
    print(f"  description : {protocol.description[:80]}...")

    # ── M1: ProbeAgent — agy selects analyzers from the protocol ─────────────
    probe_agent = ProbeAgent(judge=judge, max_analyzers=args.max_analyzers)

    # ── M2: StatsAnalysisAgent — selects stats tools + agy writes narrative ──
    # M2 now runs a statistical-tool layer (signal/label association, McNemar +
    # e-value, Friedman, single-rate e-value, rank corr) selected from the
    # catalog, e-BH FDR-corrects across them, and (with figure_dir) saves a
    # forest plot of effect sizes. The judge writes a conclusion grounded in
    # those verdicts. Falls back to threshold rules when cases are unlabeled.
    stats_agent = StatsAnalysisAgent(
        judge=None if args.analysis_only else judge,
        figure_dir=str(Path(args.run_dir) / "logs" / "figures"),
    )

    # ── M3: DiagnosisAgent — agy proposes hypotheses ─────────────────────────
    diagnosis_agent = None
    if not args.analysis_only:
        diagnosis_agent = DiagnosisAgent(judge=judge)

    # ── M5: HypothesisTester — agy checks protocol consistency ───────────────
    hypothesis_tester = None
    if not args.analysis_only:
        hypothesis_tester = HypothesisTester(judge=judge, min_effect=0.05)

    # ── M4: SurgeryAgent — agy writes and runs the fix script ────────────────
    surgery_agent = None
    if not args.analysis_only:
        writer_cfg = ExperimentWriterConfig(
            cli_agent=CliAgentConfig(provider="antigravity", timeout_sec=120),
            exec_fix_timeout_sec=60,
        )
        surgery_agent = SurgeryAgent(judge=judge, writer_config=writer_cfg)

    # ── Run directory + verbose logger ────────────────────────────────────────
    print(f"\nOutput directory: {run_dir.resolve()}")
    print("  logs/run_log.jsonl   ← one JSON line per M1/M2/M3/M5 event")
    print("  logs/artifacts/      ← per-cycle analyzer artifacts (.npy / .json)")

    logger = VerboseRunLogger(run_dir=run_dir / "logs")

    # ── VLDiagnoseLoop (M1→M2→M3→M5) ─────────────────────────────────────────
    loop = VLDiagnoseLoop(
        model=model,
        protocol=protocol,
        probe_agent=probe_agent,
        stats_agent=stats_agent,
        diagnosis_agent=diagnosis_agent,
        hypothesis_tester=hypothesis_tester,
        surgery_agent=surgery_agent,   # stored but NOT called inside run()
        max_cycles=args.max_cycles,
        run_logger=logger,
        analysis_only=args.analysis_only,
    )

    print(f"\n{'='*64}")
    print(f"VLDiagnoseLoop  model={args.model}  max_cycles={args.max_cycles}")
    print(f"{'='*64}")

    report = loop.run(cases)
    _write_report_artifacts(run_dir, report, cases)

    # ── Print verified hypotheses ─────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"LOOP RESULT  stopped_by={report.stopped_by}  cycles={report.cycles}")
    print(f"{'='*64}")
    print(f"  total hypotheses proposed : {len(report.all_hypotheses)}")
    print(f"  verified (protocol-consistent + statistically supported): "
          f"{len(report.verified_hypotheses)}")
    for vr in report.verified_hypotheses:
        print(f"    [{vr.status.value}] {vr.hypothesis.statement}")
        print(f"           effect={vr.effect_size}  confidence={vr.confidence:.2f}"
              f"  protocol_ok={vr.is_consistent_with_protocol}")
        print(f"           {vr.verdict}")

    # ── M4: post-loop fix proposal ────────────────────────────────────────────
    if surgery_agent is not None:
        print(f"\n{'='*64}")
        print("M4  Fix proposal (post-loop)")
        print(f"{'='*64}")
        if report.verified_hypotheses:
            fix = loop.run_m4(report, cases)
            if fix is not None:
                print(f"  hypothesis : {fix.hypothesis.statement}")
                print(f"  status     : {fix.status.value}  fixed={fix.fixed}")
                ev = fix.evidence or {}
                for k, v in list(ev.items())[:6]:
                    print(f"  {k:20s}: {v}")
            else:
                print("  SurgeryAgent returned None (no verified hypotheses to act on)")
        else:
            print("  No verified hypotheses — skipping M4.")

    print("\nDone.")


if __name__ == "__main__":
    main()
