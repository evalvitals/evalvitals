"""VLDiagnoseLoop: scale-dependent visual perception failure.

This example reproduces the failure mode studied in arXiv 2502.17422:
MLLMs systematically fail on fine-grained visual tasks when the spatial
structure is too small for the visual encoder to resolve, while correctly
answering when the same structure is large.

Real-model cases (448×448 px):
  FAIL  — image with N thin coloured bands (3 px each) on a grey canvas.
           At 3 px per band, each band occupies ~0.2 ViT patches (14 px);
           patch averaging makes them invisible — the image looks uniform grey.
  PASS  — same N bands but 50 px each — each band spans multiple patches,
           clearly distinguishable as alternating coloured stripes.
  Question: "How many coloured bands are in this image?" Expected: str(N).

Smoke-test cases (lightweight, no GPU):
  FAIL  — tiny colored square (4 px) / tiny dots (r=4 px)
  PASS  — large colored square (190 px) / large circles (r=55 px)
  (Smoke VLM uses pixel inspection, so color cases work for wiring tests.)

The agent is given no information about the root cause or any proposed fix.
It must discover on its own why the model fails and how it can be corrected.

Usage (via Docker — preferred):
    docker compose up

Usage (direct):
    python run.py
    python run.py --smoke-test     # fast pipeline wiring check, no GPU
    python run.py --model qwen3-vl-2b-instruct --device cuda
    python run.py --max-cycles 3 --max-analyzers 3
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

_OUTPUTS_DIR = Path(__file__).parent / "outputs"
_CANVAS = 448


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------

def _color_sq(size: int, fill: tuple, corner: str = "upper-left"):  # -> PIL.Image
    """448×448 neutral-grey canvas with a single solid-color square."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (_CANVAS, _CANVAS), color=(210, 210, 210))
    if corner == "center":
        x0 = y0 = (_CANVAS - size) // 2
    else:
        x0, y0 = 10, 10
    ImageDraw.Draw(img).rectangle([x0, y0, x0 + size - 1, y0 + size - 1], fill=fill)
    return img


def _dots_img(n: int, r: int):  # -> PIL.Image
    """448×448 cream canvas with n filled dark circles of radius r."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (_CANVAS, _CANVAS), color=(240, 235, 215))
    draw = ImageDraw.Draw(img)
    centres = [
        (80, 80), (224, 80), (368, 80),
        (80, 224), (224, 224), (368, 224),
    ]
    for cx, cy in centres[:n]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 30, 30))
    return img


# ---------------------------------------------------------------------------
# Band-counting image generator (real-model cases)
# ---------------------------------------------------------------------------

_BAND_COLORS = [
    (80,  80,  200),  # blue
    (200, 80,  80),   # red
    (60,  170, 60),   # green
    (210, 150, 30),   # amber
    (160, 50,  200),  # purple
    (40,  180, 180),  # teal
    (200, 200, 50),   # yellow
]

# Tiny: 2 px per band → 0.14 ViT patches each → invisible after patch averaging.
# Large: 60 px per band → ~4 ViT patches each → clearly visible.
_TINY_BAND_PX = 2
_LARGE_BAND_PX = 60

# FAIL counts: high enough that a random guess has ~0% chance of being correct,
# and the sub-patch tiny bands are completely invisible to the model.
_FAIL_COUNTS = [12, 14, 16, 18, 20]
# PASS counts: 1-5 large bands — each ~4 ViT patches wide, easily countable.
# Empirically, Qwen3-VL 4B reliably counts 1-5 stripes but undercounts at 6+.
# Using N=1..5 keeps all PASS cases clean.
_PASS_COUNTS = [1, 2, 3, 4, 5]

_BAND_PROMPT = (
    "How many coloured horizontal stripes can you count in this image? "
    "Answer with a single number only."
)


def _bands_image(n: int, band_px: int):  # -> PIL.Image
    """448×448 grey canvas with n horizontal coloured bands of band_px px each."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (_CANVAS, _CANVAS), color=(210, 210, 210))
    draw = ImageDraw.Draw(img)
    total_h = n * band_px
    y = (_CANVAS - total_h) // 2
    for i in range(n):
        draw.rectangle([0, y, _CANVAS - 1, y + band_px - 1], fill=_BAND_COLORS[i % len(_BAND_COLORS)])
        y += band_px
    return img


# ---------------------------------------------------------------------------
# Scoring helpers (word-boundary safe)
# ---------------------------------------------------------------------------

def _contains(term: str, text: str) -> bool:
    term = term.lower().strip()
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9]+", term):
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


def _score_case(case, observed):
    from evalvitals.core.case import Label

    text = re.sub(r"\s+", " ", str(observed).lower())
    expected = case.expected
    if isinstance(expected, dict):
        if any(_contains(t, text) for t in expected.get("none_of", [])):
            return Label.FAIL
        if not all(_contains(t, text) for t in expected.get("all_of", [])):
            return Label.FAIL
        any_of = expected.get("any_of", [])
        if any_of and not any(_contains(t, text) for t in any_of):
            return Label.FAIL
        return Label.PASS
    if isinstance(expected, str):
        return Label.PASS if _contains(expected, text) else Label.FAIL
    return Label.UNKNOWN


# ---------------------------------------------------------------------------
# Cases — stripe counting (used by real-model run)
# ---------------------------------------------------------------------------

def _build_band_cases():
    """10 stripe-counting cases: 5 tiny-band (FAIL) + 5 large-band (PASS).

    Tiny: 2 px per band — below ViT patch resolution (14 px).  Patch averaging
    makes individual stripes indistinguishable; the image looks like a narrow
    tinted bar.  The model cannot report the correct count.

    Large: 60 px per band — ~4 ViT patches each, clearly countable.
    """
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs

    def _meta(n: int, band_px: int) -> dict:
        scale = "tiny" if band_px <= 4 else "large"
        return {
            "scale_type": scale,
            "question_type": "stripe_counting",
            "n_stripes": n,
            "band_px": band_px,
            "tiny_target": 1 if scale == "tiny" else 0,
            "total_stripe_height_px": n * band_px,
            "total_stripe_height_pct": round(n * band_px / _CANVAS * 100, 1),
        }

    cases = []
    for n in _FAIL_COUNTS:
        cases.append(FailureCase(
            id=f"tiny_stripes_{n}",
            inputs=Inputs(prompt=_BAND_PROMPT, image=_bands_image(n, _TINY_BAND_PX)),
            expected=str(n),
            metadata=_meta(n, _TINY_BAND_PX),
        ))
    for n in _PASS_COUNTS:
        cases.append(FailureCase(
            id=f"large_stripes_{n}",
            inputs=Inputs(prompt=_BAND_PROMPT, image=_bands_image(n, _LARGE_BAND_PX)),
            expected=str(n),
            metadata=_meta(n, _LARGE_BAND_PX),
        ))
    return CaseBatch(cases)


# ---------------------------------------------------------------------------
# Cases — color + count (used by smoke test only)
# ---------------------------------------------------------------------------

def _build_smoke_cases():
    """10 color/count cases for the smoke test: pixel-inspectable by SmokeVLM."""
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs

    RED   = (200, 50,  50)
    GREEN = (50,  160, 70)
    BLUE  = (50,  80,  200)

    def _meta(scale: str, qtype: str, size_px: int) -> dict:
        return {
            "scale_type": scale,
            "question_type": qtype,
            "target_size_px": size_px,
            "target_area_pct": round((size_px / _CANVAS) ** 2 * 100, 3),
        }

    _TINY_SZ = 4
    _LARGE_SZ = 190
    _TINY_R = 4
    _LARGE_R = 55

    cases = [
        # ── FAIL: tiny targets ────────────────────────────────────────────
        FailureCase(
            id="tiny_color_red",
            inputs=Inputs(
                prompt="Look carefully. There is a tiny colored square in the upper-left corner. "
                       "What color is it? Answer with one word only.",
                image=_color_sq(_TINY_SZ, RED, corner="upper-left"),
            ),
            expected={"any_of": ["red", "crimson", "scarlet"]},
            metadata=_meta("tiny", "color", _TINY_SZ),
        ),
        FailureCase(
            id="tiny_color_green",
            inputs=Inputs(
                prompt="Look carefully. There is a tiny colored square in the upper-left corner. "
                       "What color is it? Answer with one word only.",
                image=_color_sq(_TINY_SZ, GREEN, corner="upper-left"),
            ),
            expected={"any_of": ["green", "lime", "olive"]},
            metadata=_meta("tiny", "color", _TINY_SZ),
        ),
        FailureCase(
            id="tiny_color_blue",
            inputs=Inputs(
                prompt="Look carefully. There is a tiny colored square in the upper-left corner. "
                       "What color is it? Answer with one word only.",
                image=_color_sq(_TINY_SZ, BLUE, corner="upper-left"),
            ),
            expected={"any_of": ["blue", "navy", "cobalt", "indigo"]},
            metadata=_meta("tiny", "color", _TINY_SZ),
        ),
        FailureCase(
            id="tiny_count_3",
            inputs=Inputs(
                prompt="Count the number of tiny dark dots scattered across this image. "
                       "Answer with a single digit only.",
                image=_dots_img(3, r=_TINY_R),
            ),
            expected={"any_of": ["3", "three"]},
            metadata=_meta("tiny", "count", _TINY_R * 2),
        ),
        FailureCase(
            id="tiny_count_2",
            inputs=Inputs(
                prompt="Count the number of tiny dark dots scattered across this image. "
                       "Answer with a single digit only.",
                image=_dots_img(2, r=_TINY_R),
            ),
            expected={"any_of": ["2", "two"]},
            metadata=_meta("tiny", "count", _TINY_R * 2),
        ),
        # ── PASS: large targets ───────────────────────────────────────────
        FailureCase(
            id="large_color_red",
            inputs=Inputs(
                prompt="Look at this image. There is a large colored square in the center. "
                       "What color is it? Answer with one word only.",
                image=_color_sq(_LARGE_SZ, RED, corner="center"),
            ),
            expected={"any_of": ["red", "crimson", "scarlet"]},
            metadata=_meta("large", "color", _LARGE_SZ),
        ),
        FailureCase(
            id="large_color_green",
            inputs=Inputs(
                prompt="Look at this image. There is a large colored square in the center. "
                       "What color is it? Answer with one word only.",
                image=_color_sq(_LARGE_SZ, GREEN, corner="center"),
            ),
            expected={"any_of": ["green", "lime", "olive"]},
            metadata=_meta("large", "color", _LARGE_SZ),
        ),
        FailureCase(
            id="large_color_blue",
            inputs=Inputs(
                prompt="Look at this image. There is a large colored square in the center. "
                       "What color is it? Answer with one word only.",
                image=_color_sq(_LARGE_SZ, BLUE, corner="center"),
            ),
            expected={"any_of": ["blue", "navy", "cobalt", "indigo"]},
            metadata=_meta("large", "color", _LARGE_SZ),
        ),
        FailureCase(
            id="large_count_3",
            inputs=Inputs(
                prompt="Count the number of large dark circles visible in this image. "
                       "Answer with a single digit only.",
                image=_dots_img(3, r=_LARGE_R),
            ),
            expected={"any_of": ["3", "three"]},
            metadata=_meta("large", "count", _LARGE_R * 2),
        ),
        FailureCase(
            id="large_count_2",
            inputs=Inputs(
                prompt="Count the number of large dark circles visible in this image. "
                       "Answer with a single digit only.",
                image=_dots_img(2, r=_LARGE_R),
            ),
            expected={"any_of": ["2", "two"]},
            metadata=_meta("large", "count", _LARGE_R * 2),
        ),
    ]
    return CaseBatch(cases)


def _build_candidate_cases(for_smoke: bool = False):
    """Return stripe-counting cases for real runs; color/count for smoke test."""
    return _build_smoke_cases() if for_smoke else _build_band_cases()


# ---------------------------------------------------------------------------
# Smoke-test stubs
# ---------------------------------------------------------------------------

class _SmokeVLM:
    """Deterministic VLM stand-in used only during --smoke-test.

    Works on the color/count smoke cases: fails on tiny targets (wrong answer),
    succeeds on large targets by inspecting center pixels and counting dark pixels.
    """

    def __init__(self) -> None:
        from evalvitals.core.capability import Capability

        self.capabilities = frozenset({Capability.GENERATE, Capability.ATTENTION})
        self.modalities = frozenset({"text", "image"})

    def __repr__(self) -> str:
        return "SmokeVLM()"

    @staticmethod
    def _center_color(image) -> str:
        import numpy as np

        arr = np.array(image)
        cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
        r, g, b = int(arr[cy, cx, 0]), int(arr[cy, cx, 1]), int(arr[cy, cx, 2])
        if r > 150 and g < 100 and b < 100:
            return "red"
        if g > 100 and r < 80 and b < 100:
            return "green"
        if b > 150 and r < 80 and g < 100:
            return "blue"
        return "grey"

    @staticmethod
    def _count_circles(image, r: int) -> int:
        import numpy as np

        arr = np.array(image)
        lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        dark = int((lum < 60).sum())
        circle_area = max(1, int(3.14159 * r * r))
        return max(0, round(dark / circle_area))

    def generate(self, inputs, **kwargs) -> str:
        prompt = str(getattr(inputs, "prompt", inputs)).lower()
        image = getattr(inputs, "image", None)

        if "tiny" in prompt:
            if "color" in prompt or "colour" in prompt:
                return "grey"
            if "dots" in prompt:
                return "1"

        if "large" in prompt:
            if ("color" in prompt or "colour" in prompt) and image is not None:
                return self._center_color(image)
            if "circles" in prompt and image is not None:
                return str(self._count_circles(image, r=55))

        return "I cannot determine the answer."

    def forward(self, inputs, capture, spec=None):
        raise NotImplementedError("SmokeVLM only supports generate().")


class _SmokeProbe:
    last_schema = None

    def probe(self, model, data, **kwargs):
        from evalvitals.core.case import Label
        from evalvitals.core.result import Result
        from evalvitals.eval_agent import ProbingSchema

        self.last_schema = ProbingSchema(
            selected_analyzers=["scale_sensitivity"],
            rationale="Smoke probe: scale_type / target_size_px vs failure label.",
            protocol=kwargs.get("protocol"),
        )
        per_case = []
        for case in data:
            meta = getattr(case, "metadata", {}) or {}
            is_tiny = 1 if meta.get("scale_type") == "tiny" else 0
            per_case.append({
                "sample_id": case.id,
                "scale_type": meta.get("scale_type", "unknown"),
                "tiny_target": is_tiny,
                "target_size_px": meta.get("target_size_px", -1),
                "target_area_pct": meta.get("target_area_pct", -1.0),
                "question_type": meta.get("question_type", "unknown"),
                "label": case.label.value if case.label != Label.UNKNOWN else "unknown",
            })
        findings = {
            "mean_tiny_fail_rate": 1.0,
            "mean_large_fail_rate": 0.0,
            "per_case": per_case,
        }
        return {
            "scale_sensitivity": Result(
                analyzer="scale_sensitivity",
                model=repr(model),
                cases=data,
                findings=findings,
            )
        }


class _SmokeDiagnosisAgent:
    def diagnose(self, analysis, prior_cycles=None):
        from evalvitals.eval_agent import DiagnosisResult, Hypothesis

        h = Hypothesis(
            statement=(
                "The model consistently fails to perceive tiny visual targets "
                "that occupy only a small fraction of the image (~0.07-0.3% of "
                "area), while answering the same question type correctly when the "
                "target is large (~18-40% of area). The failure is caused by "
                "insufficient visual token coverage for small objects."
            ),
            target_model=analysis.model_name,
            predicted_failure_mode="small_object_perception",
        )
        return DiagnosisResult(
            model_name=analysis.model_name,
            hypotheses=[h],
            findings_summary={name: r.findings for name, r in analysis.raw_results.items()},
            raw_judge_output=(
                "HYPOTHESIS: Tiny targets fail due to insufficient visual token coverage.\n"
                "FAILURE_MODE: small_object_perception"
            ),
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def _build_protocol():
    from evalvitals.eval_agent import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "A vision-language model is shown a 448×448-pixel image containing "
            "N horizontal coloured stripes on a grey background, and must count "
            "and report N as a single digit. Two conditions are compared: "
            "'tiny' — each stripe is 2 px wide (sub-ViT-patch; patch averaging "
            "makes individual stripes indistinguishable; the model sees only a "
            "narrow tinted bar, not discrete stripes; N ranges from 12 to 20); "
            "'large' — each stripe is 60 px wide (~4 ViT patches; clearly visible "
            "and individually countable; N ranges from 1 to 5). "
            "The model correctly counts stripes when they are large (1-5 bands), "
            "but reports a wrong number when they are tiny (12-20 invisible bands) "
            "— even though the underlying image structure is the same. "
            "We want to understand what visual processing bottleneck causes this "
            "systematic scale-dependent failure, and whether an inference-time "
            "intervention (without retraining) can restore correct counting of "
            "fine-grained visual structures."
        ),
        task_domain="fine-grained visual perception / stripe counting",
        success_criteria=(
            "Response must contain the correct integer N as a recognisable number "
            "(e.g. '6'). Any surrounding text is acceptable."
        ),
        failure_patterns=(
            "Tiny-stripe cases: model reports a wrong count, says '1', '0', "
            "'I cannot determine', or any number other than the true N. "
            "Large-stripe cases: model reports the correct count N."
        ),
        target_modalities=frozenset({"text", "image"}),
    )


# ---------------------------------------------------------------------------
# Smoke-test runner
# ---------------------------------------------------------------------------

def _run_smoke_test(args) -> None:
    from evalvitals.eval_agent import (
        CaseDiscoveryAgent,
        HypothesisTester,
        RunLogger,
        StatsAnalysisAgent,
        StatsToolAgent,
        SurgeryAgent,
        VLDiagnoseLoop,
    )

    model = _SmokeVLM()
    protocol = _build_protocol()

    candidates = _build_candidate_cases(for_smoke=True)
    discovery = CaseDiscoveryAgent(
        scorer=_score_case,
        include_unknown=False,
    ).discover(model, candidates, protocol=protocol)
    cases = discovery.cases

    print("\nSmoke test data:")
    print(
        f"  discovered {len(cases)} labeled cases "
        f"(PASS={discovery.n_pass}, FAIL={discovery.n_fail}, UNKNOWN={discovery.n_unknown})"
    )
    if not discovery.has_m5_groups:
        raise SystemExit(
            "Smoke test requires both PASS and FAIL cases — "
            "check _SmokeVLM.generate() and case expected values."
        )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir=run_dir / "logs", verbose=True)

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
    print(f"  stopped_by={report.stopped_by}  cycles={report.cycles}")
    print(f"  verified={len(report.verified_hypotheses)}")
    if report.stopped_by != "criteria_met" or not report.verified_hypotheses:
        raise SystemExit("Smoke test failed: no verified hypothesis.")

    fix = loop.run_m4(report, cases)
    if fix is None or fix.status.value != "supported":
        raise SystemExit("Smoke test failed: M4 did not support the verified hypothesis.")

    print("  m4_status=supported")
    print("Smoke test passed.")


# ---------------------------------------------------------------------------
# Artifact writer + output index
# ---------------------------------------------------------------------------

def _write_output_index(run_dir: Path) -> None:
    """Write README.txt and print a file guide at the end of each run."""
    logs = run_dir / "logs"

    sections: list[tuple[str, list[tuple[str, str]]]] = [
        ("Start here", [
            ("summary.md", "plain-text run summary — verified hypotheses, cycles, counts"),
            ("discovery_cases.json", "what the model answered for each case (expected vs. got)"),
        ]),
        ("Diagnosis detail", [
            ("hypotheses.json", "all generated hypotheses with SUPPORTED / REFUTED / UNKNOWN"),
            ("m5_results.json", "statistical test results for each hypothesis"),
            (f"logs/figures/m2_effects.png", "bar chart of effect sizes across analyzers"),
        ]),
        ("Per-cycle analyzer data  (c0 = cycle 0, c1 = cycle 1, …)", [
            ("logs/artifacts/c*_relative_attention_diff_map_fail_minus_pass.png",
             "attention heatmap: where FAIL cases attend differently from PASS cases"),
            ("logs/artifacts/c*_attention_attentions.png",
             "raw last-layer attention weights visualised"),
            ("logs/prompts/c*_m1_selection.*", "which analyzers were chosen and why"),
            ("logs/prompts/c*_m3_diagnosis.*", "the agent's diagnosis reasoning"),
        ]),
        ("M4 experiment (mechanism verification)", [
            ("logs/experiments/post_m4_experiment.py", "script the agent wrote to verify the hypothesis"),
            ("logs/experiments/post_m4_stdout.txt",    "metric_a, metric_b, verdict from running that script"),
            ("logs/experiments/post_m4_agent_thinking.txt", "agent's reasoning while writing the script"),
            ("logs/workspace/post_m4/hypothesis.md",   "the hypothesis the experiment was designed to test"),
        ]),
        ("Raw event log (for debugging)", [
            ("logs/run_log.jsonl", "one JSON line per M1/M2/M3/M5 event"),
        ]),
    ]

    readme_lines = ["outputs/  — file guide\n"]
    for heading, entries in sections:
        readme_lines.append(f"{heading}\n{'─' * len(heading)}")
        for fname, desc in entries:
            readme_lines.append(f"  {fname}")
            readme_lines.append(f"      {desc}")
        readme_lines.append("")

    (run_dir / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")

    print("\n── OUTPUT FILES ─────────────────────────────────────────────")
    for heading, entries in sections:
        print(f"\n  {heading}")
        for fname, desc in entries:
            full = run_dir / fname.replace("*", "c0")
            marker = "  " if full.exists() or "*" in fname else "  (not written)"
            print(f"{marker}    {fname}")
            print(f"          {desc}")
    print(f"\n  Full guide → {run_dir / 'README.txt'}")


def _write_report_artifacts(run_dir: Path, report, cases) -> None:
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
        json.dumps(hypotheses, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "m5_results.json").write_text(
        json.dumps(m5_results, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "# mllms_small_object Run Summary",
        "",
        f"- stopped_by: {report.stopped_by}",
        f"- cycles: {report.cycles}",
        f"- cases: {len(cases)}",
        f"- hypotheses: {len(hypotheses)}",
        f"- verified: {summary['n_verified']}",
        "",
        "## Hypotheses",
    ]
    for h in hypotheses or [{"status": None, "failure_mode": "none", "statement": "none"}]:
        lines.append(f"- [{h['status']}] {h['failure_mode']}: {h['statement']}")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _bar(title: str, width: int = 64) -> None:
    pad = width - len(title) - 4
    print(f"\n── {title} {'─' * max(pad, 2)}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="VLDiagnoseLoop — scale-dependent visual perception failure"
    )
    parser.add_argument("--model", default="qwen3-vl-4b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--judge-model", default="Gemini 3.1 Pro (Low)",
        help="agy model for M1–M5 judge.",
    )
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--max-analyzers", type=int, default=3)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Fast local wiring test — no GPU, no model weights, no agy.",
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
        ExperimentWriterConfig,
        HypothesisTester,
        ProbeAgent,
        RunLogger,
        StatsAnalysisAgent,
        SurgeryAgent,
        VLDiagnoseLoop,
    )

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading {args.model!r} …")
    model = evalvitals.load(
        args.model,
        backend="hf_local",
        device=args.device,
        dtype=args.dtype,
        want=["attention"],
    )

    # ── Judge ─────────────────────────────────────────────────────────────
    try:
        judge = AgyModel(model=args.judge_model)
        judge_desc = f"{args.judge_model}  (antigravity)"
    except RuntimeError as _agy_err:
        import warnings as _w
        _w.warn(
            f"agy not available ({_agy_err}). Falling back to loaded model as judge.",
            stacklevel=2,
        )
        judge = model
        judge_desc = f"{args.model}  (fallback — agy unavailable)"

    protocol = _build_protocol()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"  model   {args.model}  [{args.device} · {args.dtype}]")
    print(f"  judge   {judge_desc}")
    print(f"  output  {run_dir.resolve()}")

    # ── Cases ─────────────────────────────────────────────────────────────
    candidates = _build_candidate_cases(for_smoke=False)
    discovery = CaseDiscoveryAgent(
        scorer=_score_case,
        include_unknown=False,
    ).discover(model, candidates)
    cases = discovery.cases

    discovery_rows = []
    for case in cases:
        observed = str(case.observed)
        discovery_rows.append({
            "id": case.id,
            "prompt": case.inputs.prompt,
            "expected": case.expected,
            "observed": observed,
            "label": case.label.value,
            "metadata": getattr(case, "metadata", {}),
        })
    (run_dir / "discovery_cases.json").write_text(
        json.dumps(discovery_rows, indent=2, default=str), encoding="utf-8"
    )

    _bar(f"CASES  {len(cases)} total · {discovery.n_fail} fail · {discovery.n_pass} pass")
    id_w = max((len(c.id) for c in cases), default=20)
    for case in cases:
        label = "FAIL" if case.label.value == "fail" else "pass"
        obs = textwrap.shorten(str(case.observed), width=60, placeholder="…")
        exp = str(case.expected)
        print(f"  {label}  {case.id:<{id_w}}  expected {exp!r:<6}  got {obs!r}")

    if not discovery.has_m5_groups:
        print(
            "\n  WARNING: need both FAIL and PASS cases for diagnosis. "
            "If tiny-band cases are passing, try --model qwen3-vl-2b-instruct."
        )

    # ── Agents ────────────────────────────────────────────────────────────
    probe_agent = ProbeAgent(judge=judge, max_analyzers=args.max_analyzers)

    stats_agent = StatsAnalysisAgent(
        judge=judge,
        figure_dir=str(Path(args.run_dir) / "logs" / "figures"),
    )

    diagnosis_agent = DiagnosisAgent(judge=judge)

    hypothesis_tester = HypothesisTester(judge=judge, min_effect=0.05)

    writer_cfg = ExperimentWriterConfig(
        cli_agent=CliAgentConfig(
            provider="antigravity", timeout_sec=300, model=args.judge_model
        ),
        exec_fix_timeout_sec=90,
    )
    surgery_agent = SurgeryAgent(judge=judge, writer_config=writer_cfg)

    logger = RunLogger(run_dir=run_dir / "logs", verbose=True)

    loop = VLDiagnoseLoop(
        model=model,
        protocol=protocol,
        probe_agent=probe_agent,
        stats_agent=stats_agent,
        diagnosis_agent=diagnosis_agent,
        hypothesis_tester=hypothesis_tester,
        surgery_agent=surgery_agent,
        max_cycles=args.max_cycles,
        run_logger=logger,
    )

    _bar(f"RUNNING  max {args.max_cycles} cycles · {args.max_analyzers} analyzers")

    report = loop.run(cases)
    _write_report_artifacts(run_dir, report, cases)

    # ── Diagnosis summary ─────────────────────────────────────────────────
    n_verified = len(report.verified_hypotheses)
    _bar(f"DIAGNOSIS  {report.cycles} cycle(s) · stopped: {report.stopped_by}")
    print(f"  {len(report.all_hypotheses)} hypothesis/es generated · {n_verified} verified")
    for vr in report.verified_hypotheses:
        stmt = textwrap.shorten(vr.hypothesis.statement, width=90, placeholder="…")
        print(f"\n  ✓ \"{stmt}\"")
        print(f"    effect {vr.effect_size}  confidence {vr.confidence:.2f}  "
              f"{'consistent with protocol' if vr.is_consistent_with_protocol else 'inconsistent with protocol'}")
        if vr.verdict:
            print(f"    {textwrap.shorten(str(vr.verdict), width=90, placeholder='…')}")

    # ── M4: mechanism verification experiment ─────────────────────────────
    _bar("EXPERIMENT  (M4 · mechanism verification)")
    if report.verified_hypotheses:
        fix = loop.run_m4(report, cases)
        if fix is not None:
            verdict = fix.status.value.upper()
            print(f"  verdict   {verdict}")
            for k, v in list((fix.evidence or {}).items())[:6]:
                print(f"  {k:<22}  {v}")
        else:
            print("  no result returned")
    else:
        print("  skipped — no verified hypotheses")

    # ── Fix: tiered repair applied to failure cases ───────────────────────
    _bar("FIX  (tiered repair · up to L3a)")
    if report.verified_hypotheses:
        fix_outcome = loop.run_fix(report, cases, max_tier="L3a")
        if fix_outcome is not None:
            n_fail_cases = sum(1 for c in cases if c.label.value == "fail")
            best = fix_outcome.best
            if fix_outcome.fixed and best is not None:
                cand = getattr(best, "candidate", None)
                name = getattr(cand, "name", "?")
                tier = getattr(cand, "tier", "?")
                print(f"  {best.n_fixed} of {n_fail_cases} failure cases fixed · {best.n_broken} regression(s)")
                print(f"  method  {name}  [{tier}]")
                print(f"  effect  {best.effect}")
            else:
                n_tried = len(fix_outcome.attempted or [])
                print(f"  no fix found  ({n_tried} approach(es) tried)")
            rec = fix_outcome.recommendation
            if rec:
                print(f"  recommendation  {rec}")
        else:
            print("  no result returned")
    else:
        print("  skipped — no verified hypotheses")

    _write_output_index(run_dir)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
