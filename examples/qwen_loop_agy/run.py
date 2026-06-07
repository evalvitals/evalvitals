"""VLDiagnoseLoop on Qwen3-VL-4B: protocol-guided VL failure diagnosis.

Pipeline:

    ExperimentProtocol  ← user's NL description of what to investigate
         │
    M1  ProbeAgent           protocol-guided analyzer selection + execute
    M2  StatsAnalysisAgent   threshold rules + LLM-written evidence chain
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
    python run.py --model qwen2.5-vl-7b-instruct --device cuda:0
    python run.py --analysis-only   # M1+M2 only, skip M3/M5/M4
    python run.py --max-cycles 3 --max-analyzers 3
"""

from __future__ import annotations

import argparse
import io
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
        ),
        FailureCase(
            id="q_text",
            inputs=Inputs(
                prompt="Is the phrase 'synthetic test image' visible? Answer yes or no.",
                image=image,
            ),
            expected={"all_of": ["yes"], "none_of": ["no"]},
        ),
    ])


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
        "--download-image", action="store_true",
        help="Use the demo Wikimedia image instead of the synthetic labeled image.",
    )
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Run M1+M2 only (skip M3/M5/M4)",
    )
    parser.add_argument("--run-dir", default=str(_OUTPUTS_DIR))
    args = parser.parse_args()

    import shutil

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

    # ── Image + cases ─────────────────────────────────────────────────────────
    print("\nPreparing image …")
    image = _get_image(download=args.download_image)
    candidate_cases = _build_candidate_cases(image)
    discovery = CaseDiscoveryAgent(
        include_unknown=False,
    ).discover(model, candidate_cases)
    cases = discovery.cases
    print(
        f"  discovered {len(cases)} labeled cases "
        f"(PASS={discovery.n_pass}, FAIL={discovery.n_fail}, UNKNOWN={discovery.n_unknown})"
    )
    if discovery.errors:
        print(f"  discovery errors: {len(discovery.errors)}")
    if not discovery.has_m5_groups:
        print(
            "  WARNING: M5 needs both PASS and FAIL cases for fail-rate tests; "
            "this run may stop without verified hypotheses."
        )

    # ── Experiment protocol (the human prior) ─────────────────────────────────
    protocol = ExperimentProtocol(
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
    print("\nExperimentProtocol:")
    print(f"  task_domain : {protocol.task_domain}")
    print(f"  description : {protocol.description[:80]}...")

    # ── M1: ProbeAgent — agy selects analyzers from the protocol ─────────────
    probe_agent = ProbeAgent(judge=judge, max_analyzers=args.max_analyzers)

    # ── M2: StatsAnalysisAgent — agy writes the evidence narrative ───────────
    stats_agent = StatsAnalysisAgent(
        judge=None if args.analysis_only else judge,
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
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
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
