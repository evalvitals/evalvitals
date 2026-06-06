"""VLDiagnoseLoop on Qwen3-VL-4B: protocol-guided VL failure diagnosis.

New pipeline (2026-06-05 architecture):

    ExperimentProtocol  ← user's NL description of what to investigate
         │
    M1  ProbeAgent           protocol-guided analyzer selection + execute
    M2  StatsAnalysisAgent   protocol-aware stats analysis + evidence chain
    M3  DiagnosisAgent       Qwen as judge ("AI scientist" hypothesis gen)
    M5  HypothesisTester     statistical + protocol consistency check
         │
    loop ends when M5 finds a verified, protocol-consistent hypothesis
         │
    M4  SurgeryAgent         agy/codex writes + runs targeted fix script
                             (called separately AFTER the loop)

Run infrastructure:
    - run_dir=./outputs/  → checkpoint.json, heartbeat.json, evolution/lessons.jsonl

Usage (via Docker — preferred):
    docker compose up

Usage (direct):
    python run.py
    python run.py --model qwen2.5-vl-7b-instruct --device cuda:0
    python run.py --analysis-only   # M1+M2 only, skip M3/M4/M5
"""

from __future__ import annotations

import argparse
import io
import textwrap
import urllib.request
from pathlib import Path

_SAMPLE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
    "Bikesingapore.jpg/320px-Bikesingapore.jpg"
)
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

    def log_probe(self, cycle: int, results: dict) -> None:
        print(f"\n[M1] cycle={cycle}  analyzers={list(results.keys())}", flush=True)
        for name, r in results.items():
            scalars = {
                k: round(v, 4)
                for k, v in (getattr(r, "findings", {}) or {}).items()
                if isinstance(v, (int, float))
            }
            print(f"     {name}: {dict(list(scalars.items())[:6])}", flush=True)
        self._rl.log_probe(cycle, results)

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

def _get_image():
    from PIL import Image
    try:
        with urllib.request.urlopen(_SAMPLE_URL, timeout=10) as resp:
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

def _build_cases(image):
    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
    return CaseBatch([
        FailureCase(
            id="q_count",
            inputs=Inputs(
                prompt="How many distinct objects can you see in this image?",
                image=image,
            ),
            label=Label.FAIL,
        ),
        FailureCase(
            id="q_colour",
            inputs=Inputs(
                prompt="What are the dominant colours in this image?",
                image=image,
            ),
            label=Label.FAIL,
        ),
        FailureCase(
            id="q_location",
            inputs=Inputs(
                prompt="Describe the spatial layout of objects in this image.",
                image=image,
            ),
            label=Label.FAIL,
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
        "--analysis-only", action="store_true",
        help="Run M1+M2 only (skip M3/M5/M4)",
    )
    parser.add_argument("--run-dir", default=str(_OUTPUTS_DIR))
    args = parser.parse_args()

    import evalvitals
    from evalvitals.eval_agent import (
        DiagnosisAgent,
        VLDiagnoseLoop,
    )
    from evalvitals.eval_agent.hypothesis_tester import HypothesisTester
    from evalvitals.eval_agent.probe import ModelKind, StrategyProbe
    from evalvitals.eval_agent.probe_agent import ProbeAgent
    from evalvitals.eval_agent.protocol import ExperimentProtocol
    from evalvitals.eval_agent.stats_agent import StatsAnalysisAgent

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

    # ── Image + cases ─────────────────────────────────────────────────────────
    print("\nPreparing image …")
    image = _get_image()
    cases = _build_cases(image)
    print(f"  {len(cases)} failure cases")

    # ── Experiment protocol (the human prior) ─────────────────────────────────
    protocol = ExperimentProtocol(
        description=(
            "We evaluate a VLM on image understanding tasks: object counting, "
            "colour recognition, and spatial layout description. "
            "The model is suspected to ignore visual tokens and rely primarily "
            "on text priors, leading to hallucinated or spatially incorrect answers."
        ),
        task_domain="visual question answering",
        failure_patterns=(
            "visual token attention low, image ignored, spatial confusion, "
            "hallucinated objects"
        ),
        target_modalities=frozenset({"text", "image"}),
    )
    hints = protocol.probe_hints()
    print(f"\nExperimentProtocol:")
    print(f"  task_domain    : {protocol.task_domain}")
    print(f"  probe_hints()  : {hints}")

    # ── M1: ProbeAgent (VLM priority: attention → mm_shap) ───────────────────
    vlm_probe = StrategyProbe(priority_override={
        ModelKind.VLM: ["attention", "mm_shap"],
    })
    probe_agent = ProbeAgent(probe=vlm_probe, max_analyzers=args.max_analyzers)

    # ── M2: StatsAnalysisAgent ────────────────────────────────────────────────
    # With judge=model the analysis generates an LLM-written conclusion;
    # in analysis-only mode we skip the LLM call.
    stats_agent = StatsAnalysisAgent(
        judge=None if args.analysis_only else model,
    )

    # ── M3: DiagnosisAgent ────────────────────────────────────────────────────
    diagnosis_agent = None
    if not args.analysis_only:
        diagnosis_agent = DiagnosisAgent(judge=model)
        print("  M3 DiagnosisAgent : Qwen (same model, text-only prompt)")

    # ── M5: HypothesisTester ─────────────────────────────────────────────────
    hypothesis_tester = None
    if not args.analysis_only:
        hypothesis_tester = HypothesisTester(
            judge=model,   # Qwen checks protocol consistency
            min_effect=0.05,
        )
        print("  M5 HypothesisTester : Qwen (protocol consistency check)")

    # ── M4: SurgeryAgent (post-loop fix proposal) ─────────────────────────────
    surgery_agent = None
    if not args.analysis_only:
        import shutil

        from evalvitals.eval_agent import CliAgentConfig, ExperimentWriterConfig, SurgeryAgent

        agy_bin = shutil.which("agy")
        codex_bin = shutil.which("codex")
        if agy_bin:
            writer_cfg = ExperimentWriterConfig(
                cli_agent=CliAgentConfig(provider="antigravity", timeout_sec=120),
                exec_fix_timeout_sec=60,
            )
            print(f"  M4 SurgeryAgent   : antigravity CLI ({agy_bin})")
        elif codex_bin:
            writer_cfg = ExperimentWriterConfig(
                cli_agent=CliAgentConfig(provider="codex", timeout_sec=120),
                exec_fix_timeout_sec=60,
            )
            print(f"  M4 SurgeryAgent   : codex CLI ({codex_bin})")
        else:
            writer_cfg = ExperimentWriterConfig(exec_fix_timeout_sec=60)
            print("  M4 SurgeryAgent   : LLM path (agy/codex not found on PATH)")
        surgery_agent = SurgeryAgent(judge=model, writer_config=writer_cfg)

    # ── Run directory + verbose logger ────────────────────────────────────────
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {run_dir.resolve()}")
    print("  checkpoint.json          ← not used by VLDiagnoseLoop (stateless loop)")
    print("  evolution/lessons.jsonl  ← not used by VLDiagnoseLoop (future)")

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
