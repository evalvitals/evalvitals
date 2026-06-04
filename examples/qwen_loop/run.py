"""AutoDiagnoseLoop on Qwen3-VL-4B: explore attention over a real image.

Pipeline:
    M1 · ProbeAgent   runs attention + mm_shap analyzers on the VLM
    M2 · Analysis     interprets results → severity + narrative
    M3 · Diagnosis    Qwen itself as judge (no external API key needed)
    M4 · Surgery      codex CLI writes + runs targeted diagnostic code

Run infrastructure:
    - run_dir=./runs/  → checkpoint.json, heartbeat.json, evolution/lessons.jsonl
    - git branch eval/{run_id} created if inside a git repo

Usage (via Docker, see docker-compose.yml):
    docker compose up

Usage (direct):
    python run.py
    python run.py --model qwen2.5-vl-7b-instruct --device cuda:0
    python run.py --analysis-only   # M1+M2 only, skip M3/M4
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import textwrap
import urllib.request
from pathlib import Path

_SAMPLE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
    "Bikesingapore.jpg/320px-Bikesingapore.jpg"
)
_OUTPUTS_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# Verbose logger — inherits RunLogger, overrides hooks to also print stdout
# ---------------------------------------------------------------------------

class VerboseRunLogger:
    """Subclass RunLogger to mirror each event to stdout as it happens."""

    def __init__(self, run_dir: Path) -> None:
        from evalvitals.eval_agent import RunLogger
        self._rl = RunLogger(run_dir=run_dir)

    # Forward attribute access to the real RunLogger
    def __getattr__(self, name):
        return getattr(self._rl, name)

    def log_probe(self, cycle: int, results: dict) -> None:
        print(f"\n[M1] cycle={cycle}  analyzers={list(results.keys())}", flush=True)
        for name, r in results.items():
            scalars = {k: round(v, 4) for k, v in (getattr(r, "findings", {}) or {}).items()
                       if isinstance(v, (int, float))}
            print(f"     {name}: {dict(list(scalars.items())[:6])}", flush=True)
        self._rl.log_probe(cycle, results)

    def log_analysis(self, cycle: int, analysis) -> None:
        print(f"\n[M2] cycle={cycle}  severity={analysis.severity}", flush=True)
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
        status = getattr(getattr(intervention, "status", None), "value", "?")
        print(f"\n[M4] cycle={cycle}  '{hypothesis.statement[:70]}'", flush=True)
        print(f"     status={status}  fixed={intervention.fixed}", flush=True)
        ev = getattr(intervention, "evidence", {}) or {}
        if ev:
            print(f"     evidence: {dict(list(ev.items())[:4])}", flush=True)
        self._rl.log_surgery(cycle, hypothesis, intervention)

    def log_loop_end(self, report) -> None:
        print(f"\n[DONE] cycles={report.cycles}  resolved={report.resolved}", flush=True)
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
    draw.rectangle([20, 20, 100, 100], fill=(220, 80, 60))
    draw.rectangle([124, 20, 204, 100], fill=(60, 160, 80))
    draw.rectangle([60, 130, 164, 190], fill=(80, 80, 200))
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
    parser = argparse.ArgumentParser(description="AutoDiagnoseLoop on Qwen VL + image")
    parser.add_argument("--model", default="qwen3-vl-4b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-cycles", type=int, default=2)
    parser.add_argument("--max-analyzers", type=int, default=2)
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Run M1+M2 only (skip M3 diagnosis and M4 surgery)",
    )
    parser.add_argument("--run-dir", default=str(_OUTPUTS_DIR))
    args = parser.parse_args()

    import evalvitals
    from evalvitals.eval_agent import AutoDiagnoseLoop
    from evalvitals.eval_agent.probe import ModelKind, StrategyProbe
    from evalvitals.eval_agent.probe_agent import ProbeAgent

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

    # ── Probe (VLM priority: attention → mm_shap) ─────────────────────────────
    vlm_probe = StrategyProbe(priority_override={
        ModelKind.VLM: ["attention", "mm_shap"],
    })
    probe_agent = ProbeAgent(probe=vlm_probe, max_analyzers=args.max_analyzers)

    # ── M3 DiagnosisAgent — use the loaded Qwen model as judge ────────────────
    diagnosis_agent = None
    if not args.analysis_only:
        from evalvitals.eval_agent import DiagnosisAgent
        diagnosis_agent = DiagnosisAgent(judge=model)
        print("  M3 DiagnosisAgent : Qwen (same model, text-only prompt)")

    # ── M4 SurgeryAgent — codex CLI writes the diagnostic script ─────────────
    surgery_agent = None
    if diagnosis_agent is not None:
        import shutil
        from evalvitals.eval_agent import CliAgentConfig, ExperimentWriterConfig, SurgeryAgent

        codex_bin = shutil.which("codex")
        if codex_bin:
            writer_cfg = ExperimentWriterConfig(
                cli_agent=CliAgentConfig(
                    provider="codex",
                    timeout_sec=120,
                ),
                exec_fix_timeout_sec=60,
            )
            print(f"  M4 SurgeryAgent   : codex CLI ({codex_bin})")
        else:
            # codex not on PATH — fall back to LLM path using Qwen as writer judge
            writer_cfg = ExperimentWriterConfig(exec_fix_timeout_sec=60)
            print("  M4 SurgeryAgent   : LLM path (Qwen, codex not found on PATH)")
        surgery_agent = SurgeryAgent(judge=model, writer_config=writer_cfg)

    # ── Run directory + verbose logger ────────────────────────────────────────
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {run_dir.resolve()}")
    print("  checkpoint.json          ← written after each cycle")
    print("  heartbeat.json           ← pid + last_cycle")
    print("  evolution/lessons.jsonl  ← cross-run lesson store")

    logger = VerboseRunLogger(run_dir=run_dir / "logs")
    print(f"  event log: {logger.run_dir}")

    # ── Loop ──────────────────────────────────────────────────────────────────
    loop = AutoDiagnoseLoop(
        model=model,
        probe_agent=probe_agent,
        diagnosis_agent=diagnosis_agent,
        surgery_agent=surgery_agent,
        max_cycles=args.max_cycles,
        run_logger=logger,
        run_dir=run_dir,
    )

    print(f"\n{'='*60}")
    print(f"AutoDiagnoseLoop  model={args.model}  max_cycles={args.max_cycles}")
    print(f"{'='*60}")

    report = loop.run(cases)

    # ── Summary ───────────────────────────────────────────────────────────────
    import json
    print(f"\n{'='*60}")
    print("RUN FILES")
    print(f"{'='*60}")
    if (run_dir / "checkpoint.json").exists():
        cp = json.loads((run_dir / "checkpoint.json").read_text())
        print(f"  checkpoint : cycle {cp['last_completed_cycle']}  run_id={cp['run_id']}")
    if (run_dir / "heartbeat.json").exists():
        hb = json.loads((run_dir / "heartbeat.json").read_text())
        print(f"  heartbeat  : pid={hb['pid']}  last_cycle={hb['last_cycle']}")
    ev = run_dir / "evolution" / "lessons.jsonl"
    if ev.exists():
        n = sum(1 for _ in ev.open())
        print(f"  evolution  : {n} lesson(s) in {ev}")


if __name__ == "__main__":
    main()
