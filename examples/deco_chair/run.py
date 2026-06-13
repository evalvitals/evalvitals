"""deco_chair — Mode-1 container input: frozen captions + protocol -> VLDiagnoseLoop.

Same shape as examples/deco_pope/run.py, captioning flavour:

    1. load data/cases/{model}.json   (frozen captions + CHAIR-matched mentions)
    2. drift check                    (re-caption a sample, compare mention SETS)
    3. ExperimentProtocol             (open-ended description hallucination)
    4. VLDiagnoseLoop M1→M5           (tier-(a) should pick `chair`; the
                                       mention-level layer probe is Step 3 of TODO.md)
    5. loop.run_m4(report, cases)     (stepwise DeCo fix — TODO.md Step 4)

No package modifications; bespoke probe/fix live in this directory.

Usage:
    python run.py --model qwen3-vl-2b-instruct [--smoke-test] [--skip-m4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "outputs"
CFG = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())


def load_manifest(model_key: str):
    """Image-level cases for the loop; mention records stay in raw for Step 3."""
    from PIL import Image

    from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label

    path = DATA / "cases" / f"{model_key}.json"
    if not path.exists():
        raise SystemExit(f"{path} missing — run `python mine_cases.py --model {model_key}` "
                         f"(see TODO.md Step 1)")
    raw = json.loads(path.read_text())
    gt_objects = json.loads((DATA / "gt_objects.json").read_text())
    cases = []
    for img in raw["images"]:
        pil = Image.open(DATA / "images" / img["file_name"]).convert("RGB")
        cases.append(FailureCase(
            inputs=Inputs(prompt=raw["prompt"], image=pil),
            expected="caption mentioning only objects present in the image",
            observed=img["caption"],
            label=Label.FAIL if img["hallucinated"] else Label.PASS,
            tags={"hallucination", "deco", "captioning"},
            metadata={
                "image_id": img["image_id"],
                # chair analyzer convention — lets M1 tier-(a) score CHAIR directly:
                "gt_objects": gt_objects[str(img["image_id"])],
                "mentions": img["mentions"],
                "split": img.get("split", "explore"),
            },
        ))
    return CaseBatch(cases), raw


def drift_check(model, cases, raw, n: int = 3) -> None:
    """Long generations drift easily — compare hallucinated-mention SETS, not text."""
    # TODO(Step 2): re-caption n images, re-run chair_match (mine_cases.py),
    # warn if the hallucinated-object set differs from the frozen one.


def build_judge(model_name: str, effort: str):
    """Claude CLI judge for M2/M3/M5 (agy quota exhausted — swapped 2026-06-12).
    Default claude-fable-5 at low effort; --judge-model sonnet|haiku for tests."""
    from evalvitals.eval_agent import ClaudeModel

    judge = ClaudeModel(model=model_name, effort=effort)
    if not judge.generate("Reply with exactly the word OK").strip():
        raise SystemExit(f"judge probe: claude --model {model_name} returned empty "
                         f"(rate-limited?) — try --judge-model sonnet or haiku")
    print(f"judge: claude model={model_name} effort={effort or 'default'}")
    return judge


def build_protocol():
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    return ExperimentProtocol(
        description=(
            "When asked to describe an image in detail, the VLM mentions objects "
            "that are NOT in the image but frequently co-occur with the scene "
            "(e.g. describes a 'mouse' next to a keyboard and monitor that has no "
            "mouse). Suspected mechanism (DeCo, arXiv 2410.11779): at the position "
            "where the hallucinated object is generated, intermediate layers assign "
            "higher probability to objects actually present, and the final layers "
            "suppress this in favour of the language prior. Failure cases are "
            "captions containing hallucinated COCO objects; success cases are "
            "captions from the same image pool with no hallucinated mention."
        ),
        task_domain="object hallucination in image captioning",
        success_criteria="no caption mention outside the image's ground-truth object list",
        failure_patterns=(
            "hallucinated mentions are high-co-occurrence scene partners; "
            "grounded mentions in the SAME caption are correct"
        ),
        target_modalities=frozenset({"text", "image"}),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--max-cycles", type=int, default=2)
    ap.add_argument("--max-analyzers", type=int, default=3)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--skip-m4", action="store_true")
    ap.add_argument("--judge-model", default=CFG.get("judge_model", "claude-fable-5"))
    ap.add_argument("--judge-effort", default=CFG.get("judge_effort", "low"))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.smoke_test:
        exists = (DATA / "cases" / f"{args.model}.json").exists()
        print("smoke ok" if exists else "smoke: no manifest yet (expected before Step 1)")
        return

    from evalvitals import compose
    from evalvitals.core.capability import Capability
    from evalvitals.eval_agent import RunLogger, VLDiagnoseLoop
    from evalvitals.eval_agent.stages.diagnosis import DiagnosisAgent
    from evalvitals.eval_agent.stages.probe_agent import ProbeAgent
    from evalvitals.eval_agent.stages.stats_agent import StatsAnalysisAgent

    judge = build_judge(args.judge_model, args.judge_effort)  # probe BEFORE weights load

    model = compose(args.model, "hf_local",
                    want={Capability.GENERATE, Capability.HIDDEN_STATES,
                          Capability.ATTENTION})
    cases, raw = load_manifest(args.model)
    print(f"cases={len(list(cases))}")
    drift_check(model, cases, raw)

    loop = VLDiagnoseLoop(
        model=model,
        # judge => LLM-guided analyzer selection anchored on the protocol
        # (static StrategyProbe picks generic attention analyzers, not chair)
        probe_agent=ProbeAgent(judge=judge, max_analyzers=args.max_analyzers),
        stats_agent=StatsAnalysisAgent(judge=judge),
        diagnosis_agent=DiagnosisAgent(judge=judge),
        max_cycles=args.max_cycles,
        protocol=build_protocol(),
        run_logger=RunLogger(run_dir=OUT / "logs", verbose=True),
    )
    report = loop.run(cases)
    print(f"cycles={report.cycles} stopped_by={report.stopped_by} "
          f"verified={len(report.verified_hypotheses)}/{len(report.all_test_results)}")
    for t in report.all_test_results:
        stmt = getattr(t.hypothesis, "statement", str(t.hypothesis))
        print(f" - [{t.status}] conf={t.confidence:.2f} grade={t.evidence_grade} {stmt[:110]}")

    # TODO(Step 3): mention-level layer probe (deco_probe.py) — the loop works at
    # image level; the Finding-2 replication needs per-mention prefix re-feeds.
    if not args.skip_m4:
        # TODO(Step 4): verify_fn wrapping deco_fix.py (stepwise DeCo re-caption,
        # CHAIR before/after + no-free-lunch guards).
        fix = loop.run_m4(report, cases)
        print("m4:", fix)


if __name__ == "__main__":
    main()
