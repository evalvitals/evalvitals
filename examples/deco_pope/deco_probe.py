"""deco_probe — final-norm logit lens over the frozen POPE manifest (Step 3).

Why this exists (DESIGN.md §3.4/§6): the package logit_lens analyzer skips the
final RMSNorm (trajectories distort on RMSNorm-family models) and crashed with
a cpu/cuda device mismatch in the loop run; the loop's tier-(a) analyzers also
subsample (n=32), leaving M5 underpowered. This probe runs the DeCo-style layer
sweep over the FULL manifest, one forward per case:

    p_i(set) = softmax(lm_head(norm(h_i)))[set].sum()   at pos = -1
    s_supp      = max_{i in window} [p_i(gold) - p_i(other)]
    activated_gt = s_supp >= tau                        (DeCo Eq.2 analogue)
    delta_final = max_i p_i(gold) - p_N(gold)           (late-layer suppression)
    gt_peak_layer = argmax_i p_i(gold)

`other` is the complement answer set (yes<->no). For FAIL cases this equals the
model's emitted answer (out_token_ids) — exactly DeCo's hallucinated-vs-GT
contrast; for PASS cases it is the lure, giving the control group a
non-degenerate margin (gold==out would make the FAIL/PASS contrast trivial).

Expected if residual hallucinations are DeCo-type (H_deco):
    - FAIL (esp. adversarial): high activated_gt rate, delta_final >> 0
    - PASS: delta_final ~ 0
    - group contrast tested with an image-clustered bootstrap

Usage:
    python deco_probe.py --model qwen3-vl-2b-instruct
    # writes outputs/probe_{model}.json + prints the group table
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "outputs"
CFG = yaml.safe_load((HERE / "config.yaml").read_text())


def cluster_bootstrap_diff(rows_a, rows_b, key, n_boot=2000, seed=0):
    """Bootstrap CI for mean(key, group A) - mean(key, group B), resampling
    IMAGES (clusters), not cases — probes from one image are not independent."""
    rng = random.Random(seed)

    def by_img(rows):
        d: dict[int, list[float]] = {}
        for r in rows:
            d.setdefault(r["image_id"], []).append(float(r[key]))
        return d

    ca, cb = by_img(rows_a), by_img(rows_b)
    ia, ib = list(ca), list(cb)
    if not ia or not ib:
        return None
    point = (sum(v for i in ia for v in ca[i]) / sum(len(ca[i]) for i in ia)
             - sum(v for i in ib for v in cb[i]) / sum(len(cb[i]) for i in ib))
    diffs = []
    for _ in range(n_boot):
        sa = [v for i in (rng.choice(ia) for _ in ia) for v in ca[i]]
        sb = [v for i in (rng.choice(ib) for _ in ib) for v in cb[i]]
        diffs.append(sum(sa) / len(sa) - sum(sb) / len(sb))
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    return {"diff": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_a": len(rows_a), "n_b": len(rows_b),
            "significant": (lo > 0) or (hi < 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--limit", type=int, default=0, help="probe only the first N cases (debug)")
    args = ap.parse_args()

    import torch
    import transformers
    from PIL import Image

    from evalvitals import compose
    from evalvitals.core.capability import Capability

    manifest = json.loads((DATA / "cases" / f"{args.model}.json").read_text())
    cases = manifest["cases"][: args.limit or None]

    model = compose(args.model, "hf_local", want={Capability.GENERATE})
    hf = model._loaded[0]  # underlying HF module (read-only use, as in mine_cases)
    processor = transformers.AutoProcessor.from_pretrained(model.spec.hf_repo)
    device = next(hf.parameters()).device

    lm_head = hf.lm_head
    # final RMSNorm of the text decoder; path from the spec's module layout
    lm = hf.get_submodule("model.language_model")
    norm = lm.norm
    n_layers = hf.config.text_config.num_hidden_layers
    lo_f, hi_f = CFG["window"]
    win_lo, win_hi = max(1, round(lo_f * n_layers)), min(n_layers, round(hi_f * n_layers))
    tau = CFG["tau"]
    print(f"n_layers={n_layers} window=[{win_lo},{win_hi}] tau={tau}")

    img_cache: dict[str, Image.Image] = {}
    records = []
    for idx, c in enumerate(cases):
        gold_ids = c["gt_token_ids"]
        other_ids = c["out_token_ids"] if (c["label"] == "fail" and c["out_token_ids"]) else None
        if other_ids is None:
            # PASS (or unparsed): complement answer set as the lure
            comp = "no" if c["pope_label"] == "yes" else "yes"
            # token sets are constant per manifest; find them from any case with that gold
            other_ids = next(r["gt_token_ids"] for r in manifest["cases"]
                             if r["pope_label"] == comp)
        if c["file_name"] not in img_cache:
            img_cache[c["file_name"]] = Image.open(DATA / "images" / c["file_name"]).convert("RGB")
        prompt = manifest["prompt_template"].format(obj=c["object"])
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False, **model.spec.chat_template_kwargs,
        )
        enc = processor(text=[text], images=[img_cache[c["file_name"]]],
                        return_tensors="pt").to(device)
        enc.pop("token_type_ids", None)
        with torch.no_grad():
            hs = hf(**enc, output_hidden_states=True, return_dict=True).hidden_states
            p_gold, p_other = [], []
            for i in range(1, n_layers + 1):  # decoder layers; hs[0] = embeddings
                logits = lm_head(norm(hs[i][0, -1].unsqueeze(0))).float()
                probs = torch.softmax(logits[0], dim=-1)
                p_gold.append(float(probs[gold_ids].sum()))
                p_other.append(float(probs[other_ids].sum()))

        s_supp = max(p_gold[i - 1] - p_other[i - 1] for i in range(win_lo, win_hi + 1))
        peak = max(range(n_layers), key=lambda j: p_gold[j])
        records.append({
            "id": c.get("question_id"), "image_id": c["image_id"], "object": c["object"],
            "probe_type": c["probe_type"], "pope_label": c["pope_label"],
            "pred": c["pred"], "label": c["label"], "split": c["split"],
            "s_supp": round(s_supp, 4),
            "activated_gt": s_supp >= tau,
            "delta_final": round(max(p_gold) - p_gold[-1], 4),
            "gt_peak_layer": peak + 1,
            "p_gold_final": round(p_gold[-1], 4),
            "traj_gold": [round(v, 4) for v in p_gold],
            "traj_other": [round(v, 4) for v in p_other],
        })
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(cases)}")

    # ---- group tables ----------------------------------------------------
    def rate(rows, key="activated_gt"):
        return round(sum(r[key] for r in rows) / len(rows), 3) if rows else None

    summary: dict = {"model": args.model, "n_layers": n_layers,
                     "window": [win_lo, win_hi], "tau": tau, "groups": {}, "tests": {}}
    for split in ("explore", "validate"):
        rows = [r for r in records if r["split"] == split]
        fail = [r for r in rows if r["label"] == "fail"]
        pas = [r for r in rows if r["label"] == "pass"]
        g = {
            "n_fail": len(fail), "n_pass": len(pas),
            "activated_rate_fail": rate(fail), "activated_rate_pass": rate(pas),
            "delta_final_fail_mean": round(sum(r["delta_final"] for r in fail) / len(fail), 4) if fail else None,
            "delta_final_pass_mean": round(sum(r["delta_final"] for r in pas) / len(pas), 4) if pas else None,
        }
        for pt in ("adversarial", "present", "random"):
            sub = [r for r in fail if r["probe_type"] == pt]
            g[f"activated_rate_fail_{pt}"] = rate(sub)
            g[f"n_fail_{pt}"] = len(sub)
        summary["groups"][split] = g
        summary["tests"][split] = {
            "activated_gt fail-pass": cluster_bootstrap_diff(fail, pas, "activated_gt"),
            "delta_final fail-pass": cluster_bootstrap_diff(fail, pas, "delta_final"),
        }

    OUT.mkdir(exist_ok=True)
    out = OUT / f"probe_{args.model}.json"
    out.write_text(json.dumps({**summary, "per_case": records}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_case"}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
