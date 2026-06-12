"""deco_chair probe — mention-level prefix re-feed layer sweep (TODO.md Step 3).

Faithful Finding-2 replication: for every VERIFIED mention in the frozen
manifest, re-feed ``ids[:token_index]`` (prompt encoding + stored caption token
ids — ids, not re-derived text) and read the layer trajectory at pos = -1 with
a final-norm logit lens:

    p_i(set) = softmax(lm_head(norm(h_i)))[set].sum()

Per mention (DESIGN.md §3):
    out  = the mention's own first token (what the model emitted there)
    gt   = first tokens of the image's PRESENT GT objects, restricted to the
           final-layer top-k(20) -> top-p(0.9) candidate set (DeCo Eq.2
           precondition; zjunlp/DeCo keeps candidate members, drops the rest).
           For grounded mentions (control) the mention's own category is
           excluded from gt, so both groups ask the same question: "did the
           OTHER in-image objects outweigh the emitted token mid-stack?"
    s_supp        = max_{i in window} [p_i(gt) - p_i(out)]
    activated_gt  = s_supp >= tau            (None when gt n candidates = 0)
    delta_final   = max_i p_i(gt) - p_N(gt)
    gt_peak_layer = argmax_i p_i(gt)

H_deco signature: activated_gt rate (hallucinated) >> (grounded), and
hallucinated mentions' gt_peak_layer concentrated in [0.55N, 0.85N]
(paper: layers 20-28 of 32). Mentions with verified=false are skipped —
their token_index does not reproduce the generation-time distribution.

Usage:
    python deco_probe.py --model qwen3-vl-2b-instruct
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
    """Bootstrap CI for mean(key, A) - mean(key, B), resampling images."""
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
    ap.add_argument("--limit", type=int, default=0, help="probe only first N images (debug)")
    args = ap.parse_args()

    import torch
    import transformers
    from PIL import Image

    from evalvitals import compose
    from evalvitals.core.capability import Capability

    manifest = json.loads((DATA / "cases" / f"{args.model}.json").read_text())
    images = manifest["images"][: args.limit or None]

    model = compose(args.model, "hf_local", want={Capability.GENERATE})
    hf = model._loaded[0]
    processor = transformers.AutoProcessor.from_pretrained(model.spec.hf_repo)
    device = next(hf.parameters()).device

    lm_head = hf.lm_head
    norm = hf.get_submodule("model.language_model").norm
    n_layers = hf.config.text_config.num_hidden_layers
    lo_f, hi_f = CFG["window"]
    win_lo, win_hi = max(1, round(lo_f * n_layers)), min(n_layers, round(hi_f * n_layers))
    tau, top_k, top_p = CFG["tau"], CFG["top_k"], CFG["top_p"]
    print(f"n_layers={n_layers} window=[{win_lo},{win_hi}] tau={tau} "
          f"candidates=top_k({top_k})->top_p({top_p})")

    records, n_skipped_unverified, n_no_candidates = [], 0, 0
    for ii, img in enumerate(images):
        mentions = [m for m in img["mentions"] if m.get("verified")]
        n_skipped_unverified += sum(1 for m in img["mentions"] if not m.get("verified"))
        if not mentions:
            continue
        pil = Image.open(DATA / "images" / img["file_name"]).convert("RGB")
        content = [{"type": "image"}, {"type": "text", "text": manifest["prompt"]}]
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False, **model.spec.chat_template_kwargs,
        )
        enc = processor(text=[text], images=[pil], return_tensors="pt")
        if enc["input_ids"].shape[1] != img["prompt_len"]:
            print(f"  [WARN] {img['file_name']}: prompt_len drift "
                  f"{enc['input_ids'].shape[1]} != {img['prompt_len']} — skipping image")
            continue
        enc = enc.to(device)
        full_ids = torch.cat(
            [enc["input_ids"], torch.tensor([img["caption_token_ids"]], device=device)], dim=1)
        pixel_kwargs = {k: v for k, v in enc.items()
                        if k not in ("input_ids", "attention_mask")}
        pixel_kwargs.pop("token_type_ids", None)

        for m in mentions:
            k = m["token_index"]
            out_id = m["first_token_id"]
            gt_ids = [tid for cat, tid in img["gt_first_token_ids"].items()
                      if cat != m["coco_category"]]  # control symmetry: own cat excluded
            with torch.no_grad():
                hs = hf(input_ids=full_ids[:, :k],
                        attention_mask=torch.ones_like(full_ids[:, :k]),
                        output_hidden_states=True, return_dict=True,
                        **pixel_kwargs).hidden_states
                # final-layer candidate set: top-k then top-p (DeCo Eq.2)
                final_probs = torch.softmax(
                    lm_head(norm(hs[n_layers][0, -1].unsqueeze(0))).float()[0], dim=-1)
                pv, pi = final_probs.topk(top_k)
                keep = pv.cumsum(0) - pv <= top_p  # tokens before cumulative mass exceeds top_p
                candidates = set(pi[keep].tolist())
                gt_in = [t for t in gt_ids if t in candidates]
                p_gt, p_out = [], []
                for i in range(1, n_layers + 1):
                    probs = torch.softmax(
                        lm_head(norm(hs[i][0, -1].unsqueeze(0))).float()[0], dim=-1)
                    p_gt.append(float(probs[gt_in].sum()) if gt_in else 0.0)
                    p_out.append(float(probs[out_id]))

            rec = {
                "image_id": img["image_id"], "split": img["split"],
                "coco_category": m["coco_category"], "surface": m["surface"],
                "mention_kind": m["mention_kind"], "token_index": k,
                "n_gt_objects": len(gt_ids), "n_gt_in_candidates": len(gt_in),
                "traj_gt": [round(v, 4) for v in p_gt],
                "traj_out": [round(v, 4) for v in p_out],
            }
            if gt_in:
                s_supp = max(p_gt[i - 1] - p_out[i - 1] for i in range(win_lo, win_hi + 1))
                rec.update({
                    "s_supp": round(s_supp, 4),
                    "activated_gt": s_supp >= tau,
                    "delta_final": round(max(p_gt) - p_gt[-1], 4),
                    "gt_peak_layer": max(range(n_layers), key=lambda j: p_gt[j]) + 1,
                })
            else:
                n_no_candidates += 1
                rec.update({"s_supp": None, "activated_gt": None,
                            "delta_final": None, "gt_peak_layer": None})
            records.append(rec)
        print(f"  [{ii + 1}/{len(images)}] {img['file_name']}: {len(mentions)} mentions probed")

    # ---- group tables ------------------------------------------------------
    def usable(kind, split=None):
        return [r for r in records if r["mention_kind"] == kind
                and r["activated_gt"] is not None
                and (split is None or r["split"] == split)]

    summary: dict = {"model": args.model, "n_layers": n_layers,
                     "window": [win_lo, win_hi], "tau": tau,
                     "top_k": top_k, "top_p": top_p,
                     "n_mentions_probed": len(records),
                     "n_skipped_unverified": n_skipped_unverified,
                     "n_excluded_no_gt_candidates": n_no_candidates,
                     "groups": {}, "tests": {}}
    for split in ("explore", "validate", None):
        tag = split or "all"
        hal, grd = usable("hallucinated", split), usable("grounded", split)
        g = {"n_hallucinated": len(hal), "n_grounded": len(grd)}
        for name, rows in (("hallucinated", hal), ("grounded", grd)):
            if rows:
                g[f"activated_rate_{name}"] = round(
                    sum(r["activated_gt"] for r in rows) / len(rows), 3)
                g[f"delta_final_{name}_mean"] = round(
                    sum(r["delta_final"] for r in rows) / len(rows), 4)
        if hal:
            in_band = sum(1 for r in hal
                          if 0.55 * n_layers <= r["gt_peak_layer"] <= 0.85 * n_layers)
            g["hal_peak_in_[0.55N,0.85N]"] = round(in_band / len(hal), 3)
        summary["groups"][tag] = g
        summary["tests"][tag] = {
            "activated_gt hal-grounded": cluster_bootstrap_diff(hal, grd, "activated_gt"),
            "delta_final hal-grounded": cluster_bootstrap_diff(hal, grd, "delta_final"),
        }

    OUT.mkdir(exist_ok=True)
    out = OUT / f"probe_{args.model}.json"
    out.write_text(json.dumps({**summary, "per_mention": records}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_mention"}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
