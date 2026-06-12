"""Offline miner — build the frozen fail/success manifest for one model size.

Run ONCE per model key (failure sets don't transfer across 2B/4B/8B):

    python mine_cases.py --model qwen3-vl-8b-instruct --n-images 100

Steps:
    1. download the POPE COCO adversarial/random split JSONs (pinned URLs)
    2. group probes into per-image triplets (adversarial-absent / present / random-absent)
    3. download the COCO val2014 images into data/images/
    4. greedy-generate the model's answer per probe (do_sample=False, eager attn)
    5. label PASS/FAIL via pope.parse_yes_no, attach yes/no token-id sets
    6. split 60/40 explore/validate BY IMAGE, freeze to data/cases/{model_key}.json

See DESIGN.md §4 for the rationale of every choice here.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
IMG_DIR = DATA / "images"
COCO_URL = "http://images.cocodataset.org/val2014/{file_name}"
# POPE official probe lists (Li et al. 2023). TODO(verify): pin a commit hash.
POPE_URLS = {
    "adversarial": "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_adversarial.json",
    "random": "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_random.json",
}
POPE_TMPL = "Is there a {obj} in the image? Please answer Yes or No."


def fetch_pope_probes(n_images: int) -> list[dict]:
    """Download POPE splits and assemble per-image triplets.

    Returns entries: {image_id, file_name, object, pope_label, probe_type}.
    POPE files are JSON-lines: {"question_id", "image", "text", "label"}.
    - present probes are the label=="yes" lines (shared across splits)
    - adversarial-absent / random-absent come from the respective split's "no" lines
    TODO: parse "Is there a X in the image?" -> object name; keep images that
    have all three probe types; truncate to n_images.
    """
    raise NotImplementedError  # TODO — straight JSON-lines wrangling, no model needed


def download_images(probes: list[dict]) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for fn in sorted({p["file_name"] for p in probes}):
        dest = IMG_DIR / fn
        if not dest.exists():
            urllib.request.urlretrieve(COCO_URL.format(file_name=fn), dest)


def answer_token_sets(tokenizer) -> dict[str, list[int]]:
    """Token-id equivalence sets for yes/no (DESIGN.md §3.5)."""
    def ids(variants):
        out = []
        for v in variants:
            t = tokenizer.encode(v, add_special_tokens=False)
            if len(t) == 1:          # only single-token variants are usable at pos -1
                out.append(t[0])
        return sorted(set(out))
    return {"yes": ids(["Yes", " Yes", "yes", " yes"]),
            "no": ids(["No", " No", "no", " no"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b-instruct")
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--explore-frac", type=float, default=0.6)
    args = ap.parse_args()
    random.seed(args.seed)

    from evalvitals import compose
    from evalvitals.analyzers.hallucination.pope import parse_yes_no
    from evalvitals.core.capability import Capability

    probes = fetch_pope_probes(args.n_images)
    download_images(probes)

    model = compose(args.model, "hf_local", want={Capability.GENERATE})
    # TODO: greedy decoding — pass do_sample=False / max_new_tokens=8 through
    # RuntimeConfig or generate kwargs (check hf_local defaults).
    tok_sets = answer_token_sets(model.tokenizer)  # TODO(verify): tokenizer accessor

    # By-image split BEFORE labeling, so yields per split are honest.
    image_ids = sorted({p["image_id"] for p in probes})
    random.shuffle(image_ids)
    n_explore = int(len(image_ids) * args.explore_frac)
    split_of = {iid: ("explore" if i < n_explore else "validate")
                for i, iid in enumerate(image_ids)}

    records, yields = [], {"explore": {"fail": 0, "pass": 0}, "validate": {"fail": 0, "pass": 0}}
    for p in probes:
        from evalvitals.core.case import Inputs
        ans = model.generate(Inputs(prompt=POPE_TMPL.format(obj=p["object"]),
                                    image=str(IMG_DIR / p["file_name"])))
        pred = parse_yes_no(ans)
        label = "pass" if pred == p["pope_label"] else "fail"
        split = split_of[p["image_id"]]
        yields[split][label] += 1
        records.append({
            **p, "observed": ans, "pred": pred, "label": label, "split": split,
            "gt_token_ids": tok_sets[p["pope_label"]],
            "out_token_ids": tok_sets[pred] if pred else [],
        })

    out = DATA / "cases" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "seed": args.seed, "prompt_template": POPE_TMPL,
        "decoding": {"do_sample": False, "max_new_tokens": 8},
        # TODO: record transformers/torch versions for drift checking
        "yields": yields, "cases": records,
    }, indent=2, ensure_ascii=False))
    print(f"frozen -> {out}  yields={yields}")
    # DESIGN.md §8: if explore fail count < ~15, raise --n-images or merge AMBER.


if __name__ == "__main__":
    main()
