"""Offline miner (deco_chair) — captions -> CHAIR-matched mentions -> frozen manifest.

    python mine_cases.py --model qwen3-vl-8b-instruct --n-images 50

Steps (DESIGN.md §2):
    1. pick ~50 cluttered COCO val2014 images (selection criteria coded, not handpicked)
    2. extract per-image GT object lists from instances_val2014.json (cached, not committed)
    3. greedy caption each image ("Please help me describe the image in detail.")
    4. CHAIR-match mentions vs GT (synonym table) -> hallucinated | grounded
    5. locate each mention's first-token index in the full tokenized sequence
    6. freeze to data/cases/{model_key}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"
CAPTION_PROMPT = "Please help me describe the image in detail."
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"


def select_images(n: int) -> list[dict]:
    """Cluttered-scene selection: rank val2014 images by (#distinct GT categories,
    presence of indoor super-categories kitchen/furniture/electronics), take top n.
    TODO: implement over the cached instances_val2014.json; emit
    data/image_list.json with the criteria recorded alongside the ids."""
    raise NotImplementedError


def extract_gt_objects(images: list[dict]) -> dict[int, list[str]]:
    """instances_val2014.json -> {image_id: [coco category names]} for chosen images.
    Write data/gt_objects.json (small, committed). TODO."""
    raise NotImplementedError


def chair_match(caption: str, gt: list[str], synonyms: dict) -> list[dict]:
    """Return mentions: {surface, coco_category, mention_kind, char_start}.
    Reuse the matching conventions of evalvitals.analyzers.hallucination.chair
    (vocab + synonym table) so analysis and metrics agree. TODO."""
    raise NotImplementedError


def token_index_of(tokenizer, full_ids: list[int], char_start: int, enc) -> int:
    """Map a char offset in the caption to the index of the mention's first token
    within the FULL sequence (prompt + caption). Use the processor's offset
    mapping; greedy decoding guarantees re-feeding ids[:k] reproduces the
    distribution that emitted token k (DESIGN.md §2). TODO."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b-instruct")
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from evalvitals import compose
    from evalvitals.core.capability import Capability
    from evalvitals.core.case import Inputs

    images = select_images(args.n_images)
    gt_objects = extract_gt_objects(images)
    synonyms = json.loads((DATA / "chair_synonyms.json").read_text())

    model = compose(args.model, "hf_local", want={Capability.GENERATE})
    records = []
    for img in images:
        cap = model.generate(
            Inputs(prompt=CAPTION_PROMPT, image=img["path"]),
            max_new_tokens=512,  # TODO(verify): greedy kwargs passthrough
        )
        mentions = chair_match(cap, gt_objects[img["image_id"]], synonyms)
        # TODO: tokenize prompt+caption once; fill token_index per mention;
        #       fill gt_token_ids = first tokens of present GT objects that fall
        #       inside the final-layer top-p candidate set at that position
        #       (Eq.2 precondition — verify keep/drop direction vs zjunlp/DeCo).
        records.append({
            "image_id": img["image_id"], "file_name": img["file_name"],
            "caption": cap, "mentions": mentions,
            "hallucinated": any(m["mention_kind"] == "hallucinated" for m in mentions),
        })

    # Image-level 60/40 explore/validate split, stratified by `hallucinated`.
    # TODO: split + yields accounting, mirroring deco_pope/mine_cases.py.
    out = DATA / "cases" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "prompt": CAPTION_PROMPT, "seed": args.seed,
        "decoding": {"do_sample": False, "max_new_tokens": 512},
        "images": records,
    }, indent=2, ensure_ascii=False))
    print(f"frozen -> {out}")


if __name__ == "__main__":
    main()
