"""Combine the deco_hallu M1 case files into one explore-ready dataset.

Each ``examples/diagnosis_loops/deco_hallu/data/cases/<model>.json`` is a real
M1 output: per-case VLM object-presence probe results (COCO images, "Is there
a {object} in the image?") for one Qwen3-VL checkpoint. This script merges the
three checkpoints into a single list-of-records file with a ``model`` column,
which is what ``evalvitals explore`` expects (one row per case).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
# Docker build copies the source cases in at build time (see Dockerfile);
# running this script directly from a repo checkout falls back to the
# sibling example's data directory.
_CANDIDATES = [
    HERE / "source_cases",
    HERE / ".." / ".." / "diagnosis_loops" / "deco_hallu" / "data" / "cases",
]
OUT = HERE / "data" / "cases.json"


def _source_dir() -> Path:
    for candidate in _CANDIDATES:
        if candidate.is_dir() and list(candidate.glob("*.json")):
            return candidate
    raise FileNotFoundError(f"no deco_hallu case files found in {_CANDIDATES}")


def main() -> None:
    source = _source_dir()
    rows = []
    for model_file in sorted(source.glob("*.json")):
        payload = json.loads(model_file.read_text(encoding="utf-8"))
        model = payload["model"]
        for case in payload["cases"]:
            row = dict(case)
            row["model"] = model
            row.pop("gt_token_ids", None)
            row.pop("out_token_ids", None)
            rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows), encoding="utf-8")
    print(f"wrote {len(rows)} real M1 probe cases (from {source}) to {OUT}")


if __name__ == "__main__":
    main()
