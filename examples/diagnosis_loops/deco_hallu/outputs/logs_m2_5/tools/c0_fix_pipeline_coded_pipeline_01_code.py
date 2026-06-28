import json, re

with open("fix_cases.json") as f:
    cases = json.load(f)["cases"]

YES = re.compile(r"\byes\b", re.I)
NO = re.compile(r"\bno\b|not\b|isn'?t|aren'?t|there is no|cannot", re.I)

def verdict(text):
    if text is None:
        return None
    t = text.strip().lower()
    if YES.search(t):
        return True
    if NO.search(t):
        return False
    return None

# Quadrant crops strip global scene context (attacking the co-occurrence
# language prior) while magnifying any real localized object.
QUADS = [
    {"tool": "crop_region", "params": {"box": [0.0, 0.0, 0.6, 0.6]}},
    {"tool": "crop_region", "params": {"box": [0.4, 0.0, 1.0, 0.6]}},
    {"tool": "crop_region", "params": {"box": [0.0, 0.4, 0.6, 1.0]}},
    {"tool": "crop_region", "params": {"box": [0.4, 0.4, 1.0, 1.0]}},
    {"tool": "crop_region", "params": {"box": [0.25, 0.25, 0.75, 0.75]}},
]

per_case = []
for c in cases:
    cid, prompt = c["id"], c["prompt"]

    # Baseline (prior-laden) answer on the full image.
    base = verdict(model_generate(cid, prompt=prompt))

    # If the model already says No, trust it: the Yes-bias means a No is
    # rarely a false negative. Skip extra calls.
    if base is False:
        per_case.append({"sample_id": cid, "output": "No"})
        continue

    # Object claimed present -> demand localized spatial corroboration.
    # A real object appears (magnified) in at least one context-stripped crop;
    # a prior-driven hallucination loses its scene cues and collapses to No.
    yes_votes = 0
    for op in QUADS:
        v = verdict(model_generate(cid, prompt=prompt, image_ops=[op]))
        if v is True:
            yes_votes += 1
        if yes_votes >= 2:
            break

    # Also test a global zoom_center for small central objects.
    if yes_votes < 2:
        if verdict(model_generate(cid, prompt=prompt,
                                  image_ops=[{"tool": "zoom_center",
                                              "params": {"factor": 2.0}}])) is True:
            yes_votes += 1

    final = "Yes" if yes_votes >= 2 else "No"
    per_case.append({"sample_id": cid, "output": final})

print("FIX_PIPELINE_RESULT_JSON=" + json.dumps({"per_case": per_case}))
