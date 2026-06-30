import json, re
import numpy as np

with open("m1_probe_input.json") as f:
    data = json.load(f)

def says_yes(text):
    t = (text or "").strip().lower()
    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return m.group(1) == "yes"
    # fall back: affirmative phrasing
    if re.search(r"\bthere (is|are)\b", t) and "no " not in t[:20]:
        return True
    return False

per_case = []
false_detections = 0
absent_total = 0
for c in data.get("cases", []):
    out_yes = says_yes(c.get("output"))
    exp = c.get("expected")
    # expected truth: object present -> answer should be yes
    exp_yes = None
    if isinstance(exp, bool):
        exp_yes = exp
    elif isinstance(exp, str):
        exp_yes = says_yes(exp)
    # false detection: absent object (expected No) but model said Yes
    fd = 0
    if exp_yes is False:
        absent_total += 1
        if out_yes:
            fd = 1
            false_detections += 1
    per_case.append({"sample_id": c.get("id"), "false_detection": int(fd)})

rate = float(false_detections) / absent_total if absent_total else 0.0
print(json.dumps({"absent_total": absent_total, "false_detections": false_detections}))
print("PROBE_RESULT_JSON=" + json.dumps(
    {"findings": {"false_detection_rate": round(rate, 4)}, "per_case": per_case}))
