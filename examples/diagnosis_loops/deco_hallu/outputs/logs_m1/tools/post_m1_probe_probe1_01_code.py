import json, re
import numpy as np

with open("m1_probe_input.json") as f:
    data = json.load(f)

def first_word_yes(text):
    if not text:
        return None
    m = re.search(r"[a-zA-Z]+", text)
    if not m:
        return None
    w = m.group(0).lower()
    if w in ("yes", "yeah", "yep", "correct", "true"):
        return True
    if w in ("no", "nope", "false", "not"):
        return False
    return None

per_case = []
fp = 0
absent_total = 0
for c in data.get("cases", []):
    ans = first_word_yes(c.get("output", ""))
    exp = c.get("expected")
    # determine ground truth presence: expected may be bool or string yes/no
    if isinstance(exp, bool):
        gt = exp
    elif isinstance(exp, str):
        gt = first_word_yes(exp)
    else:
        gt = None
    false_detection = 0
    if gt is False:
        absent_total += 1
        if ans is True:
            false_detection = 1
            fp += 1
    per_case.append({"sample_id": c.get("id"), "false_detection": int(false_detection)})

rate = float(fp / absent_total) if absent_total else 0.0
result = {"findings": {"false_detection_rate": rate}, "per_case": per_case}
print("PROBE_RESULT_JSON=" + json.dumps(result))
