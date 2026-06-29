"""Diagnostic: is over-detection driven by a language/co-occurrence prior that is
upstream of vision (persists text-only) and immune to prompt reformatting?

All cases here are adversarial FAILs: the probed object is ABSENT (pope_label='no')
yet the model over-detects ("Yes"). We test three conditions on the answer-token
logits P(Yes) vs P(No):
  metric_a = yes-rate WITH image, original prompt   (the over-detection itself)
  metric_b = yes-rate TEXT-ONLY (image removed)     (language prior, upstream of vision)
  metric_c = yes-rate WITH image, REFORMATTED prompt (immunity to reformatting)

SUPPORTED if the Yes bias survives without vision (metric_b high) AND survives
prompt reformatting (metric_c high) -- i.e. the bias is not produced by the image
and is not undone by rewording.
"""
import json, time
import torch
import evalvitals
from evalvitals import Capability
from evalvitals.core.case import Inputs
from PIL import Image

START = time.time()
TIME_BUDGET = 42.0  # seconds for the measurement loop

model = evalvitals.load('qwen3-vl-2b-instruct', device='cuda', dtype='bfloat16',
                        want=['logits', 'attention', 'hidden_states'])

YES_IDS = [7414, 9454]   # " Yes", "Yes"
NO_IDS  = [2308, 2753]   # " No", "No"

def says_yes(inputs):
    tr = model.forward(inputs, capture={Capability.LOGITS})
    row = tr.logits[-1].float()
    yes = max(row[i].item() for i in YES_IDS)
    no  = max(row[i].item() for i in NO_IDS)
    return yes > no

def reformat(prompt):
    # rephrase "Is there a X in the image? Please answer Yes or No."
    obj = prompt
    for a, b in [("Is there a ", ""), ("Is there an ", ""),
                 (" in the image? Please answer Yes or No.", "")]:
        obj = obj.replace(a, b)
    return (f"Look closely at the picture. Considering only what is actually "
            f"visible, does the image contain {obj}? Respond with a single "
            f"word, Yes or No.")

cases = json.load(open('cases.json'))
cases = [c for c in cases if c.get('image_path')]

img_yes = txt_yes = ref_yes = n = 0
for c in cases:
    if time.time() - START > TIME_BUDGET:
        break
    img = Image.open(c['image_path']).convert('RGB')
    p = c['prompt']
    img_yes += says_yes(Inputs(prompt=p, image=img))
    txt_yes += says_yes(Inputs(prompt=p, image=None))
    ref_yes += says_yes(Inputs(prompt=reformat(p), image=img))
    n += 1

metric_a = img_yes / n      # over-detection rate (vision present)
metric_b = txt_yes / n      # language-prior yes-rate (vision absent)
metric_c = ref_yes / n      # reformatted-prompt yes-rate (vision present)

# SUPPORTED: yes-bias is upstream of vision (survives text-only) AND immune to
# prompt reformatting. We require text-only Yes bias to be substantial and the
# reformatted yes-rate to remain close to the original (no reformatting rescue).
upstream  = metric_b >= 0.5
immune    = metric_c >= 0.5 and metric_c >= metric_a - 0.15
verdict = 1.0 if (upstream and immune) else 0.0

print(f"n_cases: {float(n)}")
print(f"metric_a: {metric_a:.4f}")
print(f"metric_b: {metric_b:.4f}")
print(f"metric_c: {metric_c:.4f}")
print(f"verdict: {verdict}")
