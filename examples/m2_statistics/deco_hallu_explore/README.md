# deco_hallu Explore — M2/M3 on real M1 output

Demonstrates `evalvitals explore` (M2 exploratory analysis + M3 hypothesis
proposal) on **real M1 data**, not a synthetic demo: the per-case VLM
object-presence probe results already committed at
[`examples/diagnosis_loops/deco_hallu/data/cases/`](../../diagnosis_loops/deco_hallu/data/cases)
(three Qwen3-VL checkpoints — 2b/4b/8b — answering "Is there a {object} in
the image?" for COCO images, `label` = pass/fail).

No GPU or new M1 run is needed — this data was already produced by that
example's `run_m1.py`. The raw per-model files are handed to the M2 agent
as-is; it reads them, figures out the shape itself (each is a dict with
scalar run metadata plus a nested `cases` list), and organizes the three
files into one tidy table before analysing it — no pre-processing script.

## Run it

```bash
cd examples/m2_statistics/deco_hallu_explore && docker compose up
```

Or directly (needs a local coding-agent CLI, e.g. `claude`):

```bash
pip install -e ".[dashboard,viz]"
bash run.sh
```

Env overrides: `CODER_PROVIDER` (default `claude_code`), `CODER_MODEL`,
`OUT_DIR` (default `outputs`), `TIMEOUT_SEC`.

## What a real run found

- **Overall**: 126/606 cases (20.8%) are hallucination failures.
- **`probe_type` is almost fully deterministic**: adversarial probes (asking
  about an absent object) fail 34.4% of the time (n=366); present-object
  probes fail 0.0% (n=240).
- **Model size is non-monotonic**: within adversarial probes, fail rate is
  2b=33.9%, 4b=30.4%, 8b=38.5% — the *largest* model has the *highest* fail
  rate, not the lowest.
- **Object identity strongly modulates failure**: `potted plant` fails 81.8%
  of adversarial trials (n=11) vs `skis` at 9.1% (n=11).

M3 hypotheses proposed from these findings (not validated):

1. High-fail objects (potted plant, handbag, bottle, truck, bowl, backpack)
   share strong scene co-occurrence priors in the training distribution
   (plants near windows, bags near people, ...) that bias the model toward
   confirming presence even when the object is absent from the image.
2. The 8b model's higher adversarial fail rate reflects a difference in
   instruction-tuning/RLHF calibration toward helpfulness (favoring
   affirmative answers) between checkpoints, rather than a genuine capability
   regression with scale.

Open the dashboard to see the charts and hypothesis cards:

```bash
evalvitals dashboard outputs
```

See [`docs/m2_analysis.md`](../../../docs/m2_analysis.md) for the general
standalone M2/M3 workflow, and
[`examples/diagnosis_loops/deco_hallu/README.md`](../../diagnosis_loops/deco_hallu/README.md)
for the full M1 → M2 → M3 → M5 → Fix loop this data was built for.
