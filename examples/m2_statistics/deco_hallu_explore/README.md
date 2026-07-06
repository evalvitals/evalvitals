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

## Attention-enriched variant (continuous per-case signals)

The raw probe files carry only categorical fields, so the analysis above can
never show FAIL-vs-PASS *distributions*. `data_attn_full/` fixes that: the same
606 cases enriched with 7 per-case attention-geometry scalars
(attention_entropy, focus_share, center_offset, edge_mass, top1_share,
max/mean_relative_weight) extracted with the `relative_attention` analyzer over
ALL cases of ALL three checkpoints (no `max_cases` cap; per-case float16
spatial maps kept under `data_attn_full/maps/`). Regenerate with:

```bash
python extract_attention_all.py --device cuda   # GPU; downloads any missing
                                                # COCO val2014 images itself
```

The enriched data ships with the repo, so exploring it needs no GPU:

```bash
bash run_attn.sh        # same env overrides as run.sh (CODER_PROVIDER/CODER_MODEL/...)
```

## Held-out hypothesis pipeline (propose → test → fix → one report)

`run_attn.sh` analyses everything in-sample. The pipeline variant splits the
data by its `split` column and walks the full arc:

```bash
bash run_attn_pipeline.sh          # SKIP_FIX=1 to stop after phase 2 (no GPU)
```

1. `prepare_splits.py` — explore half (365) / validate half (241);
2. `evalvitals explore` on the explore half only — hypotheses + frozen,
   threshold-explicit recipes;
3. `test_hypotheses.py` — each recipe re-evaluated VERBATIM on the validate
   half (`adjudicate_signals(split_label="held_out")`: a REJECT here is a real
   held-out verdict), then an LLM judge grades every hypothesis
   (supported / partial / refuted / not_testable + surgery routing);
4. `run_surgery.py` — survivors go to the diagnosis loop's M5 confirm → M4 →
   tiered fix (L1→L3b) on the loop example's frozen M1 batch (GPU).

`confirm_report.json` / `fix_report.json` land next to the exploratory report;
the dashboard then shows a fourth tab (**Held-out Verdicts & Fix**) and badges
each hypothesis card:

```bash
evalvitals dashboard outputs_pipeline/1_explore
```

A real run found: within adversarial probes, focus_share is the strongest
separator (Cohen's d≈1.3, FAIL more peaked/off-center); the same signals
separate FAIL/PASS at every checkpoint (direction is scale-invariant, magnitude
uneven: 2B ≫ 8B > 4B); after collinearity pruning only focus_share,
center_offset, mean_relative_weight and edge_mass carry independent signal.
`data_2b_attn/` is the smaller 2B-only variant whose attention scalars were
transplanted from the diagnosis-loop's frozen M1 state (32/201 coverage —
useful mainly as an informative-missingness case study).

See [`docs/m2_analysis.md`](../../../docs/m2_analysis.md) for the general
standalone M2/M3 workflow, and
[`examples/diagnosis_loops/deco_hallu/README.md`](../../diagnosis_loops/deco_hallu/README.md)
for the full M1 → M2 → M3 → M5 → Fix loop this data was built for.
