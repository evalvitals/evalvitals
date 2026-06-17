# deco_hallu — the POPE *hallucination* subset (the hard slice)

> Third sibling of [`../deco_pope`](../deco_pope) / [`../deco_miss`](../deco_miss).
> Same DeCo scenario family ([arXiv 2410.11779](https://arxiv.org/abs/2410.11779)),
> the **hardest** failure slice: a confident *false Yes*.

## 1. Why this slice is hard

`deco_miss` is the recoverable failure — a present object answered "No". The
object's evidence exists in the image, so amplifying it (describe-first,
zoom, evidence-first decoding) flips misses without cost, and the loop closes to
a **validated fix**.

The **hallucination** is the opposite: the model answers "Yes" for an object
that is **not** in the image. There is no correct latent signal to amplify — the
wrong answer is a positive assertion, not a suppressed truth. So the open
question this example poses to the loop is: *is there any intervention up to
L3b that reduces the false-Yes rate without also destroying the model's correct
"Yes" detections?* The honest answer may well be "no, escalate to L4" — which is
itself a valid, useful outcome (it is what `deco_pope`'s mixed run concluded).

## 2. Cases — the no-free-lunch guard is built into the data

[`build_cases.py`](build_cases.py) re-slices `deco_pope`'s frozen manifest (no
GPU, images shared). The batch is deliberately mixed so a degenerate "always
answer No" cannot win — every case is scored against its **own** gold label:

| group | probe / answer | gold | role |
|---|---|---|---|
| **FAIL** | adversarial-absent → "Yes" | no | the hallucination to reduce |
| PASS | adversarial-absent → "No" | no | correct rejection (control) |
| PASS | present-object → "Yes" | yes | **correct detection — the recall a fix must not break** |

Counts (per size): 2B **41** hallucinations / 4B 35 / 8B 50, plus 80 + 80
controls. The two control types are **interleaved** in document order so the fix
module's stratified validation subset always contains present-detections — the
guard is only meaningful if a recall-breaking fix has present cases to break
(verified: an 80-case subset carries 40 FAIL / 20 reject / 20 present).

A fix is therefore accepted only if, by paired McNemar across the whole mixed
batch, it flips more hallucinations (absent→No) than it breaks detections
(present→No). A skeptical prompt that just biases toward "No" nets out near zero
or negative and is correctly rejected.

## 3. Observation-only protocol

The `ExperimentProtocol.description` states only the observable behaviour — the
model answers "Yes" for an absent object, and present-object questions are
mostly answered "Yes" correctly (so suppressing those is not an improvement). It
names **no mechanism**: no co-occurrence prior, no language prior, no layers, no
suppression, no DeCo. The loop must discover any such explanation itself. (This
follows the same answer-no-leak rule applied to `deco_pope` / `deco_miss`.)

## 4. What "closing the loop" looks like here

- **Detect/analyse**: the loop selects its own analyzers; the interesting M5
  question is whether the hallucinations carry any intermediate "No" signal
  (recoverable, DeCo-style) or are confident all the way through (irrecoverable).
- **Repair**: `run_fix` proposes interventions up to L3b, validated on the mixed
  subset. **Success = a fix with a positive, significant net flip that does NOT
  break present detections. Honest failure = no such fix → escalate to L4.**
  Both are valid; the example tests whether the loop reaches the right verdict
  and resists a degenerate recall-destroying "fix".

Reference analyses live OUTSIDE this repository so the loop's coding agents
cannot read them.

## 4b. Run record — 2026-06-13 (2B, opus-4-8-low judge/coder)

201 cases (41 hallucinations + 80 rejects + 80 present-detections, interleaved),
zero OOM. The loop closed every stage AND the no-free-lunch guard demonstrably
worked.

- **M5 verified 3/4**, converged in 2 cycles (`criteria_met`). Hypotheses were
  self-critical: the top one questioned whether its own layer-25 linear probe
  was just reading answer polarity (a confound), another flagged FAIL-slice
  heterogeneity, a third claimed the false-detections are a recoverable
  decision-threshold problem.
- **M4 surgery ran a real experiment** (returncode 0, sandbox_runs 1) targeting
  the probe-leakage self-critique and **REFUTED** the over-claim (`verdict=0.0`)
  — an honest internal check that its probe signal was not the "model knows it's
  wrong" evidence it first looked like.
- **run_fix found a VALIDATED, guarded fix**: `fixed=True`, best =
  `coded_pipeline` (L3a, opus-written), **22 hallucinations fixed / 1 detection
  broken / +0.26 / significant**. The pipeline is multi-view evidence
  consistency: re-ask under attention-crop + zoom + sharpen and require a
  majority of views to confirm presence — "a hallucinated Yes is inconsistent
  across views; a real object survives magnification". `attention_guided_crop`
  (24/5) and `upscale_verify_evidence` (13/1) also validated.
- **The guard caught the degenerate fixes**: the purely skeptical candidates
  were rejected — `skeptical_majority_vote` flipped **0 hallucinations and broke
  4 detections** (−0.05), `evidence_grounded_skeptical` 2 fixed / 4 broken
  (−0.025). A "say No more often" fix nets negative once present-detections are
  in the scoring, exactly as designed.

**Reading vs the reference**: confident hallucinations carry no intermediate
"No" to recover, so DeCo-style *logit rescoring* would not help — but
*evidence-amplifying* interventions (look closer / demand cross-view
consistency) reduce false-Yes without destroying recall, and the loop found one.
This complements deco_miss (misses fixed by surfacing evidence) and contrasts
with deco_pope's mixed run (no guard, honest L4): on a guarded, focused
hallucination slice the loop produces a real, recall-preserving repair.

## 5. Files

```
examples/deco_hallu/
├── DESIGN.md          this file
├── build_cases.py     offline slice -> hallucination FAIL + interleaved controls
├── run.py             load -> observation-only protocol -> VLDiagnoseLoop -> run_fix
├── config.yaml        opus-4-8-low judge/coder, L3b fix tier, 80-case validation
└── data/cases/        frozen manifests (committed; images shared w/ deco_pope)
```
