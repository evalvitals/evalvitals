# Examples

Examples are grouped by the layer they exercise:

- `analyzer_demos/` — single analyzer or analysis demos. These run one diagnostic
  capability directly, without the M1-M5 diagnosis loop.
- `m2_statistics/` — standalone M2 (exploratory analysis) + M3 (hypothesis
  proposal) examples via `evalvitals explore`, outside the loop.
- `diagnosis_loops/` — full diagnosis-loop examples (`AutoDiagnoseLoop`,
  `VLDiagnoseLoop`, DeCo/Qwen scenarios, and related loop demos).

Run each example from its own directory, for example:

```bash
cd examples/analyzer_demos/qwen_attention && docker compose up
cd examples/m2_statistics/synthetic_yield_explore && docker compose up
cd examples/m2_statistics/deco_hallu_explore && docker compose up
cd examples/m2_statistics/deco_hallu_explore && bash run_attn.sh          # attention-enriched variant (no GPU)
cd examples/m2_statistics/deco_hallu_explore && bash run_attn_pipeline.sh # full held-out pipeline (SKIP_FIX=1 → no GPU)
cd examples/m2_statistics/deco_hallu_explore && bash run_web.sh           # upload-a-.zip web workbench (M2+M3 per upload)
cd examples/diagnosis_loops/qwen_loop_agy && docker compose up
```

The `deco_hallu_explore` example has three runnable variants: the raw probe
data (categorical signals only); `run_attn.sh` on `data_attn_full/` — the same
606 cases enriched with per-case attention-geometry scalars for all three
checkpoints (committed with the repo), which unlocks FAIL/PASS distribution
views and cross-checkpoint attention comparisons; and `run_attn_pipeline.sh` —
the complete propose → held-out test (frozen recipes + LLM judge) →
surgery/tiered-fix arc ending in one five-tab web report (proposal,
held-out verdicts and fix each get their own tab). See its
[README](m2_statistics/deco_hallu_explore/README.md).

For the general standalone exploratory analysis workflow, see
[`docs/m2_analysis.md`](../docs/m2_analysis.md).
