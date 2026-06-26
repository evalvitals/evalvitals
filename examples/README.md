# Examples

Examples are grouped by the layer they exercise:

- `analyzer_demos/` — single analyzer or analysis demos. These run one diagnostic
  capability directly, without the M1-M5 diagnosis loop.
- `m2_statistics/` — standalone statistical/M2 examples. These focus on statistical
  comparison or the public M2 analysis API outside the loop.
- `diagnosis_loops/` — full diagnosis-loop examples (`AutoDiagnoseLoop`,
  `VLDiagnoseLoop`, DeCo/Qwen scenarios, and related loop demos).

Run each example from its own directory, for example:

```bash
cd examples/analyzer_demos/qwen_attention && docker compose up
cd examples/m2_statistics/stats_compare && docker compose up
cd examples/diagnosis_loops/qwen_loop_agy && docker compose up
```
