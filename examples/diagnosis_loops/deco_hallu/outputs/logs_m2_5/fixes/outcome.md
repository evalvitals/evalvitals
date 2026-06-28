# Fix outcome

**Result:** NOT FIXED
**Max tier allowed:** L3b
**Best candidate:** —
**Recommendation:** escalate to L4 — verified hypotheses route to L4 (L4 (parameter space: dataset construction + fine-tuning)), beyond the allowed L3b
**Re-diagnose:** 'cooccurrence_debias' repaired 3 case(s) but broke 1 — the failure mode is likely subset-specific. Re-diagnose: what distinguishes the helped cases from the hurt ones, and gate the fix on that predicate.

## Attempts (10)

| # | tier | candidate | verdict | n_fixed | n_broken | coverage | effect | sig |
|---|------|-----------|---------|---------|----------|----------|--------|-----|
| 01 | L1 | evidence_grounding | partial | 3 | 0 | 100% | 0.125 | no |
| 02 | L1 | cooccurrence_debias | partial | 3 | 1 | 100% | 0.0833 | no |
| 03 | L1 | skeptical_default_no | partial | 3 | 0 | 100% | 0.125 | no |
| 04 | L2 | skeptical_vote | partial | 3 | 0 | 100% | 0.125 | no |
| 05 | L2 | upscale_evidence_vote | partial | 4 | 0 | 100% | 0.1667 | no |
| 06 | L2 | zoom_skeptic_vote | partial | 4 | 0 | 100% | 0.1667 | no |
| 07 | L3a | coded_pipeline | partial | 4 | 0 | 100% | 0.1667 | no |
| 08 | L3b | visual_embedding_boost | partial | 1 | 0 | 100% | 0.0417 | no |
| 09 | L3b | visual_embedding_boost | no_effect | 1 | 1 | 100% | 0.0 | no |
| 10 | L3b | visual_embedding_boost | partial | 1 | 0 | 100% | 0.0417 | no |

Each attempt's full record.md + result.json is in its own folder above (`NN_<tier>_<name>/`).