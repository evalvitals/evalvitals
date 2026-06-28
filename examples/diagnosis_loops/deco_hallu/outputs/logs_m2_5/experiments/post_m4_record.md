# Experiment — M4  (SUPPORTED)

**Hypothesis:** Over-detection is driven by a language/co-occurrence prior ("Yes" bias for objects commonly co-present with the scene) that is upstream of vision, so it persists even when attention is non-pathological and is immune to prompt reformatting.
**Failure mode:** hallucination (language_prior)

**Verdict:** 1.0    **Fixed:** True

## Metrics
- n_cases: 64.0
- metric_a: 0.7031
- metric_b: 0.9219
- metric_c: 0.6875
- verdict: 1.0

## How it ran
- provider: claude_code
- return code: 0
- timed out: False
- LLM calls: 0
- sandbox runs: 1

## Files
- `experiments/post_m4_experiment.py`  — experiment.py
- `experiments/post_m4_stdout.txt`  — stdout
- `experiments/post_m4_agent_thinking.txt`  — cli_raw_output
- `experiments/post_m4_phase_log.txt`  — validation_log
