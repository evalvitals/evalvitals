# Quickstart

This page shows the common ways to run EvalVitals.

## Bring Your Own Model

If you already have a loaded Hugging Face causal LM, wrap it — no registry key
needed. The wrapped model is the same object `evalvitals.load(...)` returns, so
every capability-compatible analyzer works on it.

```python
import evalvitals
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalvitals.analyzers.lens.logit_lens import LogitLensAnalyzer

model = AutoModelForCausalLM.from_pretrained("my-org/my-llama")
tokenizer = AutoTokenizer.from_pretrained("my-org/my-llama")

wrapped = evalvitals.wrap(model, tokenizer)
result = LogitLensAnalyzer().run(wrapped, "The capital of France is")
print(result.summary())
```

Capabilities (attention, hidden states, logits, …) are inferred from the live
model. Attention capture needs eager attention; `wrap` switches the model to it
when it can, otherwise load with `attn_implementation="eager"`. White-box capture
currently supports text decoder-only models (VLM capture is Stage 2).

## One-Liner Model Load

```python
import evalvitals
from evalvitals.analyzers.attention.summary import AttentionAnalyzer

model = evalvitals.load("qwen2.5-7b-instruct")
result = AttentionAnalyzer(layer=-1, top_k=5).run(
    model,
    "The Eiffel Tower is in",
)

print(result.summary())
print(result.findings)
```

By default, `evalvitals.load` uses the `hf_local` backend unless the spec is
API-only.

## Config-Driven Run

```yaml
model: qwen2.5-7b-instruct
analysis: attention
analysis_kwargs:
  layer: -1
  top_k: 5
```

```python
from evalvitals import load_config, run

config = load_config("configs/qwen_attention.yaml")
result = run(config, "The Eiffel Tower is in")
```

## Explicit Backend Selection

Use `compose` when you want to control the runtime and negotiate capabilities
before model weights load.

```python
from evalvitals import Capability
from evalvitals.models import compose

model = compose(
    "qwen2.5-7b-instruct",
    "hf_local",
    want={Capability.ATTENTION},
)
```

If the selected backend cannot provide the requested capability, EvalVitals
raises a `CapabilityError` before constructing the model.

## Discovery

```python
import evalvitals

print(evalvitals.list_specs())
print(evalvitals.registry.analyzers.list())
print(evalvitals.registry.analyzers.names_compatible_with(model))
```

This is the same discovery surface intended for an automated evaluation agent.

## Convenience Shim

Models expose `call_<analysis>` methods dynamically through the analyzer
registry:

```python
result = model.call_attention(
    "The Eiffel Tower is in",
    layer=-1,
    top_k=5,
)
```

This is convenience syntax for:

```python
from evalvitals.analyzers.attention.summary import AttentionAnalyzer

result = AttentionAnalyzer(layer=-1, top_k=5).run(model, data)
```

Prefer direct analyzer construction in reusable code because it makes parameters
and dependencies explicit.

---

## Hallucination Analysis (POPE + CHAIR)

Both analyzers are black-box (`GENERATE`-only) and work with any vision-capable
model, including API endpoints.

**POPE** — yes/no object-presence probes; reports accuracy, precision, recall, F1.
Each case needs `metadata["pope_label"]` = `"yes"` or `"no"`.

> Paper: Li et al., EMNLP 2023 — <https://arxiv.org/abs/2305.10355>  
> Code: <https://github.com/AoiDragon/POPE>

```python
from evalvitals.analyzers.hallucination.pope import POPEAnalyzer
from evalvitals.core.case import CaseBatch, FailureCase, Inputs

cases = CaseBatch([
    FailureCase(inputs=Inputs(prompt="Is there a cat? Answer yes or no.", image=img),
                metadata={"pope_label": "yes"}),
    FailureCase(inputs=Inputs(prompt="Is there a plane? Answer yes or no.", image=img),
                metadata={"pope_label": "no"}),
])
result = POPEAnalyzer().run(model, cases)
# result.findings → {"accuracy": 0.83, "f1": 0.86, "yes_rate": 0.50, ...}
print(result.summary())
```

**CHAIR** — caption hallucination rate vs a fixed object vocabulary.
Each case needs `metadata["gt_objects"]` = list of gold object strings.

> Paper: Rohrbach et al., EMNLP 2018 — <https://arxiv.org/abs/1809.02156>

```python
from evalvitals.analyzers.hallucination.chair import CHAIRAnalyzer

COCO_VOCAB = ["cat", "dog", "car", "chair", ...]  # 80 COCO categories
cases = CaseBatch([
    FailureCase(inputs=Inputs(prompt="Describe the image.", image=img),
                metadata={"gt_objects": ["cat", "chair"]}),
])
result = CHAIRAnalyzer(object_vocab=COCO_VOCAB).run(model, cases)
# result.findings → {"chair_i": 0.25, "chair_s": 0.50, "n": 2}
# chair_i: mean per-caption hallucination rate; chair_s: fraction of captions with ≥1 hallucination
```

Full runnable example: `examples/hallucination/` (launch with `docker compose up`).

---

## Shapley Attribution — MM-SHAP + VL-SHAP

Both analyzers require `LOGPROBS` capability and use Monte-Carlo Shapley sampling.

**MM-SHAP** — decomposes model reliance between text tokens and the image.
`mm_score` near 1.0 ⇒ image-driven; near 0.0 ⇒ text-driven. Measures *reliance*,
not correctness.

> Paper: Parcalabescu & Frank, ACL 2022 — <https://arxiv.org/abs/2212.08158>  
> Code: <https://github.com/coastalcph/mm-shap>

```python
from evalvitals.analyzers.perturbation.mm_shap import MMShapAnalyzer
from evalvitals.core.case import CaseBatch, FailureCase, Inputs

case = FailureCase(inputs=Inputs(prompt="What color is the car?", image=img))
result = MMShapAnalyzer(n_samples=64, top_k=5).run(model, CaseBatch([case]))
# result.findings → {
#   "mm_score": 0.62,              # 0=text-reliant, 1=image-reliant
#   "text_contribution": 0.38,
#   "image_contribution": 0.62,
#   "top_text_tokens": [{"token": "color", "shapley": 0.12}, ...]
# }
```

**VL-SHAP** — spatial Shapley attribution over a grid of image regions.
Ranks which regions most influenced the model's output logprob.

> Based on: Lundberg & Lee, NeurIPS 2017 — <https://arxiv.org/abs/1705.07874>  
> Applied via MM-SHAP framework (Parcalabescu & Frank, ACL 2022)

```python
from evalvitals.analyzers.perturbation.vl_shap import VLShapAnalyzer

result = VLShapAnalyzer(n_regions=16, n_samples=64, top_k=3).run(model, CaseBatch([case]))
# result.findings → {
#   "grid_side": 4,           # 4×4 grid
#   "top_regions": [{"region": 5, "shapley": 0.31}, ...],
#   "total_abs_attribution": 1.84
# }
```

Full runnable example: `examples/mm_shap/` (launch with `docker compose up`).

---

## Logprob Entropy (black-box uncertainty)

`LogprobEntropyAnalyzer` computes sequence perplexity and per-token predictive
entropy from output-token logprobs — works on any API model that returns
`top_logprobs` (e.g. OpenAI).  No white-box access needed.

> Predictive entropy: Gal & Ghahramani, ICML 2016 — <https://arxiv.org/abs/1506.02142>  
> LLM self-knowledge: Kadavath et al. 2022 — <https://arxiv.org/abs/2207.05221>

```python
from evalvitals.analyzers.uncertainty.logprob_entropy import LogprobEntropyAnalyzer
from evalvitals.core.case import CaseBatch, FailureCase, Inputs

case = FailureCase(inputs=Inputs(prompt="The capital of France is"))
result = LogprobEntropyAnalyzer().run(model, CaseBatch([case]))
# result.findings → {
#   "n_tokens": 3,
#   "perplexity": 1.12,        # low ⇒ model is confident
#   "mean_logprob": -0.11,
#   "mean_top_entropy": 0.08,  # high ⇒ broad uncertainty at that step
#   "min_token_logprob": -0.31
# }
```

Wire an OpenAI endpoint as the `logprobs_fn`:

```python
from evalvitals.models.backends.api import parse_openai_logprobs
from evalvitals.models.backends.base import RuntimeConfig

def logprobs_fn(prompt, *, model="gpt-4o-mini", max_new_tokens=40, top_k=5, **_):
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens, logprobs=True, top_logprobs=top_k,
    )
    return parse_openai_logprobs(resp.choices[0].logprobs.content)

rt = RuntimeConfig(generate_fn=..., logprobs_fn=logprobs_fn)
```

Full runnable example: `examples/logprob_entropy/` (launch with `docker compose up`).

---

## Statistical Comparison — `stats.compare`

`compare` is the single entry point for pairwise A/B comparison. It **never**
returns a bare p-value: it always gives effect size + clustered bootstrap CI +
anytime-valid e-value + corrected reject decision.

> McNemar: McNemar (1947) <https://doi.org/10.1007/BF02295996>  
> E-values: Grünwald et al. (2022) <https://arxiv.org/abs/1906.07801>  
> e-BH FDR: Wang & Ramdas (2022) <https://arxiv.org/abs/2009.02824>

```python
from evalvitals.stats import compare

# success_a / success_b: list of bool (one per example, same order)
r = compare(success_a, success_b, paired=True, alpha=0.05,
            min_effect=0.03, cluster_by=task_ids)

print(r.summary())
# [mcnemar + e-value] effect=+0.12 (B>A) CI=[+0.04, +0.20] e=18.4 → REJECT H0

print(r.effect)      # float: mean(B) - mean(A)
print(r.ci)          # (lo, hi) 95% clustered bootstrap CI
print(r.e_value)     # float: anytime-valid evidence (reject when e ≥ 1/alpha)
print(r.reject)      # bool: corrected decision
print(r.underpowered)  # bool: True when CI width > 2 × min_effect
```

For **3+ strategies** use `compare_multiple` (Friedman omnibus + Nemenyi post-hoc):

> Friedman + Nemenyi: Demšar (2006) <https://jmlr.org/papers/v7/demsarar06a.html>

```python
from evalvitals.stats import compare_multiple

mr = compare_multiple({"A": success_a, "B": success_b, "C": success_c}, alpha=0.05)
print(mr.reject_global)      # bool: at least one strategy differs
print(mr.avg_ranks)          # {"A": 2.1, "B": 1.7, "C": 2.2}
print(mr.significant_pairs)  # [("A", "B"), ...] — pairs that pass Nemenyi CD
```

Full runnable example: `examples/stats_compare/` (no API key, `docker compose up`).

---

## AutoDiagnoseLoop — automated false-attribution pipeline

`AutoDiagnoseLoop` closes the analysis→diagnosis→intervention cycle automatically.
It needs a **judge model** (any instruction-following model with `GENERATE`) and
the model under evaluation.

```python
from evalvitals import Capability
from evalvitals.eval_agent import AutoDiagnoseLoop, DiagnosisAgent
from evalvitals.models import compose
from evalvitals.models.backends.base import RuntimeConfig

# 1. The model under evaluation (local VLM with white-box access)
model = compose("qwen3-vl-8b-instruct", "hf_local", want={Capability.ATTENTION})

# 2. A capable judge (API model — needs only GENERATE)
judge = compose("qwen3-8b", "api", RuntimeConfig(generate_fn=my_generate_fn))

# 3. Build the loop
loop = AutoDiagnoseLoop(
    model=model,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    max_cycles=3,     # max M1→M4 iterations
    max_analyzers=4,  # analyzers per cycle (ranked by diagnostic priority)
)

# 4. Run on your failure cases
report = loop.run(failure_cases)

print(report.resolved)                        # True if an intervention fixed it
for h in report.final_hypotheses:
    print(h.statement, "→", h.status)        # SUPPORTED / REFUTED / INCONCLUSIVE
print(report.final_results.keys())            # analyzers run in the last cycle
```

To persist the run (event log, figures, report files, fix/experiment trial
folders), wrap the loop in a `RunContext` — see "Log and persist a diagnosis
run" in [Extending](extending.md).

### Analysis-only mode

Omit `diagnosis_agent` to run M1+M2 only — useful when you want the probe's
ranked analysis without automated hypothesis generation:

```python
loop = AutoDiagnoseLoop(model=model, max_analyzers=5)
report = loop.run(cases)
for name, result in report.final_results.items():
    print(name, result.summary())
```

### Controlling analyzer selection

`StrategyProbe` automatically detects model kind and ranks analyzers:

```python
from evalvitals.eval_agent import StrategyProbe, ModelKind

probe = StrategyProbe()
probe.detect_kind(model)                     # ModelKind.VLM / AGENT / LLM
probe.select(model, max_analyzers=4)         # e.g. ["pope", "chair", "attention", "mm_shap"]
```

### Custom intervention (SurgeryAgent)

Override the default label-correlation verification with your own logic:

```python
from evalvitals.eval_agent import SurgeryAgent, InterventionResult, HypothesisStatus

def my_verify(hypothesis, model, results, data):
    # domain-specific logic — return True when fixed
    fixed = run_my_intervention(model, data)
    return InterventionResult(
        hypothesis=hypothesis,
        status=HypothesisStatus.SUPPORTED if fixed else HypothesisStatus.INCONCLUSIVE,
        fixed=fixed,
        evidence={"custom": True},
    )

loop = AutoDiagnoseLoop(
    model=model,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    surgery_agent=SurgeryAgent(verify_fn=my_verify),
)
```

Or trigger a **param sweep** — re-run analyzers with modified settings to compare
before/after findings:

```python
loop = AutoDiagnoseLoop(
    model=model,
    diagnosis_agent=DiagnosisAgent(judge=judge),
    surgery_agent=SurgeryAgent(analyzer_params={"attention": {"top_k": 20}}),
)
```

---

## Eval Agent — Pre-Registered A/B Loop

`EvalOrchestrator` enforces selective-inference safety: mine on `explore`,
pre-register a falsifiable hypothesis (hash + timestamp) **before** unblinding,
test once on `validate`, lock `confirm` for the final report.

```python
from evalvitals.eval_agent import EvalOrchestrator, PreregisteredHypothesis, DataSplit

hyp = PreregisteredHypothesis(
    predicate="cluttered scenes",
    statement="chain-of-thought prompt improves accuracy on multi-step questions",
    direction="B>A",
    min_effect=0.03,
    alpha=0.05,
    split="validate",
)

orch = EvalOrchestrator(split=DataSplit(explore_frac=0.5, validate_frac=0.3))
report = orch.run(cases, hyp, strategy_a, strategy_b)

# report keys: prereg_hash, decision, effect, ci, e_value, split, hypothesis
print(report["prereg_hash"])   # proves hypothesis was fixed before unblinding
print(report["decision"])      # "REJECT H0" or "inconclusive"
print(report["effect"])        # float
```

**CounterfactualReplay** ranks agent trajectory steps by causal influence (flip-rate):

> Causal framework: Pearl (2000) *Causality*  
> Applied to NLP: Vig et al. 2020 — <https://arxiv.org/abs/2004.12265>

```python
from evalvitals.analyzers.agent.counterfactual import CounterfactualReplay

def rerun_fn(trajectory, step_idx, seed):
    # wrap your live agent + verifier here; return True/False (success)
    ...

analyzer = CounterfactualReplay(rerun_fn=rerun_fn, n_replays=5)
result = analyzer._run(model, CaseBatch([case_with_trajectory]))
# result.findings["per_case"][0]["steps"] →
#   [{"step": 1, "action": "extract_answer", "flip_rate": 0.80}, ...]
# High flip_rate ⇒ that step was causally influential.
```

Full runnable example: `examples/eval_agent/` (no API key needed, `docker compose up`).
