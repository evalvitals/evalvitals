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
