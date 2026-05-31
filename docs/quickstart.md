# Quickstart

This page shows the common ways to run EvalVitals.

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
