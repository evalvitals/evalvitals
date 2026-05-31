# Extending EvalVitals

EvalVitals is designed to grow through small extension points: analyzers, model
specs, backends, datasets, and statistical routines.

## Add an Analyzer

An analyzer should behave like an sklearn estimator:

- constructor parameters are explicit,
- required capabilities are declared on the class,
- `run(model, data)` returns a `Result`,
- model-specific logic is avoided whenever possible.

Skeleton:

```python
from evalvitals.core import Analyzer, Capability, Result, register_analyzer


@register_analyzer("my_analysis")
class MyAnalyzer(Analyzer):
    name = "my_analysis"
    requires = frozenset({Capability.LOGITS})

    def __init__(self, top_k: int = 5):
        super().__init__(top_k=top_k)

    def _run(self, model, cases):
        case = cases[0]
        trace = model.forward(case.inputs, capture={Capability.LOGITS})
        findings = {
            "top_k": self.top_k,
            "seq_len": trace.seq_len,
        }
        return Result(
            analyzer=self.name,
            model=repr(model),
            findings=findings,
            artifacts={},
            cases=cases,
        )
```

Guidelines:

- Use `Capability` checks instead of `isinstance(model, SomeModel)`.
- Request only the internals needed for the analysis.
- Keep heavy artifacts out of `findings`; put them in `artifacts`.
- Make `summary()` useful for humans and `findings` useful for agents.

## Use a Custom or Fine-Tuned Model

If you have a model that is already loaded in memory — a fine-tuned checkpoint,
a research model, or anything `from_pretrained` returns — use `evalvitals.wrap`
instead of registering a spec:

```python
import evalvitals
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("my-org/my-llama")
tokenizer = AutoTokenizer.from_pretrained("my-org/my-llama")

wrapped = evalvitals.wrap(model, tokenizer)

# Discover which analyzers are compatible
print(evalvitals.registry.analyzers.names_compatible_with(wrapped))

# Run any compatible analyzer
from evalvitals.analyzers.lens.logit_lens import LogitLensAnalyzer
result = LogitLensAnalyzer().run(wrapped, "The capital of France is")
```

`wrap()` infers capabilities from the live model: attention, hidden states, and
logits are available for any text decoder-only model. Attention capture requires
eager attention — `wrap` enables it automatically when the model supports it; if
not, reload with `attn_implementation="eager"`.

`wrap()` currently supports text decoder-only (causal LM) models. VLM internals
capture is Stage 2. If your model has an unusual architecture not supported by
automatic inference, add a `ModelSpec` (see below) and use `evalvitals.load`.

## Add a Model Spec

Add model identity to `evalvitals/specs.py`:

```python
_add(ModelSpec(
    key="new-model-key",
    family="new_family",
    model_type="new_model_type",
    hf_repo="org/repo",
    auto_class="AutoModelForCausalLM",
    processor_class="AutoTokenizer",
    min_transformers="4.50.0",
    module_paths=ModulePaths(decoder_layers="model.layers"),
))
```

Specs should describe architecture facts and caveats. They should not load model
weights or import heavy runtime dependencies.

## Add a Backend

A backend turns a `ModelSpec` into a concrete `Model` and declares what it can
provide.

Skeleton:

```python
from evalvitals.core import Capability
from evalvitals.models.backends.base import Backend


class MyBackend(Backend):
    kind = "my_backend"
    capabilities = frozenset({Capability.GENERATE})

    def build(self, spec, runtime):
        return MyModel(spec=spec, runtime=runtime)
```

Then register it in `evalvitals/models/backends/__init__.py`:

```python
BACKENDS["my_backend"] = MyBackend
```

Guidelines:

- Keep heavy imports inside `build` or inside the concrete model class.
- Fail early when the backend cannot support a requested spec.
- Let capabilities describe runtime behavior, not model identity.

## Add a Dataset Loader

Dataset loaders should produce `FailureCase` or `CaseBatch`. This keeps raw
benchmarks, hand-authored cases, and agent-generated cases interoperable.

```python
from evalvitals.core import FailureCase, CaseBatch


def load_cases(path) -> CaseBatch:
    cases = [
        FailureCase.from_prompt(
            "...",
            label="fail",
            metadata={"source": str(path)},
        )
    ]
    return CaseBatch(cases)
```

## Add Statistical Evaluation

Statistical routines should consume `Result` objects or collections of results.
They should avoid depending on analyzer-specific internals unless that contract
is explicit.

Good inputs:

- `Result.findings`
- experiment ids,
- case ids,
- model/backend/spec metadata,
- repeated-run measurements.

Avoid making statistical code depend on raw tensors unless the test is
specifically about tensor-level measurements.
