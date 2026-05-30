"""EvalVitals core — the sklearn-like substrate.

Public contracts every other module builds on:

  Capability   the vocabulary connecting models and analyzers
  Model        analyzable model:  generate() + forward(capture)->Trace
  Analyzer     sklearn-style estimator:  Analyzer(**params).run(model, data)->Result
  FailureCase  the central data unit;  CaseBatch is a collection of them
  Result       uniform, agent-readable output (findings + artifacts)
  registry     discovery: list/match models and analyzers by capability
  Pipeline     compose analyzers
  Experiment   declarative spec the agent layer targets
"""

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.case import (
    CaseBatch,
    FailureCase,
    Inputs,
    Label,
    Provenance,
    Source,
    as_casebatch,
)
from evalvitals.core.experiment import Experiment, ExperimentRunner
from evalvitals.core.model import Model, Trace
from evalvitals.core.pipeline import Pipeline
from evalvitals.core.registry import (
    Registry,
    register_analyzer,
    register_model,
    registry,
)
from evalvitals.core.result import Result

__all__ = [
    "Capability",
    "CapabilityError",
    "Model",
    "Trace",
    "Analyzer",
    "FailureCase",
    "CaseBatch",
    "Inputs",
    "Label",
    "Provenance",
    "Source",
    "as_casebatch",
    "Result",
    "registry",
    "Registry",
    "register_model",
    "register_analyzer",
    "Pipeline",
    "Experiment",
    "ExperimentRunner",
]
