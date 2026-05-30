"""White-box analyzers — import all so they self-register.

Torch-free stub/analyzer modules import unconditionally; ``attention`` imports
torch+numpy at module load, so it is optional — on the light (pure-API) install
it is simply absent and its analyzer is not registered.
"""

from evalvitals.analysis.whitebox.activation import ActivationAnalyzer
from evalvitals.analysis.whitebox.embedding_geometry import EmbeddingGeometryAnalyzer
from evalvitals.analysis.whitebox.probing import ProbingAnalyzer
from evalvitals.analysis.whitebox.saliency import SaliencyAnalyzer
from evalvitals.analysis.whitebox.shapley import ShapleyAnalyzer
from evalvitals.analysis.whitebox.uncertainty import (
    TokenEntropyAnalyzer,
    UncertaintyResult,
)

try:  # torch/numpy only required to RUN attention; keep import light without them
    from evalvitals.analysis.whitebox.attention import AttentionAnalyzer, AttentionResult
except ImportError:  # pragma: no cover - light install
    AttentionAnalyzer = None  # type: ignore
    AttentionResult = None  # type: ignore

__all__ = [
    "AttentionAnalyzer",
    "AttentionResult",
    "SaliencyAnalyzer",
    "ProbingAnalyzer",
    "ShapleyAnalyzer",
    "ActivationAnalyzer",
    "EmbeddingGeometryAnalyzer",
    "TokenEntropyAnalyzer",
    "UncertaintyResult",
]
