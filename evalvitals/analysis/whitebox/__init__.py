"""White-box analyzers — import all so they self-register."""

from evalvitals.analysis.whitebox.activation import ActivationAnalyzer
from evalvitals.analysis.whitebox.attention import AttentionAnalyzer, AttentionResult
from evalvitals.analysis.whitebox.embedding_geometry import EmbeddingGeometryAnalyzer
from evalvitals.analysis.whitebox.probing import ProbingAnalyzer
from evalvitals.analysis.whitebox.saliency import SaliencyAnalyzer
from evalvitals.analysis.whitebox.shapley import ShapleyAnalyzer

__all__ = [
    "AttentionAnalyzer",
    "AttentionResult",
    "SaliencyAnalyzer",
    "ProbingAnalyzer",
    "ShapleyAnalyzer",
    "ActivationAnalyzer",
    "EmbeddingGeometryAnalyzer",
]
