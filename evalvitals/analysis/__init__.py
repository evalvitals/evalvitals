"""Analysis modules.

Importing this package imports every analyzer submodule so their
``@register_analyzer`` decorators run and the registry is fully populated.
"""

from evalvitals.analysis import agent, blackbox, whitebox  # noqa: F401
from evalvitals.analysis.base import Analyzer, Result

__all__ = ["Analyzer", "Result"]
