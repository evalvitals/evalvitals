"""Loader for the vendored ``eval-chart-style`` theme (``eval_viz_theme.py``).

The skill ships as package-data under ``skills/eval-chart-style/assets/``, a
directory that is NOT an importable package (hyphenated name). This loads that
asset by path, once, and hands it to the two host consumers:

- ``charts.py`` — the static-PNG renderer, which reads its ``PALETTE`` and
  ``matplotlib_rcparams()`` (no plotly needed; the asset imports plotly lazily).
- ``dashboard_app.py`` — the Streamlit dashboard, which calls its plotly chart
  builders (forest / violin / logistic / scatter / class-balance).

It returns ``None`` gracefully when the asset (or, for the plotly path, plotly
itself) is unavailable, so callers can fall back to matplotlib.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_ASSET = (
    Path(__file__).resolve().parent
    / "skills" / "eval-chart-style" / "assets" / "eval_viz_theme.py"
)

_MODULE_CACHE: Any = None
_LOADED = False


def load_viz_theme():
    """Load and cache the ``eval_viz_theme`` module, or ``None`` if unavailable.

    Loading does NOT require plotly — the asset imports plotly lazily inside its
    builders — so ``charts.py`` can read the palette/rcParams even under the
    ``[viz]``-only (matplotlib) install. The plotly builders raise when called
    without plotly; the dashboard guards that via :func:`load_plotly_theme`.
    """
    global _MODULE_CACHE, _LOADED
    if _LOADED:
        return _MODULE_CACHE
    _LOADED = True
    try:
        spec = importlib.util.spec_from_file_location(
            "evalvitals_eval_viz_theme", _ASSET
        )
        if spec is None or spec.loader is None:
            _MODULE_CACHE = None
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE_CACHE = module
    except Exception:
        _MODULE_CACHE = None
    return _MODULE_CACHE


def load_plotly_theme():
    """Return the theme module ONLY if plotly is importable and the template
    registers (``viz.apply()`` succeeds); otherwise ``None``.

    This is the dashboard's plotly-availability gate: a non-``None`` result means
    the plotly builders are safe to call and the ``evalvitals`` template is set.
    """
    viz = load_viz_theme()
    if viz is None:
        return None
    try:
        viz.apply()  # registers the plotly template; raises if plotly is missing
    except Exception:
        return None
    return viz


def reset_cache() -> None:
    """Clear the module cache (used by tests that swap the asset)."""
    global _MODULE_CACHE, _LOADED
    _MODULE_CACHE = None
    _LOADED = False
