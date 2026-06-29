"""Host-side, deterministic chart rendering for exploratory artifacts.

The explorer (a CLI coding agent) only *proposes* chart specifications — a small
dict ``{name, kind, data, x, y, title}`` where ``data`` points to a CSV table it
wrote. This module renders those specs to PNG **on the host**, from the spec and
the CSV alone: it NEVER executes LLM-authored plotting code. That keeps the
visual layer auditable and reproducible (same spec + same CSV → same figure).

Rendering is best-effort and never raises into a pipeline:
- ``matplotlib`` missing → every spec is returned with a ``render_skipped`` note
  and a free-text ``description``; callers fall back to the text.
- A spec that can't be rendered (missing CSV, unknown columns, empty table) is
  returned annotated, not dropped — discovery output is preserved.

This is the single visualization core shared by every single-shot entry
(``evalvitals explore``, the fused pipeline, and the in-loop M3 view).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KINDS = {"bar", "line", "scatter", "timeseries"}


def render_chart_specs(
    charts: list[dict[str, Any]] | None,
    tables_dir: str | Path | None,
    out_dir: str | Path,
) -> list[dict[str, Any]]:
    """Render each chart spec to a PNG under *out_dir*, deterministically.

    Args:
        charts:     Explorer chart specs. Each is ``{name, kind, data, x, y,
                    title}`` where ``data`` is a CSV path (relative to
                    *tables_dir*, or absolute).
        tables_dir: Directory the spec ``data`` CSVs live in (the explorer's
                    ``tables/``). ``None`` resolves CSVs relative to *out_dir*'s
                    parent.
        out_dir:    Directory to write ``figures/`` PNGs into (created lazily).

    Returns:
        A new list of chart dicts, each a shallow copy of the input spec plus:
        - ``figure_path``: absolute PNG path when rendered;
        - ``description``: a one-line textual summary (always set);
        - ``render_skipped``: reason string when no PNG was produced.

        The input list is not mutated; ordering is preserved.
    """
    specs = [dict(c) for c in (charts or []) if isinstance(c, dict)]
    if not specs:
        return specs

    out_dir = Path(out_dir)
    tdir = Path(tables_dir) if tables_dir else None
    plt = _import_matplotlib()
    style = _load_nature_style()

    rendered: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        rows, load_err = _load_table(spec.get("data"), tdir, out_dir)
        x = spec.get("x")
        y = spec.get("y")
        spec.setdefault("description", _describe(spec, rows, x, y))

        if plt is None:
            spec["render_skipped"] = "matplotlib not installed (pip install 'evalvitals[viz]')"
            rendered.append(spec)
            continue

        ok, reason = _can_render(rows, x, y)
        if not ok:
            spec["render_skipped"] = load_err or reason
            rendered.append(spec)
            continue

        try:
            png = _render_one(plt, spec, rows, x, y, out_dir, idx, style)
            spec["figure_path"] = str(png)
            spec.pop("render_skipped", None)
        except Exception as exc:  # rendering must never sink the caller
            logger.warning("render_chart_specs: chart %d failed: %s", idx, exc)
            spec["render_skipped"] = f"render error: {exc}"
        rendered.append(spec)

    return rendered


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless, no display needed
        # Arial/Helvetica are usually absent on servers; DejaVu Sans is the
        # deterministic fallback. Silence the per-figure findfont warning spam.
        import logging as _logging

        import matplotlib.pyplot as plt
        _logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
        return plt
    except Exception:
        return None


# --- nature-figure design tokens (single source of truth: the vendored skill) ---
# Mirrored here as a deterministic fallback; the live values are read from the
# vendored skill when present so the host-rendered charts and any agent-authored
# figures share ONE visual language.
_NATURE_RC_FALLBACK: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}
_NATURE_COLORS_FALLBACK = ["#0F4D92", "#8BCF8B", "#B64342", "#42949E", "#9A4D8E", "#CFCECE"]
# On-screen single panels: keep nature's clean frame but a readable size (the
# skill reserves font.size 7 for dense multi-panel print figures).
_SCREEN_RC: dict[str, Any] = {
    "font.size": 9.0, "axes.titlesize": 11.0, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
}

_SKILL_DIR = Path(__file__).resolve().parent / "skills" / "nature-figure"
_NATURE_STYLE_CACHE: dict[str, Any] | None = None


def _balanced(text: str, open_idx: int, open_ch: str, close_ch: str) -> str | None:
    """Return the balanced ``open_ch..close_ch`` group starting at *open_idx*."""
    depth = 0
    for j in range(open_idx, len(text)):
        if text[j] == open_ch:
            depth += 1
        elif text[j] == close_ch:
            depth -= 1
            if depth == 0:
                return text[open_idx:j + 1]
    return None


def _literal_after(text: str, marker: str, open_ch: str, close_ch: str):
    """ast.literal_eval the first balanced literal after *marker* (or None)."""
    import ast

    k = text.find(marker)
    if k < 0:
        return None
    o = text.find(open_ch, k)
    if o < 0:
        return None
    blob = _balanced(text, o, open_ch, close_ch)
    if blob is None:
        return None
    try:
        return ast.literal_eval(blob)
    except Exception:
        return None


def _load_nature_style() -> dict[str, Any]:
    """Render style sourced from the vendored nature-figure skill (rcParams +
    PALETTE), with a hardcoded fallback. Cached. Never raises."""
    global _NATURE_STYLE_CACHE
    if _NATURE_STYLE_CACHE is not None:
        return _NATURE_STYLE_CACHE

    rc = dict(_NATURE_RC_FALLBACK)
    colors = list(_NATURE_COLORS_FALLBACK)
    try:
        import re

        py = (_SKILL_DIR / "static" / "fragments" / "backend" / "python.md").read_text(encoding="utf-8")
        loaded_rc = _literal_after(py, "rcParams.update(", "{", "}")
        if isinstance(loaded_rc, dict):
            rc = {str(k): v for k, v in loaded_rc.items() if k != "font.size"}

        api = (_SKILL_DIR / "references" / "api.md").read_text(encoding="utf-8")
        palette = _literal_after(api, "PALETTE = {", "{", "}")
        m = api.find("DEFAULT_COLORS")
        if isinstance(palette, dict) and m >= 0:
            blob = _balanced(api, api.find("[", m), "[", "]") or ""
            keys = re.findall(r'PALETTE\["([^"]+)"\]', blob)
            resolved = [palette[k] for k in keys if k in palette]
            if resolved:
                colors = resolved
    except Exception:
        pass  # fall back to the mirrored tokens

    _NATURE_STYLE_CACHE = {"rc": {**rc, **_SCREEN_RC}, "colors": colors}
    return _NATURE_STYLE_CACHE


def _load_table(
    data: Any, tables_dir: Path | None, out_dir: Path
) -> tuple[list[dict[str, str]] | None, str]:
    """Resolve and read the spec's CSV into a list of row dicts.

    Returns ``(rows, "")`` on success, ``(None, reason)`` otherwise.
    """
    if not data:
        return None, "no data table referenced"
    if not isinstance(data, str):
        # inline list-of-dicts table (rare; the explorer usually writes CSV)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return [{str(k): str(v) for k, v in r.items()} for r in data], ""
        return None, "unsupported inline data shape"

    path = Path(data)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if tables_dir is not None:
            candidates.append(tables_dir / path)            # data="tables/foo.csv"
            candidates.append(tables_dir / path.name)       # data="foo.csv"
            candidates.append(tables_dir / "tables" / path.name)  # tables_dir=sandbox
        candidates.append(out_dir / path)
        candidates.append(out_dir / "tables" / path.name)
        candidates.append(out_dir.parent / path)
    resolved = next((p for p in candidates if p.exists()), None)
    if resolved is None:
        return None, f"CSV not found: {data}"
    if resolved.suffix.lower() != ".csv":
        return None, f"not a CSV: {resolved.name}"
    try:
        # errors="replace" tolerates non-UTF-8 cells (VL logs carry latin-1/binary
        # text); the broad except covers csv.Error (oversized field) so a malformed
        # CSV degrades to a (None, reason) annotation instead of raising — the
        # module's "never raises into a pipeline" contract.
        with resolved.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, UnicodeError, csv.Error) as exc:
        return None, f"could not read {resolved.name}: {exc}"
    if not rows:
        return None, "empty CSV"
    return rows, ""


def _can_render(rows: list[dict[str, str]] | None, x: Any, y: Any) -> tuple[bool, str]:
    if not rows:
        return False, "no table data"
    if not x or not y:
        return False, "spec missing x or y column"
    cols = rows[0].keys()
    if x not in cols:
        return False, f"x column {x!r} not in table"
    if y not in cols:
        return False, f"y column {y!r} not in table"
    return True, ""


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_one(plt, spec, rows, x, y, out_dir, idx, style) -> Path:
    kind = str(spec.get("kind", "bar")).lower()
    if kind not in _KINDS:
        kind = "bar"

    xs_raw = [r.get(x, "") for r in rows]
    ys = [_to_float(r.get(y, "")) for r in rows]
    # Drop rows whose y is non-numeric so the plot stays well-defined.
    pairs = [(xr, yv) for xr, yv in zip(xs_raw, ys) if yv is not None]
    if not pairs:
        raise ValueError(f"y column {y!r} has no numeric values")
    xs_raw, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    xs_num = [_to_float(v) for v in xs_raw]
    x_is_num = all(v is not None for v in xs_num)

    rc = (style or {}).get("rc", {})
    colors = (style or {}).get("colors") or _NATURE_COLORS_FALLBACK
    primary = colors[0]
    title = str(spec.get("title") or spec.get("name") or f"chart_{idx}")

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    name = _safe_filename(spec.get("name") or spec.get("title") or f"chart_{idx}")
    png = figures / f"{idx:02d}_{name}.png"

    # Apply the nature-figure style in a scoped rc_context (no global leak; fully
    # deterministic → same spec + CSV yields byte-identical PNGs).
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        if kind == "scatter":
            ax.scatter(xs_num if x_is_num else range(len(xs_raw)), ys, s=26,
                       color=primary, edgecolor="white", linewidth=0.4, zorder=3)
            if not x_is_num:
                ax.set_xticks(range(len(xs_raw)))
                ax.set_xticklabels([str(v) for v in xs_raw], rotation=45, ha="right")
        elif kind in {"line", "timeseries"}:
            ax.plot(xs_num if x_is_num else range(len(xs_raw)), ys, marker="o",
                    color=primary, linewidth=1.8, markersize=5, zorder=3)
            if not x_is_num:
                ax.set_xticks(range(len(xs_raw)))
                ax.set_xticklabels([str(v) for v in xs_raw], rotation=45, ha="right")
        else:  # bar
            positions = range(len(xs_raw))
            ax.bar(positions, ys, color=primary, width=0.72, zorder=3)
            ax.set_xticks(list(positions))
            ax.set_xticklabels([str(v) for v in xs_raw], rotation=45, ha="right")

        if kind in {"bar", "line", "timeseries"}:
            ax.grid(axis="y", linewidth=0.6, alpha=0.25, zorder=0)
            ax.set_axisbelow(True)

        ax.set_xlabel(str(x))
        ax.set_ylabel(str(y))
        ax.set_title(title, fontweight="bold")
        fig.tight_layout()
        # Pin metadata so the same spec + CSV yields byte-identical PNGs.
        fig.savefig(png, dpi=130, metadata={"Software": "evalvitals", "Creation Time": None})
        plt.close(fig)
    return png


def _safe_filename(name: Any) -> str:
    text = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(name)).strip("_")
    return (text or "chart")[:48]


def _describe(spec: dict[str, Any], rows, x, y) -> str:
    """One-line textual summary of a chart, used when the image can't render
    and as the caption M3 sees alongside the attached PNG."""
    existing = spec.get("description")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    kind = str(spec.get("kind", "bar")).lower()
    title = str(spec.get("title") or spec.get("name") or "chart")
    if x and y:
        body = f"{kind} of {y} by {x}"
    elif x:
        body = f"{kind} over {x}"
    else:
        body = f"{kind} chart"
    n = len(rows) if rows else 0
    return f"{title}: {body}" + (f" ({n} rows)" if n else "")
