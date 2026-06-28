"""Phase A — host-side deterministic chart rendering (render_chart_specs)."""

from __future__ import annotations

import pytest

from evalvitals.analysis import charts as charts_mod
from evalvitals.analysis.charts import render_chart_specs

_HAVE_MPL = charts_mod._import_matplotlib() is not None


def _write_table(d):
    (d / "tables").mkdir()
    (d / "tables" / "t.csv").write_text("grp,val\na,3\nb,7\nc,2\n", encoding="utf-8")
    return [{"name": "g1", "kind": "bar", "data": "tables/t.csv",
             "x": "grp", "y": "val", "title": "Vals by group"}]


def test_description_always_set_and_input_not_mutated(tmp_path):
    charts = _write_table(tmp_path)
    original = dict(charts[0])
    out = render_chart_specs(charts, tmp_path / "tables", tmp_path / "out")
    assert out[0]["description"]                 # always synthesized
    assert charts[0] == original                 # caller's list untouched
    assert out is not charts


def test_empty_input_returns_empty():
    assert render_chart_specs(None, None, "/tmp/x") == []
    assert render_chart_specs([], None, "/tmp/x") == []


def test_missing_csv_is_annotated_not_dropped(tmp_path):
    charts = [{"name": "m", "kind": "bar", "data": "tables/none.csv", "x": "g", "y": "v"}]
    out = render_chart_specs(charts, tmp_path / "tables", tmp_path / "out")
    assert len(out) == 1
    assert "not found" in out[0]["render_skipped"].lower()
    assert "figure_path" not in out[0]


def test_graceful_fallback_when_matplotlib_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(charts_mod, "_import_matplotlib", lambda: None)
    charts = _write_table(tmp_path)
    out = render_chart_specs(charts, tmp_path / "tables", tmp_path / "out")
    assert "matplotlib" in out[0]["render_skipped"]
    assert "figure_path" not in out[0]
    assert out[0]["description"]                 # text fallback still present


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_renders_png_for_each_kind(tmp_path):
    from pathlib import Path

    _write_table(tmp_path)
    specs = [
        {"name": "bar", "kind": "bar", "data": "tables/t.csv", "x": "grp", "y": "val"},
        {"name": "line", "kind": "line", "data": "tables/t.csv", "x": "grp", "y": "val"},
        {"name": "sca", "kind": "scatter", "data": "tables/t.csv", "x": "val", "y": "val"},
    ]
    out = render_chart_specs(specs, tmp_path / "tables", tmp_path / "out")
    for spec in out:
        assert Path(spec["figure_path"]).exists()
        assert "render_skipped" not in spec


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_render_is_deterministic(tmp_path):
    from pathlib import Path

    charts = _write_table(tmp_path)
    a = render_chart_specs(charts, tmp_path / "tables", tmp_path / "a")[0]["figure_path"]
    b = render_chart_specs(charts, tmp_path / "tables", tmp_path / "b")[0]["figure_path"]
    # Same spec + same CSV -> byte-identical PNG (pinned metadata, no timestamp).
    assert Path(a).read_bytes() == Path(b).read_bytes()


@pytest.mark.skipif(not _HAVE_MPL, reason="matplotlib not installed")
def test_unknown_x_column_skips_without_raising(tmp_path):
    _write_table(tmp_path)
    out = render_chart_specs(
        [{"name": "bad", "kind": "bar", "data": "tables/t.csv", "x": "nope", "y": "val"}],
        tmp_path / "tables", tmp_path / "out",
    )
    assert "not in table" in out[0]["render_skipped"]
