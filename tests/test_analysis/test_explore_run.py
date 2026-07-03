"""Phase 0 — single-shot explore persistence (write_report_artifacts + render).

This is the non-interactive path that replaces the retired chat REPL: it writes
the report JSON, copies figures/tables, and renders the explorer's chart specs to
PNG (host-side) so the persisted report carries each chart's figure_path.
"""

from __future__ import annotations

import json

from evalvitals.analysis.explore_run import _verdict_suffix, write_report_artifacts
from evalvitals.analysis.explorer import CandidateSignal, ExploratoryAnalysisReport
from evalvitals.viz import renderer as charts_mod

_HAVE_MPL = charts_mod._import_matplotlib() is not None


def _report_with_workdir(workdir):
    (workdir / "tables").mkdir(parents=True)
    (workdir / "tables" / "t.csv").write_text("grp,val\na,3\nb,7\n", encoding="utf-8")
    (workdir / "records.json").write_text(
        json.dumps([{"case_id": "c0", "label": "pass"}, {"case_id": "c1", "label": "fail"}]),
        encoding="utf-8",
    )
    return ExploratoryAnalysisReport(
        question="q",
        ok=True,
        observations=["small objects fail"],
        charts=[{"name": "g", "kind": "bar", "data": "tables/t.csv",
                 "x": "grp", "y": "val", "title": "Vals"}],
        caveats=["explore split only"],
        workdir=str(workdir),
        code="print('hi')",
    )


def test_write_report_artifacts_persists_and_renders(tmp_path):
    workdir = tmp_path / "wd"
    out = tmp_path / "out"
    report = _report_with_workdir(workdir)

    write_report_artifacts(report, out)

    saved = json.loads((out / "exploratory_report.json").read_text())
    assert saved["ok"] is True
    assert (out / "tables" / "t.csv").exists()         # tables copied
    assert (out / "analysis.py").exists()              # code persisted
    # raw loaded records travel with the report so the dashboard can browse them
    records = json.loads((out / "records.json").read_text())
    assert records == [{"case_id": "c0", "label": "pass"}, {"case_id": "c1", "label": "fail"}]
    chart = saved["charts"][0]
    if _HAVE_MPL:
        assert chart.get("figure_path")                # rendered host-side
        from pathlib import Path
        assert Path(chart["figure_path"]).exists()
    else:
        assert "matplotlib" in chart.get("render_skipped", "")
    assert chart["description"]                         # text fallback always present


def test_verdict_suffix_tags_descriptive_vs_adjudicated():
    descriptive = CandidateSignal(name="s", sufficient={"kind": "two_group", "a": [0], "b": [1]})
    assert _verdict_suffix(descriptive) == "  [descriptive]"

    adjudicated = CandidateSignal(name="s", host_adjudicated=True, reject=True, e_value=51.2)
    tag = _verdict_suffix(adjudicated)
    assert "REJECT H0" in tag and "e-BH family" in tag
