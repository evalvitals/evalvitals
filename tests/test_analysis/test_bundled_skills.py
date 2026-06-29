"""Bundled (package-vendored) Agent Skills travel with the repo and are applied
by `explore` on the claude/agy backends by default.

This is what makes a skill like nature-figure available to anyone who clones the
repo (or pip-installs it) — no per-machine ~/.claude setup required.
"""

from __future__ import annotations

from pathlib import Path

from evalvitals.agent_assets.skills import bundled_skill_paths
from evalvitals.analysis import explore_run


def test_nature_figure_is_bundled_in_the_package():
    paths = bundled_skill_paths()
    names = {Path(p).name for p in paths}
    assert "nature-figure" in names
    nf = next(Path(p) for p in paths if Path(p).name == "nature-figure")
    assert (nf / "SKILL.md").is_file()
    assert (nf / "LICENSE").is_file()  # Apache-2.0 attribution preserved


def test_explore_applies_bundled_skills_on_claude_by_default(monkeypatch, tmp_path):
    captured = {}

    class _FakeAgent:
        def __init__(self, *, cli_config, **kw):
            captured["cli_config"] = cli_config

        def explore_path(self, *a, **k):
            from evalvitals.analysis.explorer import ExploratoryAnalysisReport
            return ExploratoryAnalysisReport(question="q", ok=True, workdir=str(tmp_path))

    monkeypatch.setattr(explore_run, "M2ExplorerAgent", _FakeAgent)
    explore_run.run_explore(tmp_path, coder_provider="claude_code", out=tmp_path / "o")

    cfg = captured["cli_config"]
    assert any(Path(s).name == "nature-figure" for s in cfg.skills)
    assert cfg.allow_skills is True  # implied by bundled skills


def test_no_skills_flag_disables_bundled(monkeypatch, tmp_path):
    captured = {}

    class _FakeAgent:
        def __init__(self, *, cli_config, **kw):
            captured["cli_config"] = cli_config

        def explore_path(self, *a, **k):
            from evalvitals.analysis.explorer import ExploratoryAnalysisReport
            return ExploratoryAnalysisReport(question="q", ok=True, workdir=str(tmp_path))

    monkeypatch.setattr(explore_run, "M2ExplorerAgent", _FakeAgent)
    explore_run.run_explore(
        tmp_path, coder_provider="claude_code", out=tmp_path / "o", use_bundled_skills=False
    )
    assert captured["cli_config"].skills == ()


def test_non_skill_backend_does_not_vendor_skills(monkeypatch, tmp_path):
    captured = {}

    class _FakeAgent:
        def __init__(self, *, cli_config, **kw):
            captured["cli_config"] = cli_config

        def explore_path(self, *a, **k):
            from evalvitals.analysis.explorer import ExploratoryAnalysisReport
            return ExploratoryAnalysisReport(question="q", ok=True, workdir=str(tmp_path))

    monkeypatch.setattr(explore_run, "M2ExplorerAgent", _FakeAgent)
    # codex ignores skills; we don't vendor them there.
    explore_run.run_explore(tmp_path, coder_provider="codex", out=tmp_path / "o")
    assert captured["cli_config"].skills == ()
