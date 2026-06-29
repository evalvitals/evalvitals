"""Agent-Skill wiring for CLI coding backends.

Skills (e.g. a nature-figure plot-styling skill) are vendored into the sandbox
so an ``--add-dir`` Claude Code / agy run auto-discovers them, and the ``Skill``
tool is added to the allowlist so the agent may invoke them. Skills style the
agent-authored ``figures/*.png`` only — they never touch the host-rendered,
deterministic chart specs.
"""

from __future__ import annotations

from pathlib import Path

from evalvitals.analysis.explorer import _skills_hint
from evalvitals.eval_agent.cli_agent import (
    ClaudeCodeAgent,
    CliAgentConfig,
    create_cli_agent,
)


def _make_skill(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "SKILL.md").write_text(f"# {name}\nStyle figures nicely.\n", encoding="utf-8")
    (d / "style.mplstyle").write_text("axes.grid: True\n", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_skills_enabled_implied_by_skill_dirs():
    assert CliAgentConfig().skills_enabled is False
    assert CliAgentConfig(allow_skills=True).skills_enabled is True
    assert CliAgentConfig(skills=("/some/skill",)).skills_enabled is True


# ---------------------------------------------------------------------------
# vendoring: skills copied into <workdir>/.claude/skills/<name>/
# ---------------------------------------------------------------------------

def test_install_skills_vendors_into_workdir(tmp_path):
    skill = _make_skill(tmp_path, "nature-figure")
    workdir = tmp_path / "wd"
    agent = ClaudeCodeAgent(binary_path="claude", skills=[str(skill)])
    agent._install_skills(workdir)

    dest = workdir / ".claude" / "skills" / "nature-figure"
    assert (dest / "SKILL.md").exists()
    assert (dest / "style.mplstyle").exists()


def test_install_skills_missing_dir_is_skipped_not_fatal(tmp_path):
    workdir = tmp_path / "wd"
    agent = ClaudeCodeAgent(binary_path="claude", skills=["/no/such/skill"])
    agent._install_skills(workdir)  # must not raise
    assert not (workdir / ".claude" / "skills" / "skill").exists()


def test_no_skills_is_noop(tmp_path):
    workdir = tmp_path / "wd"
    ClaudeCodeAgent(binary_path="claude")._install_skills(workdir)
    assert not (workdir / ".claude").exists()


# ---------------------------------------------------------------------------
# allowlist: the Skill tool is added only when skills are enabled
# ---------------------------------------------------------------------------

def test_claude_cmd_adds_skill_tool_only_when_enabled(tmp_path):
    wd = tmp_path / "wd"

    plain = ClaudeCodeAgent(binary_path="claude")._build_cmd("hi", wd)
    allowed_plain = plain[plain.index("--allowed-tools") + 1]
    assert "Skill" not in allowed_plain

    withskill = ClaudeCodeAgent(binary_path="claude", allow_skills=True)._build_cmd("hi", wd)
    allowed = withskill[withskill.index("--allowed-tools") + 1]
    assert "Skill" in allowed
    assert "Bash" in allowed and "Write" in allowed  # base tools preserved


def test_create_cli_agent_threads_skills():
    cfg = CliAgentConfig(provider="claude_code", binary_path="claude",
                         skills=("/a/nature-figure",), allow_skills=True)
    agent = create_cli_agent(cfg)
    assert agent._skills == ["/a/nature-figure"]
    assert agent._allow_skills is True


# ---------------------------------------------------------------------------
# prompt hint: the explorer tells the agent it may use skills for figures
# ---------------------------------------------------------------------------

def test_skills_hint_empty_without_skills():
    assert _skills_hint(None) == ""
    assert _skills_hint(CliAgentConfig(provider="claude_code")) == ""


def test_skills_hint_names_the_skill_and_scopes_to_figures():
    cfg = CliAgentConfig(provider="claude_code", skills=("/x/nature-figure",))
    hint = _skills_hint(cfg)
    assert "/nature-figure" in hint
    assert "figures/" in hint
    # must NOT invite changing the data/analysis/result
    assert "styling only" in hint
