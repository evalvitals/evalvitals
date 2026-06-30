"""Bundled Agent Skills shipped with the package."""

from __future__ import annotations

from pathlib import Path

BUNDLED_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
SKILL_BACKENDS = {"claude_code", "antigravity"}


def bundled_skill_paths() -> list[str]:
    """Absolute paths of bundled Agent-Skill directories containing ``SKILL.md``."""
    if not BUNDLED_SKILLS_DIR.is_dir():
        return []
    return [
        str(path)
        for path in sorted(BUNDLED_SKILLS_DIR.iterdir())
        if (path / "SKILL.md").is_file()
    ]

