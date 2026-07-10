from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_AGENT = ROOT / "evalvitals" / "eval_agent"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class_names(path: Path) -> set[str]:
    return {node.name for node in ast.walk(_tree(path)) if isinstance(node, ast.ClassDef)}


def _function_names(path: Path) -> set[str]:
    return {node.name for node in ast.walk(_tree(path)) if isinstance(node, ast.FunctionDef)}


def test_cli_agent_is_compatibility_facade_only():
    path = EVAL_AGENT / "cli_agent.py"
    source = path.read_text(encoding="utf-8")

    assert len(source.splitlines()) <= 90
    assert _class_names(path) == set()
    assert _function_names(path) == set()
    assert "Compatibility facade" in source


def test_provider_and_model_implementations_live_in_their_packages():
    facade = (EVAL_AGENT / "cli_agent.py").read_text(encoding="utf-8")
    assert "class ClaudeCodeAgent" not in facade
    assert "class CodexAgent" not in facade
    assert "class AgyModel" not in facade
    assert "class ClaudeModel" not in facade

    assert "ClaudeCodeAgent" in _class_names(EVAL_AGENT / "providers" / "claude_code.py")
    assert "CodexAgent" in _class_names(EVAL_AGENT / "providers" / "codex.py")
    assert "AgyModel" in _class_names(EVAL_AGENT / "models" / "agy.py")
    assert "ClaudeModel" in _class_names(EVAL_AGENT / "models" / "claude.py")


def test_codegen_runner_is_the_stage_cli_invocation_boundary():
    production_paths = [
        *(ROOT / "evalvitals" / "analysis").glob("*.py"),
        *(EVAL_AGENT / "stages").glob("*.py"),
        EVAL_AGENT / "nl_runner.py",
    ]

    offenders: list[str] = []
    for path in production_paths:
        source = path.read_text(encoding="utf-8")
        if "create_cli_agent" in source or ".run(prompt" in source:
            if "CodegenRunner" not in source:
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_skill_policy_is_not_embedded_in_explorer_or_provider_adapters():
    explorer = (ROOT / "evalvitals" / "analysis" / "explorer.py").read_text(
        encoding="utf-8"
    )
    prompt_module = (EVAL_AGENT / "prompts" / "explorer.py").read_text(encoding="utf-8")

    assert "def _skills_hint" not in explorer
    assert "def _skills_hint" not in prompt_module
    assert "def skills_hint" in (EVAL_AGENT / "skills" / "prompt_policy.py").read_text(
        encoding="utf-8"
    )


def test_stage_prompt_templates_live_in_prompt_modules():
    prompt_modules = {
        "case_discovery.py",
        "diagnosis.py",
        "experiment_writer.py",
        "fix_agent.py",
        "hypothesis_tester.py",
        "nl_runner.py",
        "probe_agent.py",
        "probe_generator.py",
        "stats_agent.py",
        "stats_tool_generator.py",
        "whitebox_probe_generator.py",
    }
    assert prompt_modules <= {p.name for p in (EVAL_AGENT / "prompts").glob("*.py")}

    offenders: list[str] = []
    for path in (EVAL_AGENT / "stages").glob("*.py"):
        tree = _tree(path)
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                text = value.value
                if len(text) > 250 and any(
                    marker in text
                    for marker in ("You are", "Return ONLY", "Reply with ONLY")
                ):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert offenders == []
