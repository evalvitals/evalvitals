from __future__ import annotations

from evalvitals.reporting.stages import STAGE_SPECS, stage_specs_as_dicts
from evalvitals.viz.prompts import DASHBOARD_STORYBOARD_SYSTEM_PROMPT


def test_stage_specs_define_m1_to_m5_dashboard_roles():
    ids = [s.id for s in STAGE_SPECS]
    assert ids == ["M1", "M2", "M3", "M4", "M5"]
    rows = stage_specs_as_dicts()
    assert rows[0]["dashboard_role"].startswith("Problem Setting")
    assert "Analysis" in rows[1]["dashboard_role"]
    assert "Hypotheses" in rows[2]["dashboard_role"]


def test_dashboard_storyboard_prompt_is_three_panel_and_stage_aware():
    prompt = DASHBOARD_STORYBOARD_SYSTEM_PROMPT
    assert "Problem Setting" in prompt
    assert "Analysis" in prompt
    assert "Hypotheses & Artifacts" in prompt
    assert "M1" in prompt and "M2" in prompt and "M3-M5" in prompt
    assert "display_name" in prompt
