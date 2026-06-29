"""Prompt contract text for exploratory visualization outputs."""

VISUALIZATION_OUTPUT_CONTRACT = """\
Before writing plotting code, emit a visual_plan. For every deterministic chart,
write its plotted data as CSV under tables/ and report a chart spec with
{name, kind, data, x, y, title}. The host renders chart specs from CSV only.
Also emit chart_readings that describe what a human should see and what the
chart cannot prove.
"""

