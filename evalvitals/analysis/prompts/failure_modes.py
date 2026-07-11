"""Prompt template for naming failure-mode clusters."""

from __future__ import annotations

NAME_CLUSTERS_PROMPT = """\
You are an ML failure-analysis expert. Below are groups of failing cases that
were clustered by similarity. For each cluster, propose a short, specific
failure-mode name and a one-sentence description of what these cases have in
common.

{clusters_block}

Reply with EXACTLY ONE JSON array, one object per cluster listed above:
[{{"cluster_id": <int>, "name": "<short_snake_case_tag>", "description": "<one sentence>"}}, ...]

Return ONLY the JSON array — no prose outside it."""
