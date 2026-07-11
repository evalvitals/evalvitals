"""Prompt template for the standalone M3 HypothesisAgent."""

from __future__ import annotations

PROPOSE_PROMPT = """\
You are an expert data analyst. Based on the exploratory analysis below,
propose specific, falsifiable hypotheses that could explain the patterns
found. A hypothesis is a candidate explanation or mechanism — not a
restatement of a finding, and not a claim you are asked to prove here.

Question investigated: {question}

Key takeaways from the exploratory analysis (title: analysis):
{takeaways_text}

Observations:
{observations_text}

Candidate signals already noted:
{signals_text}

Propose 1-3 hypotheses. For each write exactly three lines:
HYPOTHESIS: <one-sentence falsifiable claim explaining a pattern above>
BASIS: <which takeaway(s)/signal(s) above this is grounded in>
TEST: <what evidence/analysis would confirm or refute this claim>

Do not repeat a takeaway verbatim — propose a CAUSE or MECHANISM behind what
was observed. If the findings are too thin to support any falsifiable
hypothesis, respond with exactly: NO_HYPOTHESIS"""
