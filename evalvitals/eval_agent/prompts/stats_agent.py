"""Prompt templates for M2 statistical analysis."""

_ANALYSIS_PROMPT = """\
You are an expert in ML failure analysis for vision-language models and agentic systems.

Experiment protocol:
{protocol_text}

Task domain: {task_domain}

Analyzer summary:
{narrative}

Raw per-analyzer findings (JSON):
{findings_json}

Based on the protocol and the findings above, write:

CONCLUSION: <one paragraph — what is the root cause of failures, given what was tested>
EVIDENCE_CHAIN:
- <step 1: which analyzer and metric first caught your attention, and why>
- <step 2: how it connects to the protocol's stated failure patterns>
- <step 3: any corroborating or contradicting signals from other analyzers>
QUALITATIVE:
- <observation 1: a pattern not captured by numbers alone>
- <observation 2: anything unexpected or surprising>

Keep each bullet to one sentence. If the model looks healthy given the protocol, say so clearly."""

_TOOL_SELECT_PROMPT = """\
You are selecting statistical tools to test why a model fails, given the data on hand.

Experiment protocol:
{protocol_text}

Available statistical tools:
{tool_catalog}

Data shape available for testing:
{data_shape}

Pick the tools whose data requirements are satisfied and that best test the \
protocol's question. Return ONLY a JSON object, no other text:
{{"tools": ["name1", "name2", ...], "rationale": "one sentence"}}"""
