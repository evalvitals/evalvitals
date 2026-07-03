"""General-purpose exploratory data analysis agent (LAMBDA-style).

A coding agent writes Python over arbitrary tabular records — not just M1
pass/fail diagnosis logs — EvalVitals runs it locally in a sandbox, and the
result is parsed into a structured exploratory report. The prompt adapts its
framing to whatever the data's outcome column actually is (binary,
multi-class, continuous, or none — see :func:`_framing_block`), so a dataset
with no FAIL/PASS label gets unsupervised EDA instead of an invented split.

M1 integration is one caller among others, not a special case: the diagnosis
loop's per-case records already carry a ``label`` column, which the name
heuristic in :func:`evalvitals.analysis.profile.profile_records` recognizes
as the outcome automatically, so it still gets the same binary FAIL/PASS
framing it always did (callers with an arbitrarily-named target can pass
``outcome_col=`` explicitly instead of relying on the heuristic). Findings
from this module are candidates only; use ``StatsAnalysisAgent`` for
confirmatory effect/CI/e-value/FDR verdicts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalvitals.analysis.profile import describe_outcome, profile_records
from evalvitals.eval_agent.sandbox import ExperimentSandbox, SandboxResult

if TYPE_CHECKING:
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.cli_agent import CliAgentConfig

logger = logging.getLogger(__name__)

RECORDS_FILENAME = "records.json"  # also read by explore_run.py / dashboard_app.py
_RESULT_MARKER = "EXPLORATORY_RESULT_JSON="

_GENERATE_PROMPT = """\
You are an exploratory data-analysis agent (Lambda-style): given ANY tabular
dataset — with a binary outcome, a multi-class outcome, a continuous outcome,
or no outcome at all — you write Python that discovers and charts the
structure that actually matters, adapting the story to what the data is.

Question:
{question}

A JSON file named "{input_filename}" is in the current working directory.
It contains a list of row dictionaries. Data profile (the host already
inferred column roles/dtypes and classified the outcome below — trust this
over re-guessing from raw values):
{data_profile}

Write a self-contained Python script that performs a THOROUGH, Lambda-style
exploratory analysis and PRODUCES A RICH SET OF CHARTS BY DEFAULT.

Setup:
- reads "{input_filename}" from the current working directory
- may use only local Python packages (pandas / numpy / matplotlib are fine); no
  network and no repo mutation
- the profile's "columns" block gives each column's role (id / outcome / group
  / time / predictor) and dtype; its "outcome" block ({{"present","column","kind"}})
  tells you whether there is one and what kind it is. Do not invent an outcome
  or a FAIL/PASS split when "outcome.present" is false.

{framing}

VISUAL ANALYSIS — before writing plotting code, make an explicit intermediate
visualization plan. The plan is part of the machine-readable output and should
show that YOU selected plot types from the data semantics, not from a fixed
template. Aim for 6-12 charts/plots that together tell the dataset's story.

First build a "visual_plan" list. Each item should be a dict:
  {{
    "name": "<stable artifact/chart name>",
    "display_name": "<short human title, no raw generated/probe id>",
    "question": "<what this visual answers>",
    "data_shape": "<numeric-vs-binary | numeric-vs-categorical | numeric-vs-numeric | many-numeric | paired | unsupervised | ...>",
    "plot_kind": "<chosen plot type, e.g. bar, line, scatter, box, violin, heatmap, paired_slope>",
    "fallback_kind": "<bar|line|scatter when a deterministic host chart is useful>",
    "required_columns": ["..."],
    "rationale": "<why this plot type fits the data and avoids misleading summaries>"
  }}

Use these decision principles:
  - categorical/binary outcome: rate/count bar with n annotated in the table.
  - numeric predictor vs categorical/binary outcome: prefer distribution views
    (box/violin/strip) when writing rich PNG plots; include a deterministic
    summary chart only when useful.
  - binned numeric trend (event rate, or mean of a continuous outcome): line
    over ordered bins/percentiles.
  - numeric vs numeric: scatter, optionally colored/stratified by outcome or group.
  - many numeric signals: ranked effect/association bar plus correlation heatmap.
  - paired/intervention data: paired slope or discordant-count visual.
  - no outcome column: prioritize distributions, missingness, correlation
    structure, and group contrasts over any label-vs-label story.
  - skip a planned visual when required columns are absent or sample size makes it
    misleading; say so in caveats.

For EVERY chart you report in "charts":
- write its plotted data as a CSV under "tables/<name>.csv"
- add a spec {{"name","display_name","kind","data","x","y","title"}} with data="tables/<name>.csv"
  and kind in {{"bar","line","scatter"}}. The HOST renders these deterministically,
  so PRE-AGGREGATE distributions into the CSV (histogram = bin->count; outcome
  rate or mean-outcome curve = bin->value; group comparison = group->value) —
  never rely on a raw dump.
ADDITIONALLY you MAY draw richer figures (box / violin / heatmap / scatter-matrix)
directly as PNG under "figures/" and list them in "plots"; a figure-styling skill
(when available) will make these publication-quality.

This is PURE EXPLORATORY DATA ANALYSIS. Describe what the data shows. Do NOT
propose causal explanations, do NOT claim anything is "confirmed" or
"validated", and do NOT frame findings as hypotheses to be tested — hypothesis
generation and validation are a different, separate step that this tool does
not perform. Stick to descriptive, evidence-grounded statements.

Takeaways (THE PRIMARY OUTPUT — this is what a reader sees first):
- "takeaways": a ranked list of 4-8 dicts, most important/surprising finding
  first, each shaped exactly like:
    {{"title": "<one punchy sentence — the finding itself, with real numbers>",
      "chart_names": ["<name(s) from 'charts' or 'plots' that support it>"],
      "table_names": ["<key(s) from 'tables' that support it, if any>"],
      "analysis": "<2-4 sentences explaining WHY this matters, citing the
                    actual numbers/columns behind the chart(s)>",
      "caveat": "<what this does NOT show, or '' if nothing notable>"}}
  EVERY important chart/plot you produce should be referenced by at least one
  takeaway's "chart_names" — never leave a chart orphaned with no explanation,
  and never write a takeaway with no supporting chart/table unless the data
  genuinely has none to offer (rare).

Report/dashboard contract:
- Emit ONE "dashboard_storyboard" panel dict (a list with one entry) orienting
  the reader on the data/question before the takeaways:
    {{"id": "problem_setting", "title": "Problem Setting", "summary": "...",
      "items": ["..."], "artifact_refs": ["data_profile"]}}
  Do not add "analysis" or "hypotheses" panels — "takeaways" already covers
  that ground, and this tool does not generate hypotheses.

Secondary fields (for programmatic consumers such as a downstream confirmatory
pipeline — NOT the primary reader-facing narrative; keep these terse):
- add "chart_readings": one short dict per important visual with
  {{"chart": "<name/title>", "reading": "<what a human should see>",
  "do_not_infer": "<what this chart cannot prove>"}}.
- add "claims" only for carefully worded descriptive/confirmable statements. Each
  claim must cite chart/signal identifiers in "evidence_ids"; set status to
  "descriptive" (never "supported" — this tool does not confirm anything).
- add "critique": agent self-audit notes about leakage, small n, double-dipping,
  missingness, misleading plot choices, or alternative explanations.
- Never use raw internal IDs like "generated_probe1_false_detection" as user-facing
  chart titles or claim text. Use display names such as "Sanity check: probe
  false-detection flag", and demote probe-derived fields to sanity-check
  evidence rather than explanatory findings.
- PREFERRED: for any composite / threshold / interaction signal that is a
  DETERMINISTIC FUNCTION of the numeric columns, attach a "recipe" so a
  downstream confirmatory pipeline can compute it on a HELD-OUT split later
  (this tool itself does not run that confirmation):
    "recipe": {{"name": "<new signal key>", "kind": "expr",
                "expr": "<boolean/numeric expression over the numeric columns above>"}}
  The expr may use the columns BY NAME, comparisons (< <= > >= == !=), and/or/not,
  arithmetic (+ - * / %), and abs/min/max/float/int/len. It must NOT reference the
  label/outcome column (a recipe is a PREDICTOR, never the answer). Example:
    "recipe": {{"name": "small_and_peripheral", "kind": "expr", "expr": "(obj_size < 40) and (focus_share < 0.3)"}}
  Emit a recipe rather than prose whenever the candidate is computable from the columns.
- ALTERNATIVELY, you MAY attach host-adjudicable "sufficient" statistics computed
  from the rows, as ONE of these shapes:
    {{"kind": "two_group", "a": [0/1, ...], "b": [0/1, ...]}}   # is_fail indicators among signal-ABSENT (a) vs signal-PRESENT (b) cases
    {{"kind": "paired_binary", "b": <int>, "c": <int>}}          # discordant counts of a paired intervention (b = flips the good way, c = the bad way)
  Do NOT emit "reject"/"e_value"/"p_value" anywhere — this tool never adjudicates
  a verdict itself; a self-declared verdict is ignored. Omit both for
  descriptive-only signals.
- prints the final result as the LAST stdout line exactly like (note "charts" is a
  RICH list here, one entry per CSV you wrote). The example below illustrates the
  JSON SHAPE using a binary-outcome dataset; KEEP THE SAME KEYS but replace the
  FAIL/PASS/fail_rate wording with language that matches the ACTUAL outcome kind
  from the profile above (categorical classes, a continuous outcome's mean/curve,
  or plain unsupervised structure when there is no outcome):
  {marker}{{
    "observations": ["..."],
    "visual_plan": [
      {{"name": "failrate_by_objsize",
        "display_name": "Failure rate by object size",
        "question": "Does object size change failure risk?",
        "data_shape": "numeric-vs-binary",
        "plot_kind": "line",
        "fallback_kind": "line",
        "required_columns": ["obj_size", "label"],
        "rationale": "Ordered bins show risk trend without assuming linearity."}}
    ],
    "takeaways": [
      {{"title": "Small objects fail far more often (18% vs 4%, n=120).",
        "chart_names": ["failrate_by_objsize"],
        "table_names": [],
        "analysis": "The fail rate rises sharply below obj_size=40 (18% vs a 4% baseline above it), across 120 rows. This is the single strongest split in the ranked-discriminator chart.",
        "caveat": "Descriptive only — object size and other factors may be confounded; no causal claim is made."}}
    ],
    "chart_readings": [
      {{"chart": "failrate_by_objsize",
        "reading": "Failure rate rises in the smallest object-size bins.",
        "do_not_infer": "This does not prove object size causes the error."}}
    ],
    "claims": [
      {{"id": "C1",
        "text": "Small object size is a descriptive failure correlate.",
        "status": "descriptive",
        "evidence_ids": ["chart:failrate_by_objsize"],
        "interpretation": "A candidate signal for downstream confirmatory analysis.",
        "do_not_infer": "Not causal; not yet confirmed by any statistical test."}}
    ],
    "dashboard_storyboard": [
      {{"id": "problem_setting", "title": "Problem Setting",
        "summary": "Labeled FAIL/PASS cases with per-case signals.",
        "items": ["FAIL means false positive on absent object."],
        "artifact_refs": ["data_profile"]}}
    ],
    "candidate_signals": [
      {{"name": "...", "display_name": "<human-readable signal label>",
        "rationale": "...", "suggested_test": "...",
        "recipe": {{"name": "...", "kind": "expr",
                   "expr": "(col_a < 40) and (col_b < 0.3)"}}}}
    ],
    "plots": ["figures/corr_heatmap.png"],
    "tables": {{}},
    "charts": [
      {{"name": "class_balance", "kind": "bar",
        "display_name": "FAIL/PASS case balance",
        "data": "tables/class_balance.csv", "x": "outcome", "y": "count",
        "title": "FAIL vs PASS"}},
      {{"name": "failrate_by_objsize", "kind": "line",
        "display_name": "Failure rate by object size",
        "data": "tables/failrate_by_objsize.csv", "x": "obj_size_bin",
        "y": "fail_rate", "title": "Fail rate by object size"}},
      {{"name": "top_discriminators", "kind": "bar",
        "data": "tables/top_discriminators.csv", "x": "signal",
        "y": "separation", "title": "Top FAIL/PASS discriminators"}}
    ],
    "caveats": ["..."],
    "critique": ["..."],
    "recommended_confirmatory_tests": ["..."]
  }}
{skills_hint}
Return ONLY the Python code{fences_hint}."""

_REPAIR_PROMPT = """\
The exploratory analysis script failed or produced an invalid result.

Question:
{question}

Data profile:
{data_profile}

Previous code:
```python
{code}
```

Sandbox stdout:
{stdout}

Sandbox stderr:
{stderr}

Parser/execution error:
{error}

Rewrite the script. It must read "{input_filename}" and print a final
{marker} JSON line with the required keys. Return ONLY Python code{fences_hint}."""


@dataclass
class Takeaway:
    """One finding paired with its supporting evidence — the primary UI unit.

    The dashboard renders each takeaway as: title -> chart(s)/table(s) ->
    analysis, so a reader never sees an orphaned chart or a claim with no
    evidence next to it. Always descriptive: this agent does not generate or
    validate hypotheses, so a takeaway is a description of the data, not a
    claim of causation or statistical confirmation.
    """

    title: str = ""
    analysis: str = ""
    chart_names: list[str] = field(default_factory=list)
    table_names: list[str] = field(default_factory=list)
    caveat: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "analysis": self.analysis,
            "chart_names": self.chart_names,
            "table_names": self.table_names,
            "caveat": self.caveat,
        }


@dataclass
class CandidateSignal:
    """A signal worth testing later with a confirmatory pipeline
    (:class:`~evalvitals.analysis.stats_agent.StatsAnalysisAgent`, or the
    diagnosis loop) — not something this agent tests or confirms itself.

    The explorer PROPOSES this signal; it has no adjudication authority. When the
    explorer attaches host-adjudicable ``sufficient`` statistics, the host
    (:func:`evalvitals.analysis.adjudicate.adjudicate_report`) recomputes the
    verdict with the validated, multiplicity-aware core and fills in the
    ``effect`` / ``ci`` / ``e_value`` / ``reject`` / ``host_adjudicated`` fields.
    Any ``reject`` / ``e_value`` the explorer self-declares is IGNORED — proposing
    is not validating; only ``sufficient`` (or, in Phase B, ``recipe``) is read.
    """

    name: str
    display_name: str = ""
    rationale: str = ""
    suggested_test: str = ""
    # --- proposed by the explorer (no authority over the verdict) ---
    sufficient: dict[str, Any] | None = None  # host-adjudicable sufficient stats
    recipe: dict[str, Any] | None = None       # Phase B operationalization recipe
    # --- filled in by the host adjudication pass (authoritative) ---
    effect: float | None = None
    ci: tuple[float, float] | None = None
    e_value: float | None = None
    reject: bool | None = None
    underpowered: bool = False
    host_adjudicated: bool = False
    fdr_corrected: bool = False
    descriptive_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "rationale": self.rationale,
            "suggested_test": self.suggested_test,
            "sufficient": self.sufficient,
            "recipe": self.recipe,
            "effect": self.effect,
            "ci": list(self.ci) if self.ci is not None else None,
            "e_value": self.e_value,
            "reject": self.reject,
            "underpowered": self.underpowered,
            "host_adjudicated": self.host_adjudicated,
            "fdr_corrected": self.fdr_corrected,
            "descriptive_only": self.descriptive_only,
        }


@dataclass
class ExploratoryAnalysisReport:
    """Output of the exploratory analysis agent: a data profile, a ranked list
    of takeaways (each paired with its supporting chart/table), and the raw
    charts/tables/code behind them. Purely descriptive — this agent does not
    generate or validate hypotheses itself; ``hypotheses`` (if present) is
    filled in by a separate downstream stage (M3, see
    ``evalvitals.analysis.hypothesis_agent.HypothesisAgent``), not by this
    class — proposal only, still no validation; see ``StatsAnalysisAgent``
    for confirmatory testing.
    """

    question: str = ""
    ok: bool = False
    observations: list[str] = field(default_factory=list)
    takeaways: list[Takeaway] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    visual_plan: list[dict[str, Any]] = field(default_factory=list)
    chart_readings: list[dict[str, Any]] = field(default_factory=list)
    dashboard_storyboard: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    candidate_signals: list[CandidateSignal] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)
    tables: dict[str, Any] = field(default_factory=dict)
    charts: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    critique: list[str] = field(default_factory=list)
    recommended_confirmatory_tests: list[str] = field(default_factory=list)
    data_profile: dict[str, Any] = field(default_factory=dict)
    # Host adjudication family metadata (method/alpha/split/n_in_family/rejected),
    # filled by evalvitals.analysis.adjudicate; empty until adjudicated.
    adjudication: dict[str, Any] = field(default_factory=dict)
    code: str = ""
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    attempts: int = 0
    workdir: str = ""
    raw_outputs: list[str] = field(default_factory=list)

    @property
    def candidate_signal_names(self) -> list[str]:
        return [s.name for s in self.candidate_signals if s.name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ok": self.ok,
            "observations": self.observations,
            "takeaways": [t.to_dict() for t in self.takeaways],
            "hypotheses": self.hypotheses,
            "visual_plan": self.visual_plan,
            "chart_readings": self.chart_readings,
            "dashboard_storyboard": self.dashboard_storyboard,
            "claims": self.claims,
            "candidate_signals": [s.to_dict() for s in self.candidate_signals],
            "plots": self.plots,
            "tables": self.tables,
            "charts": self.charts,
            "caveats": self.caveats,
            "critique": self.critique,
            "recommended_confirmatory_tests": self.recommended_confirmatory_tests,
            "data_profile": self.data_profile,
            "adjudication": self.adjudication,
            "code": self.code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "attempts": self.attempts,
            "workdir": self.workdir,
        }


class ExploratoryAnalysisAgent:
    """Backend-only exploratory analysis agent.

    Args:
        judge:      LLM-like object with ``generate(prompt) -> str`` used to
                    write the initial analysis script.
        inspector:  Optional LLM-like object used for repair prompts. When not
                    provided, ``judge`` repairs its own code.
        cli_config: Optional CLI coding-agent backend. If set to a non-``llm``
                    provider, it writes ``analysis.py`` locally.
        sandbox:    Local execution sandbox. A durable temp sandbox is created
                    when omitted so generated code and figures remain available.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        inspector: "Model | None" = None,
        cli_config: "CliAgentConfig | None" = None,
        sandbox: ExperimentSandbox | None = None,
        timeout_sec: int = 60,
        max_attempts: int = 2,
    ) -> None:
        self._judge = judge
        self._inspector = inspector
        self._cli_config = cli_config
        self._sandbox = sandbox or ExperimentSandbox(cleanup=False)
        self._timeout_sec = timeout_sec
        self._max_attempts = max(1, max_attempts)

    @property
    def available(self) -> bool:
        return self._judge is not None or (
            self._cli_config is not None and self._cli_config.provider != "llm"
        )

    def explore_records(
        self,
        records: Any,
        *,
        question: str = "Explore this dataset and surface the patterns that matter.",
        outcome_col: str | None = None,
    ) -> ExploratoryAnalysisReport:
        """Run local exploratory analysis over plain records.

        ``outcome_col`` optionally names the target/label column explicitly
        (e.g. M1 always passes ``"label"``). When omitted, the outcome (if
        any) is auto-detected by name heuristics; arbitrary datasets with no
        recognizable outcome column fall back to unsupervised EDA instead of
        having a FAIL/PASS split forced onto them.
        """
        rows = _records_to_rows(records)
        profile = _profile_rows(rows, outcome_col=outcome_col)
        self._write_input(rows)

        if not self.available:
            return ExploratoryAnalysisReport(
                question=question,
                ok=False,
                data_profile=profile,
                error="no code-writing backend configured (judge or cli_config)",
                workdir=str(self._sandbox.workdir),
            )

        raw_outputs: list[str] = []
        code = ""
        last_result: SandboxResult | None = None
        last_error = ""

        for attempt in range(1, self._max_attempts + 1):
            try:
                if attempt == 1:
                    code, raw = self._write_code(question, profile)
                else:
                    code, raw = self._repair_code(
                        question, profile, code, last_result, last_error
                    )
                raw_outputs.append(raw)
            except Exception as exc:  # noqa: BLE001
                last_error = f"code writing failed: {exc}"
                logger.warning("ExploratoryAnalysisAgent: %s", last_error)
                break

            if not code.strip():
                last_error = "backend produced no code"
                continue

            last_result = self._sandbox.run(code, timeout_sec=self._timeout_sec)
            report, last_error = _report_from_sandbox(
                question=question,
                profile=profile,
                code=code,
                result=last_result,
                attempts=attempt,
                workdir=Path(self._sandbox.workdir),
            )
            report.raw_outputs = raw_outputs
            if report.ok:
                return report

        stdout = last_result.stdout if last_result is not None else ""
        stderr = last_result.stderr if last_result is not None else ""
        return ExploratoryAnalysisReport(
            question=question,
            ok=False,
            data_profile=profile,
            code=code,
            stdout=stdout,
            stderr=stderr,
            error=last_error or "exploratory analysis failed",
            attempts=min(self._max_attempts, len(raw_outputs)),
            workdir=str(self._sandbox.workdir),
            raw_outputs=raw_outputs,
        )

    def explore_path(
        self,
        path: str | Path,
        *,
        question: str = "Explore this dataset and surface the patterns that matter.",
        max_rows: int = 2000,
        max_files: int = 200,
        include_tool_calls: bool = False,
        outcome_col: str | None = None,
    ) -> ExploratoryAnalysisReport:
        """Load JSON/JSONL records from *path* and run exploratory analysis.

        This is the no-code entrypoint behind the CLI. It recursively loads
        structured log records, samples large directories deterministically, and
        passes normalized row dictionaries to :meth:`explore_records`.
        """
        rows = load_records_from_path(
            path,
            max_rows=max_rows,
            max_files=max_files,
            include_tool_calls=include_tool_calls,
        )
        report = self.explore_records(rows, question=question, outcome_col=outcome_col)
        report.data_profile.setdefault("source_path", str(Path(path)))
        report.data_profile.setdefault("loaded_rows", len(rows))
        report.data_profile.setdefault(
            "folder_scan",
            scan_folder(path, max_files=max_files, include_tool_calls=include_tool_calls),
        )
        return report

    def _write_input(self, rows: list[dict[str, Any]]) -> None:
        path = Path(self._sandbox.workdir) / RECORDS_FILENAME
        path.write_text(json.dumps(rows, default=str), encoding="utf-8")

    def _write_code(self, question: str, profile: dict[str, Any]) -> tuple[str, str]:
        prompt = _GENERATE_PROMPT.format(
            question=question,
            input_filename=RECORDS_FILENAME,
            data_profile=json.dumps(profile, indent=2, default=str),
            framing=_framing_block(profile.get("outcome") or {}),
            marker=_RESULT_MARKER,
            skills_hint=_skills_hint(self._cli_config),
            fences_hint=_fences_hint(self._cli_config),
        )
        return self._run_writer(prompt, use_inspector=False)

    def _repair_code(
        self,
        question: str,
        profile: dict[str, Any],
        code: str,
        result: SandboxResult | None,
        error: str,
    ) -> tuple[str, str]:
        prompt = _REPAIR_PROMPT.format(
            question=question,
            data_profile=json.dumps(profile, indent=2, default=str),
            code=code,
            stdout=(result.stdout if result is not None else "")[-2000:],
            stderr=(result.stderr if result is not None else "")[-2000:],
            error=error,
            input_filename=RECORDS_FILENAME,
            marker=_RESULT_MARKER,
            fences_hint=_fences_hint(self._cli_config),
        )
        return self._run_writer(prompt, use_inspector=True)

    def _run_writer(self, prompt: str, *, use_inspector: bool) -> tuple[str, str]:
        if self._cli_config is not None and self._cli_config.provider != "llm":
            return self._run_cli_writer(prompt)
        model = self._inspector if use_inspector and self._inspector is not None else self._judge
        raw = model.generate(prompt)  # type: ignore[union-attr]
        raw_text = str(raw)
        return _extract_code(raw_text), raw_text

    def _run_cli_writer(self, prompt: str) -> tuple[str, str]:
        from evalvitals.eval_agent.cli_agent import create_cli_agent

        agent = create_cli_agent(self._cli_config)  # type: ignore[arg-type]
        res = agent.run(prompt, workdir=Path(self._sandbox.workdir), timeout_sec=self._timeout_sec)
        if not res.ok:
            return "", res.raw_output or (res.error or "")
        if "analysis.py" in res.files:
            return res.files["analysis.py"], res.raw_output
        py_files = {n: c for n, c in res.files.items() if n.endswith(".py")}
        if not py_files:
            return "", res.raw_output
        return max(py_files.values(), key=len), res.raw_output


def _records_to_rows(records: Any) -> list[dict[str, Any]]:
    if records is None:
        return []
    if hasattr(records, "to_dict"):
        try:
            data = records.to_dict(orient="records")
            if isinstance(data, list):
                return [dict(r) for r in data if isinstance(r, dict)]
        except TypeError:
            pass
    rows: list[dict[str, Any]] = []
    for row in list(records):
        if isinstance(row, dict):
            rows.append(dict(row))
        elif hasattr(row, "_asdict"):
            rows.append(dict(row._asdict()))
        elif hasattr(row, "__dict__"):
            rows.append(dict(vars(row)))
        else:
            rows.append({"value": row})
    return rows


def _discover_json_files(
    root: Path, *, max_files: int, include_tool_calls: bool
) -> tuple[list[Path], int]:
    """Find the JSON/JSONL files ``load_records_from_path`` would read.

    Returns ``(files_used, n_matching_before_filter_and_cap)`` — the second
    value counts every ``*.json`` file on disk (including ``tool_calls_*``),
    so :func:`scan_folder` can show how many were excluded/capped, not just
    how many survived."""
    if root.is_file():
        return [root], 1
    all_json = sorted(root.rglob("*.json"))
    files = all_json
    if not include_tool_calls:
        files = [p for p in files if not p.name.startswith("tool_calls_")]
    return files[:max_files], len(all_json)


def scan_folder(
    path: str | Path,
    *,
    max_files: int = 200,
    include_tool_calls: bool = False,
    max_listing: int = 60,
    max_scan_entries: int = 200_000,
) -> dict[str, Any]:
    """Filesystem-level summary of what a folder actually contains.

    This is deliberately independent of :class:`~evalvitals.analysis.profile.
    DatasetProfile` (which describes parsed row/column structure): it answers
    "what did the agent see on disk" — file/dir counts, extension mix, and how
    many of the discovered JSON files were actually sampled — since that
    differs from folder to folder and is useful to see before any row-level
    parsing happens.

    Uses a single ``os.walk`` pass rather than ``Path.rglob`` + per-entry
    ``is_file()``/``is_dir()`` checks: real run directories can hold tens of
    thousands of files (one JSON per tool call), and ``rglob`` results lose
    the cached direntry type, so each check becomes its own ``stat()`` — on
    networked storage that made this function slower than the actual row
    loader it's meant to summarize. ``max_scan_entries`` is a hard backstop so
    an unexpectedly huge tree degrades (partial counts, flagged as capped)
    instead of hanging.
    """
    root = Path(path)
    if root.is_file():
        return {
            "root": str(root),
            "is_file": True,
            "n_files_total": 1,
            "n_dirs": 0,
            "extensions": {root.suffix.lower() or "(no extension)": 1},
            "json_files_found": 1,
            "json_files_used": 1,
            "entries": [root.name],
            "truncated": False,
            "scan_capped": False,
        }

    n_files = 0
    n_dirs = 0
    n_json_found = 0
    n_json_after_filter = 0
    ext_counts: dict[str, int] = {}
    entries: list[str] = []
    scan_capped = False
    scanned = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        rel_dir = Path(dirpath).relative_to(root)

        for name in dirnames:
            n_dirs += 1
            scanned += 1
            if len(entries) < max_listing:
                rel = name if rel_dir == Path(".") else str(rel_dir / name)
                entries.append(rel + "/")

        for name in filenames:
            n_files += 1
            scanned += 1
            ext = Path(name).suffix.lower() or "(no extension)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            if name.endswith(".json"):
                n_json_found += 1
                if include_tool_calls or not name.startswith("tool_calls_"):
                    n_json_after_filter += 1
            if len(entries) < max_listing:
                rel = name if rel_dir == Path(".") else str(rel_dir / name)
                entries.append(rel)

        if scanned >= max_scan_entries:
            scan_capped = True
            break

    entries.sort()
    return {
        "root": str(root),
        "is_file": False,
        "n_files_total": n_files,
        "n_dirs": n_dirs,
        "extensions": dict(sorted(ext_counts.items(), key=lambda kv: -kv[1])),
        "json_files_found": n_json_found,
        "json_files_used": min(n_json_after_filter, max_files),
        "entries": entries[:max_listing],
        "truncated": (n_files + n_dirs) > max_listing,
        "scan_capped": scan_capped,
    }


def load_records_from_path(
    path: str | Path,
    *,
    max_rows: int = 2000,
    max_files: int = 200,
    include_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Load a bounded sample of JSON/JSONL records from a file or directory."""
    root = Path(path)
    files, _ = _discover_json_files(root, max_files=max_files, include_tool_calls=include_tool_calls)
    per_file_limit = max(1, max_rows // max(1, len(files)))
    rows: list[dict[str, Any]] = []
    for file_path in files:
        n_from_file = 0
        for row in _load_json_records(file_path):
            flat = _flatten_record(row)
            flat["_source_file"] = str(file_path)
            flat["_source_dir"] = file_path.parent.name
            rows.append(flat)
            n_from_file += 1
            if len(rows) >= max_rows:
                return rows
            if n_from_file >= per_file_limit:
                break
    return rows


def _load_json_records(path: Path) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    stripped = text.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except json.JSONDecodeError:
        pass
    records: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _flatten_record(record: Any, *, max_text: int = 500) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {"value": _shorten(record, max_text)}

    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = _shorten(value, max_text)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_key = f"{key}.{sub_key}"
                if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                    out[flat_key] = _shorten(sub_value, max_text)
        elif isinstance(value, list):
            out[f"{key}._len"] = len(value)
            if key == "model_answer":
                out["model_answer_text"] = _shorten(_join_text_blocks(value), max_text)
        else:
            out[str(key)] = _shorten(value, max_text)

    if "is_correct" in out:
        out.setdefault("label", "pass" if bool(out["is_correct"]) else "fail")
    attempts = record.get("attempts")
    if isinstance(attempts, list):
        out.setdefault("attempts_count", len(attempts))
    trace = record.get("trace")
    if isinstance(trace, list):
        out.setdefault("trace_steps", len(trace))
    return out


def _join_text_blocks(value: list[Any]) -> str:
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


def _shorten(value: Any, max_text: int) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= max_text:
        return value
    return value[:max_text] + f"... [truncated {len(value) - max_text} chars]"


def _profile_rows(
    rows: list[dict[str, Any]],
    *,
    outcome_col: str | None = None,
    sample_size: int = 5,
) -> dict[str, Any]:
    """Profile rows for the explorer prompt, including outcome-kind detection.

    Delegates role/dtype inference to :func:`profile_records` (the shared,
    domain-agnostic profiler) rather than re-deriving it from raw values, so
    the prompt can adapt to binary / categorical / continuous / no-outcome
    data instead of always assuming a FAIL/PASS split.
    """
    profile = profile_records(rows, outcome_col=outcome_col)
    outcome = describe_outcome(profile)
    numeric_columns = [
        c.name for c in profile.columns.values()
        if c.dtype == "numeric" and c.role == "predictor"
    ]
    categorical_columns = [
        c.name for c in profile.columns.values()
        if c.dtype in ("categorical", "boolean") and c.role == "predictor"
    ]
    label_like = [outcome["column"]] if outcome["present"] else []
    return {
        "n_rows": profile.n_rows,
        "grain": profile.grain,
        "columns": {name: c.to_dict() for name, c in profile.columns.items()},
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "label_like_columns": label_like,
        "id_columns": profile.id_columns,
        "group_columns": profile.group_columns,
        "time_columns": profile.time_columns,
        "outcome": outcome,
        "warnings": profile.warnings,
        "sample_rows": rows[:sample_size],
    }


def _framing_block(outcome: dict[str, Any]) -> str:
    """Build the prompt's outcome-specific framing + standard chart battery.

    This is what stops the explorer from forcing a FAIL/PASS story onto data
    that has a multi-class, continuous, or absent outcome. ``outcome`` is the
    ``describe_outcome()`` dict already embedded in the data profile.
    """
    kind = outcome.get("kind", "none")
    col = outcome.get("column")
    unique = outcome.get("unique", 0)

    if kind == "binary":
        return f"""\
OUTCOME: "{col}" is a BINARY outcome (host detected {unique} distinct values).
Call the two groups FAIL and PASS (map the two outcome values to whichever
reads as the negative/positive case) and tell the FAIL-vs-PASS story.

As a minimum, consider this standard battery when the columns exist:
  1. Class balance: count of FAIL vs PASS overall (and per group column if present).
  2. Per numeric signal — how it separates FAIL vs PASS: distribution view or
     group summary, AND a binned fail-rate curve (bin -> fail_rate).
  3. Top discriminators: a ranked bar of each signal's FAIL-vs-PASS separation
     (e.g. standardized mean difference / |meanFAIL - meanPASS| / s), largest first.
  4. Fail rate by each categorical group column (bar).
  5. Signal correlations: a correlation table and, when helpful, a heatmap PNG.
  6. 1-2 scatter plots of the most discriminative signal pairs, coloured by outcome."""

    if kind == "categorical":
        return f"""\
OUTCOME: "{col}" is a CATEGORICAL outcome with {unique} distinct classes. Tell
the per-class story — do NOT collapse it into a binary FAIL/PASS split.

As a minimum, consider this standard battery when the columns exist:
  1. Class balance: count per class overall (and per group column if present).
  2. Per numeric signal — how its distribution differs across classes (box/violin
     per class), AND a ranked bar of each signal's cross-class separation (e.g.
     between-class variance ratio or ANOVA-style F statistic), largest first.
  3. Class composition by each categorical group column (grouped/stacked bar).
  4. Signal correlations: a correlation table and, when helpful, a heatmap PNG.
  5. 1-2 scatter plots of the most discriminative signal pairs, coloured by class."""

    if kind == "continuous":
        return f"""\
OUTCOME: "{col}" is a CONTINUOUS outcome ({unique} distinct values seen). This
is a correlation/regression-style story — there is no FAIL/PASS split to
invent; do not binarize the outcome unless the question explicitly asks for it.

As a minimum, consider this standard battery when the columns exist:
  1. Outcome distribution (histogram).
  2. Per numeric signal vs outcome: scatter with a trend line, AND a binned
     mean-outcome curve (bin -> mean {col}).
  3. Top associates: a ranked bar of each signal's correlation magnitude with
     the outcome (Pearson and/or Spearman), largest first.
  4. Outcome distribution by each categorical group column (box/violin per group).
  5. Signal correlations among predictors: a correlation table and heatmap PNG.
  6. 1-2 scatter plots of the most associated signal pairs, coloured/sized by outcome."""

    return """\
OUTCOME: no recognizable outcome/target column was found. This is UNSUPERVISED
exploration — describe the dataset's structure; do NOT invent a FAIL/PASS or
any other label that is not actually in the data.

As a minimum, consider this standard battery when the columns exist:
  1. Missingness overview: non-null rate per column (bar).
  2. Per numeric column: distribution (histogram/box).
  3. Per categorical column: value counts (bar); skip columns with very high
     cardinality (say >20 distinct values) as a bar chart and note it in caveats.
  4. Signal correlations: a correlation table and, when helpful, a heatmap PNG
     among the numeric columns.
  5. 1-2 scatter plots of the most correlated numeric pairs.
  6. If a group or time column exists, contrast numeric distributions across
     its groups/periods; otherwise skip this item."""


def _report_from_sandbox(
    *,
    question: str,
    profile: dict[str, Any],
    code: str,
    result: SandboxResult,
    attempts: int,
    workdir: Path,
) -> tuple[ExploratoryAnalysisReport, str]:
    if not result.ok:
        err = (result.stderr or "").strip() or "sandbox run failed"
        return (
            ExploratoryAnalysisReport(
                question=question,
                ok=False,
                data_profile=profile,
                code=code,
                stdout=result.stdout,
                stderr=result.stderr,
                error=err,
                attempts=attempts,
                workdir=str(workdir),
            ),
            err,
        )

    parsed, err = _parse_result_json(result.stdout)
    if err:
        return (
            ExploratoryAnalysisReport(
                question=question,
                ok=False,
                data_profile=profile,
                code=code,
                stdout=result.stdout,
                stderr=result.stderr,
                error=err,
                attempts=attempts,
                workdir=str(workdir),
            ),
            err,
        )

    plots = _normalize_plot_paths(parsed.get("plots", []), workdir)
    takeaways = [
        Takeaway(
            title=str(item.get("title", "")),
            analysis=str(item.get("analysis", "")),
            chart_names=[str(x) for x in item.get("chart_names", []) or []],
            table_names=[str(x) for x in item.get("table_names", []) or []],
            caveat=str(item.get("caveat", "")),
        )
        for item in parsed.get("takeaways", []) or []
        if isinstance(item, dict) and item.get("title")
    ]
    signals = [
        CandidateSignal(
            name=str(item.get("name", "")),
            display_name=str(item.get("display_name", "")),
            rationale=str(item.get("rationale", "")),
            suggested_test=str(item.get("suggested_test", "")),
            sufficient=item["sufficient"] if isinstance(item.get("sufficient"), dict) else None,
            recipe=item["recipe"] if isinstance(item.get("recipe"), dict) else None,
        )
        for item in parsed.get("candidate_signals", []) or []
        if isinstance(item, dict) and item.get("name")
    ]
    return (
        ExploratoryAnalysisReport(
            question=question,
            ok=True,
            observations=[str(x) for x in parsed.get("observations", []) or []],
            takeaways=takeaways,
            visual_plan=_normalize_visual_plan(parsed.get("visual_plan", [])),
            chart_readings=_normalize_list_of_dicts(parsed.get("chart_readings", [])),
            dashboard_storyboard=_normalize_list_of_dicts(parsed.get("dashboard_storyboard", [])),
            claims=_normalize_list_of_dicts(parsed.get("claims", [])),
            candidate_signals=signals,
            plots=plots,
            tables=dict(parsed.get("tables", {}) or {}),
            charts=[dict(x) for x in parsed.get("charts", []) or [] if isinstance(x, dict)],
            caveats=[str(x) for x in parsed.get("caveats", []) or []],
            critique=[str(x) for x in parsed.get("critique", []) or []],
            recommended_confirmatory_tests=[
                str(x) for x in parsed.get("recommended_confirmatory_tests", []) or []
            ],
            data_profile=profile,
            code=code,
            stdout=result.stdout,
            stderr=result.stderr,
            attempts=attempts,
            workdir=str(workdir),
        ),
        "",
    )


def _parse_result_json(stdout: str) -> tuple[dict[str, Any], str]:
    marker_line = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(_RESULT_MARKER):
            marker_line = stripped[len(_RESULT_MARKER):]
    if marker_line is None:
        return {}, f"no {_RESULT_MARKER} line in output"
    try:
        parsed = json.loads(marker_line)
    except json.JSONDecodeError as exc:
        return {}, f"unparseable {_RESULT_MARKER} JSON: {exc}"
    if not isinstance(parsed, dict):
        return {}, f"{_RESULT_MARKER} payload must be a JSON object"
    return parsed, ""


def _normalize_visual_plan(raw: Any) -> list[dict[str, Any]]:
    """Keep agent plot-type decisions auditable without making old reports invalid."""
    plan: list[dict[str, Any]] = []
    for idx, item in enumerate(raw or []):
        if not isinstance(item, dict):
            continue
        cleaned: dict[str, Any] = {}
        for key in (
            "name",
            "question",
            "data_shape",
            "plot_kind",
            "fallback_kind",
            "rationale",
            "status",
        ):
            if item.get(key) is not None:
                cleaned[key] = str(item.get(key))
        cols = item.get("required_columns")
        if isinstance(cols, list):
            cleaned["required_columns"] = [str(c) for c in cols]
        elif cols is not None:
            cleaned["required_columns"] = [str(cols)]
        if "name" not in cleaned:
            cleaned["name"] = f"visual_{idx}"
        plan.append(cleaned)
    return plan


def _normalize_list_of_dicts(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        cleaned: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                cleaned[str(key)] = value
            elif isinstance(value, list):
                cleaned[str(key)] = [str(x) for x in value]
            elif isinstance(value, dict):
                cleaned[str(key)] = {str(k): str(v) for k, v in value.items()}
            else:
                cleaned[str(key)] = str(value)
        out.append(cleaned)
    return out


def _normalize_plot_paths(raw: Any, workdir: Path) -> list[str]:
    paths: list[str] = []
    for item in raw or []:
        text = str(item)
        path = Path(text)
        if not path.is_absolute():
            candidate = workdir / path
            if candidate.exists():
                path = candidate
        paths.append(str(path))
    figures_dir = workdir / "figures"
    if figures_dir.exists():
        for png in sorted(figures_dir.glob("*.png")):
            p = str(png)
            if p not in paths:
                paths.append(p)
    return paths


def _extract_code(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return cleaned.strip()


def _fences_hint(cli_config: "CliAgentConfig | None") -> str:
    if cli_config is not None and cli_config.provider != "llm":
        return ", written to a file named analysis.py"
    return " inside a ```python code block"


def _skills_hint(cli_config: "CliAgentConfig | None") -> str:
    """Prompt addendum steering the agent to use available Agent Skills (e.g. a
    figure-styling skill) for the plots it writes. Empty unless skills are enabled
    on the CLI backend. Skills style the agent-authored ``figures/*.png`` only;
    the host-rendered ``charts`` (spec+CSV) stay deterministic and unstyled."""
    if cli_config is None or not getattr(cli_config, "skills_enabled", False):
        return ""
    from pathlib import Path as _P

    names = [_P(s).name for s in (cli_config.skills or [])]
    which = ("the " + ", ".join(f"`/{n}`" for n in names) + " skill(s)") if names else "any installed Agent Skills"
    return (
        "\nFIGURE STYLING: Agent Skills are available. When you write plots under "
        f"figures/, you MAY invoke {which} to produce publication-quality, "
        "well-labelled matplotlib figures. This is a non-interactive PYTHON "
        "analysis: if a skill asks you to choose a plotting backend, choose "
        "Python and proceed without pausing — never stop to ask a question. Use a "
        "skill for styling only — it must not change the data, the analysis, the "
        "sandbox workflow, or the final result JSON.\n"
    )
