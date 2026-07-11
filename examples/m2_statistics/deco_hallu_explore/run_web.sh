#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Upload-and-explore web workbench: serves a page where anyone can drop a .zip
# of results (JSON/JSONL/CSV — e.g. a zipped copy of data_attn_full/); each
# upload becomes one `evalvitals explore` run (M2 exploratory analysis + M3
# hypothesis proposal) and renders in place with the same tabs as
# `evalvitals dashboard`. Runs execute as detached subprocesses, so closing
# the browser never kills an analysis; artifacts accumulate under $WORKSPACE.
#
# Env overrides: PORT / WORKSPACE / CODER_PROVIDER / CODER_MODEL / TIMEOUT_SEC.

PORT="${PORT:-8500}"
WORKSPACE="${WORKSPACE:-web_runs}"
CODER_PROVIDER="${CODER_PROVIDER:-claude_code}"
CODER_MODEL="${CODER_MODEL:-}"

cmd=(
  evalvitals web "$WORKSPACE"
  --port "$PORT"
  --backend "$CODER_PROVIDER"
  --timeout-sec "${TIMEOUT_SEC:-1200}"
)
if [[ -n "$CODER_MODEL" ]]; then
  cmd+=(--model "$CODER_MODEL")
fi

exec "${cmd[@]}"
