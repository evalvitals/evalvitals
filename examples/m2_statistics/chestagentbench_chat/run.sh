#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-/tealab-data/rjin02/MedRAX/logs/202607/chestagentbench}"
OUT_DIR="${OUT_DIR:-/tealab-data/rjin02/MedRAX/logs/202607/chestagentbench_m2_chat}"
CODER_PROVIDER="${CODER_PROVIDER:-antigravity}"
CODER_MODEL="${CODER_MODEL:-}"
MAX_ROWS="${MAX_ROWS:-2000}"
MAX_FILES="${MAX_FILES:-20}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"

cmd=(
  evalvitals
  chat
  "$RESULTS_DIR"
  --backend "$CODER_PROVIDER"
  --out "$OUT_DIR"
  --max-rows "$MAX_ROWS"
  --max-files "$MAX_FILES"
  --timeout-sec "$TIMEOUT_SEC"
)

if [[ -n "$CODER_MODEL" ]]; then
  cmd+=(--model "$CODER_MODEL")
fi

exec "${cmd[@]}"
