#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CODER_PROVIDER="${CODER_PROVIDER:-claude_code}"
CODER_MODEL="${CODER_MODEL:-}"
OUT_DIR="${OUT_DIR:-outputs}"

python3 build_dataset.py

cmd=(
  evalvitals
  explore
  data/cases.json
  --backend "$CODER_PROVIDER"
  --outcome-col label
  --out "$OUT_DIR"
  -q "What predicts hallucination failures (label=fail) in this VLM object-presence probe? Compare across model size, probe_type (adversarial vs present), and object. Produce a variety of chart types."
  --max-attempts 3
  --timeout-sec "${TIMEOUT_SEC:-300}"
)

if [[ -n "$CODER_MODEL" ]]; then
  cmd+=(--model "$CODER_MODEL")
fi

exec "${cmd[@]}"
