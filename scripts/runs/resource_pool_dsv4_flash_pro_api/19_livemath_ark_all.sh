#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-livemath_ark_all_$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-$(cd "${SCRIPT_DIR}/../../.." && pwd)/outputs/${RUN_ID}}"

echo "[suite] system-prompt-only baseline"
RUN_ID="${RUN_ID}_system_prompt_only" \
OUT_BASE="${OUT_BASE}" \
bash "${SCRIPT_DIR}/15_livemath_ark_system_prompt_only.sh"

echo "[suite] init-skill zero-training baseline"
RUN_ID="${RUN_ID}_init_skill_no_train" \
OUT_BASE="${OUT_BASE}" \
bash "${SCRIPT_DIR}/16_livemath_ark_init_skill_no_train.sh"

echo "[suite] dynamic-tree ablations (includes the dynamic_auto main control)"
RUN_ID="${RUN_ID}_ablations" \
OUT_BASE="${OUT_BASE}/ablations" \
bash "${SCRIPT_DIR}/18_livemath_ark_dynamic_ablation.sh"

echo "[suite] all runs complete: ${OUT_BASE}"
