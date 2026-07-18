#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# These runs were produced before the follow-up suite got its own output group,
# so their best skills live alongside the original core-8 outputs.
export SKILL_ROOT="${FOLLOWUP4_ROOT:-/ai-app-vepfs/zhangfuhao/skill/SkillOpt-Tree/outputs/livemath_core8_pairs}"
export SKILL_MANIFEST="${FOLLOWUP4_MANIFEST:-${SCRIPT_DIR}/followup4_manifest.tsv}"
export EVAL_SUITE_SLUG=livemath_followup4
export EVAL_SUITE_TITLE="LiveMath follow-up-4"
export EVAL_SUMMARY_STEM=followup4_test_summary

exec bash "${SCRIPT_DIR}/../livemath_core8/run_test_core8_best_skills.sh" "$@"
