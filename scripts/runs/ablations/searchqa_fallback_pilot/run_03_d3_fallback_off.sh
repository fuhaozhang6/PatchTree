#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PILOT_NAME=p3_d3_fallback_off
export PILOT_TREE_DEPTH=3
export PILOT_FALLBACK=false
exec bash "${SCRIPT_DIR}/_run_one.sh" "$@"
