#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PILOT_NAME=p4_d3_fallback_on
export PILOT_TREE_DEPTH=3
export PILOT_FALLBACK=true
exec bash "${SCRIPT_DIR}/_run_one.sh" "$@"
