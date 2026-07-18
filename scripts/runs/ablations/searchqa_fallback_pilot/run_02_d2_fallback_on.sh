#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PILOT_NAME=p2_d2_fallback_on
export PILOT_TREE_DEPTH=2
export PILOT_FALLBACK=true
exec bash "${SCRIPT_DIR}/_run_one.sh" "$@"
