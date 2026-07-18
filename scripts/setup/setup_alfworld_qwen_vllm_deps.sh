#!/usr/bin/env bash
set -euo pipefail

# Install the dependencies needed by ALFWorld + local Qwen/vLLM runs.
# This intentionally avoids `alfworld[full]` and never downloads ALFWorld data.
#
# Typical usage on the target server:
#   cd /ai-app-vepfs/zhangfuhao/eval/demo3/SkillOpt-Tree
#   bash scripts/setup/setup_alfworld_qwen_vllm_deps.sh
#
# Useful overrides:
#   PYTHON_BIN=python3 bash scripts/setup/setup_alfworld_qwen_vllm_deps.sh
#   VLLM_PIP_SPEC='vllm==0.9.2' bash scripts/setup/setup_alfworld_qwen_vllm_deps.sh
#   INSTALL_VLLM=0 bash scripts/setup/setup_alfworld_qwen_vllm_deps.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
FORCE_INSTALL_VLLM="${FORCE_INSTALL_VLLM:-0}"
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-vllm}"

ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/data/alfworld}"
ALFWORLD_SPLIT_DIR="${ALFWORLD_SPLIT_DIR:-${PROJECT_ROOT}/data/alfworld_path_split}"

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || fail "Python interpreter not found: ${PYTHON_BIN}"
PYTHON_EXE="$(${PYTHON_BIN} - <<'PY'
import sys

print(sys.executable)
PY
)"
PYTHON_VERSION="$(${PYTHON_BIN} - <<'PY'
import sys

print(".".join(map(str, sys.version_info[:3])))
PY
)"

echo "============================================================"
echo "  Install SkillOpt ALFWorld + Qwen/vLLM dependencies"
echo "============================================================"
echo "  project_root:      ${PROJECT_ROOT}"
echo "  python:            ${PYTHON_EXE}"
echo "  python_version:    ${PYTHON_VERSION}"
echo "  alfworld_data:     ${ALFWORLD_DATA}"
echo "  split_dir:         ${ALFWORLD_SPLIT_DIR}"
echo "  install_vllm:      ${INSTALL_VLLM}"
echo "  vllm_pip_spec:     ${VLLM_PIP_SPEC}"
echo "============================================================"

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("ERROR: Python >= 3.10 is required.")
if sys.version_info >= (3, 12):
    print("[warn] Python 3.12+ may hit older ALFWorld/TextWorld compatibility edges. Python 3.10/3.11 is safer.")
PY

echo "[install] Upgrade pip/build basics"
"${PYTHON_BIN}" -m pip install -U pip "setuptools<81" wheel

echo "[install] Install SkillOpt core + ALFWorld text-only deps"
"${PYTHON_BIN}" -m pip install -e ".[alfworld]"

echo "[install] Install runtime helpers"
"${PYTHON_BIN}" -m pip install omegaconf json_repair

if truthy "${INSTALL_VLLM}"; then
  if command -v vllm >/dev/null 2>&1 && ! truthy "${FORCE_INSTALL_VLLM}"; then
    echo "[install] vLLM already found: $(command -v vllm)"
    vllm --version || true
  else
    echo "[install] Install vLLM: ${VLLM_PIP_SPEC}"
    "${PYTHON_BIN}" -m pip install "${VLLM_PIP_SPEC}"
  fi
else
  echo "[install] Skip vLLM install because INSTALL_VLLM=${INSTALL_VLLM}"
fi

echo "[check] Import core dependencies"
"${PYTHON_BIN}" - <<'PY'
import importlib

mods = [
    "skillopt",
    "alfworld",
    "gymnasium",
    "omegaconf",
    "openai",
    "json_repair",
]
for name in mods:
    importlib.import_module(name)
print("Python imports OK")
PY

if [[ -d "${ALFWORLD_DATA}/json_2.1.1" ]]; then
  echo "[check] ALFWorld data OK: ${ALFWORLD_DATA}/json_2.1.1"
else
  echo "[warn] ALFWorld data not found at ${ALFWORLD_DATA}/json_2.1.1"
  echo "       This script does not download data. Set ALFWORLD_DATA if your data is elsewhere."
fi

if [[ -f "${ALFWORLD_SPLIT_DIR}/train/items.json" ]]; then
  echo "[check] ALFWorld split OK: ${ALFWORLD_SPLIT_DIR}/train/items.json"
else
  echo "[warn] ALFWorld train split not found at ${ALFWORLD_SPLIT_DIR}/train/items.json"
fi

if command -v vllm >/dev/null 2>&1; then
  echo "[check] vLLM command OK: $(command -v vllm)"
  vllm --version || true
else
  echo "[warn] vLLM command not found. Install it before running the local Qwen target."
fi

echo "============================================================"
echo "  Dependency setup finished"
echo "============================================================"
