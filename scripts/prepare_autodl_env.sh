#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LMEVAL_ROOT="${LMEVAL_ROOT:-${WORKSPACE_ROOT}/lm-evaluation-harness}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | grep -q "legal-llm"; then
    conda create -y -n legal-llm python=3.10
  fi
  conda activate legal-llm
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r "${PROJECT_ROOT}/requirements-autodl.txt"
"${PYTHON_BIN}" -m pip install -e "${LMEVAL_ROOT}"

echo "Environment is ready."
