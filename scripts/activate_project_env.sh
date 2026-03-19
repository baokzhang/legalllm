#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/set_cache_env.sh"

PROJECT_ENV_FILE="${PROJECT_ENV_FILE:-${PROJECT_ROOT}/data/processed/project_env.sh}"
if [ -f "${PROJECT_ENV_FILE}" ]; then
  source "${PROJECT_ENV_FILE}"
fi

LEGAL_LLM_ENV_NAME="${LEGAL_LLM_ENV_NAME:-legal-llm}"
VENV_DIR="${VENV_DIR:-${AUTODL_TMP_ROOT}/envs/${LEGAL_LLM_ENV_NAME}}"

if [ -x "${VENV_DIR}/bin/python" ]; then
  source "${VENV_DIR}/bin/activate"
  export PYTHON_BIN="${VENV_DIR}/bin/python"
else
  export PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "PYTHON_BIN=${PYTHON_BIN}"
