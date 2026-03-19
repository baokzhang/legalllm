#!/usr/bin/env bash
set -euo pipefail

prepare_autodl_env() {
  local script_dir project_root workspace_root lmeval_root dep_profile req_file
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  project_root="$(cd "${script_dir}/.." && pwd)"
  workspace_root="$(cd "${project_root}/.." && pwd)"
  lmeval_root="${LMEVAL_ROOT:-${workspace_root}/lm-evaluation-harness}"
  dep_profile="${LEGAL_LLM_DEP_PROFILE:-full}"

  source "${script_dir}/set_cache_env.sh"

  if [ "${dep_profile}" = "bootstrap" ]; then
    req_file="${project_root}/requirements-bootstrap.txt"
  else
    req_file="${project_root}/requirements-autodl.txt"
  fi

  if [ "${LEGAL_LLM_ENV_BACKEND:-venv}" = "conda" ] && command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | grep -q "^legal-llm "; then
      conda create -y -n legal-llm python=3.10
    fi
    conda activate legal-llm
    export PYTHON_BIN="${PYTHON_BIN:-python}"
  else
    export PYTHON_BIN="${PYTHON_BIN:-python3}"
    export LEGAL_LLM_ENV_NAME="${LEGAL_LLM_ENV_NAME:-legal-llm}"
    export VENV_DIR="${VENV_DIR:-${AUTODL_TMP_ROOT}/envs/${LEGAL_LLM_ENV_NAME}}"
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
      "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi
    source "${VENV_DIR}/bin/activate"
    export PYTHON_BIN="${VENV_DIR}/bin/python"
  fi

  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install -r "${req_file}"

  if [ "${dep_profile}" = "full" ]; then
    "${PYTHON_BIN}" -m pip install -e "${lmeval_root}"
  fi

  echo "Environment is ready."
  echo "LEGAL_LLM_DEP_PROFILE=${dep_profile}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
}

prepare_autodl_env "$@"
