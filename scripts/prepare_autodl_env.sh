#!/usr/bin/env bash
set -euo pipefail

pick_host_python() {
  local candidate
  for candidate in "${LEGAL_LLM_PYTHON_BIN:-}" python3.10 python3.11 python3; do
    if [ -n "${candidate}" ] && command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

prepare_autodl_env() {
  local script_dir project_root workspace_root lmeval_root dep_profile req_file
  local host_python host_python_version host_python_tag env_state_file
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
    host_python="$(pick_host_python)"
    host_python_version="$("${host_python}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    host_python_tag="py$(echo "${host_python_version}" | tr -d '.')"
    export LEGAL_LLM_HOST_PYTHON="${host_python}"
    export LEGAL_LLM_ENV_NAME="${LEGAL_LLM_ENV_NAME:-legal-llm}"
    export VENV_DIR="${VENV_DIR:-${AUTODL_TMP_ROOT}/envs/${LEGAL_LLM_ENV_NAME}-${host_python_tag}}"
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
      "${host_python}" -m venv "${VENV_DIR}"
    fi
    source "${VENV_DIR}/bin/activate"
    export PYTHON_BIN="${VENV_DIR}/bin/python"
  fi

  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install -r "${req_file}"

  if [ "${dep_profile}" = "full" ]; then
    "${PYTHON_BIN}" -m pip install -e "${lmeval_root}"
  fi

  env_state_file="${project_root}/data/processed/project_env.sh"
  mkdir -p "$(dirname "${env_state_file}")"
  cat > "${env_state_file}" <<EOF
#!/usr/bin/env bash
export LEGAL_LLM_ENV_NAME="${LEGAL_LLM_ENV_NAME:-legal-llm}"
export VENV_DIR="${VENV_DIR:-}"
export PYTHON_BIN="${PYTHON_BIN}"
export LEGAL_LLM_HOST_PYTHON="${LEGAL_LLM_HOST_PYTHON:-}"
EOF

  echo "Environment is ready."
  echo "LEGAL_LLM_DEP_PROFILE=${dep_profile}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "Saved project env file to ${env_state_file}"
}

prepare_autodl_env "$@"
