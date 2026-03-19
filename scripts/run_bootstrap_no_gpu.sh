#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

source "${SCRIPT_DIR}/set_cache_env.sh"

bash "${SCRIPT_DIR}/prepare_autodl_env.sh"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/download_legal_datasets.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_legal_case_corpus.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/export_legal_lm_eval.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prefetch_hf_models.py"

echo "No-GPU bootstrap is ready."
echo "Next, after attaching a GPU, run: bash scripts/run_prepare_training_data.sh"
