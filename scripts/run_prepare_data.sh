#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export LEGAL_LLM_DEP_PROFILE=full
source "${SCRIPT_DIR}/prepare_autodl_env.sh"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/download_legal_datasets.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_legal_case_corpus.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/select_target_aligned_cases.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_legal_sft_dataset.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/build_legal_dpo_dataset.py"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/export_legal_lm_eval.py"

echo "Prepared legal data pipeline artifacts."
