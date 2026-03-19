#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/run_merge_adapter.sh <lora_dir> <output_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-${WORKSPACE_ROOT}/MedicalGPT}"

source "${SCRIPT_DIR}/activate_project_env.sh"
source "${SCRIPT_DIR}/load_prefetched_model_paths.sh"

BASE_MODEL="${BASE_MODEL:-${BASE_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}}"
LORA_DIR="$1"
OUTPUT_DIR="$2"

cd "${MEDICALGPT_ROOT}"

"${PYTHON_BIN}" merge_peft_adapter.py \
  --base_model "${BASE_MODEL}" \
  --tokenizer_path "${BASE_MODEL}" \
  --lora_model "${LORA_DIR}" \
  --output_dir "${OUTPUT_DIR}"
