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
LORA_DIR_INPUT="$1"
OUTPUT_DIR_INPUT="$2"

if [[ "${LORA_DIR_INPUT}" = /* ]]; then
  LORA_DIR="${LORA_DIR_INPUT}"
else
  LORA_DIR="${PROJECT_ROOT}/${LORA_DIR_INPUT#./}"
fi

if [[ "${OUTPUT_DIR_INPUT}" = /* ]]; then
  OUTPUT_DIR="${OUTPUT_DIR_INPUT}"
else
  OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR_INPUT#./}"
fi

if [ ! -d "${LORA_DIR}" ]; then
  echo "LoRA directory not found: ${LORA_DIR_INPUT}"
  exit 1
fi

if [ ! -f "${LORA_DIR}/adapter_config.json" ]; then
  echo "LoRA directory is missing adapter_config.json: ${LORA_DIR}"
  exit 1
fi

cd "${MEDICALGPT_ROOT}"

"${PYTHON_BIN}" merge_peft_adapter.py \
  --base_model "${BASE_MODEL}" \
  --tokenizer_path "${BASE_MODEL}" \
  --lora_model "${LORA_DIR}" \
  --output_dir "${OUTPUT_DIR}"
