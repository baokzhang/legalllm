#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/run_legal_generation_eval.sh <model_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/activate_project_env.sh"

MODEL_DIR_INPUT="$1"
if [[ "${MODEL_DIR_INPUT}" = /* ]]; then
  MODEL_DIR="${MODEL_DIR_INPUT}"
else
  MODEL_DIR="${PROJECT_ROOT}/${MODEL_DIR_INPUT#./}"
fi

if [ ! -d "${MODEL_DIR}" ]; then
  echo "Model directory not found: ${MODEL_DIR_INPUT}"
  exit 1
fi

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "Model directory is missing config.json: ${MODEL_DIR}"
  exit 1
fi

GEN_EVAL_INPUT_FILE="${GEN_EVAL_INPUT_FILE:-${PROJECT_ROOT}/data/eval/legal_generation_eval.jsonl}"
GEN_EVAL_OUTPUT_DIR="${GEN_EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/generation_eval_$(basename "${MODEL_DIR}")}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/export_legal_generation_eval.py" \
  --output_file "${GEN_EVAL_INPUT_FILE}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_legal_generation.py" \
  --model_path "${MODEL_DIR}" \
  --input_file "${GEN_EVAL_INPUT_FILE}" \
  --output_dir "${GEN_EVAL_OUTPUT_DIR}" \
  --limit "${GEN_EVAL_LIMIT:-200}" \
  --batch_size "${GEN_EVAL_BATCH_SIZE:-4}" \
  --max_new_tokens "${GEN_EVAL_MAX_NEW_TOKENS:-256}" \
  --device "${DEVICE:-cuda:0}" \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
