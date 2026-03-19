#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/run_legal_eval.sh <merged_model_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
LMEVAL_ROOT="${LMEVAL_ROOT:-${WORKSPACE_ROOT}/lm-evaluation-harness}"

source "${SCRIPT_DIR}/activate_project_env.sh"

MODEL_DIR_INPUT="$1"
if [[ "${MODEL_DIR_INPUT}" = /* ]]; then
  MODEL_DIR="${MODEL_DIR_INPUT}"
else
  MODEL_DIR="${PROJECT_ROOT}/${MODEL_DIR_INPUT#./}"
fi

if [ ! -d "${MODEL_DIR}" ]; then
  echo "Merged model directory not found: ${MODEL_DIR_INPUT}"
  exit 1
fi

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "Merged model directory is missing config.json: ${MODEL_DIR}"
  exit 1
fi

TASK_INCLUDE_PATH="${PROJECT_ROOT}/lm_eval_tasks"
EVAL_LIMIT="${EVAL_LIMIT:-1000}"
EVAL_ARGS=()
if [ -n "${EVAL_LIMIT}" ] && [ "${EVAL_LIMIT}" != "0" ]; then
  EVAL_ARGS=(--limit "${EVAL_LIMIT}")
fi

cd "${LMEVAL_ROOT}"

lm-eval run \
  --model hf \
  --model_args pretrained="${MODEL_DIR}" dtype=bfloat16 \
  --tasks legal_charge_mc \
  --apply_chat_template \
  --device "${DEVICE:-cuda:0}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --include_path "${TASK_INCLUDE_PATH}" \
  --output_path "${PROJECT_ROOT}/outputs/lm_eval_$(basename "${MODEL_DIR}")" \
  "${EVAL_ARGS[@]}"
