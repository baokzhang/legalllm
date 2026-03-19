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

source "${SCRIPT_DIR}/set_cache_env.sh"

MODEL_DIR="$1"
TASK_INCLUDE_PATH="${PROJECT_ROOT}/lm_eval_tasks"

cd "${LMEVAL_ROOT}"

lm-eval run \
  --model hf \
  --model_args pretrained="${MODEL_DIR}" dtype=bfloat16 \
  --tasks legal_charge_mc \
  --apply_chat_template \
  --device "${DEVICE:-cuda:0}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --include_path "${TASK_INCLUDE_PATH}" \
  --output_path "${PROJECT_ROOT}/outputs/lm_eval_$(basename "${MODEL_DIR}")"
