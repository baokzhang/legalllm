#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-${WORKSPACE_ROOT}/MedicalGPT}"

source "${SCRIPT_DIR}/activate_project_env.sh"
source "${SCRIPT_DIR}/load_prefetched_model_paths.sh"

BASE_MODEL="${BASE_MODEL:-${BASE_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}}"
TRAIN_FILE_DIR="${PROJECT_ROOT}/data/processed/sft"
OUTPUT_DIR="${SFT_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sft-qwen2.5-3b-lora}"
SFT_PER_DEVICE_TRAIN_BATCH_SIZE="${SFT_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
SFT_PER_DEVICE_EVAL_BATCH_SIZE="${SFT_PER_DEVICE_EVAL_BATCH_SIZE:-8}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-8}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-1}"
SFT_LOGGING_STEPS="${SFT_LOGGING_STEPS:-10}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-500}"
SFT_DO_EVAL="${SFT_DO_EVAL:-true}"
SFT_MAX_EVAL_SAMPLES="${SFT_MAX_EVAL_SAMPLES:-200}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-500}"
SFT_EVAL_STRATEGY="${SFT_EVAL_STRATEGY:-steps}"

SFT_EVAL_ARGS=()
if [ "${SFT_DO_EVAL}" = "true" ]; then
  SFT_EVAL_ARGS=(
    --do_eval
    --max_eval_samples "${SFT_MAX_EVAL_SAMPLES}"
    --per_device_eval_batch_size "${SFT_PER_DEVICE_EVAL_BATCH_SIZE}"
    --eval_steps "${SFT_EVAL_STEPS}"
    --eval_strategy "${SFT_EVAL_STRATEGY}"
  )
fi

cd "${MEDICALGPT_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" supervised_finetuning.py \
  --model_name_or_path "${BASE_MODEL}" \
  --train_file_dir "${TRAIN_FILE_DIR}" \
  --validation_file_dir "${TRAIN_FILE_DIR}" \
  --per_device_train_batch_size "${SFT_PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --do_train \
  --template_name qwen \
  --use_peft True \
  --model_max_length 2048 \
  --num_train_epochs "${SFT_NUM_TRAIN_EPOCHS}" \
  --learning_rate 2e-5 \
  --warmup_steps 20 \
  --weight_decay 0.05 \
  --logging_strategy steps \
  --logging_steps "${SFT_LOGGING_STEPS}" \
  --save_steps "${SFT_SAVE_STEPS}" \
  --save_strategy steps \
  --gradient_accumulation_steps "${SFT_GRADIENT_ACCUMULATION_STEPS}" \
  --preprocessing_num_workers 4 \
  --output_dir "${OUTPUT_DIR}" \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to tensorboard \
  --ddp_find_unused_parameters False \
  --gradient_checkpointing True \
  --cache_dir "${HF_HOME}/hub" \
  "${SFT_EVAL_ARGS[@]}"
