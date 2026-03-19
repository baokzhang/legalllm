#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-${WORKSPACE_ROOT}/MedicalGPT}"

source "${SCRIPT_DIR}/activate_project_env.sh"
source "${SCRIPT_DIR}/load_prefetched_model_paths.sh"

BASE_MODEL="${BASE_MODEL:-${BASE_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}}"
TRAIN_FILE_DIR="${PROJECT_ROOT}/data/processed/dpo"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/dpo-qwen2.5-3b-lora"
DPO_PER_DEVICE_TRAIN_BATCH_SIZE="${DPO_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
DPO_PER_DEVICE_EVAL_BATCH_SIZE="${DPO_PER_DEVICE_EVAL_BATCH_SIZE:-4}"
DPO_GRADIENT_ACCUMULATION_STEPS="${DPO_GRADIENT_ACCUMULATION_STEPS:-8}"
DPO_MAX_TRAIN_SAMPLES="${DPO_MAX_TRAIN_SAMPLES:-20000}"
DPO_MAX_EVAL_SAMPLES="${DPO_MAX_EVAL_SAMPLES:-200}"
DPO_MAX_STEPS="${DPO_MAX_STEPS:-200}"
DPO_EVAL_STEPS="${DPO_EVAL_STEPS:-100}"
DPO_SAVE_STEPS="${DPO_SAVE_STEPS:-100}"

cd "${MEDICALGPT_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" dpo_training.py \
  --model_name_or_path "${BASE_MODEL}" \
  --template_name qwen \
  --train_file_dir "${TRAIN_FILE_DIR}" \
  --validation_file_dir "${TRAIN_FILE_DIR}" \
  --per_device_train_batch_size "${DPO_PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${DPO_GRADIENT_ACCUMULATION_STEPS}" \
  --per_device_eval_batch_size "${DPO_PER_DEVICE_EVAL_BATCH_SIZE}" \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples "${DPO_MAX_TRAIN_SAMPLES}" \
  --max_eval_samples "${DPO_MAX_EVAL_SAMPLES}" \
  --max_steps "${DPO_MAX_STEPS}" \
  --eval_steps "${DPO_EVAL_STEPS}" \
  --save_steps "${DPO_SAVE_STEPS}" \
  --max_source_length 1536 \
  --max_target_length 512 \
  --output_dir "${OUTPUT_DIR}" \
  --target_modules all \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --fp16 False \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing True \
  --cache_dir "${HF_HOME}/hub"
