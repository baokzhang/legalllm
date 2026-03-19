#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-${WORKSPACE_ROOT}/MedicalGPT}"

source "${SCRIPT_DIR}/activate_project_env.sh"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_FILE_DIR="${PROJECT_ROOT}/data/processed/dpo"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/dpo-qwen2.5-3b-lora"

cd "${MEDICALGPT_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" dpo_training.py \
  --model_name_or_path "${BASE_MODEL}" \
  --template_name qwen \
  --train_file_dir "${TRAIN_FILE_DIR}" \
  --validation_file_dir "${TRAIN_FILE_DIR}" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_train_samples 20000 \
  --max_eval_samples 1000 \
  --max_steps 200 \
  --eval_steps 50 \
  --save_steps 100 \
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
