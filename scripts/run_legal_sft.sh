#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-${WORKSPACE_ROOT}/MedicalGPT}"

source "${SCRIPT_DIR}/activate_project_env.sh"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_FILE_DIR="${PROJECT_ROOT}/data/processed/sft"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/sft-qwen2.5-3b-lora"

cd "${MEDICALGPT_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" supervised_finetuning.py \
  --model_name_or_path "${BASE_MODEL}" \
  --train_file_dir "${TRAIN_FILE_DIR}" \
  --validation_file_dir "${TRAIN_FILE_DIR}" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --do_train \
  --do_eval \
  --template_name qwen \
  --use_peft True \
  --model_max_length 2048 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --warmup_steps 20 \
  --weight_decay 0.05 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 100 \
  --eval_strategy steps \
  --save_steps 200 \
  --save_strategy steps \
  --gradient_accumulation_steps 8 \
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
  --cache_dir "${HF_HOME}/hub"
