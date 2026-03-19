#!/usr/bin/env bash
set -euo pipefail

AUTODL_TMP_ROOT="${AUTODL_TMP_ROOT:-/root/autodl-tmp}"
CACHE_ROOT="${CACHE_ROOT:-${AUTODL_TMP_ROOT}/cache}"
HF_ROOT="${HF_ROOT:-${CACHE_ROOT}/huggingface}"

mkdir -p "${AUTODL_TMP_ROOT}/tmp"
mkdir -p "${CACHE_ROOT}/pip"
mkdir -p "${CACHE_ROOT}/torch"
mkdir -p "${CACHE_ROOT}/xdg"
mkdir -p "${HF_ROOT}/hub"
mkdir -p "${HF_ROOT}/datasets"
mkdir -p "${HF_ROOT}/transformers"
mkdir -p "${HF_ROOT}/sentence_transformers"

export TMPDIR="${AUTODL_TMP_ROOT}/tmp"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TORCH_HOME="${CACHE_ROOT}/torch"

export HF_HOME="${HF_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_ROOT}/hub"
export HF_DATASETS_CACHE="${HF_ROOT}/datasets"
export TRANSFORMERS_CACHE="${HF_ROOT}/transformers"
export SENTENCE_TRANSFORMERS_HOME="${HF_ROOT}/sentence_transformers"

echo "AUTODL_TMP_ROOT=${AUTODL_TMP_ROOT}"
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "HF_HOME=${HF_HOME}"
