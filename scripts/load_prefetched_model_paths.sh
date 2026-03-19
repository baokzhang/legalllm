#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREFETCH_ENV_FILE="${PREFETCH_ENV_FILE:-${PROJECT_ROOT}/data/processed/prefetched_model_paths.sh}"

if [ -f "${PREFETCH_ENV_FILE}" ]; then
  source "${PREFETCH_ENV_FILE}"
  echo "Loaded prefetched model paths from ${PREFETCH_ENV_FILE}"
fi
