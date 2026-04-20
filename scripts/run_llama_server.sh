#!/usr/bin/env bash
# Launch llama-server locally (outside Docker).
# Set MODEL_PATH in your shell/.env or pass it as an argument.
#
# Usage:
#   ./run_llama_server.sh <path-to-model.gguf>           # CPU only
#   ENABLE_CUDA=true ./run_llama_server.sh <model.gguf>  # GPU accelerated
#   MODEL_PATH=./models/model.gguf ./run_llama_server.sh  # from env
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "Usage: $0 <path-to-model.gguf> or set MODEL_PATH in environment" >&2
  exit 1
fi
ENABLE_CUDA="${ENABLE_CUDA:-false}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"

if [[ "$ENABLE_CUDA" == "true" ]]; then
  GPU_LAYERS="$N_GPU_LAYERS"
else
  GPU_LAYERS=0
fi

exec llama-server \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers "$GPU_LAYERS"
