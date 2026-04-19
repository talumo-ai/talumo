#!/usr/bin/env bash
# Launch llama-server locally (outside Docker).
# Adjust MODEL_PATH to your GGUF file.
#
# Usage:
#   ./run_llama_server.sh <path-to-model.gguf>           # CPU only
#   ENABLE_CUDA=true ./run_llama_server.sh <model.gguf>  # GPU accelerated
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <path-to-model.gguf>}"
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
