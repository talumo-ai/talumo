#!/usr/bin/env bash
# Launch llama-server locally (outside Docker).
# Adjust MODEL_PATH to your GGUF file.
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <path-to-model.gguf>}"

exec llama-server \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 99
