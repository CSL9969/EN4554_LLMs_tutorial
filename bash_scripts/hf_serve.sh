#!/usr/bin/env bash

# Hosts the model with Hugging Face Transformers' built-in OpenAI-compatible server.
# Exposes the same endpoint as vllm_serve.sh (http://localhost:8000/v1), so
# src/api_inference.py works without changes.
# Requires FastAPI and Uvicorn: pip install "transformers[serving]"
#
# Quantization to reduce GPU memory (uses bitsandbytes). Set QUANT below to one of:
#   ""          -> no quantization, bf16 (~2.2 GB)
#   "bnb-8bit"  -> 8-bit (~1.7 GB, much slower)
#   "bnb-4bit"  -> 4-bit (~1.4 GB, slightly less accurate)
QUANT="bnb-4bit"

EXTRA_ARGS=()
if [ -n "$QUANT" ]; then
  EXTRA_ARGS+=(--quantization "$QUANT")
else
  EXTRA_ARGS+=(--dtype bfloat16)
fi

transformers serve Qwen/Qwen3.5-0.8B \
  --host localhost \
  --port 8000 \
  --device auto \
  --reasoning off \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --log-level info \
  "${EXTRA_ARGS[@]}"
