#!/usr/bin/env bash

# Hosts the model with Hugging Face Transformers' built-in OpenAI-compatible server.
# Exposes the same endpoint as vllm_serve.sh (http://localhost:8000/v1), so
# src/api_inference.py works without changes.
# Requires FastAPI and Uvicorn: pip install "transformers[serving]"

transformers serve Qwen/Qwen3.5-0.8B \
  --host localhost \
  --port 8000 \
  --device auto \
  --dtype bfloat16 \
  --reasoning off \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --log-level info
