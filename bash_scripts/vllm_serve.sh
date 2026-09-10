#!/usr/bin/env bash

export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve Qwen/Qwen3.5-2B \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.5 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \