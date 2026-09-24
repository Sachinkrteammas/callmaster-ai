#!/usr/bin/env bash
# Terminal 1: start the Qwen LLM server. Start this FIRST.
# Wait for "Application startup complete" before starting the worker.
source "$(dirname "$0")/env.sh" vllm || exit 1
MODEL=$(val LLM_MODEL);                    MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}
GPU_UTIL=$(val VLLM_GPU_MEMORY_UTILIZATION); GPU_UTIL=${GPU_UTIL:-0.55}
MAX_LEN=$(val VLLM_MAX_MODEL_LEN);         MAX_LEN=${MAX_LEN:-16384}
echo "Starting vLLM: model=$MODEL gpu_memory=$GPU_UTIL max_len=$MAX_LEN (only reachable from this server)"
exec vllm serve "$MODEL" --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_LEN"
