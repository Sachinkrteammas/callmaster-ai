#!/usr/bin/env bash
# Stop API, Worker and vLLM.
#   ./scripts/stop_all.sh
for s in api worker vllm; do
  tmux kill-session -t "$s" 2>/dev/null && echo "Stopped $s"
done
echo "All stopped."
