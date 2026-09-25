#!/usr/bin/env bash
# Start everything in the background (tmux): vLLM -> Worker -> API.
#   ./scripts/start_all.sh
cd "$(dirname "$0")/.." || exit 1

if tmux has-session -t vllm 2>/dev/null; then
  echo "Already running. Stop first with: ./scripts/stop_all.sh"
  exit 1
fi

echo "Starting vLLM (Qwen)... about 2 minutes"
tmux new -d -s vllm ./scripts/start_vllm.sh
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do
  tmux has-session -t vllm 2>/dev/null || { echo "ERROR: vLLM stopped. Run ./scripts/start_vllm.sh to see why."; exit 1; }
  sleep 5
done

echo "Starting Worker (Whisper)..."
tmux new -d -s worker ./scripts/start_worker.sh
sleep 40

echo "Starting API..."
tmux new -d -s api ./scripts/start_api.sh
sleep 5

curl -s http://localhost:8080/health; echo
