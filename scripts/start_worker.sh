#!/usr/bin/env bash
# Terminal 2: start the Whisper worker. Start AFTER vLLM is ready.
source "$(dirname "$0")/env.sh" app || exit 1
exec python worker/worker.py
