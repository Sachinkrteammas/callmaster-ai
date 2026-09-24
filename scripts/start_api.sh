#!/usr/bin/env bash
# Terminal 3: start the FastAPI service (port 8080 by default).
source "$(dirname "$0")/env.sh" app || exit 1
HOST=$(val API_HOST); HOST=${HOST:-0.0.0.0}
PORT=$(val API_PORT); PORT=${PORT:-8080}
echo "Starting API on http://$HOST:$PORT"
exec uvicorn app.main:app --host "$HOST" --port "$PORT"
