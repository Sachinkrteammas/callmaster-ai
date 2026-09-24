# Load the right Python environment.  Usage (from anywhere):
#   source scripts/env.sh app     -> for worker, API, tests
#   source scripts/env.sh vllm    -> for the LLM server
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || return 1

if [ ! -f .env ]; then
  echo "ERROR: .env not found. Run:  cp .env.example .env   and edit it."
  return 1
fi

# read one value from .env:  val KEY
val() { grep -E "^$1=" "$PROJECT_DIR/.env" | tail -1 | cut -d= -f2- | sed 's/^"//; s/"$//'; }

VENV_NAME="${1:-app}"
if [ ! -f "$PROJECT_DIR/.venv-$VENV_NAME/bin/activate" ]; then
  echo "ERROR: .venv-$VENV_NAME not found. Follow README step 6."
  return 1
fi
. "$PROJECT_DIR/.venv-$VENV_NAME/bin/activate"

export HF_HOME="$PROJECT_DIR/models/hf"

if [ "$VENV_NAME" = "app" ]; then
  # Tell faster-whisper where the pip-installed CUDA libraries are
  CUDA_LIBS=$(python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))' 2>/dev/null)
  [ -n "$CUDA_LIBS" ] && export LD_LIBRARY_PATH="$CUDA_LIBS:$LD_LIBRARY_PATH"
fi
echo "Environment '$VENV_NAME' ready in $PROJECT_DIR"
