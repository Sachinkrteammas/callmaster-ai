# Callmaster AI — Self-Hosted Speech-to-Text + Audit Service (no Docker)

Replaces **Deepgram** and **OpenAI** for Callmaster audits. Runs directly on one
Ubuntu GPU server as normal programs. Callmaster itself is **not** changed.

```
Audio (.wav/.mp3/.m4a)
   ↓  FFmpeg  (stereo: channel 1 = Agent, channel 2 = Customer)
   ↓  faster-whisper large-v3-turbo  (GPU)
Transcript   "[0.00s] Agent: Sir good morning sir."
   ↓  + EXISTING audit prompt  (app/prompts/audit_prompt.txt)
   ↓  Qwen2.5-7B-Instruct-AWQ on vLLM  (GPU, on this server)
Audit JSON   (checked against app/schemas/audit_schema.json)
```

There is **no audit logic in Python**. The audit prompt is the business logic; Qwen runs it.

## 1. The 4 programs

| # | Program | What it does | How it runs |
|---|---|---|---|
| 1 | **Redis** | Job queue | Ubuntu service, starts automatically |
| 2 | **vLLM** | Runs Qwen (the LLM) on the GPU, port 8000, only reachable from the server itself | `scripts/start_vllm.sh` (terminal 1) |
| 3 | **Worker** | Loads Whisper once, transcribes audio, sends transcript + prompt to Qwen | `scripts/start_worker.sh` (terminal 2) |
| 4 | **API** | FastAPI on port 8080 — what Callmaster will call | `scripts/start_api.sh` (terminal 3) |

Start order is always: **Redis → vLLM (wait until ready) → Worker → API**.

| Endpoint | What it does |
|---|---|
| `GET /health` | Is everything running? (no key needed) |
| `POST /v1/transcribe` | Audio → transcript |
| `POST /v1/audit` | Transcript + client prompt → audit JSON |
| `POST /v1/process` | Audio → transcript → audit JSON (waits for result) |
| `POST /v1/jobs` | Same, but returns immediately; 3 retries after 30/60/120 s |
| `GET /v1/jobs/{call_id}` | `queued` / `processing` / `completed` / `failed` + result |

All `/v1/*` endpoints need the header `X-API-Key`.

## 2. Server requirements

| Item | Minimum |
|---|---|
| GPU | NVIDIA, 24 GB VRAM (L4, A10, RTX 3090/4090, A5000 …) |
| NVIDIA driver | 550 or newer |
| CPU / RAM | 8 cores / 32 GB (64 GB better) |
| Disk | 100 GB free minimum (software + models ≈ 25 GB) + audio |
| OS | Ubuntu 22.04 or 24.04, with `sudo` |

Rent a normal **virtual machine** (AWS g6/g5, Google Cloud L4, Azure, E2E Networks…)
with **Ubuntu 22.04**. A "Deep Learning" image with the NVIDIA driver pre-installed saves a step.

---

# STEP-BY-STEP SETUP

Do the steps in order. Each ends with ✅ **Check** — only continue when it passes.
If something fails, copy the full error message and ask for help.

## Step 1 — Connect to the server

On your laptop (Windows PowerShell, Mac/Linux terminal):

```bash
ssh ubuntu@YOUR_SERVER_IP
# with a key file:  ssh -i mykey.pem ubuntu@YOUR_SERVER_IP
```

✅ **Check:** prompt looks like `ubuntu@servername:~$`. All commands below are typed on the server.

## Step 2 — Install basic software

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git unzip curl tmux ffmpeg redis-server python3-venv python3-pip python3-dev
sudo systemctl enable --now redis-server
```

✅ **Check:**
```bash
ffmpeg -version | head -1      # prints "ffmpeg version ..."
redis-cli ping                 # prints PONG
python3 --version              # 3.10 or 3.12
```

## Step 3 — NVIDIA driver

```bash
nvidia-smi
```

- A table with your GPU name and **Driver Version 550 or higher** → go to Step 4.
- `command not found` (or driver lower than 550):

```bash
sudo ubuntu-drivers install
sudo reboot
```
Wait 2 minutes, `ssh` in again, run `nvidia-smi` again.

✅ **Check:** `nvidia-smi` shows the GPU, e.g. `NVIDIA L4 ... 23034MiB`.

## Step 4 — Put the project on the server

**Option A — zip file (easiest).** On your laptop:
```bash
scp callmaster-ai.zip ubuntu@YOUR_SERVER_IP:~
```
On the server:
```bash
cd ~ && unzip callmaster-ai.zip && cd callmaster-ai
```

**Option B — Git** (after pushing it, see Step 17):
```bash
cd ~ && git clone https://github.com/YOUR_ORG/callmaster-ai.git && cd callmaster-ai
```

✅ **Check:** `ls` shows `README.md  app  scripts  tests  worker ...`

## Step 5 — Settings file (.env)

```bash
cp .env.example .env
openssl rand -hex 32        # prints a long random key - copy it
nano .env                   # paste it after API_KEY=
```
Save in nano: **Ctrl+O**, **Enter**, **Ctrl+X**.

Main settings (defaults are fine to start):

| Setting | Meaning |
|---|---|
| `API_KEY` | Password callers must send as `X-API-Key` — **must change** |
| `STT_LANGUAGE=auto` | Detect Hindi/English automatically. Don't force `en`. |
| `CHANNEL_1_SPEAKER` / `CHANNEL_2_SPEAKER` | Agent/Customer per stereo channel (swap if reversed) |
| `VOCAB` | Company/product/campaign names, comma separated, one line |
| `VLLM_GPU_MEMORY_UTILIZATION` | Share of GPU for Qwen (0.55 leaves room for Whisper) |
| `VLLM_MAX_MODEL_LEN` | Max transcript + prompt + answer length for Qwen |

`.env` is never uploaded to Git.

## Step 6 — Install Python packages (two separate environments)

vLLM and Whisper need different CUDA packages, so each gets its own folder.
This downloads several GB and can take 10–20 minutes.

**6a. App environment (Whisper + API):**
```bash
cd ~/callmaster-ai
python3 -m venv .venv-app
.venv-app/bin/pip install --upgrade pip
.venv-app/bin/pip install -r app/requirements.txt -r requirements-gpu.txt
```

✅ **Check:**
```bash
source scripts/env.sh app
python -c "import ctranslate2; print('GPUs seen by Whisper:', ctranslate2.get_cuda_device_count())"
deactivate
```
Must print `GPUs seen by Whisper: 1`.

**6b. vLLM environment (Qwen):**
```bash
python3 -m venv .venv-vllm
.venv-vllm/bin/pip install --upgrade pip
.venv-vllm/bin/pip install -r requirements-vllm.txt
```

✅ **Check:**
```bash
.venv-vllm/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Must print `True NVIDIA ...`.

## Step 7 — Put in the real audit prompt and schema

**Prompt:** copy the production audit prompt from the current Callmaster/OpenAI code,
exactly as it is, into `app/prompts/audit_prompt.txt` (replace everything):
```bash
nano app/prompts/audit_prompt.txt
```
- Default: prompt is sent as the **system** message, transcript as the **user** message.
- If Callmaster today puts the transcript **inside** the prompt, write `{{TRANSCRIPT}}`
  at that place in the prompt. Match what Callmaster does today.

**Schema:** copy the production JSON schema into `app/schemas/audit_schema.json`.
Do not rename/add/remove fields or change types/enums. If Callmaster has no formal
schema (e.g. it used OpenAI "JSON mode"), one must be written from real outputs and
approved by the audit owner. Check the file:
```bash
source scripts/env.sh app
python -c "from app.audit import load_schema; load_schema(); print('schema OK')"
```

You can do the first technical test with the placeholders, but the real audit
needs these two files. Changes take effect on the next call — no restart needed.

## Step 8 — Learn tmux (keeps programs running after you close SSH)

| Action | Keys / command |
|---|---|
| Open a new named window | `tmux new -s NAME` |
| Leave it running in background | press **Ctrl+B**, release, then press **D** |
| Go back to it | `tmux attach -t NAME` |
| List them | `tmux ls` |
| Stop the program inside | **Ctrl+C** |

## Step 9 — Start vLLM (Qwen) — FIRST

```bash
cd ~/callmaster-ai
tmux new -s vllm
./scripts/start_vllm.sh
```
The first time it downloads Qwen (~5.5 GB) into `models/hf/`. Wait until you see:
**`Application startup complete`**. Then press **Ctrl+B, D**.

✅ **Check:**
```bash
curl http://127.0.0.1:8000/v1/models
```
Shows `Qwen/Qwen2.5-7B-Instruct-AWQ`.

## Step 10 — Start the Worker (Whisper)

```bash
cd ~/callmaster-ai
tmux new -s worker
./scripts/start_worker.sh
```
First time downloads Whisper (~1.6 GB) into `models/whisper/`. Wait for:
**`Worker ready: STT=large-v3-turbo ...`**. Then **Ctrl+B, D**.

✅ **Check:** `nvidia-smi` shows two processes using GPU memory (vLLM and python).

## Step 11 — Start the API

```bash
cd ~/callmaster-ai
tmux new -s api
./scripts/start_api.sh
```
Wait for `Uvicorn running on http://0.0.0.0:8080`. Then **Ctrl+B, D**.

## Step 12 — Health check

```bash
curl http://localhost:8080/health
```

✅ **Check:** `"status":"ok"` and `"checks":{"redis":true,"stt":true,"llm":true}`.
If one is `false`, that program is not running — see Step 16.

Save your key in the terminal for the next steps:
```bash
cd ~/callmaster-ai
export API_KEY=$(grep ^API_KEY= .env | cut -d= -f2)
```

## Step 13 — Test with a real recording

Copy a real call to the server (on your laptop):
```bash
scp sample_call.mp3 ubuntu@YOUR_SERVER_IP:~/callmaster-ai/data/audio/
```

**13a. Transcript only:**
```bash
curl -X POST http://localhost:8080/v1/transcribe -H "X-API-Key: $API_KEY" \
  -F "call_id=TEST001" -F "file=@data/audio/sample_call.mp3"
```

**13b. Detailed view** (duration, channels, sample rate, language, transcript):
```bash
source scripts/env.sh app
python scripts/test_audio.py data/audio/sample_call.mp3
deactivate
```

✅ **Check:**
- Hindi appears where Hindi is spoken, English where English is spoken.
- On stereo calls, `Agent` lines really are the agent. **If reversed:** swap
  `CHANNEL_1_SPEAKER` and `CHANNEL_2_SPEAKER` in `.env`, then restart the worker
  and API (Step 15).
- If Agent and Customer lines are **exact duplicates**, the dialer writes the same
  mixed audio to both channels — ask for true two-channel recordings.

**13c. Audit only:**
```bash
curl -X POST http://localhost:8080/v1/audit -H "X-API-Key: $API_KEY" \
  -F "call_id=TEST001" -F "transcript=<transcript.txt" -F "prompt=<prompt.txt"
```

**13d. Full pipeline:**
```bash
curl -X POST http://localhost:8080/v1/process -H "X-API-Key: $API_KEY" \
  -F "call_id=TEST002" -F "file=@data/audio/sample_call.mp3"
```

**13e. Background job** (how Callmaster should use it later):
```bash
curl -X POST http://localhost:8080/v1/jobs -H "X-API-Key: $API_KEY" \
  -F "call_id=TEST003" -F "file=@data/audio/sample_call.mp3"
curl http://localhost:8080/v1/jobs/TEST003 -H "X-API-Key: $API_KEY"
```
Optional `-F "webhook_url=https://..."`: if the webhook fails, the result is still
saved and not re-processed. Every result is also saved in `data/results/<call_id>.json`.

## Step 14 — Tests and benchmark

Unit tests (no GPU needed):
```bash
source scripts/env.sh app
pip install -r requirements-dev.txt
pytest
```

GPU tests against the running service:
```bash
RUN_GPU_TESTS=1 SAMPLE_AUDIO=data/audio/sample_call.mp3 pytest tests/integration -v -s
```

Benchmark (measured numbers only):
```bash
python tests/benchmark.py data/audio/call1.mp3 data/audio/call2.mp3
```
Prints audio duration, STT time, LLM time, total, real-time factor. Run on 20–50 real calls.

## Step 15 — Daily operations

| Task | Command |
|---|---|
| See running programs | `tmux ls` |
| Watch a program's output | `tmux attach -t worker` (then Ctrl+B, D to leave) |
| Log files | `tail -f data/logs/worker.log` / `data/logs/api.log` |
| GPU usage | `watch -n 2 nvidia-smi` (Ctrl+C to exit) |
| Restart worker after `.env` change | `tmux attach -t worker` → Ctrl+C → `./scripts/start_worker.sh` → Ctrl+B, D |
| Restart API after `.env` change | same with `-t api` and `./scripts/start_api.sh` |
| After a server reboot | Start Steps 9 → 10 → 11 again (Redis starts by itself) |

Logs contain call_id, audio duration, channels, STT/LLM start/end, total time and
errors — never API keys; full transcripts only if `LOG_LEVEL=DEBUG`.

Automatic start on reboot (systemd services) can be added once everything works.

## Step 16 — Troubleshooting

| Problem | Fix |
|---|---|
| `health` → `redis: false` | `sudo systemctl restart redis-server`, check `redis-cli ping` |
| `health` → `llm: false` | vLLM not running/ready: `tmux attach -t vllm` and read the error |
| `health` → `stt: false` | Worker not running: `tmux attach -t worker` |
| `ERROR: .env not found` | Step 5 |
| `ERROR: .venv-app not found` | Step 6 |
| Worker: `libcudnn_ops.so.9` / `libcublas` not found | Always start with `./scripts/start_worker.sh` (it sets the library path). Re-run Step 6a |
| `GPUs seen by Whisper: 0` | Driver problem → `nvidia-smi`, Step 3 |
| vLLM: `CUDA out of memory` / "no available memory for cache" | Stop the worker, start vLLM first, then the worker. Or lower `VLLM_MAX_MODEL_LEN` to 8192 |
| Worker: `CUDA out of memory` | Lower `VLLM_GPU_MEMORY_UTILIZATION` to 0.50, restart vLLM then worker |
| `torch.cuda.is_available()` → False | Driver older than 550 → Step 3 |
| "LLM output was cut off" | Raise `LLM_MAX_TOKENS` and/or `VLLM_MAX_MODEL_LEN` |
| vLLM error about the JSON schema | Set `LLM_GUIDED_JSON=false` in `.env` (output is still validated), restart worker + API |
| `Address already in use` | That program is already running: `tmux ls` |

**Security:**
- Port 8000 (vLLM) listens only on 127.0.0.1 — not reachable from outside.
- Allow port 8080 only from the Callmaster server:
  ```bash
  sudo ufw allow OpenSSH
  sudo ufw allow from CALLMASTER_SERVER_IP to any port 8080
  sudo ufw enable
  ```
  (Allow OpenSSH **first**, or you will lock yourself out.) Also check your cloud firewall/security group.
- Delete old audio daily (`crontab -e`):
  `0 2 * * * find /home/ubuntu/callmaster-ai/data/audio -type f -mtime +30 -delete`

## Step 17 — Git

The project already has a first commit. To put it on GitHub (on your laptop, inside the project folder):
1. Create an **empty private** repo on GitHub (no README).
2. Run:
```bash
git remote add origin https://github.com/YOUR_ORG/callmaster-ai.git
git push -u origin main
```
Updating the server later: `cd ~/callmaster-ai && git pull`, then restart worker and API (Step 15).
`.env`, audio, results, models and `.venv*` folders are never committed.

## Models

| | Speech-to-text | Audit LLM |
|---|---|---|
| Model | Whisper large-v3-turbo (MIT) | Qwen2.5-7B-Instruct-AWQ (Apache 2.0) |
| Engine | faster-whisper, float16 | vLLM 0.8.5, 4-bit AWQ, guided JSON |
| GPU memory | ≈ 3–4 GB | ≈ 55% of GPU (setting) |
| Folder | `models/whisper/` | `models/hf/` |

Internet is needed only for installing and the first model download.

## Later: Callmaster integration (not part of this phase)

Callmaster will call `POST /v1/jobs` (or `/v1/process`) with `call_id`, audio and
`webhook_url`, and receive `transcript` + `audit` with the same audit JSON structure as today.
