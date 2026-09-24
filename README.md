# Callmaster AI — Self-Hosted Speech-to-Text + Audit Service

A standalone service that replaces **Deepgram** and **OpenAI** for Callmaster audits.
It does NOT change Callmaster. Callmaster will call this service later.

```
Audio (.wav/.mp3/.m4a)
   ↓  FFmpeg (split stereo: channel 1 = Agent, channel 2 = Customer)
   ↓  faster-whisper  large-v3-turbo  (GPU)
Timestamped transcript  "[0.00s] Agent: Sir good morning sir."
   ↓  + EXISTING audit prompt (app/prompts/audit_prompt.txt)
   ↓  Qwen2.5-7B-Instruct-AWQ on vLLM  (GPU, local)
Audit JSON  (validated against app/schemas/audit_schema.json)
```

There is **no audit logic in Python**. The audit prompt is the business logic;
Qwen executes it. Python only moves data and checks the JSON format.

---

## 1. What this project does

| Endpoint | What it does |
|---|---|
| `GET /health` | Is everything running? (public, no key) |
| `POST /v1/transcribe` | Audio → transcript only |
| `POST /v1/audit` | Transcript (text) → audit JSON only |
| `POST /v1/process` | Audio → transcript → audit JSON (waits for the result) |
| `POST /v1/jobs` | Same as process, but returns immediately (background job, 3 retries: 30/60/120 s) |
| `GET /v1/jobs/{call_id}` | Status/result of a background job: `queued`, `processing`, `completed`, `failed` |

All `/v1/*` endpoints need the header `X-API-Key`.

## 2. Architecture

Four Docker containers, all on one GPU server:

| Container | Job | Uses GPU |
|---|---|---|
| `redis` | Job queue (jobs survive restarts) | No |
| `vllm` | Runs the Qwen LLM, OpenAI-compatible API on port 8000 (internal only) | Yes (~55%) |
| `worker` | Loads Whisper once, does FFmpeg + transcription + calls vLLM | Yes |
| `api` | FastAPI on port 8080 — the only thing reachable from outside | No |

Only one Whisper model is ever in GPU memory (in the worker). `/v1/transcribe`
and `/v1/process` hand the work to the worker and wait for it.

## 3. Requirements

| Item | Minimum | Preferred |
|---|---|---|
| GPU | NVIDIA with 24 GB VRAM (e.g. L4, A10, RTX 4090/3090, A5000) | same |
| NVIDIA driver | 550 or newer (supports CUDA 12.4) | latest |
| CPU | 8 cores | more |
| RAM | 32 GB | 64 GB |
| Disk | 100 GB free (images + models ≈ 25 GB) + audio | 500 GB NVMe |
| OS | Ubuntu 22.04 or 24.04 | 22.04 |
| Access | SSH login with `sudo` | |

> **Beginner note — choosing a GPU server.** You need a *virtual machine* (VM) or
> a physical server where you get full Ubuntu with `sudo`. Cloud "GPU pods" that
> are themselves Docker containers (for example many RunPod/Vast "pods") usually
> **cannot run Docker inside**, so this setup will not work on them.
> Good options: AWS `g6.xlarge` (L4) / `g5.xlarge` (A10G), Google Cloud `g2-standard-8` (L4),
> Azure NV-series, or Indian providers such as E2E Networks (keeps data in India).
> Choose an **Ubuntu 22.04** image. Many cloud "Deep Learning" images already
> include the NVIDIA driver and Docker — you can then skip steps 4.1–5.2.

---

## STEP-BY-STEP DEPLOYMENT (do these in order)

Each step ends with a ✅ **Check**. Do not continue until the check passes.

### Step 0 — Connect to the server

From your laptop (Windows PowerShell, Mac or Linux terminal):

```bash
ssh ubuntu@YOUR_SERVER_IP          # user may be "ubuntu", "root" etc. — your provider tells you
```

If you were given a key file: `ssh -i path/to/key.pem ubuntu@YOUR_SERVER_IP`

✅ **Check:** the prompt changes to something like `ubuntu@gpu-server:~$`.
Every command below is typed **on the server**, unless it says "on your laptop".

### Step 1 — Update the server

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl ca-certificates
```

## 4. NVIDIA setup

### 4.1 Is the GPU driver already installed?

```bash
nvidia-smi
```

- If you see a table with your GPU name, memory (e.g. `23034MiB`) and
  `Driver Version: 550.xx` or higher → **driver is fine, go to 5.1**.
- If you see `command not found` → install the driver:

```bash
sudo ubuntu-drivers install
sudo reboot
```

Wait 1–2 minutes, `ssh` in again and run `nvidia-smi` again.

✅ **Check:** `nvidia-smi` shows your GPU, and "CUDA Version" in the top-right is **12.4 or higher**.

## 5. Docker setup

### 5.1 Install Docker

```bash
docker --version || curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
exit
```

`ssh` in again (this activates the group change), then:

```bash
docker run --rm hello-world
docker compose version
```

✅ **Check:** "Hello from Docker!" and a Compose version (v2.x) are printed.

### 5.2 Let Docker use the GPU (NVIDIA Container Toolkit)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

✅ **Check (most important check of all):**

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You must see the same GPU table as in 4.1. If you get an error, see
**section 16 (GPU troubleshooting)** before going further.

## 6. Installation (get the code onto the server)

**Option A — through Git (recommended):** see section 19 to push the project
to GitHub/GitLab from your laptop first, then on the server:

```bash
cd ~
git clone https://github.com/YOUR_ORG/callmaster-ai.git
cd callmaster-ai
```

For a private repository GitHub will ask for a username and a **Personal Access
Token** (not your password): GitHub → Settings → Developer settings → Personal access tokens.

**Option B — copy the zip file (quickest first time).** On your laptop:

```bash
scp callmaster-ai.zip ubuntu@YOUR_SERVER_IP:~
```

On the server:

```bash
sudo apt install -y unzip && unzip callmaster-ai.zip && cd callmaster-ai
```

✅ **Check:** `ls` shows `README.md  app  docker-compose.yml  tests  worker ...`

## 7. Environment variables

```bash
cp .env.example .env
openssl rand -hex 32          # copy the long random text this prints
nano .env                     # paste it after API_KEY=   then Ctrl+O, Enter, Ctrl+X
```

Important variables:

| Variable | Meaning | Default |
|---|---|---|
| `API_KEY` | Password Callmaster must send in `X-API-Key` | must change |
| `STT_MODEL` | Whisper model | `large-v3-turbo` |
| `STT_LANGUAGE` | `auto` = detect Hindi/English. Do not force `en`. | `auto` |
| `STT_BEAM_SIZE` / `STT_BEST_OF` | Accuracy vs speed | `5` / `5` |
| `STT_VAD_FILTER` | Skip silence | `true` |
| `STT_CONDITION_ON_PREVIOUS_TEXT` | `false` reduces Whisper repeat-loops | `false` |
| `CHANNEL_1_SPEAKER` / `CHANNEL_2_SPEAKER` | Who is on which stereo channel | `Agent` / `Customer` |
| `VOCAB` | Comma-separated domain words (company, product, campaign names) | empty |
| `LLM_MODEL` | Model served by vLLM | `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| `VLLM_GPU_MEMORY_UTILIZATION` | Share of GPU memory for vLLM | `0.55` |
| `VLLM_MAX_MODEL_LEN` | Max prompt + answer length (tokens) | `16384` |
| `API_BIND` | Network address the API listens on | `0.0.0.0` |

`.env` is in `.gitignore` — it is never committed.

## 8. Prompt replacement  (do this before real testing)

Copy the **production** audit prompt from the existing Callmaster/OpenAI code,
exactly as it is, into `app/prompts/audit_prompt.txt` (replace the whole file):

```bash
nano app/prompts/audit_prompt.txt
```

How it is sent to Qwen:
- Default: the prompt is the **system** message and the transcript is the **user** message.
- If Callmaster today puts the transcript **inside** the prompt text, put the
  marker `{{TRANSCRIPT}}` where the transcript goes; the whole prompt is then sent
  as one user message. Match whatever Callmaster does today.
- Our transcript has timestamps and speaker labels (`[2.10s] Customer: Hello.`).
  Deepgram's text may have looked different; that is fine, but keep it in mind
  when comparing results.

The file is re-read for every call — no restart or rebuild needed.

## 9. Schema replacement

Copy the production audit JSON **schema** into `app/schemas/audit_schema.json`.
Do not rename, remove or add fields; do not change types or enums.

If Callmaster does **not** have a formal JSON Schema today (for example it used
OpenAI "JSON mode" and the prompt just describes the fields), the schema must be
written from real production outputs and then checked by the person who owns the
audit. You can check that the file is a valid schema with:

```bash
docker compose run --rm api python3 -c "from app.audit import load_schema; load_schema(); print('schema OK')"
```

`/health` shows `"prompt_is_placeholder": false` and `"schema_is_placeholder": false`
once both real files are in place.

## 10. Start the service

```bash
docker compose config        # checks the compose file + .env; prints the full config, no errors
docker compose build         # builds the app image (5–15 min the first time)
docker compose up -d         # starts everything in the background
docker compose logs -f vllm  # watch the LLM start
```

The **first start downloads models**: Qwen ≈ 5.5 GB, Whisper ≈ 1.6 GB, vLLM image ≈ 10 GB.
This can take 10–30 minutes. Wait until the vllm log shows
`Application startup complete`. Press **Ctrl+C** to stop watching (the service keeps running).

The worker waits for vLLM to be healthy before loading Whisper (so the GPU memory
is shared correctly). Then:

```bash
docker compose logs -f worker   # wait for "Worker ready: STT=large-v3-turbo ..."
docker compose ps               # all 4 services "running" / vllm "healthy"
```

## 11. Check health

```bash
curl http://localhost:8080/health
```

✅ **Check:** `"status":"ok"` and `"checks":{"redis":true,"stt":true,"llm":true}`.

Make your API key easy to use in the next commands:

```bash
export API_KEY=$(grep ^API_KEY= .env | cut -d= -f2)
```

## 12. Test transcription

Copy a real recording to the server (on your laptop):
`scp sample_call.mp3 ubuntu@YOUR_SERVER_IP:~/callmaster-ai/data/audio/`

```bash
curl -X POST http://localhost:8080/v1/transcribe \
  -H "X-API-Key: $API_KEY" \
  -F "call_id=TEST001" \
  -F "file=@data/audio/sample_call.mp3"
```

Or print full details (duration, channels, sample rate, language, transcript):

```bash
docker compose exec worker python3 scripts/test_audio.py /data/audio/sample_call.mp3
```

✅ **Check:**
- Transcript is readable; Hindi appears where Hindi is spoken, English where English is spoken.
- On a stereo call, lines marked `Agent` really are the agent. **If they are reversed**,
  swap `CHANNEL_1_SPEAKER` / `CHANNEL_2_SPEAKER` in `.env`, then `docker compose up -d`.
- If Agent and Customer lines are **identical duplicates**, the dialer is writing the
  same mixed audio to both channels (fake stereo). Ask for true two-channel recordings.

## 13. Test audit

```bash
curl -X POST http://localhost:8080/v1/audit \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"call_id":"TEST001","transcript":"[0.00s] Agent: Sir good morning sir.\n[2.00s] Customer: Hello."}'
```

✅ **Check:** `"success": true` and an `audit` object matching your schema.

## 14. Test full pipeline

Wait for the result:

```bash
curl -X POST http://localhost:8080/v1/process \
  -H "X-API-Key: $API_KEY" \
  -F "call_id=TEST002" \
  -F "file=@data/audio/sample_call.mp3"
```

Background job (how Callmaster should use it in production):

```bash
curl -X POST http://localhost:8080/v1/jobs -H "X-API-Key: $API_KEY" \
  -F "call_id=TEST003" -F "file=@data/audio/sample_call.mp3" \
  -F "webhook_url=https://callmaster.internal/ai-callback"      # optional

curl http://localhost:8080/v1/jobs/TEST003 -H "X-API-Key: $API_KEY"
```

Results are also saved as files: `data/results/<call_id>.json`.

If the webhook fails, the result is still saved and the job stays `completed`
(the AI work is not repeated); the error is in the logs and in `result.webhook`.

### GPU integration tests

```bash
docker compose exec -e RUN_GPU_TESTS=1 -e API_URL=http://api:8080 -e SAMPLE_AUDIO=/data/audio/sample_call.mp3 \
  worker sh -c "pip3 install -q pytest && python3 -m pytest tests/integration -v -s"
```

## 15. Check logs

```bash
docker compose logs -f api
docker compose logs -f worker
docker compose logs -f vllm
ls data/logs/            # api.log, worker.log
watch -n 2 nvidia-smi    # live GPU memory/usage (Ctrl+C to exit)
```

Logs include call_id, audio duration, channels, STT/LLM start and end, total time and errors.
API keys are never logged; full transcripts are only logged when `LOG_LEVEL=DEBUG`.

Other everyday commands:

```bash
docker compose ps           # what is running
docker compose restart      # restart all
docker compose down         # stop all
docker compose up -d --build   # after changing code or .env
```

## 16. GPU troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi: command not found` | `sudo ubuntu-drivers install && sudo reboot` |
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing/not configured → redo step 5.2, then `sudo systemctl restart docker` |
| `docker: permission denied` | You skipped `usermod -aG docker $USER`, or didn't log out/in |
| vLLM: `CUDA out of memory` or "not enough KV cache" | Lower `VLLM_MAX_MODEL_LEN` (e.g. 8192) or change `VLLM_GPU_MEMORY_UTILIZATION`; make sure nothing else uses the GPU (`nvidia-smi`) |
| worker: `CUDA out of memory` | Lower `VLLM_GPU_MEMORY_UTILIZATION` (e.g. 0.50) so Whisper has room |
| worker: `libcudnn_ops.so.9 ... cannot open` | The image's cuDNN doesn't match; keep the Dockerfile base image and `ctranslate2==4.5.0` as pinned |
| `CUDA driver version is insufficient` | Driver too old for CUDA 12.4 → install driver ≥ 550 |
| vLLM `unhealthy` for a long time | First download is big; check `docker compose logs vllm`. Healthcheck allows 15 min |
| LLM output "cut off" error | Increase `LLM_MAX_TOKENS` and/or `VLLM_MAX_MODEL_LEN` |
| vLLM rejects the schema (guided decoding error) | Some JSON Schema features aren't supported; set `LLM_GUIDED_JSON=false` (output is still validated) |
| Very long calls fail in the LLM | Transcript is longer than `VLLM_MAX_MODEL_LEN`; raise it if GPU memory allows |

## 17. Model information

| | Speech-to-text | Audit LLM |
|---|---|---|
| Model | Whisper large-v3-turbo (OpenAI weights, MIT) | Qwen2.5-7B-Instruct-AWQ (Apache 2.0) |
| Runtime | faster-whisper / CTranslate2, float16 | vLLM, 4-bit AWQ |
| GPU memory | ≈ 3–4 GB | ≈ 55% of the GPU (set by `VLLM_GPU_MEMORY_UTILIZATION`) |
| Stored in | `models/whisper/` | `models/hf/` |
| Languages | Auto-detected; handles Hindi, English, mixed | Multilingual |

Models are downloaded once and reused. Internet is needed only for the first start.

## 18. Performance benchmark

Put a few real recordings in `data/audio/`, then:

```bash
docker compose exec -e API_URL=http://api:8080 worker \
  python3 tests/benchmark.py /data/audio/call1.mp3 /data/audio/call2.mp3
```

Prints measured audio duration, STT time, LLM time, total and real-time factor (RTF).
Nothing is assumed — run it on 20–50 real calls before deciding capacity.

## Unit tests (no GPU needed — run on your laptop or the server)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

Whisper, vLLM and Redis are replaced by fakes; FFmpeg tests run if FFmpeg is installed.

## 19. Git deployment

The project already contains a first commit. To put it on GitHub/GitLab (on your laptop):

1. Create an **empty private** repository on GitHub/GitLab (no README).
2. In the project folder:

```bash
git remote add origin https://github.com/YOUR_ORG/callmaster-ai.git
git push -u origin main
```

Later updates: change code on the laptop → `git commit -am "message"` → `git push`;
on the server → `cd ~/callmaster-ai && git pull && docker compose up -d --build`.
`.env`, audio, results and models are never committed.

## Security checklist

- Change `API_KEY`. Never commit `.env`.
- Port 8000 (vLLM) is not published; only 8080 is.
- **Docker bypasses `ufw` firewall rules for published ports.** To restrict 8080,
  either set `API_BIND` in `.env` to the server's private IP, or use your cloud
  provider's security group / firewall to allow 8080 only from the Callmaster server.
- Delete old audio regularly, e.g. daily cron (`crontab -e`):
  `0 2 * * * find /home/ubuntu/callmaster-ai/data/audio -type f -mtime +30 -delete`

## Future Callmaster integration (NOT done in this phase)

Callmaster will later call `POST /v1/jobs` (or `/v1/process`) with `call_id`,
the audio file and a `webhook_url`, and receive `transcript` + `audit` (same audit
JSON structure as today). Callmaster itself is not modified by this project.
