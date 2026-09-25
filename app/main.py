"""FastAPI service.

GPU work (Whisper) only happens in the worker process, so there is exactly
one Whisper model in GPU memory. /v1/transcribe and /v1/process put a job on
the queue and wait for it; /v1/jobs returns immediately (async).
/v1/audit talks to vLLM directly (no Whisper needed).
"""
import json
import logging
import secrets
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from redis import Redis
from rq import Queue, Retry, Worker
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app import audit as audit_module
from app.config import settings, setup_logging
from app.llm import LLMClient, LLMError
from app.utils import audio
from app.utils.validation import is_valid_call_id

setup_logging("api")
log = logging.getLogger("callmaster-ai.api")

app = FastAPI(title="Callmaster AI", version="1.0.0",
              description="Self-hosted speech-to-text + audit LLM service")

STT_READY_KEY = "callmaster:stt_ready"
JOB_STATES = {
    "queued": "queued", "deferred": "queued", "scheduled": "queued",
    "started": "processing",
    "finished": "completed",
    "failed": "failed", "stopped": "failed", "canceled": "failed",
}
RETRY_INTERVALS = [30, 60, 120]


# ---------- dependencies (overridden in unit tests) ----------

@lru_cache
def _redis_singleton() -> Redis:
    return Redis.from_url(settings.redis_url)


def get_redis() -> Redis:
    return _redis_singleton()


def get_queue(redis: Redis = Depends(get_redis)) -> Queue:
    return Queue(settings.queue_name, connection=redis, default_timeout=settings.job_timeout)


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()


def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="API_KEY is not configured on the server")
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ---------- helpers ----------

def _check_call_id(call_id: str) -> None:
    if not is_valid_call_id(call_id):
        raise HTTPException(status_code=422, detail="call_id may only contain letters, digits, '_', '-', '.' (max 128)")


def _save_upload(file: UploadFile, call_id: str) -> Path:
    if not audio.is_supported(file.filename):
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Allowed: {sorted(audio.SUPPORTED_EXTENSIONS)}")
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix.lower()
    dest = settings.audio_dir / f"{call_id}_{uuid.uuid4().hex[:8]}{ext}"
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with open(dest, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File larger than {settings.max_upload_mb} MB")
            out.write(chunk)
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return dest


def _state(job: Job) -> str:
    status = job.get_status(refresh=True)
    status = getattr(status, "value", status)
    return JOB_STATES.get(str(status), str(status))


def _error_of(job: Job) -> str:
    text = job.exc_info or ""
    lines = [line for line in str(text).strip().splitlines() if line.strip()]
    return lines[-1] if lines else "unknown error"


def _fetch(job_id: str, redis: Redis) -> Optional[Job]:
    try:
        return Job.fetch(job_id, connection=redis)
    except NoSuchJobError:
        return None


def _wait(job: Job, timeout: int) -> str:
    deadline = time.time() + timeout
    while True:
        state = _state(job)
        if state in ("completed", "failed") or time.time() >= deadline:
            return state
        time.sleep(1)


def _prepare_new_call(call_id: str, queue: Queue) -> None:
    """Refuse duplicates that are still running; clear old results otherwise."""
    old = _fetch(call_id, queue.connection)
    if old is not None:
        if _state(old) in ("queued", "processing"):
            raise HTTPException(status_code=409, detail=f"call_id {call_id} is already {_state(old)}")
        old.delete()
    (settings.results_dir / f"{call_id}.json").unlink(missing_ok=True)
    (settings.results_dir / f"{call_id}.transcript.json").unlink(missing_ok=True)


# ---------- endpoints ----------

@app.get("/health")
def health(redis: Redis = Depends(get_redis), llm: LLMClient = Depends(get_llm_client)):
    checks = {"redis": False, "stt": False, "llm": False}
    queued = None
    try:
        redis.ping()
        checks["redis"] = True
        checks["stt"] = bool(redis.get(STT_READY_KEY)) and Worker.count(connection=redis) > 0
        queued = Queue(settings.queue_name, connection=redis).count
    except Exception as exc:
        log.warning("health: redis check failed: %s", exc)
    checks["llm"] = llm.is_available()

    files = {"prompt_loaded": False, "schema_loaded": False,
             "prompt_is_placeholder": None, "schema_is_placeholder": None}
    try:
        files["prompt_is_placeholder"] = audit_module.is_placeholder_prompt(audit_module.load_prompt())
        files["prompt_loaded"] = True
    except Exception as exc:
        log.warning("health: prompt not loadable: %s", exc)
    try:
        files["schema_is_placeholder"] = audit_module.is_placeholder_schema(audit_module.load_schema())
        files["schema_loaded"] = True
    except Exception as exc:
        log.warning("health: schema not loadable: %s", exc)

    return {
        "status": "ok" if all(checks.values()) else "degraded",
        "service": "callmaster-ai",
        "stt_model": settings.stt_model,
        "llm_model": settings.llm_model,
        "checks": checks,
        "audit_files": files,
        "queue_length": queued,
    }


@app.post("/v1/transcribe", dependencies=[Depends(require_api_key)])
def transcribe(call_id: str = Form(...), file: UploadFile = File(...), queue: Queue = Depends(get_queue)):
    _check_call_id(call_id)
    path = _save_upload(file, call_id)
    log.info("call_id=%s /v1/transcribe accepted", call_id)
    job = queue.enqueue("app.pipeline.transcribe_job", call_id, str(path),
                        job_id=f"transcribe-{call_id}-{uuid.uuid4().hex[:6]}",
                        job_timeout=settings.job_timeout, result_ttl=3600, failure_ttl=86400)
    state = _wait(job, settings.sync_timeout)
    if state == "completed":
        return {"success": True, **job.return_value()}
    if state == "failed":
        return JSONResponse(status_code=500, content={"success": False, "call_id": call_id, "error": _error_of(job)})
    return JSONResponse(status_code=504, content={"success": False, "call_id": call_id,
                                                  "error": "Timed out waiting for transcription"})


class AuditRequest(BaseModel):
    call_id: str
    transcript: str
    prompt: Optional[str] = None  # empty = use app/prompts/audit_prompt.txt


def _audit(call_id: str, transcript: str, prompt: Optional[str], llm: LLMClient):
    _check_call_id(call_id)
    t0 = time.time()
    try:
        result = audit_module.run_audit(transcript, call_id=call_id, llm=llm, prompt=prompt)
    except audit_module.AuditError as exc:
        return JSONResponse(status_code=422, content={"success": False, "call_id": call_id, "error": str(exc)})
    except LLMError as exc:
        return JSONResponse(status_code=502, content={"success": False, "call_id": call_id, "error": str(exc)})
    return {"success": True, "call_id": call_id, "audit": result,
            "llm_model": settings.llm_model, "processing_seconds": round(time.time() - t0, 2)}


@app.post("/v1/audit", dependencies=[Depends(require_api_key)])
def audit_endpoint(body: AuditRequest, llm: LLMClient = Depends(get_llm_client)):
    """JSON body: call_id, transcript, optional prompt."""
    return _audit(body.call_id, body.transcript, body.prompt, llm)


@app.post("/v1/audit/form", dependencies=[Depends(require_api_key)])
def audit_form_endpoint(call_id: str = Form(...), transcript: str = Form(...),
                        prompt: Optional[str] = Form(None), llm: LLMClient = Depends(get_llm_client)):
    """Same as /v1/audit but with plain form fields - easy to paste text in /docs."""
    return _audit(call_id, transcript, prompt, llm)


@app.post("/v1/process", dependencies=[Depends(require_api_key)])
def process(call_id: str = Form(...), file: UploadFile = File(...),
            webhook_url: Optional[str] = Form(None), queue: Queue = Depends(get_queue)):
    """Full pipeline, waits for the result. If it takes longer than
    SYNC_TIMEOUT_SECONDS, returns 202 and the caller polls /v1/jobs/{call_id}."""
    _check_call_id(call_id)
    _prepare_new_call(call_id, queue)
    path = _save_upload(file, call_id)
    log.info("call_id=%s /v1/process accepted", call_id)
    job = queue.enqueue("app.pipeline.process_call", call_id, str(path), webhook_url or None,
                        job_id=call_id, job_timeout=settings.job_timeout,
                        result_ttl=86400, failure_ttl=7 * 86400)
    state = _wait(job, settings.sync_timeout)
    if state == "completed":
        return {"success": True, **job.return_value()}
    if state == "failed":
        return JSONResponse(status_code=500, content={"success": False, "call_id": call_id,
                                                      "status": "failed", "error": _error_of(job)})
    return JSONResponse(status_code=202, content={"success": True, "call_id": call_id, "status": state,
                                                  "poll_url": f"/v1/jobs/{call_id}"})


@app.post("/v1/jobs", status_code=202, dependencies=[Depends(require_api_key)])
def create_job(file: UploadFile = File(...), call_id: Optional[str] = Form(None),
               webhook_url: Optional[str] = Form(None), queue: Queue = Depends(get_queue)):
    call_id = call_id or uuid.uuid4().hex
    _check_call_id(call_id)
    _prepare_new_call(call_id, queue)
    path = _save_upload(file, call_id)
    queue.enqueue("app.pipeline.process_call", call_id, str(path), webhook_url or None,
                  job_id=call_id, job_timeout=settings.job_timeout,
                  retry=Retry(max=3, interval=RETRY_INTERVALS),
                  result_ttl=86400, failure_ttl=7 * 86400)
    log.info("call_id=%s job queued", call_id)
    return {"success": True, "call_id": call_id, "status": "queued", "poll_url": f"/v1/jobs/{call_id}"}


@app.get("/v1/jobs/{call_id}", dependencies=[Depends(require_api_key)])
def get_job(call_id: str, redis: Redis = Depends(get_redis)):
    _check_call_id(call_id)
    job = _fetch(call_id, redis)
    state = _state(job) if job else None

    if state in ("queued", "processing"):
        return {"call_id": call_id, "status": state, "retries_left": job.retries_left}

    result_file = settings.results_dir / f"{call_id}.json"
    if result_file.exists():
        return {"call_id": call_id, "status": "completed",
                "result": json.loads(result_file.read_text(encoding="utf-8"))}
    if state == "failed":
        return {"call_id": call_id, "status": "failed", "error": _error_of(job)}
    if state == "completed":
        return {"call_id": call_id, "status": "completed", "result": job.return_value()}
    raise HTTPException(status_code=404, detail="Unknown call_id")
