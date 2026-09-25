"""Pipeline executed inside the GPU worker.

Audio -> FFmpeg -> faster-whisper -> timestamped transcript (Agent/Customer)
      -> existing audit prompt + transcript -> Qwen (vLLM) -> validated audit JSON
"""
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from app.audit import run_audit
from app.config import settings
from app.llm import LLMClient
from app.stt import Transcriber, format_transcript, merge_segments
from app.utils import audio

log = logging.getLogger(__name__)

_transcriber: Optional[Transcriber] = None
_llm: Optional[LLMClient] = None


def get_transcriber() -> Transcriber:
    """One Whisper instance per worker process, kept alive between calls."""
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def transcribe_audio(call_id: str, audio_path: str) -> dict:
    t0 = time.time()
    info = audio.probe_audio(audio_path)
    log.info("call_id=%s audio duration=%ss channels=%s sample_rate=%s",
             call_id, info["duration_seconds"], info["channels"], info["sample_rate"])

    transcriber = get_transcriber()
    work_dir = tempfile.mkdtemp(prefix=f"stt_{call_id}_")
    per_channel, languages, speakers = [], {}, []
    try:
        log.info("call_id=%s STT start", call_id)
        for speaker, wav in audio.prepare_channels(audio_path, work_dir, info):
            segments, meta = transcriber.transcribe_channel(wav, speaker)
            per_channel.append(segments)
            languages[speaker] = meta
            speakers.append(speaker)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    segments = merge_segments(per_channel)
    seconds = round(time.time() - t0, 2)
    log.info("call_id=%s STT end segments=%d seconds=%.2f", call_id, len(segments), seconds)
    return {
        "call_id": call_id,
        "segments": segments,
        "transcript": format_transcript(segments),
        "audio": info,
        "channel_speakers": speakers,
        "languages": languages,
        "model": settings.stt_model,
        "processing_seconds": seconds,
    }


def transcribe_job(call_id: str, audio_path: str) -> dict:
    """Used by POST /v1/transcribe."""
    return transcribe_audio(call_id, audio_path)


def deliver_webhook(url: str, payload: dict) -> dict:
    """A webhook failure never re-runs the AI work; it is only logged."""
    try:
        resp = requests.post(url, json=payload, timeout=settings.webhook_timeout)
        resp.raise_for_status()
        return {"delivered": True, "status_code": resp.status_code}
    except Exception as exc:
        log.error("call_id=%s webhook failed: %s", payload.get("call_id"), exc)
        return {"delivered": False, "error": f"{type(exc).__name__}: {exc}"}


def process_call(call_id: str, audio_path: str, webhook_url: Optional[str] = None,
                 prompt: Optional[str] = None) -> dict:
    """Used by POST /v1/process and POST /v1/jobs. `prompt` = the client's audit
    prompt; empty = use app/prompts/audit_prompt.txt."""
    t0 = time.time()

    # If a retry happens after STT succeeded (e.g. LLM was temporarily down),
    # reuse the saved transcript instead of running Whisper again.
    cache_path = settings.results_dir / f"{call_id}.transcript.json"
    stt = None
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("audio_path") == audio_path:
            stt = cached
            log.info("call_id=%s reusing transcript from previous attempt", call_id)
    if stt is None:
        stt = transcribe_audio(call_id, audio_path)
        _write_json(cache_path, {**stt, "audio_path": audio_path})

    log.info("call_id=%s LLM start", call_id)
    t_llm = time.time()
    audit_json = run_audit(stt["transcript"], call_id=call_id, llm=get_llm(), prompt=prompt)
    llm_seconds = round(time.time() - t_llm, 2)
    log.info("call_id=%s LLM end seconds=%.2f", call_id, llm_seconds)

    total = round(time.time() - t0, 2)
    duration = stt["audio"].get("duration_seconds") or 0
    result = {
        "call_id": call_id,
        "status": "completed",
        "transcript": stt["transcript"],
        "segments": stt["segments"],
        "audit": audit_json,
        "stt_model": settings.stt_model,
        "llm_model": settings.llm_model,
        "audio": stt["audio"],
        "channel_speakers": stt["channel_speakers"],
        "languages": stt["languages"],
        "timings": {
            "audio_duration_seconds": duration,
            "stt_seconds": stt["processing_seconds"],
            "llm_seconds": llm_seconds,
            "total_seconds": total,
            "real_time_factor": round(total / duration, 3) if duration else None,
        },
        "processing_seconds": total,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    result_path = settings.results_dir / f"{call_id}.json"
    _write_json(result_path, result)

    if webhook_url:
        result["webhook"] = deliver_webhook(webhook_url, {k: v for k, v in result.items() if k != "webhook"})
        _write_json(result_path, result)

    cache_path.unlink(missing_ok=True)
    log.info("call_id=%s completed total_seconds=%.2f", call_id, total)
    return result
