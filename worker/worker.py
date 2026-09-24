"""GPU worker.

Loads Whisper ONCE at start-up and keeps it in GPU memory, then processes
jobs from the Redis queue. SimpleWorker runs jobs in this same process (no
fork per job), which is what keeps the model loaded between calls.
with_scheduler=True is required for the 30/60/120-second retry delays.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redis import Redis  # noqa: E402
from rq import Queue, SimpleWorker  # noqa: E402

from app import pipeline  # noqa: E402
from app.config import settings, setup_logging  # noqa: E402

STT_READY_KEY = "callmaster:stt_ready"


def main() -> None:
    setup_logging("worker")
    log = logging.getLogger("callmaster-ai.worker")
    redis = Redis.from_url(settings.redis_url)

    # Load Whisper now (not on the first call). Errors are shown, not hidden.
    _ = pipeline.get_transcriber().model
    redis.set(STT_READY_KEY, settings.stt_model)
    log.info("Worker ready: STT=%s LLM=%s queue=%s", settings.stt_model, settings.llm_model, settings.queue_name)

    try:
        worker = SimpleWorker([Queue(settings.queue_name, connection=redis)], connection=redis)
        worker.work(with_scheduler=True)
    finally:
        redis.delete(STT_READY_KEY)


if __name__ == "__main__":
    main()
