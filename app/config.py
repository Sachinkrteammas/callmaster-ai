"""All configuration comes from environment variables (see .env.example).

Nothing here is hard-coded for a specific company, GPU or call centre.
"""
import logging
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

# Read settings from the project's .env file (values already set in the
# environment win). No Docker needed.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Settings:
    def __init__(self) -> None:
        env = os.getenv

        # Security
        self.api_key = env("API_KEY", "")

        # Speech-to-text (faster-whisper)
        self.stt_model = env("STT_MODEL", "large-v3-turbo")
        self.stt_device = env("STT_DEVICE", "cuda")
        self.stt_compute_type = env("STT_COMPUTE_TYPE", "float16")
        language = env("STT_LANGUAGE", "auto").strip()
        # None = Whisper detects the language automatically
        self.stt_language = None if language.lower() in ("", "auto") else language
        self.stt_beam_size = int(env("STT_BEAM_SIZE", "5"))
        self.stt_best_of = int(env("STT_BEST_OF", "5"))
        self.stt_vad_filter = _bool("STT_VAD_FILTER", True)
        self.stt_condition_on_previous_text = _bool("STT_CONDITION_ON_PREVIOUS_TEXT", False)
        self.stt_model_dir = env("STT_MODEL_DIR", str(PROJECT_DIR / "models" / "whisper"))
        # "hotwords" (applied to every 30 s window) or "initial_prompt" (first window only)
        self.stt_vocab_mode = env("STT_VOCAB_MODE", "hotwords")
        self.vocab = env("VOCAB", "").strip() or None

        # Speaker labels
        self.channel_1_speaker = env("CHANNEL_1_SPEAKER", "Agent")
        self.channel_2_speaker = env("CHANNEL_2_SPEAKER", "Customer")
        self.mono_speaker = env("MONO_SPEAKER", "Speaker")

        # Local LLM (vLLM, OpenAI-compatible)
        self.llm_url = env("LLM_URL", "http://127.0.0.1:8000/v1")
        self.llm_model = env("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
        self.llm_timeout = float(env("LLM_TIMEOUT_SECONDS", "300"))
        self.llm_max_tokens = int(env("LLM_MAX_TOKENS", "4096"))
        self.llm_temperature = float(env("LLM_TEMPERATURE", "0"))
        self.llm_guided_json = _bool("LLM_GUIDED_JSON", True)

        # Queue
        self.redis_url = env("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.queue_name = env("QUEUE_NAME", "calls")
        self.job_timeout = int(env("JOB_TIMEOUT_SECONDS", "1800"))
        self.sync_timeout = int(env("SYNC_TIMEOUT_SECONDS", "900"))

        # Files
        self.data_dir = Path(env("DATA_DIR", str(PROJECT_DIR / "data")))
        self.prompt_path = Path(env("AUDIT_PROMPT_PATH", str(APP_DIR / "prompts" / "audit_prompt.txt")))
        self.schema_path = Path(env("AUDIT_SCHEMA_PATH", str(APP_DIR / "schemas" / "audit_schema.json")))
        self.max_upload_mb = int(env("MAX_UPLOAD_MB", "300"))
        self.webhook_timeout = float(env("WEBHOOK_TIMEOUT_SECONDS", "15"))

        self.log_level = env("LOG_LEVEL", "INFO").upper()

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def speaker_for_channel(self, index: int) -> str:
        """index is 0-based: 0 -> CHANNEL_1_SPEAKER, 1 -> CHANNEL_2_SPEAKER."""
        return self.channel_1_speaker if index == 0 else self.channel_2_speaker


settings = Settings()


def setup_logging(name: str = "service") -> None:
    """Log to the screen and to data/logs/<name>.log.

    Never log API keys. Full transcripts are only logged at DEBUG level.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(settings.logs_dir / f"{name}.log", encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
