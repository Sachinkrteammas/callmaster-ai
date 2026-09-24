"""Unit-test setup. No GPU, no Redis server, no vLLM needed:
Whisper, vLLM and Redis are replaced by fakes."""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_DATA = tempfile.mkdtemp(prefix="callmaster_test_")
os.environ.update({
    "API_KEY": "test-key",
    "DATA_DIR": _DATA,
    "LLM_MODEL": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "STT_MODEL": "large-v3-turbo",
    "CHANNEL_1_SPEAKER": "Agent",
    "CHANNEL_2_SPEAKER": "Customer",
    "LOG_LEVEL": "WARNING",
})

import json  # noqa: E402

import fakeredis  # noqa: E402
import pytest  # noqa: E402
from rq import Queue  # noqa: E402

from app import pipeline  # noqa: E402
from app.config import settings  # noqa: E402

API_HEADERS = {"X-API-Key": "test-key"}

TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "greeting_done": {"type": "string", "enum": ["Yes", "No"]},
        "summary": {"type": "string"},
    },
    "required": ["greeting_done", "summary"],
    "additionalProperties": False,
}


class FakeLLM:
    """Returns pre-set replies in order; records what it was sent."""

    def __init__(self, replies=None, available=True):
        self.replies = list(replies or ['{"greeting_done": "Yes", "summary": "Agent greeted."}'])
        self.calls = []
        self.available = available

    def chat_json(self, system_prompt, user_content, schema):
        self.calls.append({"system": system_prompt, "user": user_content, "schema": schema})
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def is_available(self):
        return self.available


class FakeWhisperModel:
    """Mimics faster_whisper.WhisperModel.transcribe."""

    def __init__(self, per_file=None):
        self.per_file = per_file or {}
        self.last_kwargs = None

    def transcribe(self, path, **kwargs):
        self.last_kwargs = kwargs
        name = Path(path).name
        segs = self.per_file.get(name, [(0.0, 1.0, "Hello.")])
        segments = (SimpleNamespace(start=s, end=e, text=t) for s, e, t in segs)
        return segments, SimpleNamespace(language="hi", language_probability=0.87)


@pytest.fixture
def test_schema_files(tmp_path, monkeypatch):
    """Point the service at a TEST prompt/schema (never the production files)."""
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Existing audit prompt. Return JSON.", encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(TEST_SCHEMA), encoding="utf-8")
    monkeypatch.setattr(settings, "prompt_path", prompt)
    monkeypatch.setattr(settings, "schema_path", schema)
    return prompt, schema


@pytest.fixture
def fake_redis():
    return fakeredis.FakeStrictRedis()


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(pipeline, "_llm", llm)
    return llm


@pytest.fixture
def fake_stt(monkeypatch):
    """Fake Whisper + fake FFmpeg step: stereo call with Agent/Customer."""
    from app.stt import Transcriber

    model = FakeWhisperModel({
        "channel1.wav": [(0.0, 2.0, "Sir good morning sir."), (3.2, 4.5, "Ma'am good morning ma'am.")],
        "channel2.wav": [(2.1, 2.8, "Hello."), (5.1, 5.4, "हां.")],
    })
    monkeypatch.setattr(pipeline, "_transcriber", Transcriber(model=model))
    monkeypatch.setattr(pipeline.audio, "probe_audio",
                        lambda p: {"duration_seconds": 10.0, "channels": 2, "sample_rate": 8000, "codec": "pcm"})

    def fake_prepare(path, work_dir, info):
        return [("Agent", f"{work_dir}/channel1.wav"), ("Customer", f"{work_dir}/channel2.wav")]

    monkeypatch.setattr(pipeline.audio, "prepare_channels", fake_prepare)
    return model


@pytest.fixture
def client(fake_redis, fake_llm):
    from fastapi.testclient import TestClient

    from app import main

    queue = Queue(settings.queue_name, connection=fake_redis, is_async=False)
    main.app.dependency_overrides[main.get_redis] = lambda: fake_redis
    main.app.dependency_overrides[main.get_queue] = lambda: queue
    main.app.dependency_overrides[main.get_llm_client] = lambda: fake_llm
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()
