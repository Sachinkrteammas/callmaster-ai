"""GPU integration tests against the RUNNING Docker stack on the GPU server.

    export RUN_GPU_TESTS=1
    export API_KEY=...            # same as in .env
    export SAMPLE_AUDIO=/path/to/real_call.mp3
    pytest tests/integration -v

Optional: EXPECT_WORDS="madam,website,login" checks those words appear.
"""
import os

import pytest
import requests

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1 on the GPU server"),
]

URL = os.getenv("API_URL", "http://localhost:8080").rstrip("/")
HEADERS = {"X-API-Key": os.getenv("API_KEY", "")}


def test_health_all_green():
    body = requests.get(f"{URL}/health", timeout=10).json()
    assert body["checks"] == {"redis": True, "stt": True, "llm": True}, body


def test_real_recording_end_to_end():
    path = os.environ["SAMPLE_AUDIO"]
    with open(path, "rb") as f:
        r = requests.post(f"{URL}/v1/process", headers=HEADERS, timeout=3600,
                          data={"call_id": "gpu-integration-test"}, files={"file": f})
    body = r.json()
    assert r.status_code == 200, body
    assert body["transcript"].strip()
    speakers = {s["speaker"] for s in body["segments"]}
    if body["audio"]["channels"] >= 2:
        assert speakers <= {os.getenv("CHANNEL_1_SPEAKER", "Agent"), os.getenv("CHANNEL_2_SPEAKER", "Customer")}
    assert isinstance(body["audit"], dict) and body["audit"]
    for word in filter(None, os.getenv("EXPECT_WORDS", "").split(",")):
        assert word.strip().lower() in body["transcript"].lower(), f"'{word}' not in transcript"
    print("\n" + body["transcript"])
    print(body["timings"])
