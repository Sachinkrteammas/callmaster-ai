import io
import json

from app.config import settings
from tests.conftest import API_HEADERS


def _upload(name="call.wav"):
    return {"file": (name, io.BytesIO(b"RIFF....fake audio bytes"), "audio/wav")}


def test_api_key_required(client):
    assert client.post("/v1/transcribe", data={"call_id": "A1"}, files=_upload()).status_code == 401
    r = client.post("/v1/transcribe", data={"call_id": "A1"}, files=_upload(), headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    assert client.get("/v1/jobs/A1").status_code == 401
    assert client.post("/v1/audit", data={"transcript": "x", "prompt": "p"}).status_code == 401


def test_unsafe_call_id_rejected(client):
    r = client.post("/v1/transcribe", data={"call_id": "../../etc"}, files=_upload(), headers=API_HEADERS)
    assert r.status_code == 422


def test_unsupported_file_type(client):
    r = client.post("/v1/transcribe", data={"call_id": "A1"}, files=_upload("notes.txt"), headers=API_HEADERS)
    assert r.status_code == 415


def test_transcribe_endpoint(client, fake_stt):
    r = client.post("/v1/transcribe", data={"call_id": "TEST001"}, files=_upload("sample.mp3"), headers=API_HEADERS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True and body["call_id"] == "TEST001"
    assert body["model"] == "large-v3-turbo"
    assert body["transcript"].splitlines() == [
        "[0.00s] Agent: Sir good morning sir.",
        "[2.10s] Customer: Hello.",
        "[3.20s] Agent: Ma'am good morning ma'am.",
        "[5.10s] Customer: हां.",
    ]
    assert body["segments"][0] == {"speaker": "Agent", "start": 0.0, "end": 2.0, "text": "Sir good morning sir."}
    assert "processing_seconds" in body


def test_audit_endpoint(client, fake_llm):
    fake_llm.replies = ['{"Opening": 1, "SaleDone": 0}']
    r = client.post("/v1/audit", headers=API_HEADERS,
                    data={"call_id": "P1", "transcript": "[0.00s] Agent: Hi.", "prompt": "Client prompt"})
    assert r.status_code == 200, r.text
    assert r.json()["audit"] == {"Opening": 1, "SaleDone": 0}
    assert fake_llm.calls[0]["system"] == "Client prompt"
    assert fake_llm.calls[0]["user"] == "[0.00s] Agent: Hi."
    assert fake_llm.calls[0]["schema"] is None


def test_audit_call_id_optional(client, fake_llm):
    r = client.post("/v1/audit", headers=API_HEADERS, data={"transcript": "x", "prompt": "p"})
    assert r.status_code == 200, r.text
    assert r.json()["call_id"]


def test_audit_prompt_required(client, fake_llm):
    r = client.post("/v1/audit", headers=API_HEADERS, data={"transcript": "x"})
    assert r.status_code == 422


def test_audit_non_object_gives_422(client, fake_llm):
    fake_llm.replies = ['[1, 2]', '[1, 2]']
    r = client.post("/v1/audit", headers=API_HEADERS, data={"transcript": "x", "prompt": "p"})
    assert r.status_code == 422 and r.json()["success"] is False


def test_process_with_client_prompt(client, fake_stt, fake_llm, test_schema_files):
    fake_llm.replies = ['{"Opening": 1}']
    r = client.post("/v1/process", data={"call_id": "CP1", "prompt": "Client prompt"},
                    files=_upload(), headers=API_HEADERS)
    assert r.status_code == 200, r.text
    assert r.json()["audit"] == {"Opening": 1}
    assert fake_llm.calls[-1]["system"] == "Client prompt"
    assert fake_llm.calls[-1]["schema"] is None


def test_process_end_to_end_saves_result(client, fake_stt, fake_llm, test_schema_files):
    r = client.post("/v1/process", data={"call_id": "CALL123"}, files=_upload(), headers=API_HEADERS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["audit"] == {"greeting_done": "Yes", "summary": "Agent greeted."}
    assert body["stt_model"] == "large-v3-turbo"
    assert body["llm_model"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert set(body["timings"]) >= {"audio_duration_seconds", "stt_seconds", "llm_seconds", "total_seconds", "real_time_factor"}
    # transcript sent to the LLM is exactly the Whisper transcript
    assert fake_llm.calls[-1]["user"] == body["transcript"]
    saved = json.loads((settings.results_dir / "CALL123.json").read_text(encoding="utf-8"))
    assert saved["audit"] == body["audit"]
    assert not (settings.results_dir / "CALL123.transcript.json").exists()


def test_webhook_failure_keeps_result(client, fake_stt, fake_llm, test_schema_files):
    r = client.post("/v1/process", headers=API_HEADERS, files=_upload(),
                    data={"call_id": "CALLWH", "webhook_url": "http://127.0.0.1:9/nothing-here"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "completed"
    assert body["webhook"]["delivered"] is False
    assert (settings.results_dir / "CALLWH.json").exists()


def test_process_invalid_audit_marks_failed(client, fake_stt, fake_llm, test_schema_files):
    fake_llm.replies = ["not json"]
    r = client.post("/v1/process", data={"call_id": "BAD1"}, files=_upload(), headers=API_HEADERS)
    assert r.status_code == 500
    assert r.json()["status"] == "failed"
    assert "invalid audit JSON" in r.json()["error"]
    assert not (settings.results_dir / "BAD1.json").exists()


def test_async_job_create_and_fetch(client, fake_stt, fake_llm, test_schema_files):
    r = client.post("/v1/jobs", data={"call_id": "JOB1"}, files=_upload(), headers=API_HEADERS)
    assert r.status_code == 202
    assert r.json()["call_id"] == "JOB1"
    r = client.get("/v1/jobs/JOB1", headers=API_HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    assert r.json()["result"]["audit"]["greeting_done"] == "Yes"


def test_async_job_generates_call_id(client, fake_stt, fake_llm, test_schema_files):
    r = client.post("/v1/jobs", files=_upload(), headers=API_HEADERS)
    assert r.status_code == 202 and len(r.json()["call_id"]) == 32


def test_unknown_job_404(client):
    assert client.get("/v1/jobs/NOPE", headers=API_HEADERS).status_code == 404
