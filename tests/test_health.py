from app import main
from app.config import settings
from rq import Queue


def test_health_is_public_and_reports_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "callmaster-ai"
    assert body["stt_model"] == "large-v3-turbo"
    assert body["llm_model"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert set(body["checks"]) == {"redis", "stt", "llm"}
    assert body["checks"]["redis"] is True
    assert body["checks"]["llm"] is True


def test_health_degraded_when_no_worker(client):
    # No GPU worker registered in the fake Redis -> STT not ready
    body = client.get("/health").json()
    assert body["checks"]["stt"] is False
    assert body["status"] == "degraded"


def test_health_reports_placeholder_prompt_and_schema(client):
    files = client.get("/health").json()["audit_files"]
    assert files["prompt_loaded"] and files["schema_loaded"]
    # These flip to False automatically once production files are copied in
    assert isinstance(files["prompt_is_placeholder"], bool)
    assert isinstance(files["schema_is_placeholder"], bool)


def test_health_does_not_expose_secrets(client):
    assert settings.api_key not in client.get("/health").text


def test_health_when_redis_down(fake_llm):
    from fastapi.testclient import TestClient

    class DeadRedis:
        def ping(self):
            raise ConnectionError("redis down")

    main.app.dependency_overrides[main.get_redis] = lambda: DeadRedis()
    main.app.dependency_overrides[main.get_llm_client] = lambda: fake_llm
    try:
        body = TestClient(main.app).get("/health").json()
    finally:
        main.app.dependency_overrides.clear()
    assert body["checks"]["redis"] is False
    assert body["status"] == "degraded"
