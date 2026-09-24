import json
from pathlib import Path

import pytest

from app import audit
from app.config import APP_DIR
from app.utils.validation import check_schema, parse_json_text, validate_against_schema
from tests.conftest import FakeLLM, TEST_SCHEMA


# ---- the real files shipped in the repo (placeholder OR production) ----

def test_repo_prompt_file_loads():
    text = (APP_DIR / "prompts" / "audit_prompt.txt").read_text(encoding="utf-8")
    assert text.strip()


def test_repo_schema_file_is_valid_json_schema():
    schema = json.loads((APP_DIR / "schemas" / "audit_schema.json").read_text(encoding="utf-8"))
    check_schema(schema)


# ---- loading mechanism ----

def test_prompt_and_schema_are_loaded_dynamically(test_schema_files):
    prompt_path, schema_path = test_schema_files
    assert audit.load_prompt() == "Existing audit prompt. Return JSON."
    prompt_path.write_text("Changed prompt", encoding="utf-8")
    assert audit.load_prompt() == "Changed prompt"          # no restart needed
    assert audit.load_schema() == TEST_SCHEMA


def test_placeholder_detection():
    assert audit.is_placeholder_prompt("PLACEHOLDER AUDIT PROMPT\n...")
    assert not audit.is_placeholder_prompt("You are a call quality auditor...")
    assert audit.is_placeholder_schema({"title": "PLACEHOLDER_AUDIT_SCHEMA"})
    assert not audit.is_placeholder_schema(TEST_SCHEMA)


def test_prompt_sent_unchanged_as_system_and_transcript_as_user(test_schema_files):
    llm = FakeLLM()
    audit.run_audit("[0.00s] Agent: Sir good morning sir.", "C1", llm)
    call = llm.calls[0]
    assert call["system"] == "Existing audit prompt. Return JSON."
    assert call["user"] == "[0.00s] Agent: Sir good morning sir."
    assert call["schema"] == TEST_SCHEMA


def test_transcript_marker_inside_prompt(test_schema_files):
    prompt_path, _ = test_schema_files
    prompt_path.write_text("Audit this call:\n{{TRANSCRIPT}}\nReturn JSON.", encoding="utf-8")
    llm = FakeLLM()
    audit.run_audit("[0.00s] Agent: Hi", "C1", llm)
    assert llm.calls[0]["system"] == ""
    assert llm.calls[0]["user"] == "Audit this call:\n[0.00s] Agent: Hi\nReturn JSON."


# ---- validation + retry ----

def test_valid_json_returned_as_is(test_schema_files):
    out = audit.run_audit("t", "C1", FakeLLM(['{"greeting_done": "No", "summary": "x"}']))
    assert out == {"greeting_done": "No", "summary": "x"}


def test_invalid_then_valid_retries_once(test_schema_files):
    llm = FakeLLM(['{"greeting_done": "Maybe"}', '{"greeting_done": "Yes", "summary": "ok"}'])
    assert audit.run_audit("t", "C1", llm)["greeting_done"] == "Yes"
    assert len(llm.calls) == 2


def test_invalid_twice_fails_and_never_fabricates(test_schema_files):
    llm = FakeLLM(['{"greeting_done": "Yes"}'])     # missing "summary" every time
    with pytest.raises(audit.AuditError):
        audit.run_audit("t", "C1", llm)
    assert len(llm.calls) == 2


def test_non_json_reply_fails(test_schema_files):
    with pytest.raises(audit.AuditError):
        audit.run_audit("t", "C1", FakeLLM(["Sure! Here is the audit..."]))


def test_empty_transcript_rejected(test_schema_files):
    with pytest.raises(audit.AuditError):
        audit.run_audit("   ", "C1", FakeLLM())


def test_code_fence_tolerated():
    assert parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}


def test_enum_and_extra_field_errors_reported():
    errors = validate_against_schema({"greeting_done": "Maybe", "summary": "x", "extra": 1}, TEST_SCHEMA)
    assert any("Maybe" in e for e in errors)
    assert any("extra" in e for e in errors)


def test_no_audit_logic_in_python_source():
    """Guard: audit decisions must live in the prompt, not in Python."""
    src = (APP_DIR / "audit.py").read_text() + (APP_DIR / "pipeline.py").read_text()
    for forbidden in ("score =", "score+=", "score +=", "in transcript", "rude"):
        assert forbidden not in src
