"""Audit step: EXISTING audit prompt + transcript -> Qwen -> audit JSON.

IMPORTANT: there is deliberately NO audit logic in Python.
- The audit prompt (prompts/audit_prompt.txt) IS the business logic.
- The audit schema (schemas/audit_schema.json) defines the output.
- Python only loads the files, sends them to the LLM, and checks the JSON FORMAT.

Both files are re-read on every call, so replacing them needs no code change.
"""
import json
import logging
from typing import Optional, Tuple

from app.config import settings
from app.llm import LLMClient
from app.utils.validation import check_schema, parse_json_text, validate_against_schema

log = logging.getLogger(__name__)

# If the production prompt embeds the transcript inside the prompt text,
# put this marker where the transcript goes. Otherwise the prompt is sent as
# the SYSTEM message and the transcript as the USER message.
TRANSCRIPT_MARKER = "{{TRANSCRIPT}}"
PLACEHOLDER_PROMPT_PREFIX = "PLACEHOLDER AUDIT PROMPT"
PLACEHOLDER_SCHEMA_TITLE = "PLACEHOLDER_AUDIT_SCHEMA"


class AuditError(Exception):
    """The LLM did not return JSON that matches the audit schema."""


def load_prompt() -> str:
    text = settings.prompt_path.read_text(encoding="utf-8")
    if not text.strip():
        raise AuditError(f"Audit prompt file is empty: {settings.prompt_path}")
    return text


def load_schema() -> dict:
    schema = json.loads(settings.schema_path.read_text(encoding="utf-8"))
    check_schema(schema)
    return schema


def is_placeholder_prompt(prompt: str) -> bool:
    return prompt.lstrip().startswith(PLACEHOLDER_PROMPT_PREFIX)


def is_placeholder_schema(schema: dict) -> bool:
    return schema.get("title") == PLACEHOLDER_SCHEMA_TITLE


def build_messages(prompt: str, transcript: str) -> Tuple[str, str]:
    """Return (system_message, user_message). The prompt text is never modified
    except to insert the transcript at the marker if the marker is present."""
    if TRANSCRIPT_MARKER in prompt:
        return "", prompt.replace(TRANSCRIPT_MARKER, transcript)
    return prompt, transcript


def run_audit(transcript: str, call_id: str = "", llm: Optional[LLMClient] = None,
              max_attempts: int = 2) -> dict:
    """First attempt + one retry. Never fabricates missing values."""
    if not transcript or not transcript.strip():
        raise AuditError("Transcript is empty - nothing to audit")

    llm = llm or LLMClient()
    prompt = load_prompt()
    schema = load_schema()
    system_msg, user_msg = build_messages(prompt, transcript)
    log.debug("call_id=%s transcript sent to LLM:\n%s", call_id, transcript)

    errors = []
    for attempt in range(1, max_attempts + 1):
        raw = llm.chat_json(system_msg, user_msg, schema)  # LLMError propagates (connection/GPU problem)
        try:
            data = parse_json_text(raw)
            errors = validate_against_schema(data, schema)
        except ValueError as exc:
            errors = [str(exc)]
        if not errors:
            return data
        log.warning("call_id=%s attempt %d/%d: audit JSON invalid: %s",
                    call_id, attempt, max_attempts, "; ".join(errors[:5]))

    raise AuditError(f"LLM returned invalid audit JSON after {max_attempts} attempts: "
                     + "; ".join(errors[:5]))
