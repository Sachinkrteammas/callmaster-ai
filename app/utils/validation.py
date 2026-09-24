"""Validation helpers. These check FORMAT only - never audit content."""
import json
import re
from typing import Any, List

from jsonschema.validators import validator_for

# call_id is used in file names, so only safe characters are allowed.
_CALL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def is_valid_call_id(call_id: str) -> bool:
    return bool(call_id) and bool(_CALL_ID_RE.match(call_id)) and ".." not in call_id


def parse_json_text(text: str) -> Any:
    """Parse the LLM reply as JSON. Tolerates a ```json ... ``` wrapper."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    if not cleaned:
        raise ValueError("LLM returned an empty response")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc}") from exc


def check_schema(schema: dict) -> None:
    """Raise jsonschema.SchemaError if the schema itself is broken."""
    validator_for(schema).check_schema(schema)


def validate_against_schema(data: Any, schema: dict) -> List[str]:
    """Return a list of human-readable errors (empty list = valid)."""
    validator_cls = validator_for(schema)
    validator = validator_cls(schema)
    errors = []
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in err.path) or "(root)"
        errors.append(f"{where}: {err.message}")
    return errors
