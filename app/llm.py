"""Client for the LOCAL vLLM server.

The OpenAI Python SDK is used only as an HTTP client pointed at our own vLLM
container (LLM_URL). No OpenAI cloud key is used and nothing leaves the server.
"""
import logging
from typing import Optional

from app.config import settings

log = logging.getLogger(__name__)

LOCAL_DUMMY_KEY = "not-used-local-vllm"


class LLMError(Exception):
    """The local LLM could not be reached or returned an unusable reply."""


class LLMClient:
    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=settings.llm_url,
                api_key=LOCAL_DUMMY_KEY,
                timeout=settings.llm_timeout,
                max_retries=0,
            )
        return self._client

    def chat_json(self, system_prompt: str, user_content: str, schema: Optional[dict]) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        kwargs = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
        if settings.llm_guided_json and schema:
            # vLLM guided decoding: output is constrained to the schema
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "audit_result", "schema": schema},
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # report the real vLLM/GPU error, don't hide it
            raise LLMError(f"Local LLM request failed: {type(exc).__name__}: {exc}") from exc

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError("LLM output was cut off (max tokens or context length reached). "
                           "Increase LLM_MAX_TOKENS / VLLM_MAX_MODEL_LEN.")
        return choice.message.content or ""

    def is_available(self) -> bool:
        try:
            self.client.with_options(timeout=3).models.list()
            return True
        except Exception as exc:
            log.debug("LLM health check failed: %s", exc)
            return False
