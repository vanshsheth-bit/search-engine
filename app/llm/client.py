"""LLM client: talks to a local Ollama server with schema-constrained decoding.

Design goals:
- Deterministic (temperature 0).
- Constrained output via Ollama's `format` JSON-schema param.
- Robust: timeout, bounded retries, graceful fallback to a CLARIFY response
  instead of crashing when the model or server misbehaves.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import requests

from app.core.config import settings
from app.llm.json_schema import build_filter_json_schema
from app.llm.prompt import build_system_prompt
from app.models.schemas import LLMOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = build_system_prompt()
_JSON_SCHEMA = build_filter_json_schema()


class LLMError(Exception):
    """Raised when the LLM cannot produce a usable structured result."""


class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.model = model or settings.model
        self.timeout = timeout or settings.llm_timeout
        self.max_retries = max_retries or settings.llm_max_retries

    def translate(self, query: str, current_filters: list[dict]) -> LLMOutput:
        """Translate a recruiter query into structured filter JSON."""
        user_msg = (
            f"CURRENT FILTERS: {json.dumps(current_filters)}\n"
            f"NEW QUERY: {query.strip()}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "format": _JSON_SCHEMA,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": settings.num_ctx},
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                data = json.loads(content)
                return LLMOutput.model_validate(data)
            except (requests.RequestException, KeyError, json.JSONDecodeError,
                    ValueError) as exc:
                last_err = exc
                logger.warning(
                    "LLM translate attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )

        # Graceful degradation: never crash the request. Ask the user to
        # rephrase rather than guessing or 500-ing.
        logger.error("LLM translate failed after retries: %s", last_err)
        return LLMOutput(
            intent="CLARIFY",
            question="I couldn't understand that. Could you rephrase your filter?",
            options=[],
        )
