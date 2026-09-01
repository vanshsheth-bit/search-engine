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
import time
from datetime import datetime
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

    def translate(
        self,
        query: str,
        current_filters: list[dict],
        history: list[dict] | None = None,
    ) -> LLMOutput:
        """Translate a recruiter query into structured filter JSON.

        `history`: recent real conversation turns ([{"role": "user"/
        "assistant", "content": ...}, ...], oldest first), replayed as
        actual prior chat messages so a short reply ("yes", "no", "actually
        6") resolves against whatever this system itself just said -- the
        model reasoning over real context, not a hand-coded extractor for
        every clarify shape."""
        user_msg = (
            f"CURRENT FILTERS: {json.dumps(current_filters)}\n"
            f"NEW QUERY: {query.strip()}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                *(history or []),
                {"role": "user", "content": user_msg},
            ],
            "format": _JSON_SCHEMA,
            "stream": False,
            # Thinking OFF: on this CPU-only hardware, thinking adds a
            # multi-minute tax to EVERY query regardless of complexity (even
            # "hi") without reliably fixing the one failure mode it was
            # tried for (a short confirm-reply like "Yes" to a pending
            # clarify) -- see service.py's PendingClarify.value, which
            # resolves that case deterministically instead, without any LLM
            # call at all, so it can't hallucinate or need to "think".
            "think": False,
            "options": {"temperature": 0, "num_ctx": settings.num_ctx},
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            t0_wall = datetime.now().isoformat(timespec="milliseconds")
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                elapsed = time.perf_counter() - t0
                resp.raise_for_status()
                body = resp.json()
                content = body["message"]["content"]
                data = json.loads(content)
                # Ollama's own breakdown (ns -> ms) -- separates prompt
                # prefill from actual token generation, so a slow request can
                # be diagnosed instead of just seen as "one big number".
                prompt_ms = body.get("prompt_eval_duration", 0) / 1e6
                eval_ms = body.get("eval_duration", 0) / 1e6
                load_ms = body.get("load_duration", 0) / 1e6
                logger.info(
                    "LLM query=%r start=%s attempt=%d/%d total=%.1fs "
                    "(model_load=%.0fms prompt_eval=%.0fms(%d tok) "
                    "generate=%.0fms(%d tok))",
                    query, t0_wall, attempt, self.max_retries, elapsed,
                    load_ms, prompt_ms, body.get("prompt_eval_count", 0),
                    eval_ms, body.get("eval_count", 0),
                )
                return LLMOutput.model_validate(data)
            except (requests.RequestException, KeyError, json.JSONDecodeError,
                    ValueError) as exc:
                elapsed = time.perf_counter() - t0
                last_err = exc
                logger.warning(
                    "LLM query=%r start=%s attempt %d/%d FAILED after %.1fs: %s",
                    query, t0_wall, attempt, self.max_retries, elapsed, exc,
                )

        # Graceful degradation: never crash the request. Ask the user to
        # rephrase rather than guessing or 500-ing.
        logger.error("LLM translate failed after retries: %s", last_err)
        return LLMOutput(
            intent="CLARIFY",
            question="I couldn't understand that. Could you rephrase your filter?",
            options=[],
        )
