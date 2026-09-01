"""General, reusable mechanism for classifying real stored values (a company
name, a university name, a job title -- any string that already exists in
candidate data) against a new dimension a recruiter names, using the LLM's
own general knowledge, WITHOUT letting it hallucinate a confident-sounding
answer for something it doesn't actually know.

This exists because of a concrete, tested finding: asked to classify famous
companies (Google, Infosys, TCS...) as product- vs. service-based, qwen3:8b
gave correct, well-reasoned answers. Asked to classify obscure real
companies from this dataset (e.g. "Particle City", "LayereDefense"), it
NEVER once said "I don't know" -- it confidently invented a specific,
detailed-sounding justification for every one, indistinguishable in tone
from the correct answers about Google. Explicitly requiring an "Unknown"
category fixed this cleanly in the same test: famous companies still
classified correctly, obscure ones correctly came back Unknown instead of
fabricated.

Two things make this safe to use as a real filter, not just a curiosity:
1. "Unknown" is a REQUIRED, first-class category, not a hope -- the prompt
   makes fabrication explicitly disallowed, and any candidate/company that
   didn't get a real classification is treated as missing data (same
   semantics as a missing skill or an unmatched college_tier already have
   in this engine -- never a guess, never a crash).
2. Classification happens ONCE per distinct value, ever, then is cached
   permanently -- never repeated per query, and never repeated for a value
   already seen (even across server restarts, since the cache is a file, not
   just an in-memory lru_cache). Cost is proportional to how many genuinely
   NEW values show up as more data is ingested, not to how many queries or
   how many resumes exist in total.

This module only makes sense for Type-1 questions -- classifying something
that already exists in the data (a stored name) against a new dimension.
It does NOT apply to Type-2 questions with no underlying data at all
(salary, visa status, demographic traits) -- there is nothing to classify
there, and asking the model to guess would be fabricating a fact about a
real person from nothing, not classifying an existing one. Those stay
UNSUPPORTED_FILTER.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Callable

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _build_schema(categories: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string", "enum": categories + [UNKNOWN]},
                    },
                    "required": ["name", "category"],
                },
            },
        },
        "required": ["classifications"],
    }


def _default_llm_call(prompt: str, schema: dict, timeout: float) -> str:
    """Talks to Ollama directly for a one-off classification call -- separate
    from LLMClient (which is for query translation) since this is a
    different task with its own prompt/schema and no conversational state."""
    payload = {
        "model": settings.model,
        "messages": [{"role": "user", "content": prompt}],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": settings.num_ctx},
    }
    resp = requests.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def classify_new_values(
    values: list[str],
    *,
    dimension_description: str,
    categories: list[str],
    llm_call: Callable[[str, dict, float], str] = _default_llm_call,
    batch_size: int = 30,
    timeout: float = 120.0,
    max_retries: int = 2,
    on_batch_done: Callable[[dict[str, str]], None] | None = None,
) -> dict[str, str]:
    """Classify each of `values` (assumed to all be genuinely new -- callers
    are expected to filter out anything already cached first) into one of
    `categories` or UNKNOWN. Returns {original_value: category} -- but ONLY
    for values a real model response actually covered.

    Two different kinds of "we don't have an answer" must NOT be conflated:
    - The model responded, and genuinely classified a value as UNKNOWN
      (no reliable knowledge of it) -- a real, meaningful answer, safe to
      cache permanently (a company's product-vs-service nature doesn't
      change day to day).
    - The call itself failed (timeout, connection error, malformed JSON) --
      an infrastructure problem, not a judgment. Confirmed to happen in
      practice under GPU contention (a concurrent query competing for the
      same local model). These values are DROPPED from the result entirely,
      not defaulted to UNKNOWN -- caching an infrastructure failure as a
      real "unknown" classification would permanently and silently hide a
      company the model might classify correctly on a less-contended retry.
      Retried up to `max_retries` times per batch before giving up.

    `on_batch_done`, if given, is called with each batch's results as soon
    as that batch completes -- so a caller (see company_type.warm_cache) can
    persist progress incrementally. A long run (hundreds of companies,
    tens of minutes) that gets interrupted partway would otherwise lose
    everything, since nothing is written to disk until the whole call
    returns -- confirmed to matter in practice, not theoretical.
    """
    if not values:
        return {}

    schema = _build_schema(categories)
    out: dict[str, str] = {}

    for i in range(0, len(values), batch_size):
        batch = values[i:i + batch_size]
        prompt = (
            f"For each item below, classify it as {' or '.join(categories)} "
            f"regarding: {dimension_description}\n\n"
            "ONLY classify an item if you have specific, reliable knowledge "
            "of it -- not a guess based on how the name sounds. If you do "
            "not have real, specific knowledge of an exact item, you MUST "
            f"use category \"{UNKNOWN}\" for it instead of guessing. Do not "
            "invent a plausible-sounding reason for something you don't "
            "actually know.\n\n"
            "Return the \"name\" field EXACTLY as given below, unchanged, "
            "for every item -- one classification per item, same order.\n\n"
            + "\n".join(batch)
        )

        returned: dict[str, str] | None = None
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                content = llm_call(prompt, schema, timeout)
                data = json.loads(content)
                returned = {
                    _normalize(c["name"]): c["category"]
                    for c in data.get("classifications", [])
                    if isinstance(c, dict) and c.get("name") and c.get("category") in categories + [UNKNOWN]
                }
                break
            except (requests.RequestException, KeyError, json.JSONDecodeError, ValueError, TypeError) as exc:
                last_exc = exc
                logger.warning(
                    "Classification batch attempt %d/%d failed (%d items): %s",
                    attempt, max_retries, len(batch), exc,
                )

        if returned is None:
            logger.error(
                "Classification batch failed after %d attempts (%d items skipped, "
                "will retry on next warmup run): %s", max_retries, len(batch), last_exc,
            )
            continue  # do NOT cache these as Unknown -- just leave them unclassified

        batch_results = {v: returned.get(_normalize(v), UNKNOWN) for v in batch}
        out.update(batch_results)
        if on_batch_done:
            on_batch_done(batch_results)

    return out


class PersistentCache:
    """A simple on-disk {key: value} cache, loaded once and saved after every
    update -- so classification cost is paid at most once per value, ever,
    surviving server restarts. Not a database: fine for the scale here
    (thousands of distinct companies/universities, not millions)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._data: dict[str, str] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load cache %s: %s", self.path, exc)
                self._data = {}
        self._loaded = True

    def get_all(self) -> dict[str, str]:
        self._ensure_loaded()
        return dict(self._data)

    def update(self, new_entries: dict[str, str]) -> None:
        if not new_entries:
            return
        self._ensure_loaded()
        self._data.update(new_entries)
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, ensure_ascii=False, sort_keys=True)
        except OSError as exc:
            logger.warning("Failed to persist cache %s: %s", self.path, exc)
