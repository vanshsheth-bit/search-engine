"""Runtime configuration, overridable via environment variables.

Loads a `.env` file from the project root (if present) before reading any
variable, so `MODEL=qwen3:8b` etc. can just be set once in `.env` instead of
exported in every shell. Real environment variables still take precedence
over `.env` (dotenv default), so CI/deploy overrides work as expected.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    # DESIGN TARGET: qwen3:8b (or an equivalent 8B-class instruct model --
    # see the model-choice note in prompt.py). The prompt, its rules, and its
    # few-shot examples are all written assuming a model in that reasoning
    # tier -- that's what actually generalizes to phrasing nobody
    # anticipated, not endless one-off examples. qwen2.5:1.5b below is NOT a
    # second design target; it's a dev-only stand-in because this particular
    # machine has no GPU and 8GB RAM, and confirmed can't serve an 8B model
    # within a usable timeout (100% CPU, one-word replies took 3+ minutes).
    # It will misparse things an 8B model handles fine -- expected, not a
    # bug to chase. Switch via `.env` (see `.env.example`) -- no code change.
    model: str = os.getenv("MODEL", "qwen2.5:1.5b")
    # 30s suits the small dev fallback model. On real 8B-class hardware,
    # bump this to ~60s -- a heavier model's occasional slow response
    # shouldn't fall back to a generic CLARIFY when it would have answered
    # correctly given a few more seconds.
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "30"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    num_ctx: int = int(os.getenv("NUM_CTX", "4096"))
    # Session TTL in seconds (in-memory store housekeeping).
    session_ttl: int = int(os.getenv("SESSION_TTL", "3600"))


settings = Settings()
