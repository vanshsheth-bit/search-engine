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
    # anticipated, not endless one-off examples. qwen3:4b below is the
    # PRACTICAL default this project actually runs on: this machine has no
    # GPU and 8GB RAM, and confirmed can't serve qwen3:8b within a usable
    # timeout (100% CPU, one-word replies took 3+ minutes). 4B is a real,
    # workable middle ground on this hardware -- not a second design target,
    # just what's actually pulled and used. It will still misparse some
    # things an 8B model handles fine -- expected, not a bug to chase.
    # Switch via `.env` (see `.env.example`) -- no code change.
    model: str = os.getenv("MODEL", "qwen3:4b")
    # 30s suited the old 1.5B dev fallback. qwen3:4b needs much more room --
    # and more so now that NUM_CTX is correctly sized for the real ~9,400
    # token prompt instead of silently truncating it (see num_ctx below): a
    # live round-trip at num_ctx=12288 on this CPU-only/8GB machine measured
    # 204s. 120s undershoots that and just burns two failed retries (240s
    # wasted) before falling back to a generic CLARIFY the model would have
    # answered correctly given more time. On real 8B-class (GPU) hardware
    # this can come back down a lot.
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "240"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    # Separate from llm_timeout -- app/llm/skill_verify.py's verification
    # call runs with think:True (unlike the main translate() call, which
    # stays think:False for speed), and thinking is real, additional work:
    # confirmed live, a 2-candidate verification call took 225s WITH
    # thinking on this CPU-only hardware. think:False was tried there first
    # purely for speed, but confirmed broken: it made the model skip
    # reasoning entirely and rubber-stamp EVERY candidate as a match --
    # including ones whose only listed skills were "cooking, painting,
    # yoga, gardening" against a "Java" query. Correctness requires
    # thinking; thinking requires more time than the main call's timeout
    # comfortably allows for a shortlist of up to 8 candidates.
    skill_verify_timeout: float = float(os.getenv("SKILL_VERIFY_TIMEOUT", "400"))
    # Model for skill_verify.py's specific judgment call, independent of the
    # main `model` used for translation -- confirmed empirically on real
    # data that a smaller/faster model (qwen3:4b) strong at the structured
    # translation task is measurably WORSE at this different, harder
    # reasoning task ("is this candidate's real skill list genuinely
    # equivalent to the term asked for") -- e.g. matched lab/chemistry
    # technicians with zero programming skills against a plain "python"
    # search. That call is small and infrequent (a term + up to 8 skill
    # lists, not the full system prompt), so a stronger model here stays
    # fast even when `model` itself is set to something smaller/faster.
    skill_verify_model: str = os.getenv("SKILL_VERIFY_MODEL", "qwen3:8b")
    # REAL BUG, found and fixed: 4096 was silently too small. The v1 system
    # prompt (rules + FEW_SHOTS, app/llm/prompt.py) alone measures ~9,400
    # tokens (confirmed: build_system_prompt() is 37,570 chars) -- with
    # num_ctx=4096, Ollama had to truncate roughly HALF the prompt on every
    # v1-path call, before the user's query or any conversation history
    # even entered the count. Confirmed live: adding a new few-shot example
    # with the EXACT failing query text still didn't fix that query --
    # because the model was never actually seeing it. 12288 was already
    # used successfully earlier in this project (see skill_verify.py's
    # comment) but never promoted to the actual default, so a fresh
    # checkout silently regressed to the broken 4096. Leaves ~2,900 tokens
    # of headroom above the current prompt for history + query + response;
    # bump further if FEW_SHOTS grows or conversations run long.
    num_ctx: int = int(os.getenv("NUM_CTX", "12288"))
    # Session TTL in seconds (in-memory store housekeeping).
    session_ttl: int = int(os.getenv("SESSION_TTL", "3600"))
    # "v2" (default as of this measurement): the extraction-first schema --
    # the LLM only reports raw spans/buckets, and app/core/taxonomy.py
    # resolves them deterministically (app/llm/prompt_v2.py +
    # json_schema_v2.py + schema_v2_adapter.py). "v1": the older
    # direct-filter-emission prompt (app/llm/prompt.py + json_schema.py).
    #
    # Switched the default from v1 to v2 after a real, clean (no concurrent
    # requests -- see the CPU-only/8GB machine's single-request-at-a-time
    # Ollama server) latency measurement on this exact hardware, same query
    # both ways: v1's ~10,300-token prompt measured a 35.9s round-trip
    # (prompt_eval=3.5s(10803 tok), generate=30.2s(63 tok)); v2's ~4,800-
    # token prompt measured 19.9s (prompt_eval=1.8s(4786 tok),
    # generate=14.5s(71 tok)) for the same query -- ~45% faster overall.
    # The generation-phase speedup (not just prefill) is the bigger piece:
    # each decode step attends over the WHOLE context so far, so a shorter
    # system prompt speeds up every single output token, not just the
    # one-time prefill. v2 also received the fuller prompt treatment for
    # this project's two most recent features (alternative_groups,
    # seniority bands) specifically because of its larger token headroom --
    # it's no longer the less-tested path. Switch back via `.env` if v1's
    # accuracy on some query pattern is ever needed for comparison -- no
    # code change either way.
    prompt_schema: str = os.getenv("PROMPT_SCHEMA", "v2")
    # How long Ollama keeps this model resident in memory after a request,
    # before unloading it. Ollama's own default is 5 minutes -- short enough
    # that a normal gap between recruiter queries (reading results, typing
    # the next one) routinely let the model unload, so the NEXT request paid
    # a full reload (confirmed live: a "first request after idle" timed out
    # at 240s+ more than once this session, where model_load alone can be a
    # large fraction of that on this CPU-only/8GB machine). "30m" keeps it
    # loaded through a realistic idle gap between turns; raise further (or
    # use "-1" for "never unload") if this box is dedicated to this service.
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

    # Qdrant vector store for the experience index (docker-compose.yml runs
    # it locally). Collection name is versioned by embedding model + dim so
    # switching embedding models later can't silently mix incompatible
    # vectors into one collection.
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "experience_chunks_nomic768")


settings = Settings()
