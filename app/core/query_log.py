"""Persists the JSON produced for every recruiter query, one line per query.

Append-only JSONL (same convention as experience_index/*.jsonl) rather than
one file per query -- a recruiter session is many queries in a row, and a
single growing file is what you actually want to grep/tail/replay later.

Each line carries BOTH stages of translation, not just the end result:
`llm_raw` is exactly what the model returned (before any of service.py's
deterministic repairs -- _repair_incomplete_skill_experience,
_reclassify_skill_as_domain_when_its_a_position, taxonomy expansion, etc.),
and `response.filters` is what the system actually applied after those
repairs ran. Seeing both side by side in one place is what made several real
bugs this session findable at all (a skill_experience filter missing its
`skill` field, "DevOps" misrouted as a skill instead of a position) --
without the raw stage, the log only ever showed the ALREADY-repaired
result, indistinguishable from the model having gotten it right the first
time. `llm_raw` is null for a response that never called the LLM at all
(a pending lookup/confirm/clarify resolved deterministically from a short
reply, or an intent with no filter-producing branch).

Deliberately excludes `FilterResponse.candidates` (the full per-candidate
records, including real PII) -- this log exists to capture what the system
DID with a query, not to become a second copy of candidate data that then
needs its own retention/access controls. `n_candidates_returned` is kept
instead, which is enough to tell "this query matched 3 people" from a log
line without persisting who they were.

Logging must NEVER break a real request -- same fail-safe convention as
semantic.py/skill_verify.py (an unavailable log is a degraded feature, not
an outage). Every failure is caught and only logged as a warning.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
QUERY_LOG_PATH = Path(os.getenv("QUERY_LOG_PATH", str(_ROOT / "query_log.jsonl")))

_lock = threading.Lock()


def log_query(
    *,
    session_id: str,
    job_id: str,
    query: str,
    response,  # FilterResponse -- typed loosely to avoid a schemas.py import cycle risk
    raw_llm_output: dict | None = None,
) -> None:
    """Appends one JSON line: the query, the RAW LLM JSON (before any
    deterministic repair -- see module docstring; null if this query never
    reached the LLM), and the full JSON the system actually produced
    (status, repaired filters/chips, any clarify question, message) --
    everything in FilterResponse except the bulky per-candidate PII list,
    replaced with just its count."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "session_id": session_id,
        "job_id": job_id,
        "query": query,
        "llm_raw": raw_llm_output,
        "response": {
            "status": response.status,
            "total": response.total,
            "showing": response.showing,
            "logic": response.logic,
            "filters": [f.model_dump(exclude_none=True) for f in response.filters],
            "chips": [c.label for c in response.chips],
            "question": response.question,
            "options": response.options,
            "message": response.message,
            "suggestions": response.suggestions,
            "n_candidates_returned": len(response.candidates),
        },
    }
    try:
        with _lock:
            QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with QUERY_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("failed to write query log entry", exc_info=True)


def read_query_log(path: Path | str = QUERY_LOG_PATH) -> list[dict]:
    """Reads the log back, oldest first. Missing file -> [] (nothing logged
    yet), same graceful-absence convention as experience_index's loaders."""
    path = Path(path)
    if not path.is_file():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
