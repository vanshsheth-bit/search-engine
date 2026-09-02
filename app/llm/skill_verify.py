"""LLM verification pass for semantic skill-similarity shortlist candidates.

See app/core/semantic.py's term_similarities docstring for why raw vector
similarity alone isn't precise enough to merge into real, unlabeled search
results: confirmed empirically that a cybersecurity candidate's skill list
scored within 0.006 of a genuine ML-skilled candidate for a "Scikit Learn"
query -- too close for any similarity threshold to safely separate signal
from noise.

This asks the model to actually REASON about each shortlisted candidate's
real, declared skill list -- not trust a bare vector distance -- batched
into ONE call per query term (not one call per candidate) so latency stays
bounded regardless of shortlist size.
"""
from __future__ import annotations

import json
import logging

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


def _build_prompt(term: str, shortlist: list[tuple[str, list[str]]]) -> str:
    lines = [f"{i}: {', '.join(skills[:40])}" for i, (_, skills) in enumerate(shortlist)]
    return (
        f'A recruiter is searching for candidates with genuine experience in "{term}". '
        "Below is a numbered list of candidates and their real, declared skills "
        "(nothing else -- no job descriptions). For each one, judge ONLY from the "
        "skills actually listed: does this candidate have experience a reasonable "
        f'recruiter would accept as satisfying "{term}" -- the exact tool, a well-known '
        "alias of it, or a closely equivalent sibling tool in the SAME specific "
        "category (e.g. a different but comparable ML framework counts; a general "
        "data-analysis or unrelated field does NOT)? Be strict -- when genuinely "
        "unsure, exclude rather than include.\n\n"
        "Return ONLY a JSON array of the index numbers that qualify, e.g. [0, 2]. "
        "If none qualify, return [].\n\n" + "\n".join(lines)
    )


def verify_skill_candidates(term: str, shortlist: list[tuple[str, list[str]]]) -> set[str]:
    """`shortlist`: [(candidate_id, skill_names), ...], already pre-filtered to
    a small top-N by embedding similarity (see semantic.term_similarities).
    Returns the subset of candidate_ids the LLM judges as genuinely
    satisfying `term`.

    Fails safe: on ANY error (timeout, bad JSON, Ollama down), returns an
    EMPTY set rather than guessing -- an unverified semantic hit must never
    reach real, unlabeled results just because verification itself broke;
    the search still returns its exact + taxonomy-related matches either
    way, it just doesn't gain the extra semantic ones this round."""
    if not shortlist:
        return set()
    prompt = _build_prompt(term, shortlist)
    try:
        resp = requests.post(
            f"{settings.ollama_url}/api/chat",
            json={
                "model": settings.model,
                "messages": [{"role": "user", "content": prompt}],
                "format": {"type": "array", "items": {"type": "integer"}},
                "stream": False,
                "think": False,
                "options": {"temperature": 0},
            },
            timeout=settings.llm_timeout,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        indices = json.loads(content)
        return {
            shortlist[i][0] for i in indices
            if isinstance(i, int) and 0 <= i < len(shortlist)
        }
    except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
        logger.warning("verify_skill_candidates failed for term=%r: %s", term, exc)
        return set()
