"""Embedding-based semantic similarity for skill matching.

Uses Ollama's local embedding model (nomic-embed-text) to catch what the
curated taxonomy (skill_taxonomy.py) structurally can't: a resume describing
genuinely equivalent experience in completely different words -- a novel
phrasing, an unlisted alias, a tool the taxonomy has no entry for at all.
Complements, never replaces, the taxonomy's curated widening, which runs
first and is preferred wherever it has an answer.

Raw similarity alone is NOT precise enough to decide a match on its own --
confirmed empirically a cybersecurity candidate's skill list scored within
0.006 of a genuine ML-skilled candidate for a "Scikit Learn" query. This
module only computes the similarity signal; see app/llm/skill_verify.py for
the LLM-reasoning pass that actually decides whether a shortlisted candidate
counts, and service.py for how the two combine into the real result."""
from __future__ import annotations

import logging
import math
import os
from functools import lru_cache

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

_EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# Calibrated empirically against this project's real resume data (see
# scripts/ -- similarity for a genuinely related skill pairing sits ~0.5-0.7
# with this model on short skill-list text; unrelated pairings drop below
# 0.4). Kept as an env var since the right cutoff is corpus-dependent and
# worth tuning once more real recruiter queries are observed.
_MIN_SIMILARITY = float(os.getenv("SEMANTIC_MIN_SIMILARITY", "0.55"))


def _embed(texts: list[str]) -> list[list[float]] | None:
    """None on any failure (Ollama down, model not pulled, etc.) -- this
    feature degrades to "no suggestions" silently rather than breaking the
    real search, which must never depend on the embedding model being up."""
    if not texts:
        return []
    try:
        resp = requests.post(
            f"{settings.ollama_url}/api/embed",
            json={"model": _EMBED_MODEL, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            return None
        return embeddings
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Embedding call failed, skipping semantic suggestions: %s", exc)
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _candidate_text(candidate: dict) -> str:
    """A short, embeddable signal for a candidate -- their real, stored
    skill list wrapped in a natural sentence, not a bare comma-joined list.

    Confirmed empirically this matters a lot: a raw list ("AWS, C++,
    chatgpt, ...") scored a genuine ML match (who has TensorFlow/XGBoost)
    BELOW a completely unrelated candidate for a "Scikit Learn" query
    (0.494 vs 0.498) -- nomic-embed-text is tuned for natural language, and
    keyword soup reads as noise to it. The same skill list framed as a
    sentence ("This candidate has experience with the following technical
    skills: ...") correctly ranked the genuine match #1 of 99 for the same
    query. Kept to the skill list (not full resume prose) so the embedding
    call stays cheap and the signal stays about tooling, not writing style."""
    skills = candidate.get("skills")
    if isinstance(skills, dict):
        names = list(skills.keys())
    elif isinstance(skills, (list, tuple)):
        names = [str(s) for s in skills]
    else:
        names = []
    if not names:
        return ""
    return "This candidate has experience with the following technical skills: " + ", ".join(names) + "."


@lru_cache(maxsize=32)
def _job_candidate_embeddings(job_id: str) -> tuple[tuple, ...]:
    """(candidate_id, embedding) pairs for every matched candidate on this
    job, computed ONCE per job and cached for the process lifetime -- so a
    session asking several queries against the same job only pays the
    embedding cost the first time, not per query (this dataset's jobs run
    to 100+ candidates; re-embedding all of them on every request would be
    real, avoidable latency on top of the LLM call that's already the
    bottleneck -- see SESSION_NOTES.md's caching discussion)."""
    from app.core.candidates import get_matched_candidates  # local: avoid import cycle at module load

    candidates = get_matched_candidates(job_id)
    ids, texts = [], []
    for c in candidates:
        t = _candidate_text(c)
        if t and c.get("id"):
            ids.append(c["id"])
            texts.append(t)
    if not texts:
        return ()
    embeddings = _embed(texts)
    if not embeddings:
        return ()
    return tuple(zip(ids, (tuple(e) for e in embeddings)))


def term_similarities(job_id: str, term: str) -> dict[str, float]:
    """Embeds `term` once and compares it against this job's pre-computed
    candidate skill embeddings (see _job_candidate_embeddings), returning
    {candidate_id: cosine_similarity} for every candidate on the job.

    Used to widen a specific-tool skill filter to also match a candidate
    whose skills read as the same thing in different words -- a case
    skill_taxonomy.py's curated relations structurally can't catch (a novel
    phrasing, an unlisted alias, a tool with no taxonomy entry at all).
    Callers decide the similarity threshold and how to combine this with
    exact/taxonomy-related matching (see service.py's fuzzy-matching pass);
    this function only computes the raw similarity, it never itself decides
    who counts as a match. Returns {} on any embedding failure (Ollama down,
    model not pulled) -- callers must treat that as "no signal", not an
    error, so search itself never depends on the embedding model being up.
    """
    if not term:
        return {}
    pairs = _job_candidate_embeddings(job_id)
    if not pairs:
        return {}
    # Same natural-sentence framing as _candidate_text, for the same reason
    # -- a bare term embeds noisily against nomic-embed-text; "Has
    # experience with X." measurably improved true-positive ranking in
    # testing (confirmed: from below an unrelated candidate to #1 of 99).
    query_embedded = _embed([f"Has experience with {term}."])
    if not query_embedded:
        return {}
    query_vec = query_embedded[0]
    return {cid: _cosine(query_vec, list(vec)) for cid, vec in pairs}


# Similarity at/above this counts as a genuine match when merged directly
# into search results, indistinguishable from an exact one -- see
# service.py. Exposed so it's calibrated once, empirically, against this
# model's real score distribution (see _MIN_SIMILARITY above).
MIN_SIMILARITY = _MIN_SIMILARITY
