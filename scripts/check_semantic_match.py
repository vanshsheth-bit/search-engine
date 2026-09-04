"""Quick, cheap verification that the embedding pipeline is actually running
and computing real similarity numbers -- bypasses the slow LLM translation
step entirely (no 250-300s wait), so you can check this in ~5-10s.

Usage:
    .venv/Scripts/python.exe scripts/check_semantic_match.py <job_id> <term>

Example:
    .venv/Scripts/python.exe scripts/check_semantic_match.py 00000103 ETL
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from app.core.candidates import get_matched_candidates
from app.core.semantic import MIN_SIMILARITY, term_similarities
from app.core.skill_taxonomy import canonicalize, related_terms_for, skill_names_of


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    job_id, term = sys.argv[1], sys.argv[2]

    candidates = get_matched_candidates(job_id)
    if not candidates:
        print(f"No candidates found for job_id={job_id!r}")
        sys.exit(1)

    canon = canonicalize(term)
    exact, related = related_terms_for(canon)
    print(f"Taxonomy lookup for {term!r}: exact={exact} related={related}")
    if related or (exact - {canon.lower()}):
        print("  -> taxonomy already covers this term; embeddings are a moot point here.")
    else:
        print("  -> taxonomy has NOTHING for this term; only embeddings can widen it.")

    print(f"\nComputing real embeddings for {len(candidates)} candidates' skill lists...")
    sims = term_similarities(job_id, canon)
    if not sims:
        print("EMBEDDING CALL FAILED (Ollama down, or 'nomic-embed-text' not pulled).")
        print("  -> the semantic path is silently degrading to 'no extra matches' right now.")
        sys.exit(1)

    ranked = sorted(candidates, key=lambda c: -sims.get(c.get("id"), 0.0))
    print(f"\nTop 10 by similarity to {term!r} (threshold to qualify: {MIN_SIMILARITY}):\n")
    for c in ranked[:10]:
        sim = sims.get(c.get("id"), 0.0)
        skills = skill_names_of(c)
        has_literal = any(term.lower() in s.lower() for s in skills)
        flag = "(has literal term)" if has_literal else "(NO literal match -- pure semantic)"
        qualifies = "QUALIFIES" if sim >= MIN_SIMILARITY else "below threshold"
        print(f"  {sim:.3f}  {qualifies:15s}  {c['name']:30s} {flag}")


if __name__ == "__main__":
    main()
