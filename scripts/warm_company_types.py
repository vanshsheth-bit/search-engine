"""Warms the company-type classification caches. Two layers, in order of
value:

1. INDUSTRY cache (fast, high-coverage, run by default): 94.7% of this
   dataset's real companies already have a real "industry" on file in
   company_ranks.json (already streamed once for company_tier -- this reads
   the same pass, no extra file scan), and there are only ~130 distinct
   industries among them. Classifying that small, tractable set (the model
   reliably knows what "computer software" vs "banking" means) covers the
   vast majority of companies in a few batches, in minutes, not tens of
   minutes.

2. Per-company cache (slow, low-hit-rate, opt-in via --per-company): direct
   classification for individual company names. Real hit rate on this
   dataset was ~15% (most companies are small/regional -- the model
   correctly has no specific knowledge of them, honestly Unknown rather
   than guessed). Only worth running for the remainder with NO industry
   data at all -- pass --per-company to also run this (still incremental:
   already-cached names cost nothing on a re-run).

A direct per-company classification (whether from this or a manual
correction via scripts/set_company_type.py) always wins over the
industry-level inference at query time -- see company_type.py.

Requires Ollama running with the configured MODEL.

Run:
  .venv/Scripts/python.exe scripts/warm_company_types.py
  .venv/Scripts/python.exe scripts/warm_company_types.py --per-company
  .venv/Scripts/python.exe scripts/warm_company_types.py --per-company --limit 15
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

from app.core.candidates import (  # noqa: E402
    _load_company_industries_from_ranks,
    _load_raw_resumes,
    get_company_evidence,
)
from app.core.company_type import (  # noqa: E402
    warm_cache,
    warm_cache_with_evidence,
    warm_industry_cache,
)


def _progress_printer(label: str):
    done = 0

    def _progress(batch: dict) -> None:
        nonlocal done
        done += len(batch)
        sample = ", ".join(f"{k}={v}" for k, v in list(batch.items())[:3])
        print(f"  [{label}] ...{done} classified so far "
              f"(this batch: {sample}{'...' if len(batch) > 3 else ''})", flush=True)

    return _progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-company", action="store_true",
                         help="Also run the slower per-company fallback warmup "
                              "(for companies with no industry data at all)")
    parser.add_argument("--with-evidence", action="store_true",
                         help="Also run the evidence-based fallback: classify "
                              "companies still Unknown after the passes above "
                              "using real resume excerpts instead of bare "
                              "names (see company_type.warm_cache_with_evidence)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only warm the first N values (for testing)")
    args = parser.parse_args()

    print("=== Industry cache (primary, fast, high-coverage) ===")
    industry_map = _load_company_industries_from_ranks()
    industries = sorted(set(industry_map.values()))
    if args.limit:
        industries = industries[:args.limit]
    print(f"distinct industries found among real companies: {len(set(industry_map.values()))}"
          + (f" (warming first {len(industries)})" if args.limit else ""))

    t0 = time.time()
    n_new = warm_industry_cache(industries, progress_callback=_progress_printer("industry"))
    print(f"newly classified: {n_new} industries (rest already cached)")
    print(f"took {time.time() - t0:.1f}s")

    if not args.per_company:
        covered = len(industry_map)
        print(f"\nSkipping per-company fallback (pass --per-company to also run it).")
        print(f"{covered} real companies have industry data and will use it at query time.")
    else:
        print("\n=== Per-company fallback (slow, for companies with no industry data) ===")
        resumes = _load_raw_resumes()
        companies: set[str] = set()
        for r in resumes:
            for e in r.get("experience") or []:
                c = e.get("company")
                if isinstance(c, str) and c.strip():
                    companies.add(c.strip())

        # Only the ones industry-based lookup can't help at all.
        from app.core.candidates import _normalize_company  # noqa: E402
        no_industry = sorted(c for c in companies if _normalize_company(c) not in industry_map)
        if args.limit:
            no_industry = no_industry[:args.limit]
        print(f"companies with no industry data: {len(no_industry)} (of {len(companies)} total)")

        t0 = time.time()
        n_new = warm_cache(no_industry, progress_callback=_progress_printer("company"))
        print(f"newly classified: {n_new} companies (rest already cached)")
        print(f"took {time.time() - t0:.1f}s")

    if args.with_evidence:
        _run_evidence_pass(args.limit)


def _run_evidence_pass(limit: int | None) -> None:
    """Companies still Unknown (or never classified) get one more shot using
    REAL resume excerpts instead of a bare name -- see
    company_type.warm_cache_with_evidence's docstring for why this recovers
    companies the name-only passes above structurally can't (~85% Unknown
    in practice, confirmed on this dataset)."""
    print("\n=== Evidence-based fallback (companies still Unknown, using "
          "real resume excerpts) ===")
    evidence = get_company_evidence()
    if limit:
        evidence = dict(list(evidence.items())[:limit])
    print(f"companies with at least one resume excerpt available: {len(evidence)}")

    t0 = time.time()
    n_new = warm_cache_with_evidence(
        evidence, progress_callback=_progress_printer("evidence"),
    )
    print(f"newly classified: {n_new} companies (rest already cached or "
          f"had no evidence)")
    print(f"took {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
