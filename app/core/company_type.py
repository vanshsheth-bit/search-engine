"""Company product-vs-service classification -- the first real instance of
the general mechanism in llm_classifier.py. See that module's docstring for
why this is safe (mandatory "Unknown" category, cached once per company
forever) where a live per-query guess would not have been.

A candidate can have multiple companies with DIFFERENT types (e.g. Infosys
then Google) -- unlike company_tier (which picks the single best tier),
there's no "best" type, so a candidate's company_type is the SET of types
across all their companies, matched like skill/university (does the
candidate have ANY company of the asked-for type), not a single winner.

SCALE: classifying company names one at a time doesn't reach far -- most
real companies are small/regional and the model correctly has no specific
knowledge of them (confirmed: only ~15% of this dataset's real companies
got a real classification this way, the rest honestly Unknown). The fix
isn't asking harder, it's asking a SMALLER, more tractable question:
company_ranks.json (already streamed once for company_tier) carries a real
"industry" field for 94.7% of this dataset's real companies, and there are
only ~130 distinct industries among them -- a small, well-known set
("computer software", "banking") the model reliably knows, unlike specific
company names. So: classify the ~130 industries ONCE (this module), look up
each company's real industry (candidates.py, from data already on disk),
and use industry-based inference as a FALLBACK when a company has no direct
classification of its own. A direct, specific classification for a company
(from the per-company cache, or a manual correction via
scripts/set_company_type.py) always wins over the industry-level inference,
since it's more specific.
"""
from __future__ import annotations

import os
from functools import lru_cache

from app.core.llm_classifier import UNKNOWN, PersistentCache, classify_new_values

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_CACHE_PATH = os.getenv(
    "COMPANY_TYPE_CACHE_PATH", os.path.join(_ROOT, "company_type_cache.json")
)
_INDUSTRY_CACHE_PATH = os.getenv(
    "INDUSTRY_TYPE_CACHE_PATH", os.path.join(_ROOT, "industry_type_cache.json")
)

CATEGORIES = ["Product", "Service", "Both"]


@lru_cache(maxsize=1)
def _cache() -> PersistentCache:
    return PersistentCache(_CACHE_PATH)


@lru_cache(maxsize=1)
def _industry_cache() -> PersistentCache:
    return PersistentCache(_INDUSTRY_CACHE_PATH)


def company_types_for(
    company_names: list[str], industry_lookup: dict[str, str] | None = None
) -> list[str]:
    """Returns the distinct set of types (from CATEGORIES) across these
    company names, using only cached data -- never calls the LLM live
    during a request.

    `industry_lookup`: optional {normalized company name: industry} map,
    already loaded by the caller (candidates.py, from the SAME
    company_ranks.json pass that reads company_tier -- passed in rather
    than reloaded here, to avoid a second full scan of a ~900MB file).
    Used as a fallback ONLY when a company has no direct classification of
    its own: a specific classification always wins over an industry-level
    inference, since it's more specific and could be a manual correction.

    Companies with no classification available at all (direct or via
    industry) are simply omitted, same as any other missing-data field in
    this engine: absent, not guessed."""
    if not company_names:
        return []
    direct = _cache().get_all()
    industry_types = _industry_cache().get_all() if industry_lookup else {}
    types = set()
    for name in company_names:
        norm = name.strip().lower()
        t = direct.get(norm)
        if t and t != UNKNOWN:
            types.add(t)
            continue
        if industry_lookup:
            industry = industry_lookup.get(norm)
            if industry:
                it = industry_types.get(industry)
                if it and it != UNKNOWN:
                    types.add(it)
    return sorted(types)


def _warm(cache: PersistentCache, values: list[str], dimension_description: str,
          progress_callback=None) -> int:
    """Shared warmup logic for both the per-company and per-industry caches:
    classify whichever of `values` aren't already cached, persisting
    incrementally per batch (see llm_classifier.classify_new_values --
    surviving an interruption partway through matters for a run over
    hundreds of values). Returns how many NEW classifications were made."""
    existing = cache.get_all()
    distinct = {v.strip().lower(): v.strip() for v in values if v and v.strip()}
    new_values = [orig for key, orig in distinct.items() if key not in existing]
    if not new_values:
        return 0

    def _on_batch_done(batch: dict[str, str]) -> None:
        cache.update({k.strip().lower(): v for k, v in batch.items()})
        if progress_callback:
            progress_callback(batch)

    results = classify_new_values(
        new_values, dimension_description=dimension_description,
        categories=CATEGORIES, on_batch_done=_on_batch_done,
    )
    return len(results)


def warm_cache(company_names: list[str], progress_callback=None) -> int:
    """Classify whichever of these company names aren't already directly
    cached. Safe to call repeatedly (e.g. once per server startup, or from a
    standalone warmup script) -- already-cached names cost nothing.

    `progress_callback(batch_results: dict[str, str])`, if given, is called
    after each batch is persisted -- lets a caller (e.g. the warmup script)
    print progress on a long run instead of it looking hung for tens of
    minutes with no output."""
    return _warm(
        _cache(), company_names,
        dimension_description=(
            "whether the company is primarily a PRODUCT company (builds and "
            "sells its own software/product, e.g. a SaaS company) or a "
            "SERVICE company (provides IT services/consulting/outsourcing "
            "for other companies, e.g. a systems integrator), or BOTH if it "
            "genuinely does both at meaningful scale"
        ),
        progress_callback=progress_callback,
    )


def warm_industry_cache(industries: list[str], progress_callback=None) -> int:
    """Same idea, for the much smaller, much more tractable set of INDUSTRY
    categories (see module docstring) -- this is what actually makes
    company-type classification scale to thousands of companies: most real
    companies have a real industry on file already, and there are only
    ~130 distinct industries to ever classify, versus 1,200+ individual
    company names the model mostly has no specific knowledge of anyway."""
    return _warm(
        _industry_cache(), industries,
        dimension_description=(
            "whether companies in this INDUSTRY CATEGORY are typically "
            "PRODUCT companies (build and sell their own software/product), "
            "SERVICE companies (provide IT services/consulting/outsourcing/"
            "professional services for other companies), or BOTH -- if the "
            "industry itself isn't meaningfully a software product-vs-"
            "service distinction at all (e.g. banking, farming, "
            "construction, biotechnology), that's not a guess to force -- "
            "use Unknown"
        ),
        progress_callback=progress_callback,
    )
