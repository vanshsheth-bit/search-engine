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

import json
import logging
import os
from functools import lru_cache

import requests

from app.core.config import settings
from app.core.llm_classifier import UNKNOWN, PersistentCache, classify_new_values

logger = logging.getLogger(__name__)

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


_EVIDENCE_DIMENSION = (
    "whether the company is primarily a PRODUCT company (builds and sells "
    "its own software/product, e.g. a SaaS company) or a SERVICE company "
    "(provides IT services/consulting/outsourcing for other companies, e.g. "
    "a systems integrator), or BOTH if it genuinely does both at meaningful scale"
)


def _evidence_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "category": {"type": "string", "enum": CATEGORIES + [UNKNOWN]},
                    },
                    "required": ["index", "category"],
                },
            },
        },
        "required": ["classifications"],
    }


def warm_cache_with_evidence(
    evidence_by_company: dict[str, list[str]],
    progress_callback=None,
    batch_size: int = 15,
    timeout: float = 300.0,
    max_retries: int = 2,
) -> int:
    """Second-pass classification for companies STILL Unknown after
    warm_cache/warm_industry_cache -- the fix for the ~85% of companies the
    model has no specific knowledge of by bare name alone (see module
    docstring). Uses real resume evidence instead (see
    candidates.get_company_evidence): actual excerpts from people who
    worked there, e.g. "provided ... services to clients for customization
    ERP projects" (Service) vs "owned our platform's roadmap" (Product) --
    something the model can reason over instead of trying to recall.

    Confirmed on this dataset: "Thirdware Solutions Ltd" (Unknown by name
    alone) correctly resolves to "Service" once given its actual resume
    excerpt -- see the classify_new_values module docstring for why a bare
    name so often fails for real (non-famous) companies in the first place.

    Deliberately indexed by POSITION in the batch, not by echoing the
    company name back (unlike classify_new_values) -- once a name is shown
    to the model alongside a paragraph of resume text, round-tripping it
    back verbatim is fragile (truncation, minor rewording); an index is
    unambiguous regardless of what the model does with the surrounding
    text.

    Only ever called for companies ALREADY Unknown (or uncached) in the
    direct cache -- re-classifying an already-answered company with
    evidence and getting a DIFFERENT category would be a silent behavior
    change with no clear "which one is right" signal. That's out of scope
    here; use scripts/set_company_type.py for a manual correction instead.

    Same infra-failure-vs-real-Unknown distinction as classify_new_values:
    a failed batch (timeout, bad JSON) is dropped, not cached as Unknown --
    left to retry on the next warmup run."""
    cache = _cache()
    existing = cache.get_all()
    todo = [
        (name, ev) for name, ev in evidence_by_company.items()
        if ev and existing.get(name.strip().lower(), UNKNOWN) == UNKNOWN
    ]
    if not todo:
        return 0

    schema = _evidence_schema()
    total_new = 0

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        lines = []
        for idx, (name, excerpts) in enumerate(batch):
            ev_text = " | ".join(excerpts[:3])
            lines.append(
                f'{idx}. {name}\n   Resume excerpts from people who worked '
                f'there: "{ev_text}"'
            )
        prompt = (
            f"For each numbered company below, classify it as "
            f"{' or '.join(CATEGORIES)} regarding: {_EVIDENCE_DIMENSION}\n\n"
            "Use the resume excerpts as real evidence of what people who "
            "worked there actually did -- client/customer/engagement/"
            "outsourcing language points to Service; own-product/roadmap/"
            "users language points to Product. A role at a Service company "
            "can still use product-sounding language when staffed on a "
            "client's product (e.g. \"owned the roadmap for the client's "
            "platform\") -- read for the EMPLOYER's business model, not just "
            f"individual phrases. If the excerpts are genuinely ambiguous or "
            f"don't tell you enough, use \"{UNKNOWN}\" rather than guessing.\n\n"
            "Return one classification per numbered item, with \"index\" "
            "matching the number exactly, same count as items given.\n\n"
            + "\n".join(lines)
        )
        payload = {
            "model": settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": settings.num_ctx},
        }

        parsed: dict[int, str] | None = None
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    f"{settings.ollama_url}/api/chat", json=payload, timeout=timeout,
                )
                resp.raise_for_status()
                data = json.loads(resp.json()["message"]["content"])
                parsed = {
                    c["index"]: c["category"]
                    for c in data.get("classifications", [])
                    if isinstance(c, dict) and isinstance(c.get("index"), int)
                    and c.get("category") in CATEGORIES + [UNKNOWN]
                }
                break
            except (requests.RequestException, KeyError, json.JSONDecodeError,
                    ValueError, TypeError) as exc:
                last_exc = exc
                logger.warning(
                    "Evidence classification batch attempt %d/%d failed "
                    "(%d items): %s", attempt, max_retries, len(batch), exc,
                )

        if parsed is None:
            logger.error(
                "Evidence classification batch failed after %d attempts "
                "(%d items skipped, will retry on next warmup run): %s",
                max_retries, len(batch), last_exc,
            )
            continue

        batch_results = {
            name.strip().lower(): parsed[idx]
            for idx, (name, _) in enumerate(batch)
            if idx in parsed
        }
        if batch_results:
            cache.update(batch_results)
            total_new += len(batch_results)
            if progress_callback:
                progress_callback(batch_results)

    return total_new
