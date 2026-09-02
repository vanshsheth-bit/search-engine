"""Candidate data provider.

Loads REAL matched-candidate data, joined from two real DB exports:
  - `rebee_client_rebeeai.parsedresumes.json`  -- resume content (skills,
    location, education, experience) per `processId`.
  - `rebee_client_rebeeai.jdmatchresults.json` -- per-(processId, jdId) match
    scores (`rankingScore` etc), i.e. one row per candidate PER JOB.

A candidate is only searchable for a job if they have a `completed` match
result for that job's `jdId` -- this is real per-job scoping, not a
placeholder. `job_id` may be either the internal `jdId` (Mongo ObjectId hex,
what the real matching API keys on) or the human-facing `jobId` string
(e.g. "00000084") -- both are indexed.

Replace this module with your real DB / matching-service query in production.
"""
from __future__ import annotations

import csv
import json
import os
import re
from functools import lru_cache

from app.core.company_type import company_types_for
from app.core.skill_taxonomy import canonicalize
from app.core.vocabulary import EDUCATION_RANK_LABELS, education_rank

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_PARSED_RESUMES_PATH = os.getenv(
    "PARSED_RESUMES_PATH",
    os.path.join(_ROOT, "rebee_client_rebeeai.parsedresumes.json"),
)
_JD_MATCH_RESULTS_PATH = os.getenv(
    "JD_MATCH_RESULTS_PATH",
    os.path.join(_ROOT, "rebee_client_rebeeai.jdmatchresults.json"),
)
_MASTER_UNIVERSITIES_PATH = os.getenv(
    "MASTER_UNIVERSITIES_PATH",
    os.path.join(_ROOT, "master_universities.csv"),
)
_COMPANY_RANKS_PATH = os.getenv(
    "COMPANY_RANKS_PATH",
    os.path.join(_ROOT, "company_ranks.json"),
)
_LOCATION_JSON_PATH = os.getenv(
    "LOCATION_JSON_PATH",
    os.path.join(_ROOT, "Location.json"),
)

# Location normalisation: primarily a real lookup against Location.json
# (151k real cities across 210 countries), falling back to a crude heuristic
# (strip a trailing state/country token) only when nothing in the gazetteer
# matches. The gazetteer pass also fixes cases the old heuristic alone
# couldn't -- e.g. "Mumbai Maharashtra" and "Mumbai" now resolve to the same
# canonical "Mumbai" instead of silently fragmenting into two location values.
_US_STATE_ABBR = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY",
}
_TRAILING_REGIONS = {
    "ohio", "california", "illinois", "texas", "georgia",
    "india", "usa", "us", "united states", "sudan", "uk", "united kingdom",
    "canada", "australia", "germany", "singapore", "uae",
    "united arab emirates",
}

# Location.json (the city gazetteer) lists historical/colonial-era Indian
# city names (Bombay, Calcutta, Madras, ...) as their OWN separate entries
# with their OWN canonical spelling -- it has no idea they're the same city
# as their modern renamed counterpart. Confirmed against this dataset:
# without this map, a resume saying "Bombay" and one saying "Mumbai" resolve
# to two different canonical strings and silently never match each other in
# an "equals" filter, even though a recruiter means the same city either
# way. "Bangalore" is worse -- it isn't in the gazetteer as any entry at all
# (only "Bengaluru" is), so it fell all the way through to the raw-string
# fallback, unnormalized. Applied as a final step regardless of whether the
# match came from the gazetteer or the crude fallback, so either path lands
# on the one modern canonical name.
_HISTORICAL_CITY_ALIASES = {
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "bangalore": "Bengaluru",
    "poona": "Pune",
    "gurgaon": "Gurugram",
    "trivandrum": "Thiruvananthapuram",
    "mysore": "Mysuru",
    "baroda": "Vadodara",
    "cochin": "Kochi",
}


@lru_cache(maxsize=1)
def _load_location_data() -> tuple[dict[str, str], dict[str, str]]:
    """Reads Location.json ONCE, returns (city gazetteer, city->country map).
    country_name was sitting right there in every row, unused -- "candidates
    in India" was structurally impossible to answer even with a flawless
    parse, since candidate locations are stored city-level only. This is
    real, ground-truth geographic data already in this file, not an LLM
    guess -- deterministic and zero hallucination risk, unlike asking the
    model "what country is this city in" would be for obscure places."""
    canonical: dict[str, str] = {}
    country: dict[str, str] = {}
    if not os.path.isfile(_LOCATION_JSON_PATH):
        return canonical, country
    with open(_LOCATION_JSON_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = d.get("name")
            if not (isinstance(name, str) and name.strip()):
                continue
            key = name.strip().lower()
            canonical.setdefault(key, name.strip())
            country_name = d.get("country_name")
            if isinstance(country_name, str) and country_name.strip():
                country.setdefault(key, country_name.strip())
    return canonical, country


def _load_city_gazetteer() -> dict[str, str]:
    """lowercased real city name -> canonical city name, from Location.json."""
    return _load_location_data()[0]


def _country_for_city(city: str | None) -> str | None:
    """Country for an already-resolved canonical city name, via the same
    real gazetteer data -- None if the city isn't in it (e.g. it came from
    the crude fallback heuristic rather than a real gazetteer match), same
    "absent, not guessed" semantics as everywhere else in this engine."""
    if not city:
        return None
    return _load_location_data()[1].get(city.strip().lower())


def _canonicalize_city(name: str | None) -> str | None:
    if not name:
        return name
    return _HISTORICAL_CITY_ALIASES.get(name.strip().lower(), name)


def _normalize_location(raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None
    s = raw.strip()

    # Try the longest real-city match first, so multi-word cities ("Navi
    # Mumbai") resolve whole rather than truncated. Require >=3 chars so a
    # garbage fragment ("NO, 39") can't coincidentally hit some obscure
    # 2-letter place name in a 151k-row gazetteer. Split on hyphens too, not
    # just commas/whitespace -- resumes write compound city names both ways
    # ("Navi-Mumbai" as well as "Navi Mumbai"), and the gazetteer only has
    # the space-separated form.
    words = [w for w in re.split(r"[,\s\-]+", s) if w]
    gazetteer = _load_city_gazetteer()
    for end in range(len(words), 0, -1):
        candidate = " ".join(words[:end])
        if len(candidate) < 3:
            continue
        canon = gazetteer.get(candidate.lower())
        if canon:
            return _canonicalize_city(canon)

    # Fall back to the old heuristic for anything the gazetteer doesn't
    # recognise verbatim.
    if "," in s:
        city = s.split(",")[0].strip()
        return _canonicalize_city(city) if city else None
    parts = s.split()
    if len(parts) >= 2:
        last = parts[-1]
        if last.upper() in _US_STATE_ABBR or last.lower() in _TRAILING_REGIONS:
            city = " ".join(parts[:-1]).strip()
            return _canonicalize_city(city) if city else None
    return _canonicalize_city(s)


_DURATION_RE = re.compile(
    r"(?:(\d+)\s*years?)?\s*(?:(\d+)\s*months?)?", re.IGNORECASE
)


def _total_experience_years(total_experience: str | None) -> float | None:
    if not total_experience:
        return None
    m = _DURATION_RE.search(total_experience)
    if not m or (m.group(1) is None and m.group(2) is None):
        return None
    years = int(m.group(1) or 0)
    months = int(m.group(2) or 0)
    return round(years + months / 12, 1)


def _highest_education(education: list[dict]) -> str | None:
    ranks = [education_rank(entry.get("degree")) for entry in education or []]
    ranks = [r for r in ranks if r is not None]
    return EDUCATION_RANK_LABELS.get(max(ranks)) if ranks else None


# College-tier matching: resumes carry messy free-text university names
# ("KJ Somaiya School of Engineering, Mumbai, India") that rarely match
# master_universities.csv's canonical names ("SOMAIYA VIDYAVIHAR UNIVERSITY")
# exactly. Best-effort fuzzy match: try the full normalised name first, then
# fall back to a distinctive-keyword overlap. Not a geocoder-grade matcher --
# returns None (no tier) rather than guessing when nothing lines up.
_INSTITUTION_STOPWORDS = {
    "college", "university", "institute", "school", "of", "the", "and",
    "engineering", "arts", "science", "commerce", "management", "technology",
    "india", "for", "in", "polytechnic", "academy", "degree", "womens",
    "mens", "college's",
}
_TIER_RANK = {"Low": 1, "Medium": 2, "High": 3}


def _normalize_institution(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _load_university_tiers() -> tuple[dict[str, str], dict[str, str]]:
    """Returns (exact_name -> tier, distinctive_keyword -> tier)."""
    exact: dict[str, str] = {}
    keywords: dict[str, str] = {}
    if not os.path.isfile(_MASTER_UNIVERSITIES_PATH):
        return exact, keywords
    with open(_MASTER_UNIVERSITIES_PATH, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tier = (row.get("Tier") or "").strip()
            if tier not in ("Low", "Medium", "High"):
                continue
            norm = _normalize_institution(row.get("institution_name") or "")
            if not norm:
                continue
            exact.setdefault(norm, tier)
            for word in norm.split():
                if len(word) >= 5 and word not in _INSTITUTION_STOPWORDS:
                    keywords.setdefault(word, tier)
    return exact, keywords


def _college_tier_for(university_names: list[str]) -> str | None:
    if not university_names:
        return None
    exact, keywords = _load_university_tiers()

    best_rank, best_tier = 0, None
    for raw in university_names:
        norm = _normalize_institution(raw)
        if not norm:
            continue
        tier = exact.get(norm)
        if tier is None:
            for word in norm.split():
                if len(word) >= 5 and word not in _INSTITUTION_STOPWORDS and word in keywords:
                    tier = keywords[word]
                    break
        if tier and _TIER_RANK[tier] > best_rank:
            best_rank, best_tier = _TIER_RANK[tier], tier
    return best_tier


# Company-tier matching: company_ranks.json has ~7M rows (862MB) -- keeping
# all of it resident in memory would be wasteful (est. 1GB+) for what is, in
# practice, a lookup against the few thousand company names that actually
# appear in our resumes. So: collect the company names we actually need
# first, then stream the file once and keep only matching rows. Exact-name
# match only (no keyword fallback like universities) -- company_ranks.json
# is dominated by small/local businesses with generic overlapping words
# ("solutions", "group", "services"), where fuzzy keyword matching would
# produce real false positives, unlike the much smaller, more distinctive
# university list.
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|plc|pvt|private)\b\.?"
)


def _normalize_company(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    text = _COMPANY_SUFFIX_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# When a resume's "company" field isn't a real, disclosed company at all
# ("Self-employed", "Startup (Confidential)", "Freelance") it must not be
# tier-matched -- confirmed a real false positive this way: company_ranks.json
# (a ~7M-row scraped dataset) contains its OWN junk rows for these exact
# placeholder phrases (someone else's messy "company" field became a row in
# it too), so "Startup (Confidential)" matched a real but meaningless entry
# literally named "startup" at Medium tier, and "Self-employed / Freelance"
# matched one named "self employed" at Low tier -- both via a 2-WORD prefix,
# not just the single-word case the length>=5 threshold already guards
# against. Checked as a substring of the full normalized name, before any
# prefix matching is attempted, so it can't be worked around by trailing
# text either way.
_COMPANY_PLACEHOLDER_MARKERS = (
    "confidential", "undisclosed", "stealth", "self employed", "freelance",
    "unemployed", "not applicable", "independent contractor",
    "career break", "various clients", "multiple clients",
)


def _is_company_placeholder(norm: str) -> bool:
    return any(marker in norm for marker in _COMPANY_PLACEHOLDER_MARKERS)


# Real resume company fields are frequently the real company name PLUS
# trailing junk a recruiter never typed -- a department, a city, a country
# ("Syngene International Ltd. Discovery & Med.Chem Bangalore India" instead
# of just "Syngene International"). An exact-only match against that misses
# ~59% of real companies in practice (confirmed against this dataset), even
# extremely well-known ones like Amazon or HSBC, purely because of trailing
# noise. Falling back to progressively shorter PREFIXES (not a keyword-bag
# like universities) recovers most of these while staying safe: a prefix
# match preserves word order and requires the match to start at the
# beginning of the name, so "Godrej Industries Ltd Thane" can only ever match
# "godrej industries" or "godrej", never coincidentally match an unrelated
# company that merely shares a later word. Still requires >=2 words, or a
# single word of >=5 chars, to avoid matching on a lone generic/short token.
def _company_prefixes(norm: str) -> list[str]:
    words = norm.split()
    return [
        " ".join(words[:end])
        for end in range(len(words), 0, -1)
        if end >= 2 or len(words[0]) >= 5
    ]


def _needed_company_names(raw_records: list[dict]) -> set[str]:
    names = set()
    for r in raw_records:
        for e in r.get("experience") or []:
            c = e.get("company")
            if isinstance(c, str) and c.strip():
                norm = _normalize_company(c)
                if norm and not _is_company_placeholder(norm):
                    names.update(_company_prefixes(norm))
    return names


@lru_cache(maxsize=1)
def _load_company_ranks_data() -> tuple[dict[str, str], dict[str, str]]:
    """Streams company_ranks.json (~900MB) ONCE, returns (tier map, industry
    map) together -- industry sits right there in the same rows as tier,
    previously read and discarded. This is what makes company-type
    classification scale: instead of asking an LLM to know ~1,200 individual
    (mostly obscure) company names, look up each one's real, already-present
    industry (94.7% of real companies in this dataset have one here) and
    classify the ~130 distinct INDUSTRY categories once instead -- a small,
    tractable, well-known set ("computer software", "banking") the model
    reliably knows, unlike specific small companies."""
    needed = _needed_company_names(_load_raw_resumes())
    tiers: dict[str, str] = {}
    industries: dict[str, str] = {}
    if not needed or not os.path.isfile(_COMPANY_RANKS_PATH):
        return tiers, industries
    with open(_COMPANY_RANKS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = d.get("name")
            if not isinstance(name, str):
                continue
            norm = _normalize_company(name)
            if norm not in needed:
                continue
            tier = d.get("company_tier")
            if tier in ("LOW", "MEDIUM", "HIGH"):
                tiers.setdefault(norm, tier.title())  # -> "Low"/"Medium"/"High", matching college_tier's casing
            industry = d.get("industry")
            if isinstance(industry, str) and industry.strip():
                industries.setdefault(norm, industry.strip().lower())
    return tiers, industries


def _load_company_tiers() -> dict[str, str]:
    return _load_company_ranks_data()[0]


def _load_company_industries_from_ranks() -> dict[str, str]:
    return _load_company_ranks_data()[1]


def _industry_for_company(raw: str) -> str | None:
    """Resolve one company's industry the SAME way _company_tier_for resolves
    tier (normalize + progressively shorter prefixes) against the SAME
    underlying map (_load_company_industries_from_ranks, built with
    _normalize_company-normalized keys) -- a flat `.strip().lower()` lookup
    against that map would almost never hit, since company_ranks.json names
    are keyed post-suffix-stripping."""
    norm = _normalize_company(raw)
    if _is_company_placeholder(norm):
        return None
    industries = _load_company_industries_from_ranks()
    for prefix in _company_prefixes(norm):
        industry = industries.get(prefix)
        if industry:
            return industry
    return None


def _industry_lookup_for(companies: list[str]) -> dict[str, str]:
    """{raw company name, normalized the way company_type.company_types_for
    expects (.strip().lower(), no suffix-stripping): resolved industry}.

    company_types_for does a flat `.strip().lower()` lookup into whatever
    industry_lookup dict it's given -- it has no normalization/prefix-
    matching logic of its own (and company_type.py can't import
    _normalize_company/_company_prefixes here without a circular import, so
    that resolution has to happen on this side). Passing
    _load_company_industries_from_ranks() straight through used to mean
    company_types_for's lookup could only hit by coincidence (when a name
    had no suffix to strip) -- confirmed 353 real misses on this dataset,
    including "Tata Consultancy Services Limited" silently never resolving
    its real, present industry ("information technology and services").
    Resolving here first, keyed the way the caller actually looks it up,
    fixes that without changing company_types_for's simple contract."""
    out: dict[str, str] = {}
    for c in companies:
        industry = _industry_for_company(c)
        if industry:
            out[c.strip().lower()] = industry
    return out


def _company_tier_for(company_names: list[str]) -> str | None:
    if not company_names:
        return None
    exact = _load_company_tiers()
    best_rank, best_tier = 0, None
    for raw in company_names:
        norm = _normalize_company(raw)
        if _is_company_placeholder(norm):
            continue
        tier = None
        for prefix in _company_prefixes(norm):
            tier = exact.get(prefix)
            if tier:
                break
        if tier and _TIER_RANK[tier] > best_rank:
            best_rank, best_tier = _TIER_RANK[tier], tier
    return best_tier


def _current_location(personal: dict, experience: list[dict]) -> str | None:
    # Prefer the resume's own stated location over one inferred from job
    # history -- more direct, less guesswork.
    direct = _normalize_location(personal.get("personal_location"))
    if direct:
        return direct
    ongoing = [e for e in experience if e.get("is_ongoing")]
    for entry in ongoing + experience:
        loc = _normalize_location(entry.get("current_location") or entry.get("location"))
        if loc:
            return loc
    return None


def _adapt_resume(raw: dict) -> dict:
    personal = raw.get("personalInfo", {}) or {}
    experience = raw.get("experience", []) or []
    education = raw.get("education", []) or []

    # Canonicalize each stored skill through the taxonomy's aliases (safe,
    # identity-preserving only -- see skill_taxonomy.canonicalize) so naming
    # variants of the SAME tool ("react.js" / "ReactJS" / "React JS") all
    # collapse to one consistent spelling. Confirmed a real gap otherwise: a
    # candidate whose resume only used those variants was invisible to a
    # plain "React" query, while an identical candidate who happened to have
    # the exact string "React" matched fine -- same skill, same person in
    # substance, different outcome purely from resume formatting.
    raw_skills = raw.get("skillsNormalized") or raw.get("skills") or []
    skills = list(dict.fromkeys(canonicalize(s) for s in raw_skills))
    universities = list(dict.fromkeys(
        (e.get("university") or "").strip() for e in education if e.get("university")
    ))
    companies = list(dict.fromkeys(
        (e.get("company") or "").strip() for e in experience if e.get("company")
    ))
    job_titles = list(dict.fromkeys(
        (e.get("position") or "").strip() for e in experience if e.get("position")
    ))
    certifications = list(dict.fromkeys(
        (c.get("name") or "").strip() for c in (raw.get("certifications") or []) if c.get("name")
    ))
    # Longest single gap, not cumulative -- "no gap over 6 months" is about
    # one continuous absence, not total time away across a career. Defaults
    # to 0 (not omitted) for candidates with no recorded gap, so a "gap <= N"
    # query correctly keeps them instead of failing them via missing-data
    # semantics (see matches_filter's handling of value is None).
    gaps = raw.get("gaps") or []
    longest_gap_months = max((g.get("gap_months") or 0) for g in gaps) if gaps else 0

    resolved_location = _current_location(personal, experience)
    candidate = {
        "id": raw.get("processId"),
        "name": personal.get("name"),
        "location": resolved_location,
        # From the SAME real gazetteer data as location, not an LLM guess --
        # deterministic, and absent (not guessed) when the location came
        # from the crude fallback heuristic rather than a real match.
        "country": _country_for_city(resolved_location),
        "experience": _total_experience_years(raw.get("totalExperience")),
        "education": _highest_education(education),
        "university": universities,
        "college_tier": _college_tier_for(universities),
        "company": companies,
        "company_tier": _company_tier_for(companies),
        # The SET of types across all their companies, not a single "best"
        # winner (unlike company_tier) -- someone who worked at both
        # Infosys (Service) and Google (Product) genuinely has both, and a
        # "product-based" filter should find them via either. Only ever
        # populated from persistent caches (direct per-company classification,
        # falling back to industry-based inference -- see company_type.py's
        # module docstring for why that's what makes this scale) -- never
        # classified live during a request.
        "company_type": company_types_for(companies, _industry_lookup_for(companies)),
        "skills": skills,
        "job_title": job_titles,
        "certification": certifications,
        "employment_gap_months": longest_gap_months,
    }
    # Drop keys with no data rather than asserting a false "field is present".
    return {k: v for k, v in candidate.items() if v not in (None, [], "")}


@lru_cache(maxsize=1)
def _load_raw_resumes() -> list[dict]:
    with open(_PARSED_RESUMES_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _load_resumes_by_process_id() -> dict[str, dict]:
    raw_records = _load_raw_resumes()
    return {r["processId"]: _adapt_resume(r) for r in raw_records if r.get("processId")}


# Caps keep a single well-attested company's evidence from blowing up prompt
# size (already the scarce resource on this hardware -- see llm/client.py).
_MAX_EVIDENCE_PER_COMPANY = 5
_MAX_EVIDENCE_CHARS = 400


@lru_cache(maxsize=1)
def _load_company_evidence() -> dict[str, list[str]]:
    """{company name, normalized the SAME way as company_type.py's cache key
    (.strip().lower(), no suffix-stripping -- these two normalizations must
    match exactly or a lookup here silently misses the cache entry):
    real resume description excerpts, one per DISTINCT PERSON who worked
    there (not one per job-match row -- the same person's resume appears
    once per job they were matched against, so naively counting rows over-
    counts a single person's account as if it were independent evidence).

    Feeds company_type.py's evidence-based classification fallback: a bare
    company name is often something the model has no specific knowledge of
    (see that module's docstring -- ~85% Unknown in practice), but real
    excerpts of what people who worked there actually did are a genuine,
    scaling signal an LLM can reason over instead of recall."""
    raw_records = _load_raw_resumes()
    evidence: dict[str, dict[str, str]] = {}  # company -> {processId: excerpt}
    for r in raw_records:
        pid = r.get("processId")
        if not pid:
            continue
        for e in r.get("experience") or []:
            company = e.get("company")
            desc = (e.get("description") or "").strip()
            if not (isinstance(company, str) and company.strip() and desc):
                continue
            norm = company.strip().lower()
            bucket = evidence.setdefault(norm, {})
            if pid not in bucket:
                bucket[pid] = desc[:_MAX_EVIDENCE_CHARS]
    return {
        name: list(by_person.values())[:_MAX_EVIDENCE_PER_COMPANY]
        for name, by_person in evidence.items()
    }


def _identity_key(process_id: str, raw_by_pid: dict) -> tuple:
    """A best-effort real-person identity, independent of processId --
    confirmed a real, live data issue this exists to fix: the SAME resume
    (same email, same phone, same experience, uploaded minutes apart) got
    parsed multiple times, producing multiple DISTINCT processIds for one
    actual person, several of which then independently matched the same
    job -- surfacing as duplicate candidate cards for one real human.
    Email is the most reliable natural key here; name+phone is a fallback
    for records missing one. Deliberately does NOT fall back to name alone
    -- two different real people can share a name, and merging those would
    be a much worse bug (silently losing a real, distinct candidate) than
    the duplicate-card problem this fixes."""
    raw = raw_by_pid.get(process_id) or {}
    personal = raw.get("personalInfo") or {}
    email = (personal.get("email") or "").strip().lower()
    if email:
        return ("email", email)
    phone = (personal.get("phone") or "").strip()
    name = (personal.get("name") or "").strip().lower()
    if name and phone:
        return ("name_phone", name, phone)
    return ("pid", process_id)


@lru_cache(maxsize=1)
def _load_matches_by_job() -> dict[str, list[dict]]:
    """Join match results onto resume content, grouped by job. Only
    `completed` matches are included -- a failed match has no real score, so
    surfacing one would be worse than omitting the candidate for that job.

    Deduped per job by real-person identity (see _identity_key), keeping
    the highest-scoring of any duplicate uploads -- without this, a
    candidate whose resume was accidentally uploaded/parsed twice (a real,
    confirmed case in this dataset) shows up as two separate cards for the
    same person in the same search."""
    resumes = _load_resumes_by_process_id()
    raw_by_pid = {r["processId"]: r for r in _load_raw_resumes() if r.get("processId")}

    with open(_JD_MATCH_RESULTS_PATH, "r", encoding="utf-8") as fh:
        raw_matches = json.load(fh)

    by_job: dict[str, dict[tuple, dict]] = {}
    for m in raw_matches:
        if m.get("status") != "completed":
            continue
        process_id = m.get("processId")
        resume = resumes.get(process_id)
        if resume is None:
            continue

        candidate = dict(resume)
        candidate["match_score"] = m.get("rankingScore", 0)
        identity = _identity_key(process_id, raw_by_pid)

        jd_id = (m.get("jdId") or {}).get("$oid")
        job_id = m.get("jobId")
        for key in (jd_id, job_id):
            if not key:
                continue
            bucket = by_job.setdefault(key, {})
            existing = bucket.get(identity)
            if existing is None or candidate["match_score"] > existing["match_score"]:
                bucket[identity] = candidate

    return {job_key: list(cands.values()) for job_key, cands in by_job.items()}


def get_matched_candidates(job_id: str) -> list[dict]:
    """Real per-job matched candidates: everyone with a completed match
    result against this job (looked up by either internal jdId or the
    human-facing jobId), each carrying their real rankingScore as
    `match_score`. Empty list if the job has no completed matches."""
    return _load_matches_by_job().get(job_id, [])


def get_company_evidence() -> dict[str, list[str]]:
    """Public accessor for scripts/warm_company_types.py -- see
    _load_company_evidence's docstring."""
    return _load_company_evidence()


def get_available_fields(job_id: str) -> set[str]:
    """Infer which filterable fields the dataset actually provides, so we never
    invent attributes. Maps raw candidate keys onto vocabulary field names."""
    candidates = get_matched_candidates(job_id)
    available: set[str] = set()
    for c in candidates:
        if "location" in c:
            available.add("location")
        if "country" in c:
            available.add("country")
        if "experience" in c:
            available.add("experience")
        if "education" in c:
            available.add("education")
        if "university" in c:
            available.add("university")
        if "college_tier" in c:
            available.add("college_tier")
        if "company" in c:
            available.add("company")
        if "company_tier" in c:
            available.add("company_tier")
        if "relocation" in c:
            available.add("relocation")
        if any(k in c for k in ("notice_period_days", "notice_period")):
            available.add("notice_period")
        if "skills" in c:
            available.add("skill")
        if "job_title" in c:
            available.add("job_title")
        if "certification" in c:
            available.add("certification")
        if "employment_gap_months" in c:
            available.add("employment_gap_months")
        if "company_type" in c:
            available.add("company_type")
    return available
