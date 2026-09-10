"""Skill-concept expansion, so "machine learning" can match a candidate whose
resume only lists "TensorFlow, PyTorch" -- a real, quantified gap (confirmed
against this dataset: 0 resumes contain the literal phrase "machine
learning", yet 31 clearly have ML tooling; broader concepts like "cloud" miss
~40% of resumes that plainly have the relevant tools).

Two layers, in priority order:
1. `info.json` + `data/<Domain>.json` -- a curated, weighted tool/alias/
   related-tools taxonomy across 11 real domains (Engineering, Finance,
   Legal, Healthcare, HR, Design, Marketing, Media, Operations, Leadership,
   Presales -- ~16,800 tools total). Deterministic and auditable: every
   expansion traces to a specific tool name and a specific weight in files
   you can inspect and edit. Used whenever a term is covered.

   Replaces the old `merged_tools.json` (kept only as a fallback if
   `info.json` is missing): that file's ~6,400 entries were, on inspection,
   effectively ONE domain (5,903 "Engineering" + 480 "Software Engineering"
   + 10 "ML/AI" -- despite its generic name) -- a real, total blind spot for
   any Legal/Finance/HR/Healthcare/Design/Marketing/Media/Operations/
   Leadership/Presales skill. Confirmed near-identical content for the
   domain they DO share (Engineering): the same tool, same aliases, same
   related_tools/weights, e.g. ".NET MAUI" -- strongly suggesting
   merged_tools.json was an Engineering-only export of this same source.
2. The LLM's own general knowledge, for concepts NOT in the taxonomy (e.g.
   "cloud", "frontend" -- umbrella category words this file doesn't model as
   entries at all). See prompt.py's rule for how the LLM is asked to propose
   a handful of concrete tool names itself when it recognizes a broad
   concept the taxonomy doesn't cover. That LLM-proposed list is trusted
   as-is ONLY when the taxonomy has nothing to say about the term -- if the
   taxonomy *does* have an entry, it always wins over whatever the LLM
   guessed, since it's curated and the LLM has already shown real routing
   mistakes on simpler tasks (see prompt.py's model-choice notes).
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from app.models.schemas import Filter

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_INFO_JSON_PATH = os.getenv("INFO_JSON_PATH", os.path.join(_ROOT, "info.json"))
# Fallback only -- see module docstring for why info.json's per-domain files
# are strictly broader/richer and preferred whenever present.
_MERGED_TOOLS_PATH = os.getenv(
    "MERGED_TOOLS_PATH", os.path.join(_ROOT, "merged_tools.json")
)

# Below this weight, a "related tool" is too loosely associated to safely
# stand in for the concept itself (e.g. a 0.5-weight tangential tool).
DEFAULT_MIN_WEIGHT = 0.8

# Stricter than DEFAULT_MIN_WEIGHT: this threshold gates silently widening a
# SPECIFIC single-tool ask ("knows PyTorch") rather than an explicit umbrella
# concept the recruiter themselves named ("machine learning"). Widening a
# concrete ask is a bigger inferential leap -- e.g. Python's related tools
# include Django and SQL, which must NEVER stand in for "knows Python" -- so
# it only kicks in for genuinely close siblings (PyTorch/TensorFlow at 0.94,
# scikit-learn/XGBoost at 0.89), never a loosely-associated tool. Every
# candidate that only matches via this widening is labeled "related, not
# exact" (see annotate_related_skill_matches), so nothing is ever silently
# blended in -- the recruiter always sees which is which.
RELATED_TOOL_MIN_WEIGHT = 0.85


def _norm(s: str) -> str:
    # Strip spaces AND hyphens, not just collapse whitespace -- confirmed a
    # real miss otherwise: a resume's own skill-extraction pipeline stored
    # "scikit-learn" as "scikitlearn" (hyphen dropped entirely, not even
    # replaced with a space), which matched NEITHER this taxonomy's
    # "scikit-learn" canonical name NOR its "scikit learn" alias under plain
    # whitespace normalization -- zero recognition at all for a candidate
    # who had the EXACT literal tool. Checked this is safe taxonomy-wide:
    # it adds 59 new alias collisions on top of the ~1,346 that already
    # exist under the old normalization (a separate, pre-existing data-
    # quality issue in merged_tools.json -- e.g. "Fluentbit" and "Fluent
    # Bit" are already two distinct canonical entries for the same tool).
    # Inspected the 59: overwhelmingly the same kind of pre-existing near-
    # duplicate entries this merge is naturally exposing, not genuinely
    # different tools being wrongly conflated.
    return re.sub(r"[\s\-]+", "", s.lower().strip())


def _accumulate_entry(
    name: str, aliases: list, related: list[tuple[str, float]],
    alias_to_canonical: dict[str, str],
    canonical_to_related: dict[str, dict[str, float]],
    canonical_to_aliases: dict[str, list[str]],
) -> None:
    """Folds one tool entry into the three shared accumulators, regardless of
    which source format it came from. `setdefault` for the canonical spelling
    (first file processed wins -- order is Engineering-first, see
    `info.json`'s `domains` dict, but this only affects which of two
    IDENTICALLY-cased spellings wins, never whether a term is recognized) and
    max-weight merge for related tools -- the same de-dup logic the old
    single-file loader already used for a tool appearing under more than one
    subdomain, extended here to a tool appearing in more than one domain
    file."""
    alias_to_canonical.setdefault(_norm(name), name)
    canonical_to_aliases.setdefault(name, [])
    for a in aliases or []:
        alias_to_canonical.setdefault(_norm(a), name)
        canonical_to_aliases[name].append(a)

    related_acc = canonical_to_related.setdefault(name, {})
    for rtool, weight in related:
        related_acc[rtool] = max(related_acc.get(rtool, 0.0), weight)


def _load_from_info_json() -> tuple[dict, dict, dict] | None:
    """None if info.json isn't present -- caller falls back to
    merged_tools.json. `domains` in info.json maps a domain key to its file
    (e.g. "engineering" -> "data/Engineering.json"); loads every one of them
    into ONE global taxonomy, since skill matching today has no notion of
    "which domain is this job in" -- a Finance job asking for a Legal skill
    should still resolve it."""
    if not os.path.isfile(_INFO_JSON_PATH):
        return None
    with open(_INFO_JSON_PATH, "r", encoding="utf-8") as fh:
        info = json.load(fh)

    alias_to_canonical: dict[str, str] = {}
    canonical_to_related: dict[str, dict[str, float]] = {}
    canonical_to_aliases: dict[str, list[str]] = {}

    for rel_path in (info.get("domains") or {}).values():
        abs_path = os.path.join(_ROOT, rel_path)
        if not os.path.isfile(abs_path):
            continue
        with open(abs_path, "r", encoding="utf-8") as fh:
            domain_data = json.load(fh)
        for tool in (domain_data.get("tools_by_id") or {}).values():
            name = tool.get("name")
            if not name:
                continue
            related = [
                (r["name"], float(r["weight"]))
                for r in (tool.get("related_tools") or [])
                if r.get("name") and r.get("weight") is not None
            ]
            _accumulate_entry(
                name, tool.get("aliases") or [], related,
                alias_to_canonical, canonical_to_related, canonical_to_aliases,
            )

    canonical_to_related_sorted = {
        name: sorted(rel.items(), key=lambda kv: -kv[1])
        for name, rel in canonical_to_related.items()
    }
    return alias_to_canonical, canonical_to_related_sorted, canonical_to_aliases


def _load_from_merged_tools() -> tuple[dict, dict, dict]:
    alias_to_canonical: dict[str, str] = {}
    canonical_to_related: dict[str, dict[str, float]] = {}
    canonical_to_aliases: dict[str, list[str]] = {}

    if not os.path.isfile(_MERGED_TOOLS_PATH):
        return {}, {}, {}

    with open(_MERGED_TOOLS_PATH, "r", encoding="utf-8") as fh:
        entries = json.load(fh)

    for entry in entries:
        name = entry.get("tool")
        if not name:
            continue
        related = [
            (r["tool"], float(r["weight"]))
            for sub in (entry.get("subdomain_data") or {}).values()
            for r in (sub.get("related_tools") or [])
            if r.get("tool") and r.get("weight") is not None
        ]
        _accumulate_entry(
            name, entry.get("aliases") or [], related,
            alias_to_canonical, canonical_to_related, canonical_to_aliases,
        )

    canonical_to_related_sorted = {
        name: sorted(rel.items(), key=lambda kv: -kv[1])
        for name, rel in canonical_to_related.items()
    }
    return alias_to_canonical, canonical_to_related_sorted, canonical_to_aliases


@lru_cache(maxsize=1)
def _load_taxonomy() -> tuple[dict[str, str], dict[str, list[tuple[str, float]]], dict[str, list[str]]]:
    """Returns:
    - alias_to_canonical: normalized tool name OR alias -> canonical tool name
    - canonical_to_related: canonical tool name -> [(related_tool, weight), ...],
      merged across all of that tool's subdomains/domains, deduped keeping
      max weight
    - canonical_to_aliases: canonical tool name -> its own alias list (real
      names, not normalized) -- these always count as the same thing, no
      weight threshold needed, e.g. "ML pipeline" IS "machine learning".

    Prefers info.json's per-domain files (see module docstring for why);
    falls back to the old single-file merged_tools.json only if info.json
    isn't present, so an environment that hasn't picked up the new data yet
    doesn't silently lose skill matching entirely.
    """
    result = _load_from_info_json()
    if result is not None:
        return result
    return _load_from_merged_tools()


# A recruiter naming a real, well-known concept very often wraps it in one
# of these ("machine learning concepts", "Python skills", "AWS knowledge")
# -- the taxonomy only ever indexes the bare term itself, so an exact-match
# lookup on the wrapped phrase always misses even though the meaning is
# completely unambiguous. Real, reported live bug: "knows machine learning
# concepts" resolved to a literal, unrecognized "machine learning concepts"
# skill filter that matched ZERO candidates -- the SAME query without
# "concepts" correctly expands via the real taxonomy relationship. Suffix-
# only (not prefix) -- covers the actual failure pattern seen, and prefix
# phrasing ("knowledge of X") is normally already stripped by the LLM's own
# raw-span extraction before it ever reaches this module.
_SKILL_NOISE_SUFFIXES = {
    "concepts", "concept", "skills", "skill", "knowledge", "expertise",
    "fundamentals", "basics", "principles", "techniques", "background",
    "exposure", "experience",
}


def _strip_skill_noise(term: str) -> str | None:
    """Strips ONE trailing generic noise word from `term` if present,
    returning the shortened phrase to retry against the taxonomy -- None if
    no such suffix is present (so a caller can tell "nothing to strip" apart
    from "stripped down to an empty string", though the length check below
    also guards that directly)."""
    words = term.strip().split()
    if len(words) < 2:
        return None
    if words[-1].lower() not in _SKILL_NOISE_SUFFIXES:
        return None
    return " ".join(words[:-1])


def _bounded_levenshtein(a: str, b: str, max_dist: int) -> int:
    """Edit distance between `a` and `b`, capped at `max_dist` -- returns
    max_dist + 1 (a cheap "too far, don't care exactly how far" sentinel)
    the moment it's provably exceeded, so this stays fast even compared
    against every known term in the taxonomy, not just a handful."""
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            row_min = min(row_min, cur[j])
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


@lru_cache(maxsize=1)
def _fuzzy_candidates_by_length() -> dict[int, list[tuple[str, str]]]:
    """{normalized-string length: [(normalized_canonical_name, canonical_name),
    ...]} -- grouped by length so a fuzzy lookup only scans candidates whose
    length could possibly be within max_dist of the query term.

    CANONICAL NAMES ONLY, not the full alias set -- real, measured
    difference: this taxonomy has 153,806 total alias entries but only
    14,774 distinct canonical names (>10x fewer) -- most of that bulk is
    each tool's own near-duplicate aliases (the same category of noise
    expand_skill_term already excludes from its OUTPUT, see that function's
    docstring). Scanning all 153,806 measured 200-550ms per lookup even
    with length-bucketing; canonical-only cuts that by the same >10x and is
    also the semantically right target -- a recruiter typing a typo means
    the common name ("Kubernetes"), not one of its obscure aliases, and
    matching against the full alias set only added more near-duplicate ties
    (which _fuzzy_typo_match refuses to guess between) without adding real
    typo-catching power."""
    alias_to_canonical, _, _ = _load_taxonomy()
    by_length: dict[int, list[tuple[str, str]]] = {}
    for canonical in set(alias_to_canonical.values()):
        by_length.setdefault(len(_norm(canonical)), []).append((_norm(canonical), canonical))
    return by_length


def _fuzzy_typo_match(term: str) -> str | None:
    """Last-resort typo tolerance: if `term` isn't recognized even after
    noise-word stripping, look for a SINGLE, unambiguous known term within
    a small edit distance -- real, reported live bug: "muchine learning"
    (one substituted letter) resolved to nothing, even though "machine
    learning" was obviously meant.

    Deliberately conservative in two ways that matter more than catching
    every possible typo:
    - Skipped entirely for short terms (< 10 normalized characters), raised
      from an initial 5 after finding a REAL false positive at that
      threshold: "safety" (6 chars, an ordinary complete English word, not
      anything that reads as a typo) fuzzy-matched an obscure unrelated
      taxonomy entry ("SAFETI") at edit distance 1 purely by coincidence,
      wrongly making is_known_tool("safety") true. A short, ordinary word
      has too little "surface area" for edit-distance-1/2 proximity to mean
      anything -- there are always SOME taxonomy entries that close by pure
      chance among ~15,000 canonical names. A longer phrase doesn't have
      that problem (astronomically less likely to coincidentally sit near
      an unrelated entry), which is also exactly where the real, reported
      case ("muchine learning", 16 normalized characters) lives -- so this
      trades away short single-word typo tolerance (a real but smaller
      loss, and this module already works around several short-name
      collisions elsewhere, see candidates._AMBIGUOUS_FOR_YEARS_TEXT_
      MATCH's "R"/"C" note) for not corrupting ordinary short words.
    - Returns None -- never guesses -- when more than one DISTINCT
      canonical term ties for closest, rather than silently picking one and
      risking a wrong, confident-looking correction. A missed typo (falls
      through to "not found", same as today) is a far smaller problem than
      a silently WRONG one."""
    norm_term = _norm(term)
    if len(norm_term) < 10:
        return None
    max_dist = 2
    by_length = _fuzzy_candidates_by_length()
    best_dist = max_dist + 1
    best_canonicals: set[str] = set()
    for length in range(len(norm_term) - max_dist, len(norm_term) + max_dist + 1):
        for alias_norm, canonical in by_length.get(length, ()):
            d = _bounded_levenshtein(norm_term, alias_norm, max_dist)
            if d < best_dist:
                best_dist, best_canonicals = d, {canonical}
            elif d == best_dist:
                best_canonicals.add(canonical)
    if best_dist <= max_dist and len(best_canonicals) == 1:
        return next(iter(best_canonicals))
    return None


@lru_cache(maxsize=4096)
def _resolve_canonical(term: str, fuzzy: bool = True) -> str | None:
    """Canonical tool name for `term`, or None if the taxonomy has nothing
    to say. Tries, in order: the exact (normalized) term; the term with ONE
    trailing noise word stripped (see _strip_skill_noise, exact match only);
    a conservative typo-tolerant fuzzy match (see _fuzzy_typo_match) on the
    ORIGINAL term only -- never on the stripped form. Shared by
    canonicalize/is_known_tool/expand_skill_term so all three treat
    "machine learning concepts" and "muchine learning" exactly as well as
    "machine learning" itself, instead of each silently failing to look
    past the wrapper word or typo on its own.

    Fuzzy-matching the STRIPPED form was tried and reverted after finding a
    real false positive: "Analytical Skills" (a common resume soft-skill
    phrase) stripped to "Analytical", which then fuzzy-matched "Analytica"
    (an obscure, unrelated BI tool) at edit distance 1 -- stacking two
    heuristic transformations (strip a suffix, THEN tolerate a typo)
    compounds the odds of a coincidental collision well past what either
    alone produces. Fuzzy-matching only the untouched original term keeps
    the real reported case working ("muchine learning" has no noise word to
    strip in the first place) while dropping the narrower, riskier combined
    case (a typo INSIDE a noise-wrapped phrase, e.g. "muchine learning
    concepts") -- an accepted, deliberate trade-off.

    `fuzzy=False` skips the typo tier entirely (exact + noise-strip only).
    Used by the bulk resume-INGESTION path (candidates._adapt_resume), for
    two independent reasons:
    - Correctness: the typo tier exists for what a RECRUITER typed -- a
      one-off, visible, self-correcting mistake ("muchine learning"). A
      resume's stored skill list is data, not a typed query; silently
      "correcting" one there mis-files a real candidate's real skill
      permanently and invisibly, at dataset scale. Both false positives
      this tier has produced so far ("safety" -> "SAFETI", "Analytical
      Skills" -> "Analytica") were exactly this shape: ordinary resume
      vocabulary, not typos, landing near an unrelated obscure entry by
      chance. A query-side miss falls through harmlessly to "not found";
      a data-side miss corrupts the record everything else is matched
      against.
    - Cost: measured on the real dataset, fuzzy-matching resume skill
      strings was ~29s of a ~97s cold start (1.2M bounded-Levenshtein
      calls over ~1,000 unrecognized strings) -- paid to "fix" data that
      shouldn't be fuzzily rewritten in the first place.

    Memoized: the fuzzy-match fallback (still thousands of candidates even
    after canonical-only + length-bucketing) is milliseconds, not free, and
    the SAME term routinely gets asked about more than once -- once each
    from canonicalize/is_known_tool/expand_skill_term on a single query, and
    (far more) once per raw skill string across thousands of candidates
    during dataset load (see candidates._adapt_resume)."""
    alias_to_canonical, _, _ = _load_taxonomy()
    canonical = alias_to_canonical.get(_norm(term))
    if canonical is not None:
        return canonical
    stripped = _strip_skill_noise(term)
    if stripped is not None:
        canonical = alias_to_canonical.get(_norm(stripped))
        if canonical is not None:
            return canonical
    return _fuzzy_typo_match(term) if fuzzy else None


# Confirmed live: expanding "Kubernetes" (a tool with unusually rich
# taxonomy data) with NO cap on related_tools pulled in 66 items -- not
# just Kubernetes' own aliases (safe, same technology, e.g. "k8s",
# "kubectl") but genuinely DIFFERENT, merely-commonly-adjacent tools
# (Docker, Helm, Prometheus, Grafana, Rancher, ...). A candidate with ONLY
# Prometheus (a monitoring tool) would wrongly satisfy a filter meant to
# mean "Kubernetes or Terraform". This directly contradicts rule 3's own
# stated guidance elsewhere in this prompt ("4-6 CONCRETE, real, well-known
# technologies") -- the expansion logic itself had no such limit. Capping
# related_tools (never the alias list -- an alias is always the exact same
# technology by definition, unlimited and always safe) keeps every
# umbrella-concept expansion this small, bounded, more precise set,
# matching what the prompt already promises the recruiter.
_MAX_RELATED_TOOLS = 6


def expand_skill_term(term: str, min_weight: float = DEFAULT_MIN_WEIGHT) -> list[str] | None:
    """If `term` (a tool name, or any of its aliases) is in the taxonomy,
    return [canonical name, up to _MAX_RELATED_TOOLS concrete related tools
    >= min_weight] -- the canonical name plus a small, bounded set of real,
    DISTINCT technologies that satisfy the concept. Returns None (not [])
    when the term isn't covered at all, so callers can distinguish "found,
    no strong related tools" from "taxonomy has nothing to say -- fall back
    to the LLM's own knowledge".

    Does NOT include the term's own ALIASES (near-duplicate self-referential
    phrasings, e.g. "machine learning" -> "ML pipeline", "ML system",
    "predictive modeling", ...) -- confirmed real, live bug: for a well-
    aliased concept like "machine learning" these outnumbered the genuine
    related tools 17-to-10, rendering as a bewildering "machine learning +27
    more" checklist row for one bare mention (same root cause already
    diagnosed for "Python" pulling in 74 alias entries, see
    expand_skill_filters's docstring -- that fix only guarded the SAME-FIELD-
    OR call path, never this function itself, which the v2 (now-default)
    schema's "expand" match_mode calls directly with no such guard, see
    taxonomy._tool_to_filter). Safe to drop: every candidate's own skill
    list is ALREADY canonicalized at load time (see candidates._adapt_resume),
    so a candidate whose resume literally says "ML pipeline" is already
    stored as "machine learning" -- the alias never needs to appear in a
    filter's own value list to be matched.

    Also excludes `_GENERIC_SUPPORT_LIBS` (NumPy, pandas, git, ...) from the
    related-tools portion, same curated exclusion related_terms_for already
    applies and for the identical reason: a library that's "related" to
    nearly everything in its ecosystem is useless signal for "did this
    person do the specific thing asked about". And caps the related-tools
    portion to _MAX_RELATED_TOOLS -- confirmed live, "Kubernetes" (a tool
    with unusually rich taxonomy data) with no cap pulled in 66 items, not
    just close siblings but genuinely different, merely-commonly-adjacent
    tools (Docker, Helm, Prometheus, Grafana, Rancher, ...) that would
    wrongly satisfy a filter meant to mean "Kubernetes or Terraform".

    Recognizes `term` wrapped in a generic noise word too ("machine
    learning concepts" resolves exactly like "machine learning") -- see
    _resolve_canonical."""
    _, canonical_to_related, _ = _load_taxonomy()
    canonical = _resolve_canonical(term)
    if canonical is None:
        return None

    expanded = [canonical]
    # canonical_to_related is already sorted by weight descending (see
    # _load_taxonomy) -- taking the first _MAX_RELATED_TOOLS after the
    # threshold/generic-lib filter keeps the highest-relevance related
    # tools, not an arbitrary subset.
    expanded.extend([
        rtool for rtool, weight in canonical_to_related.get(canonical, [])
        if weight >= min_weight and _norm(rtool) not in _GENERIC_SUPPORT_LIBS
    ][:_MAX_RELATED_TOOLS])
    # de-dupe, preserve order (canonical first, most-relevant related next)
    seen, out = set(), []
    for t in expanded:
        key = _norm(t)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def is_known_tool(term: str, fuzzy: bool = True) -> bool:
    """True if `term` (or a normalized variant of it) is a real, recognized
    tool or alias somewhere in the taxonomy. Used to tell a genuine
    tool/skill name apart from a broader practice-area/position label that
    merely LOOKS like a "skill" in a casual query -- e.g. "DevOps" is not a
    tool anyone "has" (confirmed: absent from the entire ~16,800-tool
    taxonomy), it's a practice area a person works IN. See service.py's
    _reclassify_skill_as_domain_when_its_a_position, which only ever
    reclassifies a term this function says is NOT a real tool -- a genuine
    tool match always wins. Recognizes `term` wrapped in a generic noise
    word too ("Python skills" is a known tool exactly like "Python") -- see
    _resolve_canonical.

    `fuzzy=False` (used by the bulk resume-ingestion path) drops the
    typo-tolerance tier -- see _resolve_canonical for why."""
    return _resolve_canonical(term, fuzzy) is not None


@lru_cache(maxsize=1)
def _load_subcategory_index() -> dict[str, list[str]]:
    """{normalized hierarchy.sub_category label: [real tool names tagged
    with it]}, built from the SAME info.json + data/<Domain>.json files as
    the main tool taxonomy (see _load_from_info_json) -- but indexed by
    PRACTICE AREA instead of by tool name, since this answers a different
    question: "what tools does the DevOps practice area actually involve"
    rather than "what does this tool name resolve to." Empty if info.json
    isn't present -- no merged_tools.json fallback, since that file has no
    hierarchy/sub_category data at all."""
    if not os.path.isfile(_INFO_JSON_PATH):
        return {}
    with open(_INFO_JSON_PATH, "r", encoding="utf-8") as fh:
        info = json.load(fh)

    index: dict[str, list[str]] = {}
    for rel_path in (info.get("domains") or {}).values():
        abs_path = os.path.join(_ROOT, rel_path)
        if not os.path.isfile(abs_path):
            continue
        with open(abs_path, "r", encoding="utf-8") as fh:
            domain_data = json.load(fh)
        for tool in (domain_data.get("tools_by_id") or {}).values():
            name = tool.get("name")
            sub_category = (tool.get("hierarchy") or {}).get("sub_category")
            if not name or not sub_category:
                continue
            for label in (sub_category if isinstance(sub_category, list) else [sub_category]):
                if label:
                    index.setdefault(label.lower(), []).append(name)
    return index


def tools_for_subdomain(subdomain: str) -> list[str]:
    """Real, curated tool names tagged with this exact practice-area label
    in the tool taxonomy (e.g. "DevOps" -> Docker, Kubernetes, Terraform,
    Jenkins, Ansible, ... -- confirmed 188 real entries). Empty list if the
    taxonomy has no tools filed under this exact label. Used to offer a
    recruiter real skill choices for a vague domain-only query -- see
    service.py's _domain_skill_options, which further narrows this to only
    the ones actually present among a specific job's real candidates."""
    return _load_subcategory_index().get(subdomain.lower(), [])


def canonicalize(term: str, fuzzy: bool = True) -> str:
    """Safe, identity-preserving normalization ONLY -- resolves a naming
    variant to its canonical spelling (e.g. an alias -> its tool's real
    name) if the taxonomy recognizes it, otherwise returns `term` unchanged.
    Deliberately does NOT touch `related_tools` -- unlike an alias (which is
    always the exact same thing by definition), a related tool is merely
    "commonly seen in the same context" and is very often a DIFFERENT,
    sometimes competing technology (confirmed against this taxonomy: Python's
    related tools include Django and SQL; React's include Angular and
    Vue.js). Using those as if they were interchangeable would silently
    match a candidate who knows a different tool than the one asked for.
    This is the only expansion applied to a single specific-tool query.
    Recognizes `term` wrapped in a generic noise word too ("Python skills"
    canonicalizes to "Python") -- see _resolve_canonical. The fallback only
    ever fires when the EXACT phrase isn't itself already recognized, so a
    genuinely different, real skill/entry name is never touched by it.

    `fuzzy=False` (used by the bulk resume-ingestion path) drops the
    typo-tolerance tier -- see _resolve_canonical for why stored resume
    data must not be fuzzily rewritten the way a typed query can be."""
    return _resolve_canonical(term, fuzzy) or term


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for t in items:
        key = _norm(str(t))
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def expand_skill_filters(filters: list[Filter]) -> list[Filter]:
    """Applied to every "skill" filter the LLM produces, right after
    translation.

    The operator shape tells us what the LLM already decided (per the rule
    in prompt.py):
    - "contains"/"not_contains" (a single value) -- the LLM judged this to be
      ONE specific named tool ("Python", "AWS"). Only safe, identity-
      preserving alias canonicalization is applied here -- never
      `related_tools`, since that relationship means "commonly seen
      together", not "interchangeable" (confirmed: Python's related tools
      include Django and SQL; React's include Angular and Vue.js -- treating
      those as OR-equivalent would match a candidate who knows a different,
      sometimes competing tool than the one actually asked for).
    - "in"/"not_in" (already a list) is AMBIGUOUS between two genuinely
      different intents that share this one wire shape, and must be told
      apart before deciding whether to expand anything:
        (a) a genuine UMBRELLA CONCEPT ("machine learning", "cloud")
            satisfiable by several different concrete tools, where the
            LLM's first item is the broad concept phrase itself (often not
            a real recognized tool -- "cloud"/"devops" aren't in the
            taxonomy) followed by tools it proposed itself (see prompt.py's
            rule 3). Here it's safe to also pull in the taxonomy's
            `related_tools` for whichever proposed term(s) it recognizes,
            augmenting the model's own guess with curated, weighted data.
        (b) a same-field "OR" of SPECIFIC named alternatives the recruiter
            actually said ("Python or Java", "AWS or Azure" -- see
            prompt.py's rule 4). Every term here is already a real,
            recognized tool the recruiter explicitly named -- expanding the
            first one via `related_tools` would be WRONG twice over: it
            silently widens "Python or Java" into "Python or Java or
            Django or ...", a match the recruiter never asked for, and (a
            real, confirmed bug) `expand_skill_term` mixes in the tool's
            own near-duplicate ALIASES ("py", "cpython", "python3", ...)
            alongside genuine related tools -- for a common term like
            "Python" this pulled in 74 items, none of which added real
            signal beyond canonicalize() already provides.
      Distinguished by `is_known_tool`: if EVERY term in the list is
      already a real, recognized tool, this is case (b) -- treat it exactly
      like the single-tool "contains" case above (canonicalize each item,
      no expansion). Only when at least one term is NOT independently
      recognized (the hallmark of a genuine, still partly-unresolved
      concept) does case (a)'s concept-expansion apply.
    """
    out = []
    for f in filters:
        if f.field != "skill" or f.operator not in {"contains", "not_contains", "in", "not_in"}:
            out.append(f)
            continue

        if f.operator in {"contains", "not_contains"}:
            canon = canonicalize(str(f.value))
            out.append(f if canon == f.value else f.model_copy(update={"value": canon}))
            continue

        terms = f.value if isinstance(f.value, list) else [f.value]

        if all(is_known_tool(str(t)) for t in terms):
            # Case (b): a same-field OR of specific named tools the
            # recruiter actually said -- canonicalize each individually,
            # exactly like a single "contains" filter, no expansion.
            merged = _dedupe([canonicalize(str(t)) for t in terms])
            out.append(f.model_copy(update={"value": merged}))
            continue

        # Case (a): a genuine umbrella concept, with the concept phrase
        # listed FIRST (per prompt.py's rule) and the model's own
        # specific-tool suggestions after. Only the concept phrase itself
        # gets the full taxonomy expansion (canonical + aliases + related
        # tools) -- expanding EVERY proposed term's own related_tools too
        # would cascade into unrelated things (e.g. TensorFlow's related
        # tools pull in CUDA, Hugging Face, DeepSpeed -- a different, much
        # wider net than "machine learning" itself warrants). The model's
        # other suggested tools are just canonicalized, not re-expanded.
        concept, rest = terms[0], terms[1:]

        concept_expansion = expand_skill_term(str(concept))
        head = concept_expansion if concept_expansion is not None else [concept]
        tail = [canonicalize(str(t)) for t in rest]

        merged = _dedupe(head + tail)
        out.append(f.model_copy(update={"value": merged}))
    return out


def _raw_skill_names(candidate: dict) -> list[str]:
    """Candidate `skills` may be a flat list or a {name: {...}} dict (same
    flexible shape engine.py's _skills_map handles) -- return real names,
    original casing, regardless of which."""
    skills = candidate.get("skills")
    if isinstance(skills, dict):
        return list(skills.keys())
    if isinstance(skills, (list, tuple)):
        return [str(s) for s in skills]
    return []


# Ubiquitous supporting libraries the taxonomy weights highly (e.g. NumPy is
# 0.87-0.95 "related" to nearly every Python data/ML tool) but which are a
# USELESS signal on their own for "did this person do the specific thing
# asked about" -- confirmed against real data: almost every candidate with
# ANY scikit-learn/pandas-adjacent skill has NumPy, which would otherwise
# turn "knows scikit-learn" into "knows any Python data tool at all". Never
# counted as a qualifying related tool by itself.
_GENERIC_SUPPORT_LIBS = {
    "numpy", "pandas", "scipy", "matplotlib", "seaborn", "requests", "git",
    "jupyter", "jupyternotebook",
}

# General-purpose programming languages -- the taxonomy's weighted
# related_tools scores treat two languages used in similar PROBLEM domains
# (e.g. Python and MATLAB both showing up in data/numerical-analysis work,
# Python and C++ both showing up in systems/perf-sensitive contexts) as
# "related" at >= the normal fuzzy-match threshold. That is a real signal for
# widening a CONCEPT ("machine learning" -> its constituent libraries) but
# the wrong signal for a candidate asked about ONE SPECIFIC LANGUAGE: real,
# reported live bug -- a controls/mechanical engineer whose only point of
# contact with "Python" was knowing MATLAB (a completely different language,
# not a Python library/framework) got surfaced as a fuzzy "Python" match.
# Recruiters treat a specific language name as close to a hard requirement,
# not a stylistic preference -- knowing C++ or MATLAB is not evidence of
# Python experience the way TensorFlow is evidence of "ML tooling" or Flask
# is evidence of "Python web experience". A relation between two entries
# BOTH in this set is therefore never counted as "related" for fuzzy-match
# purposes (see related_terms_for) -- language-to-FRAMEWORK/LIBRARY
# relations (Python -> Django/Flask/FastAPI) are unaffected.
_PROGRAMMING_LANGUAGES = {
    "python", "c", "c++", "c#", "java", "javascript", "typescript", "lua",
    "r", "matlab", "julia", "go", "golang", "rust", "swift", "kotlin", "php",
    "ruby", "scala", "perl", "objective-c", "dart", "groovy", "haskell",
    "erlang", "elixir", "f#", "visual basic", "vb.net", "cobol", "fortran",
    "bash", "powershell", "sql",
}
# _norm'd once here (matching how _GENERIC_SUPPORT_LIBS's own entries are
# already pre-normalized) so the membership check in related_terms_for can
# compare like-for-like against _norm(canon)/_norm(rtool) below.
_PROGRAMMING_LANGUAGES_NORM = {_norm(lang) for lang in _PROGRAMMING_LANGUAGES}


def skill_names_of(candidate: dict) -> list[str]:
    """Public wrapper on _raw_skill_names -- a candidate's real skill names,
    original casing, regardless of whether `skills` is stored as a flat list
    or a {name: {...}} dict."""
    return _raw_skill_names(candidate)


def related_terms_for(
    term: str, min_weight: float = RELATED_TOOL_MIN_WEIGHT,
) -> tuple[set[str], set[str]]:
    """For a specific tool name, returns (exact_terms_lower, related_terms_lower):
    - exact_terms: the tool's own canonical name + real aliases -- always the
      same thing, safe to treat identically.
    - related_terms: other tools genuinely close enough (>= min_weight, not
      a generic supporting library -- see _GENERIC_SUPPORT_LIBS, and not a
      DIFFERENT programming language from `term` itself -- see
      _PROGRAMMING_LANGUAGES) to reasonably stand in for it, e.g. "PyTorch"
      -> also TensorFlow, Keras, Hugging Face.

    Used to widen a specific-tool skill filter so a candidate who has a
    close sibling tool counts as a match too -- confirmed necessary against
    real data (a job's matched pool had candidates with TensorFlow/XGBoost
    but literally nobody with the exact words "scikit-learn"/"pytorch").
    Callers decide how to combine this with the candidate's real data (see
    service.py's fuzzy-matching pass) -- this function only looks up the
    taxonomy relationship, it never itself decides who counts as a match.
    """
    canon = canonicalize(term)
    _, canonical_to_related, canonical_to_aliases = _load_taxonomy()
    exact = {canon.lower()} | {a.lower() for a in canonical_to_aliases.get(canon, [])}
    term_is_language = _norm(canon) in _PROGRAMMING_LANGUAGES_NORM
    related = {
        rtool.lower() for rtool, w in canonical_to_related.get(canon, [])
        if w >= min_weight and rtool.lower() not in exact
        and _norm(rtool) not in _GENERIC_SUPPORT_LIBS
        and not (term_is_language and _norm(rtool) in _PROGRAMMING_LANGUAGES_NORM)
    }
    return exact, related
