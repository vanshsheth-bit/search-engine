"""Skill-concept expansion, so "machine learning" can match a candidate whose
resume only lists "TensorFlow, PyTorch" -- a real, quantified gap (confirmed
against this dataset: 0 resumes contain the literal phrase "machine
learning", yet 31 clearly have ML tooling; broader concepts like "cloud" miss
~40% of resumes that plainly have the relevant tools).

Two layers, in priority order:
1. `merged_tools.json` -- a curated, weighted tool/alias/related-tools
   taxonomy (~6,400 tools, 60k+ names incl. aliases). Deterministic and
   auditable: every expansion traces to a specific tool name and a specific
   weight in a file you can inspect and edit. Used whenever a term is
   covered.
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


@lru_cache(maxsize=1)
def _load_taxonomy() -> tuple[dict[str, str], dict[str, list[tuple[str, float]]], dict[str, list[str]]]:
    """Returns:
    - alias_to_canonical: normalized tool name OR alias -> canonical tool name
    - canonical_to_related: canonical tool name -> [(related_tool, weight), ...],
      merged across all of that tool's subdomains, deduped keeping max weight
    - canonical_to_aliases: canonical tool name -> its own alias list (real
      names, not normalized) -- these always count as the same thing, no
      weight threshold needed, e.g. "ML pipeline" IS "machine learning".
    """
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
        alias_to_canonical.setdefault(_norm(name), name)

        aliases = entry.get("aliases") or []
        canonical_to_aliases.setdefault(name, [])
        for a in aliases:
            alias_to_canonical.setdefault(_norm(a), name)
            canonical_to_aliases[name].append(a)

        related_acc = canonical_to_related.setdefault(name, {})
        for sub in (entry.get("subdomain_data") or {}).values():
            for r in sub.get("related_tools") or []:
                rtool, weight = r.get("tool"), r.get("weight")
                if not rtool or weight is None:
                    continue
                related_acc[rtool] = max(related_acc.get(rtool, 0.0), float(weight))

    canonical_to_related_sorted = {
        name: sorted(rel.items(), key=lambda kv: -kv[1])
        for name, rel in canonical_to_related.items()
    }
    return alias_to_canonical, canonical_to_related_sorted, canonical_to_aliases


def expand_skill_term(term: str, min_weight: float = DEFAULT_MIN_WEIGHT) -> list[str] | None:
    """If `term` (a tool name, or any of its aliases) is in the taxonomy,
    return [canonical name, its aliases, related tools >= min_weight] --
    every real-world phrasing of the same thing, plus concrete tools that
    satisfy the concept. Returns None (not []) when the term isn't covered
    at all, so callers can distinguish "found, no strong related tools" from
    "taxonomy has nothing to say -- fall back to the LLM's own knowledge"."""
    alias_to_canonical, canonical_to_related, canonical_to_aliases = _load_taxonomy()
    canonical = alias_to_canonical.get(_norm(term))
    if canonical is None:
        return None

    expanded = [canonical]
    expanded.extend(canonical_to_aliases.get(canonical, []))
    expanded.extend(
        rtool for rtool, weight in canonical_to_related.get(canonical, [])
        if weight >= min_weight
    )
    # de-dupe, preserve order (canonical first, most-relevant related next)
    seen, out = set(), []
    for t in expanded:
        key = _norm(t)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def canonicalize(term: str) -> str:
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
    This is the only expansion applied to a single specific-tool query."""
    alias_to_canonical, _, _ = _load_taxonomy()
    return alias_to_canonical.get(_norm(term), term)


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
    - "in"/"not_in" (already a list) -- the LLM judged this to be a genuine
      UMBRELLA CONCEPT ("machine learning", "cloud") satisfiable by several
      different concrete tools, and proposed some itself. Here it's safe to
      also pull in the taxonomy's `related_tools` for whichever proposed
      term(s) it recognizes, augmenting the model's own guess with curated,
      weighted data -- because the "OR any one of these" semantics the model
      already chose is exactly what `related_tools` is being used to serve.
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

        # "in" / "not_in": the model already decided this is a multi-term
        # concept expansion, with the original concept phrase listed FIRST
        # (per prompt.py's rule) and its own specific-tool suggestions after.
        # Only the concept phrase itself gets the full taxonomy expansion
        # (canonical + aliases + related_tools) -- expanding EVERY proposed
        # term's own related_tools too would cascade into unrelated things
        # (e.g. TensorFlow's related_tools pull in CUDA, Hugging Face,
        # DeepSpeed -- a different, much wider net than "machine learning"
        # itself warrants). The model's other suggested tools are just
        # canonicalized (safe, identity-preserving), not re-expanded.
        terms = f.value if isinstance(f.value, list) else [f.value]
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
    - related_terms: other tools genuinely close enough (>= min_weight, and
      not a generic supporting library -- see _GENERIC_SUPPORT_LIBS) to
      reasonably stand in for it, e.g. "PyTorch" -> also TensorFlow, Keras,
      Hugging Face.

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
    related = {
        rtool.lower() for rtool, w in canonical_to_related.get(canon, [])
        if w >= min_weight and rtool.lower() not in exact
        and _norm(rtool) not in _GENERIC_SUPPORT_LIBS
    }
    return exact, related
