"""Runs the subdomain classifier over one experience and returns structured
metadata about it.

Thin wrapper, on purpose. `subdomain.py` at the repo root is the classifier
(n-gram indicator matching + embedding similarity against per-subdomain
anchors + negative-indicator penalty + disambiguation tie-break); this module
does not reimplement any of it. What it adds is the three things needed to
run it over a whole corpus instead of a handful of demo sentences:

  1. An embedding backend that is actually available here. `subdomain.py`
     ships `TfidfBackend` (no semantic generalization -- its own docstring
     calls it a correctness stand-in) and `SentenceTransformerBackend`
     (needs torch + a model download). This project already runs
     `nomic-embed-text` on Ollama for `semantic.py`, so `OllamaBackend`
     below implements the same swappable `EmbeddingBackend` interface
     against it -- real sentence embeddings, no new heavyweight dependency.
  2. A context built ONCE (212 subdomains, ~5.8k anchor phrases, ~5.3k
     negative phrases) and reused across all 3,689 experiences, instead of
     per call.
  3. Aggregation of the per-sentence results into one compact metadata
     record per experience, keeping only what a downstream filter can use --
     `classify_sentence()` returns a score for all 212 subdomains per
     sentence, which is ~10x the size of the source resume if persisted.

This module NEVER touches the embeddings written to the experience index.
Its output is stored in a separate file, joined by `experience_id`. See
`experience_text.embedding_text` for why.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core import experience_text
from app.core.embeddings import EMBED_MODEL, EmbeddingCache, default_cache, l2_normalize

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_CLASSIFIER_PATH = Path(os.getenv("SUBDOMAIN_CLASSIFIER_PATH", str(_ROOT / "subdomain.py")))
_DATASET_PATH = Path(
    os.getenv("SUBDOMAINS_DATASET_PATH", str(_ROOT / "master_subdomains_fixed.json"))
)
# "ollama" (real sentence embeddings, needs Ollama up) or "tfidf" (offline
# stand-in from subdomain.py, no network, materially worse -- see its
# docstring). Kept switchable so the pipeline and its tests can run with no
# model server at all.
_BACKEND_NAME = os.getenv("CLASSIFIER_EMBED_BACKEND", "ollama").strip().lower()

UNKNOWN = "Unknown"

# How many ranked alternatives to persist per experience. The classifier
# ranks all 212 subdomains; beyond the top few the scores are noise, and
# storing them all would make the metadata file larger than the corpus.
_KEEP_ALTERNATIVES = 5

# Per-sentence subdomain scores kept for each experience. This is what makes
# the expensive run reusable: `aggregate()` combines per-sentence scores into
# the final pick, and that combining rule is the part most likely to want
# tuning (confirmed on real data -- a Salesforce technical consultant whose
# sentences label as Technical Consulting + Discovery + Full Stack + Team
# Leadership lands on HR/Learning & Development, because two correct-but-
# DIFFERENT presales subdomains split the vote while a consistent
# second-place accumulates). Keeping the top-k per sentence means a new
# aggregation rule can be evaluated over the whole corpus in seconds instead
# of re-running a 45-minute classification pass.
_KEEP_SENTENCE_SCORES = 5


@lru_cache(maxsize=1)
def _load_classifier_module():
    """Loads `subdomain.py` as a module by path.

    By path rather than a plain import because it lives at the repo root as
    a runnable script, not inside the `app` package -- a bare `import
    subdomain` would depend on the process's working directory, which is
    exactly the kind of thing that works in the dev shell and breaks under
    uvicorn or pytest."""
    if not _CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(f"subdomain classifier not found at {_CLASSIFIER_PATH}")
    spec = importlib.util.spec_from_file_location("_subdomain_classifier", _CLASSIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OllamaBackend:
    """`subdomain.EmbeddingBackend` implemented against this project's
    existing Ollama embedding model, with the disk cache from
    `embeddings.py` behind it.

    The cache matters more than it looks: `classify_sentence` embeds the
    same sentence up to three times (similarity stage, negative penalty,
    tie-break), and the ~11k anchor texts are identical on every rebuild.
    Without it a rebuild costs ~15 minutes of pure re-embedding before it
    reaches the first experience."""

    def __init__(self, cache: EmbeddingCache | None = None):
        self.cache = cache or default_cache()
        self.model = self.cache.model

    def fit(self, corpus: list) -> None:
        pass  # pretrained; nothing to fit

    def embed_many(self, texts: list) -> np.ndarray:
        # Normalized once here so every cosine downstream is a dot product
        # and cannot be skewed by magnitude differences between a one-line
        # bullet and a 300-word anchor description.
        return l2_normalize(self.cache.embed(list(texts)))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(np.asarray(vec_a).ravel(), np.asarray(vec_b).ravel()))


def prewarm(texts: list[str]) -> None:
    """Embed `texts` in large batches so later per-text lookups are cache
    hits.

    This is the difference between a usable pipeline and an unusable one.
    `classify_sentence` embeds one sentence at a time (three times over, in
    fact -- similarity, negative penalty, tie-break), and Ollama's per-request
    overhead makes a one-text request cost ~2.1s against ~64ms for the same
    text inside a batch of 256. Left alone that is ~34x the necessary work,
    which on this corpus is the difference between ~80 minutes and ~15 hours.
    Nothing about the classifier's logic changes -- only whether the vector
    it asks for is already in the cache when it asks."""
    _prewarm_with(get_context()["backend"], texts)


def _prewarm_with(backend, texts: list[str]) -> None:
    """Same, against an explicit backend -- used from inside `get_context`,
    where calling `prewarm` would recurse into the context being built."""
    if isinstance(backend, OllamaBackend) and texts:
        backend.cache.embed(texts)


def _build_backend(name: str):
    subdomain = _load_classifier_module()
    if name == "tfidf":
        return subdomain.TfidfBackend()
    if name == "sentence-transformers":
        return subdomain.SentenceTransformerBackend()
    if name == "ollama":
        return OllamaBackend()
    raise ValueError(f"unknown CLASSIFIER_EMBED_BACKEND: {name!r}")


@lru_cache(maxsize=1)
def get_context(backend_name: str = _BACKEND_NAME) -> dict:
    """The classifier's precomputed dataset structures, built once per
    process. With the Ollama backend on a cold embedding cache this is the
    expensive step (~11k anchor + negative phrases); on a warm cache it is
    a few seconds of disk reads."""
    subdomain = _load_classifier_module()
    dataset = subdomain.load_dataset(str(_DATASET_PATH))
    backend = _build_backend(backend_name)
    logger.info(
        "building classifier context: %d subdomains, backend=%s", len(dataset), backend_name
    )
    ctx = subdomain.build_context(dataset, backend)

    # Pre-embed every disambiguation string. `tie_break` embeds these one at
    # a time, on demand, whenever the top two subdomains land within its
    # margin -- which on real resume text is common, and each one would
    # otherwise cost a full single-text request (~2.1s). They are a small
    # fixed set from the dataset, so warming them once here removes that
    # entire failure mode.
    disambiguation_texts = sorted({
        text
        for siblings in ctx["disambig_lookup"].values()
        for text in siblings.values()
        if text
    })
    _prewarm_with(backend, disambiguation_texts)
    logger.info("pre-warmed %d disambiguation texts", len(disambiguation_texts))

    ctx["_backend_name"] = backend_name
    ctx["_n_subdomains"] = len(dataset)
    return ctx


def _winning_indicator_terms(per_sentence: list[dict], subdomain_name: str) -> list[str]:
    """Distinct indicator spans that actually fired for the predicted
    subdomain -- the human-readable "why" behind the label, and the only
    part of the classifier's internals worth persisting per experience."""
    terms: dict[str, None] = {}
    for result in per_sentence:
        for hit in result.get("indicator_hits") or []:
            if hit.get("subdomain") == subdomain_name:
                span = hit.get("matched_span")
                if span:
                    terms.setdefault(span, None)
    return list(terms)


def classify_text(text: str, ctx: dict | None = None) -> dict:
    """Classify one experience's ORIGINAL text into domain/subdomain
    metadata.

    Returns a flat, storable record. `subdomain` is `"Unknown"` when the
    classifier's own confidence gate found no real signal -- that gate is
    deliberately preserved rather than forcing a pick, because a wrong
    confident label is worse for filtering than an honest gap.
    """
    subdomain_mod = _load_classifier_module()
    ctx = ctx if ctx is not None else get_context()

    sentences = experience_text.classifier_sentences(text)
    if not sentences:
        return {
            "domain": UNKNOWN,
            "subdomain": UNKNOWN,
            "alternatives": [],
            "domain_scores": {},
            "indicator_terms": [],
            "n_sentences": 0,
            "n_sentences_with_signal": 0,
            "signal_ratio": 0.0,
            "margin": 0.0,
            "mean_similarity": 0.0,
            "sentence_labels": [],
        }

    # One batched request for this experience's sentences, so the three
    # embed_one calls each sentence makes inside classify_sentence all hit
    # the cache. Callers processing many experiences should pre-warm across
    # a whole window instead (see scripts/build_experience_index.py) -- one
    # experience averages ~6 sentences, which is still a small batch.
    _prewarm_with(ctx["backend"], sentences)
    per_sentence = [subdomain_mod.classify_sentence(s, ctx) for s in sentences]
    agg = subdomain_mod.aggregate(per_sentence, ctx)

    ranked = agg.get("ranked_subdomains") or []
    sub_to_domain = ctx["sub_to_domain"]
    predicted_sub = agg.get("predicted_subdomain", UNKNOWN)
    predicted_domain = agg.get("predicted_domain", UNKNOWN)

    # Domain-level rollup: the 212 subdomains collapse to 11 domains, and a
    # coarse "is this person in Engineering at all" filter is both the more
    # common recruiter question and far more robust than the subdomain pick,
    # since sibling subdomains stealing score from each other doesn't move
    # the domain total.
    domain_scores: dict[str, float] = defaultdict(float)
    for row in ranked:
        domain_scores[sub_to_domain.get(row["subdomain"], UNKNOWN)] += row["score"]
    total = sum(domain_scores.values()) or 1.0
    domain_scores = {d: round(v / total, 4) for d, v in
                     sorted(domain_scores.items(), key=lambda kv: -kv[1])}

    with_signal = [r for r in per_sentence if r.get("has_signal")]
    n_signal = len(with_signal)
    # `aggregate` normalizes the winner to 1.0, so the top score itself says
    # nothing. These three do: how far clear the winner is, how much of the
    # text produced any signal, and how strong the raw anchor similarity was
    # before normalization.
    margin = round(ranked[0]["score"] - ranked[1]["score"], 4) if len(ranked) > 1 else (
        1.0 if ranked else 0.0
    )
    mean_similarity = round(
        sum(r.get("raw_max_embedding_sim", 0.0) for r in with_signal) / n_signal, 4
    ) if n_signal else 0.0

    return {
        "domain": predicted_domain,
        "subdomain": predicted_sub,
        "alternatives": [
            {
                "subdomain": row["subdomain"],
                "domain": sub_to_domain.get(row["subdomain"], UNKNOWN),
                "score": row["score"],
            }
            for row in ranked[1:1 + _KEEP_ALTERNATIVES]
        ],
        "domain_scores": domain_scores,
        "indicator_terms": _winning_indicator_terms(per_sentence, predicted_sub),
        "n_sentences": len(sentences),
        "n_sentences_with_signal": n_signal,
        "signal_ratio": round(n_signal / len(sentences), 4),
        "margin": margin,
        "mean_similarity": mean_similarity,
        "sentence_labels": [
            {
                "index": i,
                "subdomain": r["predicted_subdomain"],
                "domain": r["predicted_domain"],
                "has_signal": bool(r.get("has_signal")),
                # [subdomain, blended score] pairs, as arrays rather than
                # objects purely to keep the file small at ~22k sentences.
                "top": [
                    [row["subdomain"], row["score"]]
                    for row in (r.get("ranked_subdomains") or [])[:_KEEP_SENTENCE_SCORES]
                ],
            }
            for i, r in enumerate(per_sentence)
        ],
    }


def classifier_fingerprint(backend_name: str = _BACKEND_NAME) -> dict:
    """Identifies WHICH classifier produced a stored record, so a rebuild
    after a dataset or model change is detectable instead of silently
    mixing two generations of labels in one file."""
    stat = _DATASET_PATH.stat() if _DATASET_PATH.is_file() else None
    return {
        "dataset": _DATASET_PATH.name,
        "dataset_bytes": stat.st_size if stat else None,
        "backend": backend_name,
        "embed_model": EMBED_MODEL if backend_name == "ollama" else None,
    }
