"""
classify_v2.py — multi-sentence domain classifier (rewrite)

WHAT CHANGED FROM classify.py AND WHY
======================================
1. Dropped the spaCy dependency-tree noun extraction (to_tree/extract_nouns/
   match_noun) entirely. It mis-handled multi-word technical terms, and the
   depth-based weighting fought against the dataset's actual signal.
   -> Replaced with n-gram (1-4 word) scanning against indicator_words,
      with singular/plural fallback. See ngram_matches().

2. common_phrases now drives a real similarity stage using a swappable
   embedding backend, not spaCy's averaged-vector .similarity(). The
   original's SIMILARITY_THRESHOLD=0.75 was tuned against weak GloVe-style
   averaged vectors; with a real sentence embedding model this stage is
   far more reliable. See score_sentence_against_subdomains().

3. description fields are now embedded and used as a stable per-subdomain
   anchor, blended with common_phrases similarity. The original script
   never used `description` at all.

4. negative_indicators now use the SAME embedding backend instead of
   substring containment (`phrase in sentence_norm`), which almost never
   fired because those phrases are 15-25 word sentences a resume bullet
   will never literally contain. See apply_negative_penalty().

5. related_subdomains[].disambiguation text is now used as a tie-breaker
   when the top-2 candidates are close. This was the single most valuable
   piece of hand-written content in the dataset and was completely unused
   before. See tie_break().

6. Co-occurrence bonus: if multiple DIFFERENT indicator terms for the same
   subdomain fire in one sentence, score gets a small multiplicative boost
   rather than pure summation, rewarding dense technical sentences (which
   is what the dataset's own common_phrases look like) over single
   keyword coincidences.

SWAPPING IN A REAL EMBEDDING MODEL
====================================
This file ships with TfidfBackend by default because the dev sandbox this
was built in had no network access to download sentence-transformers.
In your real environment:

    pip install sentence-transformers --break-system-packages

then change ONE line in main():
    backend = TfidfBackend()
to:
    backend = SentenceTransformerBackend("BAAI/bge-small-en-v1.5")

Everything else is unchanged -- both backends implement the same
EmbeddingBackend interface (fit/embed_many/embed_one/similarity).
TfidfBackend works but has no real semantic generalization; it's a
correctness-testing stand-in, not a quality bar.

Usage:
    python classify_v2.py "sentence 1" "sentence 2" ...
    python classify_v2.py          # runs built-in demo sentences
"""
import sys, json, re
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = "master_subdomains_fixed.json"

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING BACKEND ABSTRACTION
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingBackend:
    def fit(self, corpus: list) -> None:
        raise NotImplementedError

    def embed_many(self, texts: list) -> np.ndarray:
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        a = vec_a.reshape(1, -1)
        b = vec_b.reshape(1, -1)
        return float(cosine_similarity(a, b)[0, 0])


class TfidfBackend(EmbeddingBackend):
    """
    Default fallback backend. Works with zero extra dependencies but has
    no real semantic generalization (won't know "LLM" relates to "large
    language model" unless the fitted corpus links those tokens via
    co-occurring n-grams). Use SentenceTransformerBackend in production.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self._fitted = False

    def fit(self, corpus: list) -> None:
        self.vectorizer.fit(corpus)
        self._fitted = True

    def embed_many(self, texts: list) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before embedding with TfidfBackend.")
        # Keep the TF-IDF matrix sparse. This dataset's fitted vocabulary is
        # ~85k terms; densifying it (.toarray()) blows up to several GB
        # (rows * 85k * 8 bytes) even though the real data is <1% non-zero.
        # cosine_similarity/reshape/vstack all work directly on sparse input.
        return self.vectorizer.transform(texts)


class SentenceTransformerBackend(EmbeddingBackend):
    """
    PRODUCTION BACKEND. Requires:
        pip install sentence-transformers --break-system-packages
    Recommended models:
        'BAAI/bge-small-en-v1.5'                     (best quality/speed)
        'sentence-transformers/all-MiniLM-L6-v2'     (smaller/faster)
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, corpus: list) -> None:
        pass  # pretrained, nothing to fit

    def embed_many(self, texts: list) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALISATION (kept minimal — no spaCy dependency needed anywhere now)
# ─────────────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _singular(tok: str) -> str:
    """Cheap singularizer for matching purposes only, not grammatically rigorous."""
    if len(tok) > 3 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 2 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def expand_slash_term(term: str) -> list:
    return [normalise(p.strip()) for p in term.split("/") if p.strip()]


GENERIC_NOUNS = {
    "devices", "device", "events", "event", "services", "service",
    "systems", "system", "data", "process", "processes", "tasks", "task",
    "tools", "tool", "platform", "platforms", "solutions", "solution",
    "applications", "application", "features", "feature", "components",
    "component", "modules", "module", "workflows", "workflow", "queries",
    "query", "requests", "request", "responses", "response", "operations",
    "operation", "functions", "function", "endpoints", "endpoint",
}
GENERIC_PENALTY = 0.3
MIN_TERM_LEN_FOR_GENERIC_CHECK = 2  # only down-weight single/double generic words


def is_fully_generic(span: str) -> bool:
    tokens = span.split()
    return len(tokens) <= MIN_TERM_LEN_FOR_GENERIC_CHECK and all(t in GENERIC_NOUNS for t in tokens)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list:
    # Use explicit UTF-8 (with BOM support) to avoid platform-default decode issues.
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_indicator_index(dataset: list) -> tuple:
    """
    term (normalised, possibly multi-word) -> [{subdomain, domain, weight, raw_term}, ...]
    Returns (index, max_ngram_len).
    """
    index = defaultdict(list)
    max_len = 1
    for entry in dataset:
        for iw in entry.get("indicator_words", []):
            for term in expand_slash_term(iw["term"]):
                if not term:
                    continue
                n = len(term.split())
                max_len = max(max_len, n)
                index[term].append({
                    "subdomain": entry["subdomain"],
                    "domain": entry["domain"],
                    "weight": iw["weight"],
                    "raw_term": iw["term"],
                })
    return index, max_len


def build_subdomain_anchor_texts(dataset: list) -> dict:
    anchors = {}
    for entry in dataset:
        anchors[entry["subdomain"]] = {
            "description": entry.get("description", ""),
            "common_phrases": entry.get("common_phrases", []),
            "domain": entry["domain"],
        }
    return anchors


def build_negative_anchor_texts(dataset: list) -> dict:
    neg = {}
    for entry in dataset:
        phrases = []
        for indicator in entry.get("negative_indicators", []):
            if isinstance(indicator, str):
                phrase = indicator.strip()
            elif isinstance(indicator, dict):
                phrase = indicator.get("phrase", "").strip()
            else:
                phrase = ""
            if phrase:
                phrases.append(phrase)
        neg[entry["subdomain"]] = phrases
    return neg


def build_disambiguation_lookup(dataset: list) -> dict:
    """{subdomain: {related_subdomain: disambiguation_text}}"""
    lookup = {}
    for entry in dataset:
        sub = entry["subdomain"]
        related = entry.get("related_subdomains", {})
        disambiguation_map = {}

        if isinstance(related, dict):
            for sibling, info in related.items():
                if isinstance(info, dict) and info.get("disambiguation"):
                    disambiguation_map[sibling] = info["disambiguation"]
        elif isinstance(related, list):
            # Some datasets provide a plain list (without disambiguation text).
            # Support richer list-of-dicts schemas when available.
            for item in related:
                if not isinstance(item, dict):
                    continue
                sibling = item.get("subdomain") or item.get("related_subdomain") or item.get("name")
                disambiguation = item.get("disambiguation")
                if sibling and disambiguation:
                    disambiguation_map[sibling] = disambiguation

        lookup[sub] = disambiguation_map
    return lookup


def build_sub_to_domain(dataset: list) -> dict:
    return {entry["subdomain"]: entry["domain"] for entry in dataset}


# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTATION (do all embedding work ONCE at startup, not per sentence)
# ─────────────────────────────────────────────────────────────────────────────

def _stack_vecs(vec_list: list, dim: int, sparse: bool):
    """np.stack, but sparse-aware: vec rows may be scipy sparse (TfidfBackend)
    or dense numpy (SentenceTransformerBackend)."""
    if not vec_list:
        return sp.csr_matrix((0, dim)) if sparse else np.zeros((0, dim))
    return sp.vstack(vec_list) if sparse else np.stack(vec_list)


def precompute_embeddings(anchors: dict, backend: EmbeddingBackend) -> dict:
    all_texts, index_map = [], []
    for sub, data in anchors.items():
        all_texts.append(data["description"])
        index_map.append((sub, "desc"))
        for i, phrase in enumerate(data["common_phrases"]):
            all_texts.append(phrase)
            index_map.append((sub, i))

    backend.fit(all_texts)
    vecs = backend.embed_many(all_texts) if all_texts else np.zeros((0, 1))
    sparse = sp.issparse(vecs)

    result = defaultdict(lambda: {"desc_vec": None, "phrase_vecs": []})
    for (sub, kind), vec in zip(index_map, vecs):
        if kind == "desc":
            result[sub]["desc_vec"] = vec
        else:
            result[sub]["phrase_vecs"].append(vec)

    dim = vecs.shape[1] if vecs.shape[0] else 1
    for sub in result:
        result[sub]["phrase_vecs"] = _stack_vecs(result[sub]["phrase_vecs"], dim, sparse)
    return dict(result)


def precompute_negative_embeddings(neg_anchors: dict, backend: EmbeddingBackend) -> dict:
    all_texts, index_map = [], []
    for sub, phrases in neg_anchors.items():
        for phrase in phrases:
            all_texts.append(phrase)
            index_map.append(sub)
    if not all_texts:
        return {}
    vecs = backend.embed_many(all_texts)
    sparse = sp.issparse(vecs)
    dim = vecs.shape[1]
    grouped = defaultdict(list)
    for sub, vec in zip(index_map, vecs):
        grouped[sub].append(vec)
    return {sub: _stack_vecs(v, dim, sparse) for sub, v in grouped.items()}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: N-GRAM INDICATOR MATCHING (replaces dependency-tree noun extraction)
# ─────────────────────────────────────────────────────────────────────────────

def ngram_matches(sentence: str, index: dict, max_len: int) -> list:
    norm = normalise(sentence)
    tokens = norm.split()
    hits = []
    seen_spans = set()

    for n in range(max_len, 0, -1):
        for i in range(len(tokens) - n + 1):
            span_tokens = tokens[i:i + n]
            span = " ".join(span_tokens)
            key = (i, i + n)
            if any(s <= i and j >= i + n for s, j in seen_spans):
                continue

            matched_entries = index.get(span)
            matched_via = "exact"
            if not matched_entries:
                sing_span = " ".join(_singular(t) for t in span_tokens)
                matched_entries = index.get(sing_span)
                matched_via = "singular"

            if matched_entries:
                seen_spans.add(key)
                for entry in matched_entries:
                    hits.append({**entry, "matched_span": span, "ngram_len": n, "matched_via": matched_via})
    return hits


def score_indicator_hits(hits: list, cooccurrence_bonus: float = 0.15) -> dict:
    """
    Sum weights per subdomain, with a co-occurrence multiplier: if N
    DIFFERENT indicator terms hit for the same subdomain, scale up rather
    than just summing -- rewards dense, specific technical sentences.
    Generic single/double-word terms get GENERIC_PENALTY applied.
    """
    by_sub = defaultdict(lambda: {"raw_score": 0.0, "distinct_terms": set()})
    for h in hits:
        weight = h["weight"]
        if is_fully_generic(h["matched_span"]):
            weight *= GENERIC_PENALTY
        by_sub[h["subdomain"]]["raw_score"] += weight
        by_sub[h["subdomain"]]["distinct_terms"].add(h["matched_span"])

    scores = {}
    for sub, data in by_sub.items():
        n_distinct = len(data["distinct_terms"])
        multiplier = 1 + cooccurrence_bonus * max(0, n_distinct - 1)
        scores[sub] = data["raw_score"] * multiplier
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: DESCRIPTION + COMMON_PHRASES SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def score_embedding_similarity(sentence: str, precomputed: dict, backend: EmbeddingBackend,
                                desc_weight: float = 0.4, phrase_weight: float = 0.6) -> dict:
    sent_vec = backend.embed_one(sentence)
    scores = {}
    for sub, data in precomputed.items():
        desc_sim = backend.similarity(sent_vec, data["desc_vec"]) if data["desc_vec"] is not None else 0.0
        if data["phrase_vecs"].shape[0] > 0:
            phrase_sims = cosine_similarity(sent_vec.reshape(1, -1), data["phrase_vecs"])[0]
            max_phrase_sim = float(phrase_sims.max())
        else:
            max_phrase_sim = 0.0
        scores[sub] = desc_weight * desc_sim + phrase_weight * max_phrase_sim
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: SEMANTIC NEGATIVE INDICATOR PENALTY
# ─────────────────────────────────────────────────────────────────────────────

def apply_negative_penalty(sentence: str, scores: dict, neg_vecs: dict, backend: EmbeddingBackend,
                            penalty_strength: float = 0.5, trigger_threshold: float = 0.35) -> tuple:
    sent_vec = backend.embed_one(sentence)
    adjusted, penalty_log = {}, {}

    for sub, score in scores.items():
        if sub not in neg_vecs or score <= 0:
            adjusted[sub] = score
            continue
        sims = cosine_similarity(sent_vec.reshape(1, -1), neg_vecs[sub])[0]
        max_sim = float(sims.max())
        if max_sim > trigger_threshold:
            excess = (max_sim - trigger_threshold) / (1 - trigger_threshold)
            penalty_factor = max(1 - (penalty_strength * excess), 0.1)
            adjusted[sub] = score * penalty_factor
            penalty_log[sub] = {"max_negative_sim": round(max_sim, 4), "penalty_factor": round(penalty_factor, 4)}
        else:
            adjusted[sub] = score
    return adjusted, penalty_log


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: DISAMBIGUATION TIE-BREAKER (related_subdomains[].disambiguation)
# ─────────────────────────────────────────────────────────────────────────────

def get_disambiguation_pair(disambig_lookup: dict, sub_a: str, sub_b: str) -> dict:
    texts = {}
    if sub_b in disambig_lookup.get(sub_a, {}):
        texts[sub_a] = disambig_lookup[sub_a][sub_b]
    if sub_a in disambig_lookup.get(sub_b, {}):
        texts[sub_b] = disambig_lookup[sub_b][sub_a]
    return texts


def tie_break(sentence: str, ranked_scores: list, disambig_lookup: dict,
              backend: EmbeddingBackend, margin: float = 0.08) -> tuple:
    """
    Single consistent rule: a disambiguation string OWNED by subdomain S
    describes "prefer S's sibling when [conditions]". High similarity to
    that string is evidence FOR THE SIBLING, not for S. Evidence is
    accumulated for both directions (if both exist) and the candidate with
    more accumulated evidence wins the tie-break.
    """
    if len(ranked_scores) < 2:
        return ranked_scores, {"tie_broken": False}

    (top_sub, top_score), (second_sub, second_score) = ranked_scores[0], ranked_scores[1]
    if top_score - second_score > margin:
        return ranked_scores, {"tie_broken": False, "reason": "margin too large"}

    disambig_texts = get_disambiguation_pair(disambig_lookup, top_sub, second_sub)
    if not disambig_texts:
        return ranked_scores, {"tie_broken": False, "reason": "no disambiguation text available"}

    sent_vec = backend.embed_one(sentence)
    evidence_for = {top_sub: 0.0, second_sub: 0.0}
    raw_sims = {}

    for owner_sub, text in disambig_texts.items():
        sibling = second_sub if owner_sub == top_sub else top_sub
        text_vec = backend.embed_one(text)
        sim = backend.similarity(sent_vec, text_vec)
        raw_sims[owner_sub] = round(sim, 4)
        evidence_for[sibling] += sim

    winner = max(evidence_for, key=evidence_for.get)
    meta = {"tie_broken": True, "evidence_for": evidence_for, "raw_similarities": raw_sims}

    if winner != top_sub:
        meta["reason"] = f"disambiguation evidence favored {winner}"
        return [(second_sub, second_score), (top_sub, top_score)] + ranked_scores[2:], meta
    meta["reason"] = "disambiguation evidence confirmed original order"
    return ranked_scores, meta


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE PER SENTENCE
# ─────────────────────────────────────────────────────────────────────────────

def classify_sentence(sentence: str, ctx: dict,
                       indicator_weight: float = 0.45, embedding_weight: float = 0.55,
                       tie_break_margin: float = 0.08, min_confidence: float = 0.12) -> dict:
    """
    ctx is a dict bundling all the precomputed dataset structures, built
    once in main() and reused across every sentence. See build_context().

    min_confidence: if the top blended score (pre-normalization, i.e. the
    RAW best-subdomain score before we divide everything by the max) is
    below this, we report "Unknown" rather than forcing a pick among
    subdomains that all scored near zero. Without this, an empty string
    or off-topic gibberish still "wins" some subdomain by arbitrary
    dict-ordering or embedding noise, which is worse than admitting we
    don't know.
    """
    # Stage 1
    hits = ngram_matches(sentence, ctx["indicator_index"], ctx["max_ngram_len"])
    indicator_scores = score_indicator_hits(hits)

    # Stage 2
    embedding_scores = score_embedding_similarity(sentence, ctx["precomputed_embeddings"], ctx["backend"])

    # track the RAW (pre-normalization) maximum embedding similarity, since
    # that's the only meaningful absolute-scale number that survives
    # normalization wiping out the difference between "noisy near-zero"
    # and "one real anchor + everything else genuinely near zero"
    raw_max_embedding_sim = max(embedding_scores.values(), default=0.0)
    raw_max_indicator_score = max(indicator_scores.values(), default=0.0)

    # Normalize each component to 0..1 before blending (different scales otherwise)
    def normalize(d):
        if not d:
            return {}
        mx = max(d.values()) or 1.0
        return {k: v / mx for k, v in d.items()}

    norm_indicator = normalize(indicator_scores)
    norm_embedding = normalize(embedding_scores)

    all_subs = set(norm_indicator) | set(norm_embedding) | set(ctx["sub_to_domain"])
    blended = {
        sub: indicator_weight * norm_indicator.get(sub, 0.0) + embedding_weight * norm_embedding.get(sub, 0.0)
        for sub in all_subs
    }

    # Stage 3: negative penalty
    blended, penalty_log = apply_negative_penalty(sentence, blended, ctx["negative_embeddings"], ctx["backend"])

    # rank, then tie-break top-2
    ranked = sorted(blended.items(), key=lambda x: -x[1])
    ranked, tie_meta = tie_break(sentence, ranked, ctx["disambig_lookup"], ctx["backend"], margin=tie_break_margin)

    # Confidence gate: empty/no-indicator-hit/low-similarity input should
    # not produce a confident-looking pick. Without ANY indicator hit and
    # without a real embedding anchor signal, refuse to guess.
    has_signal = (raw_max_indicator_score > 0) or (raw_max_embedding_sim > min_confidence)

    if not has_signal or not ranked:
        predicted_subdomain = "Unknown"
        predicted_domain = "Unknown"
    else:
        predicted_subdomain = ranked[0][0]
        predicted_domain = ctx["sub_to_domain"].get(predicted_subdomain, "Unknown")

    return {
        "sentence": sentence,
        "indicator_hits": hits,
        "indicator_scores": {k: round(v, 4) for k, v in indicator_scores.items()},
        "embedding_scores": {k: round(v, 4) for k, v in embedding_scores.items()},
        "blended_scores": {k: round(v, 4) for k, v in blended.items()},
        "negative_penalty_log": penalty_log,
        "tie_break_meta": tie_meta,
        "ranked_subdomains": [{"subdomain": s, "score": round(sc, 4)} for s, sc in ranked[:10]],
        "raw_max_embedding_sim": round(raw_max_embedding_sim, 4),
        "has_signal": has_signal,
        "predicted_subdomain": predicted_subdomain,
        "predicted_domain": predicted_domain,
    }


def aggregate(per_sentence_results: list, ctx: dict, min_confidence: float = 0.12) -> dict:
    agg = defaultdict(float)
    n_with_signal = 0
    for r in per_sentence_results:
        if not r.get("has_signal", True):
            continue  # don't let no-signal sentences contribute noise to the aggregate
        n_with_signal += 1
        for sub, score in r["blended_scores"].items():
            agg[sub] += max(score, 0.0)

    if n_with_signal == 0 or not agg:
        return {"ranked_subdomains": [], "predicted_subdomain": "Unknown", "predicted_domain": "Unknown"}

    mx = max(agg.values(), default=1.0) or 1.0
    ranked = sorted(
        [{"subdomain": k, "score": round(v / mx, 4)} for k, v in agg.items()],
        key=lambda x: -x["score"]
    )
    predicted_subdomain = ranked[0]["subdomain"] if ranked else "Unknown"
    return {
        "ranked_subdomains": ranked[:10],
        "predicted_subdomain": predicted_subdomain,
        "predicted_domain": ctx["sub_to_domain"].get(predicted_subdomain, "Unknown"),
        "n_sentences_with_signal": n_with_signal,
        "n_sentences_total": len(per_sentence_results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDING (run ONCE)
# ─────────────────────────────────────────────────────────────────────────────

def build_context(dataset: list, backend: EmbeddingBackend) -> dict:
    indicator_index, max_ngram_len = build_indicator_index(dataset)
    anchors = build_subdomain_anchor_texts(dataset)
    precomputed_embeddings = precompute_embeddings(anchors, backend)
    neg_anchors = build_negative_anchor_texts(dataset)
    negative_embeddings = precompute_negative_embeddings(neg_anchors, backend)
    disambig_lookup = build_disambiguation_lookup(dataset)
    sub_to_domain = build_sub_to_domain(dataset)

    return {
        "indicator_index": indicator_index,
        "max_ngram_len": max_ngram_len,
        "precomputed_embeddings": precomputed_embeddings,
        "negative_embeddings": negative_embeddings,
        "disambig_lookup": disambig_lookup,
        "sub_to_domain": sub_to_domain,
        "backend": backend,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT
# ─────────────────────────────────────────────────────────────────────────────

SEP, SEP2 = "-" * 70, "=" * 70


def print_sentence_result(i: int, result: dict) -> None:
    print(f"\n{SEP}\n  Sentence {i}: {result['sentence']}\n{SEP}")

    if result["indicator_hits"]:
        print("\n  Indicator matches:")
        by_sub = defaultdict(set)
        for h in result["indicator_hits"]:
            by_sub[h["subdomain"]].add(f"{h['matched_span']} (w={h['weight']}, via={h['matched_via']})")
        for sub, spans in by_sub.items():
            print(f"    {sub}: {sorted(spans)}")

    if result["negative_penalty_log"]:
        print(f"\n  Negative penalties applied: {result['negative_penalty_log']}")

    if result["tie_break_meta"].get("tie_broken"):
        print(f"\n  Tie-break: {result['tie_break_meta']['reason']}")

    if not result.get("has_signal", True):
        print(f"\n  [!] No reliable signal detected (raw_max_embedding_sim={result.get('raw_max_embedding_sim')}) "
              f"-- refusing to guess.")

    print("\n  Ranked subdomains:")
    for r in result["ranked_subdomains"]:
        print(f"    {r['subdomain']:50s} {r['score']:.4f}")

    print(f"\n  -> Predicted Domain    : {result['predicted_domain']}")
    print(f"  -> Predicted Subdomain : {result['predicted_subdomain']}")


def print_aggregate(agg: dict, n: int) -> None:
    print(f"\n{SEP2}\n  FINAL AGGREGATE ({n} sentences)\n{SEP2}")
    for r in agg["ranked_subdomains"]:
        print(f"    {r['subdomain']:50s} {r['score']:.4f}")
    print(f"\n  * Final Domain    : {agg['predicted_domain']}")
    print(f"  * Final Subdomain : {agg['predicted_subdomain']}")
    print(f"{SEP2}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sentences = sys.argv[1:] if len(sys.argv) > 1 else [
      "recorded and mixed multitrack sessions for studio albums using Pro Tools",
 
  "operated front of house console for live concerts and touring productions",
 
  "designed sound effects and ambiance beds for film and TV post-production",
 
  "performed ADR sessions and dialogue editing for feature films",
 
  "implemented interactive audio systems using Wwise for AAA game titles"
    ]

    print("[*] Loading dataset...", file=sys.stderr)
    dataset = load_dataset(DATASET_PATH)

    print("[*] Initializing embedding backend...", file=sys.stderr)
    # --- SWAP THIS LINE in your real environment for better results: ---
    # backend = SentenceTransformerBackend("BAAI/bge-small-en-v1.5")
    backend = TfidfBackend()

    print("[*] Precomputing embeddings for descriptions, common_phrases, "
          "and negative_indicators (one-time cost)...", file=sys.stderr)
    ctx = build_context(dataset, backend)

    print(f"[*] Loaded {len(dataset)} subdomains. Processing {len(sentences)} sentence(s)...\n", file=sys.stderr)

    per_sentence = []
    for i, sentence in enumerate(sentences, 1):
        result = classify_sentence(sentence, ctx)
        per_sentence.append(result)
        print_sentence_result(i, result)

    if len(per_sentence) > 1:
        agg = aggregate(per_sentence, ctx)
        print_aggregate(agg, len(per_sentence))
    elif per_sentence:
        r = per_sentence[0]
        print(f"\n  * Domain    : {r['predicted_domain']}")
        print(f"  * Subdomain : {r['predicted_subdomain']}\n")