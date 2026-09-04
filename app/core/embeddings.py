"""Batched, disk-cached embedding client (Ollama).

Why this exists alongside `semantic.py`'s `_embed`: that one is a per-request
helper for live search -- small inputs, failure means "no suggestions", and
nothing is persisted. The experience index is the opposite workload: tens of
thousands of texts embedded ONCE in an offline batch, where the same text
recurs constantly (the subdomain classifier embeds the same sentence three
times per classification -- similarity stage, negative penalty, tie-break --
and the ~11k dataset anchor texts would be re-embedded on every rebuild).
Measured on this machine `nomic-embed-text` runs ~78ms/text, so re-embedding
the anchors alone costs ~15 minutes per run. The cache turns that into a
one-time cost; batching turns per-text HTTP overhead into per-64-text.

Storage is an append-only raw float32 blob plus a JSONL key index, not one
big `.npz` rewritten on each save: the cache reaches ~35k vectors (~107MB),
and rewriting that on every flush would cost more than the embeddings it
saves. Appending is O(new rows).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent

EMBED_MODEL = os.getenv("EXPERIENCE_EMBED_MODEL", os.getenv("EMBED_MODEL", "nomic-embed-text"))
# Ollama charges a large FIXED cost per /api/embed request -- measured on this
# machine: 2,152ms for a 1-text request vs 63.5ms/text at 256 texts, a 34x
# difference for identical work. Batch size is therefore the single most
# important knob in this pipeline, not a micro-optimisation:
#     32 -> 123ms   64 -> 87ms   128 -> 72ms   256 -> 63.5ms
# Past 256 the curve flattens while request latency keeps growing (16s at
# 256), so that is where this sits. Callers that embed one text at a time in
# a loop pay the 2.1s overhead every time -- see `prewarm` in
# experience_classifier.py for how the classifier avoids that.
_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))
_EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "300"))
# Retries for transient backend failures -- see _ollama_embed's docstring for
# the specific one this exists to survive.
_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "4"))
_RETRY_BASE_DELAY = float(os.getenv("EMBED_RETRY_DELAY", "2"))
_CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", str(_ROOT / ".embed_cache")))


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend is unreachable or misbehaving.

    Unlike live search (which degrades to "no semantic suggestions" when
    Ollama is down), an offline index build must fail loudly: silently
    writing an index with missing vectors would look like a successful run
    and poison every downstream lookup."""


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Append-only text -> vector store on disk, keyed by (model, text) hash."""

    def __init__(self, directory: Path | str = _CACHE_DIR, model: str = EMBED_MODEL):
        self.dir = Path(directory)
        self.model = model
        self.dir.mkdir(parents=True, exist_ok=True)
        self._vec_path = self.dir / "vectors.f32"
        self._key_path = self.dir / "keys.jsonl"
        self._meta_path = self.dir / "meta.json"
        self._lock = threading.Lock()
        self._dim: int | None = None
        self._rows: dict[str, int] = {}
        # `_buf` is a CAPACITY buffer holding `_n` live rows, grown
        # geometrically. It was a plain array re-created with np.vstack on
        # every append, which is O(total rows) per append -- measured 10ms
        # per single-vector append at only 9.6k rows, getting worse as the
        # cache grows. Rows are appended one at a time during classification,
        # so that quietly became the second-biggest cost in the pipeline.
        self._buf: np.ndarray | None = None
        self._n = 0
        self._load()

    # ---- persistence -----------------------------------------------------
    def _load(self) -> None:
        if self._meta_path.is_file():
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._dim = meta.get("dim")
        if self._key_path.is_file():
            with self._key_path.open("r", encoding="utf-8") as fh:
                for row, line in enumerate(fh):
                    line = line.strip()
                    if line:
                        self._rows[line] = row
        if self._dim and self._vec_path.is_file():
            raw = np.fromfile(self._vec_path, dtype=np.float32)
            n_full = raw.size // self._dim
            # A crash mid-append can leave a partial trailing row: drop it,
            # and drop the key rows it no longer backs, rather than ever
            # serving a truncated vector as if it were real.
            self._buf = raw[: n_full * self._dim].reshape(n_full, self._dim)
            self._n = n_full
            if len(self._rows) > n_full:
                self._rows = {k: r for k, r in self._rows.items() if r < n_full}

    def _append(self, keys: list[str], vecs: np.ndarray) -> None:
        if not keys:
            return
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        if self._dim is None:
            self._dim = int(vecs.shape[1])
            self._meta_path.write_text(
                json.dumps({"model": self.model, "dim": self._dim}), encoding="utf-8"
            )
        with self._vec_path.open("ab") as fh:
            vecs.tofile(fh)
        with self._key_path.open("a", encoding="utf-8") as fh:
            for k in keys:
                fh.write(k + "\n")
        start = self._n
        for i, k in enumerate(keys):
            self._rows[k] = start + i
        self._grow(vecs.shape[0])
        self._buf[start:start + vecs.shape[0]] = vecs
        self._n += vecs.shape[0]

    def _grow(self, extra: int) -> None:
        """Double the capacity buffer when it fills, so appending N rows
        total costs O(N) copies overall instead of O(N^2)."""
        needed = self._n + extra
        if self._buf is None:
            self._buf = np.empty((max(needed, 1024), self._dim), dtype=np.float32)
        elif needed > self._buf.shape[0]:
            grown = np.empty((max(needed, self._buf.shape[0] * 2), self._dim), dtype=np.float32)
            grown[: self._n] = self._buf[: self._n]
            self._buf = grown

    # ---- public API ------------------------------------------------------
    def embed(self, texts: list[str], progress: str | None = None) -> np.ndarray:
        """(len(texts), dim) float32 matrix in input order. Cache hits are
        free; misses are embedded in batches and persisted before returning.
        Texts repeated within one call are embedded once."""
        if not texts:
            return np.zeros((0, self._dim or 1), dtype=np.float32)
        keys = [_key(self.model, t) for t in texts]
        with self._lock:
            missing: dict[str, str] = {}  # key -> text, de-duplicated
            for k, t in zip(keys, texts):
                if k not in self._rows and k not in missing:
                    missing[k] = t
            if missing:
                miss_keys = list(missing)
                miss_texts = [missing[k] for k in miss_keys]
                for start in range(0, len(miss_texts), _BATCH_SIZE):
                    batch_keys = miss_keys[start:start + _BATCH_SIZE]
                    batch_texts = miss_texts[start:start + _BATCH_SIZE]
                    self._append(batch_keys, _ollama_embed(batch_texts, self.model))
                    if progress:
                        done = min(start + _BATCH_SIZE, len(miss_texts))
                        logger.info("%s: embedded %d/%d new texts", progress, done, len(miss_texts))
            return np.stack([self._buf[self._rows[k]] for k in keys])

    def embed_one(self, text: str) -> np.ndarray:
        """Prefer `embed()` with every text you need. A one-text call that
        misses costs a full ~2.1s request; the same text inside a batch of
        256 costs ~64ms."""
        return self.embed([text])[0]

    def __len__(self) -> int:
        return self._n

    @property
    def dim(self) -> int | None:
        return self._dim


def _ollama_embed(texts: list[str], model: str) -> np.ndarray:
    """One /api/embed call, retried through transient backend failures.

    The retry is not defensive boilerplate -- it is the difference between
    finishing a build and losing it. Ollama runs the model in a child
    process it can evict and restart under memory pressure (real here: 8GB
    RAM, with API servers also resident), and a request landing in that
    window comes back as an HTTP 400 whose body is a dial error:

        Post "http://127.0.0.1:56009/tokenize": ... actively refused

    A 400 normally means "your request is malformed, do not retry", so this
    one is genuinely misleading -- confirmed transient by bisecting the
    failing batch, where both halves then passed individually. Without a
    retry that blip aborts an hour-long run at whatever point it happens.
    Persistent failure still raises, so a truly bad request fails loudly
    (see EmbeddingError) rather than writing an index with missing vectors.
    """
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{settings.ollama_url}/api/embed",
                json={"model": model, "input": texts},
                timeout=_EMBED_TIMEOUT,
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings")
            if not embeddings or len(embeddings) != len(texts):
                raise EmbeddingError(
                    f"backend returned {len(embeddings or [])} vectors for {len(texts)} inputs"
                )
            return np.asarray(embeddings, dtype=np.float32)
        except (requests.RequestException, ValueError, EmbeddingError) as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "embed call failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1, _MAX_RETRIES + 1, delay, str(exc)[:200],
                )
                time.sleep(delay)
    raise EmbeddingError(
        f"embedding call to {settings.ollama_url} failed after "
        f"{_MAX_RETRIES + 1} attempts: {last_error}"
    ) from last_error


_default_cache: EmbeddingCache | None = None


def default_cache() -> EmbeddingCache:
    """Process-wide cache, created lazily so importing this module never
    touches the filesystem or the network."""
    global _default_cache
    if _default_cache is None:
        _default_cache = EmbeddingCache()
    return _default_cache


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise unit norm, so cosine similarity is a plain dot product.
    Zero rows stay zero instead of turning into NaN."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)
