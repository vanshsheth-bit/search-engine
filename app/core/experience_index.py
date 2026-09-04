"""The experience index: two separate stores, joined by id.

    experience_index/
      classifications.jsonl   one row per experience  (structured metadata)
      chunks.jsonl            one row per chunk       (embedded text + ids)
      embeddings.f32          raw float32 matrix, row i <-> chunks.jsonl line i
      manifest.json           what produced this build

They are separate FILES, not two columns of one record, because they are
separate concerns with different lifecycles: re-running the classifier after
a taxonomy change must not force re-embedding 3,689 chunks, and swapping the
embedding model must not invalidate the labels. `experience_id` is the join
key; `chunks.jsonl` carries `candidate_id` and `experience_id` on every row,
so a vector hit resolves to its classification (and its candidate) with a
dict lookup and no scan.

Vectors live in a flat binary blob rather than inside the JSONL because
3,689 x 768 float32 is 11MB binary versus ~56MB of JSON text that has to be
parsed float-by-float on every load.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from app.core.embeddings import default_cache, l2_normalize

_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = Path(os.getenv("EXPERIENCE_INDEX_DIR", str(_ROOT / "experience_index")))

CLASSIFICATIONS_FILE = "classifications.jsonl"
CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "embeddings.f32"
MANIFEST_FILE = "manifest.json"


class IndexPaths:
    def __init__(self, directory: Path | str = INDEX_DIR):
        self.dir = Path(directory)
        self.classifications = self.dir / CLASSIFICATIONS_FILE
        self.chunks = self.dir / CHUNKS_FILE
        self.vectors = self.dir / VECTORS_FILE
        self.manifest = self.dir / MANIFEST_FILE

    def exists(self) -> bool:
        return self.manifest.is_file()


# ---------------------------------------------------------------------------
# WRITE
# ---------------------------------------------------------------------------

def write_classifications(rows: list[dict], directory: Path | str = INDEX_DIR) -> Path:
    paths = IndexPaths(directory)
    paths.dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(paths.classifications, rows)
    return paths.classifications


def write_embeddings(chunk_rows: list[dict], vectors: np.ndarray,
                     directory: Path | str = INDEX_DIR) -> Path:
    """`chunk_rows[i]` describes `vectors[i]`. The alignment is the whole
    contract of this store, so it is checked here rather than trusted."""
    if len(chunk_rows) != (0 if vectors.size == 0 else vectors.shape[0]):
        raise ValueError(
            f"chunk/vector misalignment: {len(chunk_rows)} rows vs "
            f"{0 if vectors.size == 0 else vectors.shape[0]} vectors"
        )
    paths = IndexPaths(directory)
    paths.dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(paths.chunks, chunk_rows)
    tmp = paths.vectors.with_suffix(".f32.tmp")
    np.ascontiguousarray(vectors, dtype=np.float32).tofile(tmp)
    os.replace(tmp, paths.vectors)
    return paths.vectors


def write_manifest(manifest: dict, directory: Path | str = INDEX_DIR) -> Path:
    paths = IndexPaths(directory)
    paths.dir.mkdir(parents=True, exist_ok=True)
    tmp = paths.manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, paths.manifest)
    return paths.manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write-then-rename, so an interrupted build leaves the previous index
    intact instead of a half-written file that still parses."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(directory: Path | str = INDEX_DIR) -> dict:
    paths = IndexPaths(directory)
    if not paths.manifest.is_file():
        return {}
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def load_classifications(directory: Path | str = INDEX_DIR) -> dict[str, dict]:
    """{experience_id: classification row}."""
    return {r["experience_id"]: r for r in _read_jsonl(IndexPaths(directory).classifications)}


def load_chunks(directory: Path | str = INDEX_DIR) -> list[dict]:
    """Chunk manifest, in the same order as the vector matrix rows."""
    return _read_jsonl(IndexPaths(directory).chunks)


def load_vectors(directory: Path | str = INDEX_DIR) -> np.ndarray:
    paths = IndexPaths(directory)
    if not paths.vectors.is_file():
        return np.zeros((0, 0), dtype=np.float32)
    dim = load_manifest(directory).get("embedding_dim")
    if not dim:
        raise ValueError(f"{paths.manifest} has no embedding_dim; cannot read {paths.vectors}")
    raw = np.fromfile(paths.vectors, dtype=np.float32)
    return raw.reshape(-1, int(dim))


def classifications_by_candidate(directory: Path | str = INDEX_DIR) -> dict[str, list[dict]]:
    """{candidate_id: [classification rows]}, in resume order -- the shape a
    candidate-level filter ("has any Backend Engineering experience") wants."""
    grouped: dict[str, list[dict]] = {}
    for row in _read_jsonl(IndexPaths(directory).classifications):
        grouped.setdefault(row["candidate_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["experience_index"])
    return grouped


def search(query: str, top_k: int = 10, directory: Path | str = INDEX_DIR,
           with_classification: bool = True) -> list[dict]:
    """Pure semantic search over the experience chunks.

    Note what this deliberately does NOT do: it never consults the
    classifier's labels to rank. The vectors encode only the candidate's own
    words, so a hit here is evidence independent of the classifier -- which
    is what makes combining the two (semantic recall, then structured
    filtering on `classification`) meaningful rather than circular.
    """
    chunks = load_chunks(directory)
    vectors = load_vectors(directory)
    if not chunks or vectors.size == 0:
        return []
    query_vec = l2_normalize(default_cache().embed([query]))[0]
    scores = l2_normalize(vectors) @ query_vec
    order = np.argsort(-scores)[:top_k]
    labels = load_classifications(directory) if with_classification else {}
    results = []
    for i in order:
        row = dict(chunks[int(i)])
        row["score"] = round(float(scores[int(i)]), 4)
        if with_classification:
            row["classification"] = labels.get(row["experience_id"], {}).get("classification")
        results.append(row)
    return results
