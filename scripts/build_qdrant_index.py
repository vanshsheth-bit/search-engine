"""One-time (re-runnable) loader: push the already-built experience index
(experience_index/{chunks.jsonl,embeddings.f32,classifications.jsonl}, see
app/core/experience_index.py's module docstring for that store's own shape)
into Qdrant as one point per experience chunk.

This does NOT change how search happens today -- app/core/experience_index.py's
numpy cosine search is untouched and still what the app actually queries.
This script only builds the Qdrant-side copy of the same vectors, so it can
be evaluated/queried independently before any decision to switch the app
over to it.

Usage:
    python scripts/build_qdrant_index.py
    python scripts/build_qdrant_index.py --recreate   # drop + rebuild collection
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core import experience_index
from app.core.config import settings

_UPSERT_BATCH = 256


def ensure_collection(client: QdrantClient, collection: str, dim: int,
                       recreate: bool = False, log=print) -> None:
    """Shared setup used by both the full `sync()` and the incremental
    `upsert_chunks()` -- a fresh collection needs to exist before either can
    push a single point into it."""
    exists = client.collection_exists(collection)
    if exists and recreate:
        log(f"Dropping existing collection {collection!r} (--recreate)")
        client.delete_collection(collection)
        exists = False

    if not exists:
        log(f"Creating collection {collection!r} (dim={dim}, cosine)")
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def _point(i: int, chunk: dict, vector, classifications: dict) -> PointStruct:
    classification = classifications.get(chunk["experience_id"], {}).get("classification")
    payload = {
        "candidate_id": chunk["candidate_id"],
        "experience_id": chunk["experience_id"],
        "chunk_id": chunk["chunk_id"],
        "experience_index": chunk["experience_index"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk["text"],
        "classification": classification,
    }
    return PointStruct(id=i, vector=vector.tolist() if hasattr(vector, "tolist") else vector,
                        payload=payload)


def upsert_chunks(
    chunk_rows: list[dict], vectors, start_id: int,
    classifications: dict[str, dict], dim: int,
    client: QdrantClient | None = None, log=print,
) -> int:
    """Push exactly THESE chunk rows/vectors into Qdrant, right after they
    were computed -- the direct-write path `build_experience_index.py` calls
    for newly-embedded chunks, instead of the old flow (finish the WHOLE
    numpy/JSONL rewrite, then have a completely separate pass re-read the
    ENTIRE store back off disk and re-upsert every point, changed or not, on
    every single run). Point id = `start_id + row offset`, matching `sync()`'s
    own row-position convention -- callers pass `start_id` = however many
    chunks existed before this batch, so ids stay stable across runs as long
    as existing chunks are never reordered (see build_experience_index.py's
    stable-prefix check before calling this).

    Owns its own QdrantClient (and ensures the collection exists) when the
    caller doesn't already have one open, so a script that only ever embeds
    incrementally never needs to import connection setup from `sync()`.
    Returns the collection's point count after upserting -- same contract as
    `sync()`, so callers can log/verify identically."""
    owns_client = client is None
    if client is None:
        client = QdrantClient(url=settings.qdrant_url)
    collection = settings.qdrant_collection
    try:
        ensure_collection(client, collection, dim, log=log)
        points: list[PointStruct] = []
        for offset, chunk in enumerate(chunk_rows):
            points.append(_point(start_id + offset, chunk, vectors[offset], classifications))
            if len(points) >= _UPSERT_BATCH:
                client.upsert(collection_name=collection, points=points)
                points = []
                log(f"  upserted {offset + 1}/{len(chunk_rows)} new chunks")
        if points:
            client.upsert(collection_name=collection, points=points)
        return client.count(collection).count
    finally:
        if owns_client:
            client.close()


def sync(directory=experience_index.INDEX_DIR, recreate: bool = False,
          log=print) -> int:
    """Push EVERYTHING currently in `directory` (experience_index.py's
    numpy/JSONL store) into Qdrant, overwriting any point with the same id --
    a full resync, not the routine path. `build_experience_index.py`'s normal
    run instead calls `upsert_chunks()` directly on just the newly-embedded
    chunks the moment they're computed; this function stays around as the
    manual repair/recovery tool for `--recreate`, a corrupted collection, or
    catching Qdrant up after it was down during a normal incremental build
    (see that script's own exception handler). Returns the collection's point
    count after syncing, or -1 if no index exists yet."""
    if not experience_index.index_exists(directory):
        log(f"No experience index found at {directory} -- nothing to sync.")
        return -1

    manifest = experience_index.load_manifest(directory)
    chunks = experience_index.load_chunks(directory)
    vectors = experience_index.load_vectors(directory)
    classifications = experience_index.load_classifications(directory)
    dim = int(manifest["embedding_dim"])

    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            f"chunk/vector misalignment: {len(chunks)} chunks vs "
            f"{vectors.shape[0]} vectors -- index looks corrupt, rebuild it."
        )

    log(f"Syncing {len(chunks)} chunks ({dim}-dim) from {directory} to Qdrant")

    client = QdrantClient(url=settings.qdrant_url)
    try:
        ensure_collection(client, settings.qdrant_collection, dim, recreate=recreate, log=log)
        started = time.monotonic()
        count = upsert_chunks(chunks, vectors, 0, classifications, dim, client=client, log=log)
        elapsed = time.monotonic() - started
        log(f"Done in {elapsed:.1f}s -- collection {settings.qdrant_collection!r} "
            f"now has {count} points.")
        return count
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true",
                         help="Drop the collection first if it already exists.")
    args = parser.parse_args()
    result = sync(recreate=args.recreate)
    if result == -1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
