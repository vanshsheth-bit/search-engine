"""Build the experience index: classify every experience, embed every
experience, store the two results separately.

    .venv/Scripts/python.exe scripts/build_experience_index.py

For each candidate, for each of their experiences:

  * the subdomain classifier runs over the experience's ORIGINAL text and
    its output is written to `classifications.jsonl` as structured metadata;
  * that same original text is chunked (one chunk per experience; split
    further only if it exceeds the embedding model's window) and embedded,
    into `chunks.jsonl` + `embeddings.f32`.

The classifier's output never enters the embedded text -- see
`app/core/experience_text.py`. The two stores are joined by `experience_id`.

The run is resumable by default: experiences already classified with the
same classifier fingerprint are kept and skipped, so an interrupted build
(this is a ~45 minute job on a cold cache) picks up where it stopped rather
than starting over. Pass --rebuild to force everything.

As soon as this run's embeddings are computed, the genuinely NEW ones (a new
resume's chunks, appended after everything already indexed) are pushed
directly into Qdrant (see scripts/build_qdrant_index.py's `upsert_chunks`) --
not a separate pass that reloads the ENTIRE store back off disk and
re-upserts every point, changed or not. Pass --skip-qdrant to skip that
(e.g. no Qdrant running locally); a failed push never fails the overall
build, since the numpy store -- what the app actually queries today -- is
already written and valid by that point. `build_qdrant_index.py --recreate`
remains available as a manual full-resync/repair tool.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import experience_classifier, experience_index, experience_text  # noqa: E402
from app.core.embeddings import EMBED_MODEL, default_cache, l2_normalize  # noqa: E402
from scripts import build_qdrant_index  # noqa: E402

logger = logging.getLogger("build_experience_index")

_ROOT = Path(__file__).resolve().parent.parent
_PARSED_RESUMES_PATH = Path(os.getenv(
    "PARSED_RESUMES_PATH", str(_ROOT / "rebee_client_rebeeai.parsedresumes.json")
))

# Flush interval for the classification store. The classifier is the slow
# half of this job (one embedding round-trip per sentence on a cold cache);
# checkpointing means a crash at minute 40 costs at most this many
# experiences, not the whole run.
_CHECKPOINT_EVERY = 100

# Experiences whose sentences are embedded in one batched pass before any of
# them is classified. Ollama's per-request overhead dwarfs its per-text cost
# (2,152ms for one text vs 63.5ms/text at 256), so the batch has to be built
# ACROSS experiences -- a single experience averages ~6 sentences, which is
# still a tiny, expensive request. 48 experiences is ~300 sentences, landing
# in the flat part of that curve.
_PREWARM_WINDOW = 48

# Original resume fields carried into the classification store alongside the
# label. These are NOT classifier output -- they are the structured columns a
# recruiter filter needs next to the predicted domain ("Backend Engineering
# roles at a current employer"), and keeping them here saves every consumer
# a second join back into the resume file.
_CARRIED_FIELDS = (
    "company", "position", "start_date", "end_date",
    "duration_months", "duration_years", "is_ongoing",
)


def load_resumes(limit: int | None = None) -> list[dict]:
    with _PARSED_RESUMES_PATH.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if limit:
        records = records[:limit]
    return records


def collect_units(resumes: list[dict]) -> list[dict]:
    """Flatten resumes into one work unit per experience, carrying the ids
    that tie both stores back to the same candidate and experience."""
    units = []
    for resume in resumes:
        cid = experience_text.candidate_id_of(resume)
        if not cid:
            continue
        for exp_id, index, raw_exp in experience_text.iter_experiences(resume):
            units.append({
                "candidate_id": cid,
                "experience_id": exp_id,
                "experience_index": index,
                "text": experience_text.embedding_text(raw_exp),
                "raw": raw_exp,
            })
    return units


def _carried(raw_exp: dict) -> dict:
    out = {}
    for field in _CARRIED_FIELDS:
        value = raw_exp.get(field)
        if value not in (None, ""):
            out[field] = value.strip() if isinstance(value, str) else value
    return out


def classify_units(units: list[dict], out_dir: Path, existing: dict[str, dict]) -> list[dict]:
    ctx = experience_classifier.get_context()
    rows: list[dict] = []
    todo = [u for u in units if u["experience_id"] not in existing]
    logger.info("classifying %d experiences (%d reused from previous run)",
                len(todo), len(units) - len(todo))

    started = time.time()
    done = 0
    pending = list(todo)  # experiences still needing a pre-warm batch
    for unit in units:
        cached = existing.get(unit["experience_id"])
        if cached is not None:
            rows.append(cached)
            continue
        if pending and pending[0]["experience_id"] == unit["experience_id"]:
            window, pending = pending[:_PREWARM_WINDOW], pending[_PREWARM_WINDOW:]
            experience_classifier.prewarm([
                s for w in window for s in experience_text.classifier_sentences(w["text"])
            ])
        rows.append({
            "candidate_id": unit["candidate_id"],
            "experience_id": unit["experience_id"],
            "experience_index": unit["experience_index"],
            **_carried(unit["raw"]),
            "classification": experience_classifier.classify_text(unit["text"], ctx),
        })
        done += 1
        if done % _CHECKPOINT_EVERY == 0:
            experience_index.write_classifications(rows, out_dir)
            rate = done / max(time.time() - started, 1e-9)
            remaining = (len(todo) - done) / rate if rate else 0
            logger.info("  classified %d/%d (%.1f/s, ~%.0f min left)",
                        done, len(todo), rate, remaining / 60)
    experience_index.write_classifications(rows, out_dir)
    return rows


def build_chunks(units: list[dict]) -> list[dict]:
    """One chunk per experience, except for the rare experience whose text
    exceeds the embedding window (22 of 3,689 in this dataset)."""
    chunk_rows = []
    for unit in units:
        pieces = experience_text.chunk_text(unit["text"])
        for i, piece in enumerate(pieces):
            chunk_rows.append({
                "chunk_id": experience_text.chunk_id(unit["experience_id"], i),
                "experience_id": unit["experience_id"],
                "candidate_id": unit["candidate_id"],
                "experience_index": unit["experience_index"],
                "chunk_index": i,
                "n_chunks": len(pieces),
                "n_chars": len(piece),
                "text": piece,
            })
    return chunk_rows


def embed_chunks(chunk_rows: list[dict]) -> np.ndarray:
    texts = [r["text"] for r in chunk_rows]
    logger.info("embedding %d chunks with %s", len(texts), EMBED_MODEL)
    # task="search_document" -- must match the task="search_query" prefix
    # experience_index._score_chunks uses at search time (nomic-embed-text's
    # asymmetric-retrieval convention; see EmbeddingCache.embed's docstring
    # for the confirmed live score-ordering bug this fixes).
    return l2_normalize(default_cache().embed(texts, progress="chunks", task="search_document"))


def _push_new_chunks_to_qdrant(
    previous_chunks: list[dict], chunk_rows: list[dict], vectors: np.ndarray,
    class_rows: list[dict], out_dir: Path,
) -> None:
    """Direct-write path: the moment this run's embeddings exist, push only
    the genuinely NEW chunks straight into Qdrant, instead of a separate pass
    that reloads the WHOLE store back off disk and re-upserts every point,
    changed or not, on every single run (that full path still exists as
    `build_qdrant_index.sync()`, now a manual repair/recovery tool rather
    than the routine one).

    Relies on `chunk_rows` extending `previous_chunks` as a stable prefix --
    true as long as an already-indexed candidate's experiences are never
    reordered/removed between runs, which matches how a new resume actually
    arrives (appended to the parsed-resumes file, so `collect_units` puts its
    units at the END, after everything already there). Verified rather than
    assumed: falls back to a full `build_qdrant_index.sync()` if the prefix
    doesn't match, rather than risk upserting a new vector under a point id
    that used to mean something else."""
    n_old = len(previous_chunks)
    stable_prefix = (
        [c["chunk_id"] for c in chunk_rows[:n_old]]
        == [c["chunk_id"] for c in previous_chunks]
    )
    if not stable_prefix:
        logger.warning(
            "Chunk ordering changed since the last build (not a plain "
            "append) -- falling back to a full Qdrant resync instead of a "
            "targeted push."
        )
        build_qdrant_index.sync(out_dir, log=logger.info)
        return

    new_chunk_rows = chunk_rows[n_old:]
    if not new_chunk_rows:
        logger.info("No new chunks since the last build -- nothing to push to Qdrant.")
        return

    classifications = {r["experience_id"]: r for r in class_rows}
    dim = int(vectors.shape[1])
    count = build_qdrant_index.upsert_chunks(
        new_chunk_rows, vectors[n_old:], n_old, classifications, dim, log=logger.info,
    )
    logger.info(
        "Pushed %d new chunk(s) directly to Qdrant -- collection now has %d points.",
        len(new_chunk_rows), count,
    )


def reusable_classifications(out_dir: Path, fingerprint: dict) -> dict[str, dict]:
    """Previous classifications, but only if the same classifier produced
    them. A dataset or backend change makes old labels a different
    generation, and mixing two generations in one file is worse than
    redoing the work."""
    manifest = experience_index.load_manifest(out_dir)
    if manifest.get("classifier") != fingerprint:
        if manifest:
            logger.info("classifier fingerprint changed; reclassifying from scratch")
        return {}
    return experience_index.load_classifications(out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(experience_index.INDEX_DIR))
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N candidates (smoke tests)")
    parser.add_argument("--rebuild", action="store_true",
                        help="ignore any previous run and reclassify everything")
    parser.add_argument("--skip-classify", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--skip-qdrant", action="store_true",
                        help="don't sync the result into Qdrant afterwards")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    started = time.time()

    resumes = load_resumes(args.limit)
    units = collect_units(resumes)
    n_candidates = len({u["candidate_id"] for u in units})
    logger.info("%d resumes -> %d experiences across %d candidates with experience",
                len(resumes), len(units), n_candidates)

    fingerprint = experience_classifier.classifier_fingerprint()

    if args.skip_classify:
        class_rows = list(experience_index.load_classifications(out_dir).values())
    else:
        existing = {} if args.rebuild else reusable_classifications(out_dir, fingerprint)
        class_rows = classify_units(units, out_dir, existing)

    # Snapshot of what was already indexed BEFORE this run's write below --
    # the baseline _push_new_chunks_to_qdrant diffs against to find only the
    # NEW chunks. Empty on a first-ever build, which naturally makes
    # "new chunks" mean "all of them" there -- one code path for both cases.
    previous_chunks = experience_index.load_chunks(out_dir)

    if args.skip_embed:
        chunk_rows = experience_index.load_chunks(out_dir)
        vectors = experience_index.load_vectors(out_dir)
    else:
        chunk_rows = build_chunks(units)
        vectors = embed_chunks(chunk_rows)
        experience_index.write_embeddings(chunk_rows, vectors, out_dir)

    experience_index.write_manifest({
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": _PARSED_RESUMES_PATH.name,
        "n_resumes": len(resumes),
        "n_candidates": n_candidates,
        "n_experiences": len(class_rows),
        "n_chunks": len(chunk_rows),
        "embedding_model": EMBED_MODEL,
        "embedding_dim": int(vectors.shape[1]) if vectors.size else None,
        "chunking": {
            "granularity": "one chunk per experience",
            "max_chunk_chars": experience_text.MAX_CHUNK_CHARS,
        },
        "classifier": fingerprint,
        "separation": (
            "chunks.jsonl/embeddings.f32 contain candidate text only; "
            "classifier output lives solely in classifications.jsonl, "
            "joined on experience_id"
        ),
        "build_seconds": round(time.time() - started, 1),
    }, out_dir)

    labelled = sum(1 for r in class_rows
                   if r.get("classification", {}).get("subdomain") not in (None, "Unknown"))
    logger.info("done in %.1f min -> %s", (time.time() - started) / 60, out_dir)
    logger.info("  %d experiences, %d labelled (%.1f%%), %d chunks",
                len(class_rows), labelled,
                100 * labelled / max(len(class_rows), 1), len(chunk_rows))

    if not args.skip_qdrant:
        if args.skip_embed:
            logger.info(
                "Skipping Qdrant push (--skip-embed): nothing new was embedded this run."
            )
        else:
            try:
                _push_new_chunks_to_qdrant(previous_chunks, chunk_rows, vectors, class_rows, out_dir)
            except Exception:
                logger.exception(
                    "Qdrant push failed -- the numpy/JSONL index above is still "
                    "valid and complete; re-run scripts/build_qdrant_index.py "
                    "once Qdrant is reachable to catch it up."
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
