"""Identity and text extraction for a single work experience.

This module is deliberately the ONLY place that decides what text represents
an experience, because two consumers need different things from it and the
whole design depends on them staying separate:

  * `embedding_text()` -- what gets embedded. Must be the candidate's own
    words and nothing else. No classifier output, no predicted domain or
    subdomain, no derived tier/type metadata. If classifier output leaked
    in here, the vector space would start encoding the classifier's guesses
    and semantic search would quietly become "search for things the
    classifier already labelled the same way" -- which is circular, and
    means a classifier error becomes an unfixable retrieval error too.
  * `classifier_sentences()` -- what the subdomain classifier reads. Also
    the candidate's own words, split into sentences because
    `subdomain.classify_sentence()` scores one sentence at a time.

Both derive from the same raw resume fields; neither ever sees the other's
output. See `tests/test_experience_text.py` for the leak test that enforces
this.

ID scheme (the reference that ties the two stores back together):

    candidate_id   proc_808c4f72-...        the resume's `processId`
    experience_id  proc_808c4f72-...#exp0   candidate + position in the
                                            resume's `experience` array
    chunk_id       proc_808c4f72-...#exp0#chunk0

Experience index is positional, so it stays stable as long as the resume is
re-parsed the same way, and is meaningful without a lookup table.
"""
from __future__ import annotations

import re

# nomic-embed-text truncates past ~2048 tokens (~8k chars). Real data here
# tops out at 7,466 chars with only 22 of 3,689 experiences over 4,000, so a
# 4,000-char budget keeps essentially every experience as ONE chunk (the
# granularity asked for: 3 experiences -> 3 chunks) while still splitting the
# rare outlier that would otherwise be silently truncated mid-text.
MAX_CHUNK_CHARS = 4000

# Cap on sentences fed to the classifier per experience. Long descriptions
# are bullet lists whose later bullets repeat the same domain signal; each
# extra sentence costs a full embedding round-trip, and the aggregate is a
# normalized sum, so the tail changes the prediction rarely and the cost
# always.
MAX_CLASSIFIER_SENTENCES = 16

MIN_SENTENCE_CHARS = 12

# Resume bullets arrive with a mix of real sentence punctuation and bullet
# glyphs used INSTEAD of punctuation ("* Did X * Did Y" with no full stops),
# so splitting on "." alone leaves one 2,000-char pseudo-sentence.
_BULLET_CHARS = "•●▪◦‣⁃·∙*−–—"
_SENTENCE_SPLIT_RE = re.compile(rf"(?<=[.!?;])\s+|[\n\r]+|\s*[{re.escape(_BULLET_CHARS)}]+\s*")
_WS_RE = re.compile(r"[ \t ]+")

ID_SEP = "#"


def candidate_id_of(raw_resume: dict) -> str | None:
    """The id every other store in this project keys candidates on."""
    pid = raw_resume.get("processId")
    return pid if isinstance(pid, str) and pid.strip() else None


def experience_id(candidate_id: str, index: int) -> str:
    return f"{candidate_id}{ID_SEP}exp{index}"


def chunk_id(exp_id: str, chunk_index: int) -> str:
    return f"{exp_id}{ID_SEP}chunk{chunk_index}"


def parse_experience_id(exp_id: str) -> tuple[str, int]:
    """Inverse of `experience_id`. Raises ValueError on anything else, so a
    malformed reference fails at the lookup instead of silently resolving to
    the wrong candidate."""
    candidate, _, suffix = exp_id.partition(f"{ID_SEP}exp")
    if not candidate or not suffix.isdigit():
        raise ValueError(f"not an experience id: {exp_id!r}")
    return candidate, int(suffix)


def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _WS_RE.sub(" ", value.replace("​", "")).strip()


def embedding_text(raw_experience: dict) -> str:
    """The text that gets embedded: ONLY the candidate's original words.

    Position, company and description are all verbatim resume fields. They
    are framed as a short natural-language block rather than concatenated
    raw, for the reason already established empirically in `semantic.py`:
    `nomic-embed-text` is tuned for prose and reads bare field soup as
    noise. Dates and durations are deliberately excluded -- they are
    numerics that belong in structured filters, and they add embedding
    noise without adding meaning a vector search can use.
    """
    position = _clean(raw_experience.get("position"))
    company = _clean(raw_experience.get("company"))
    description = _clean(raw_experience.get("description"))

    parts = []
    if position and company:
        parts.append(f"{position} at {company}.")
    elif position:
        parts.append(f"{position}.")
    elif company:
        parts.append(f"Worked at {company}.")
    if description:
        parts.append(description)
    return " ".join(parts).strip()


def _split_sentences(text: str) -> list[str]:
    """Every non-empty piece, nothing discarded. Used for chunking, where
    dropping a fragment would silently drop the candidate's words from the
    embedded text."""
    return [s for s in (_clean(raw) for raw in _SENTENCE_SPLIT_RE.split(text)) if s]


def classifier_sentences(text: str, limit: int = MAX_CLASSIFIER_SENTENCES) -> list[str]:
    """Split original experience text into the sentences the subdomain
    classifier scores. Fragments shorter than `MIN_SENTENCE_CHARS` are
    dropped: they carry no indicator n-grams and their embeddings are pure
    noise the classifier's own confidence gate would reject anyway. Chunking
    uses `_split_sentences` instead, precisely because it must NOT drop
    anything."""
    if not text:
        return []
    sentences = [s for s in _split_sentences(text) if len(s) >= MIN_SENTENCE_CHARS]
    if not sentences and _clean(text):
        # A single short experience ("QA Analyst at Infosys.") still deserves
        # one shot at classification rather than being dropped entirely.
        sentences = [_clean(text)]
    return sentences[:limit]


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """One chunk per experience, splitting only when the text exceeds the
    embedding model's usable window. Splits on sentence boundaries so a
    chunk is never cut mid-clause; falls back to a hard character split only
    for a single sentence longer than the whole budget."""
    text = _clean(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces = _split_sentences(text) or [text]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        while len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece[:max_chars])
            piece = piece[max_chars:]
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def iter_experiences(raw_resume: dict):
    """Yields (experience_id, index, raw_experience) for one resume, in
    resume order. Experiences with no usable text at all are skipped -- they
    would produce a zero-signal classification and a meaningless vector."""
    cid = candidate_id_of(raw_resume)
    if not cid:
        return
    for index, raw_experience in enumerate(raw_resume.get("experience") or []):
        if not isinstance(raw_experience, dict):
            continue
        if not embedding_text(raw_experience):
            continue
        yield experience_id(cid, index), index, raw_experience
