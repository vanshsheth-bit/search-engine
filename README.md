# Candidate Filter — NL → Structured Filter (Local LLM)

Production-ready natural-language candidate filtering. A local LLM (via Ollama)
translates recruiter queries into structured JSON; a **deterministic engine**
applies the filters. The LLM never touches candidate data, never runs queries,
and never changes match scores.

```
Query → LLM (NL→JSON) → merge w/ session → validate → engine → results
```

## Features

- Local model via **Ollama** with **JSON-schema-constrained decoding**
- Deterministic filter engine (AND / OR / NOT, 10 operators)
- Validation gate (unknown field, wrong operator, type coercion, missing data)
- Session state so follow-ups merge ("actually, Bangalore instead" *replaces*)
- Ambiguity handling (returns `clarify` with options, never guesses)
- No-match handling with actionable suggestions
- Deterministic chip removal endpoint (no LLM call)
- Match scores never recalculated
- Full unit tests that run **without** a model

## 1. Prerequisites

Install [Ollama](https://ollama.com), then pull the recommended model:

```bash
ollama pull qwen2.5:1.5b
```

**Design target is `qwen3:8b`** -- the prompt in `app/llm/prompt.py` (rules +
few-shot examples) is written for an 8B-class instruct model's reasoning, not
for `qwen2.5:1.5b`. Set `MODEL=qwen3:8b`, but only on hardware with a GPU or
enough RAM/CPU to serve it within a few seconds. On a laptop-class CPU with no
GPU (confirmed here: 8GB RAM, no discrete GPU) an 8B model can take minutes
per query -- not usable for an interactive search bar, so `qwen2.5:1.5b` is
this repo's *dev-only* default, not a second design target. It will misparse
phrasing an 8B model generalizes to correctly (see `prompt.py`'s
`MODEL_CHOICE_NOTE` for a concrete example and free-model alternatives worth
comparing: `llama3.1:8b-instruct`, `gemma2:9b`).

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run the tests (no Ollama needed)

```bash
pytest -q
```

## 4. Run the API

```bash
uvicorn app.api.main:app --reload --port 8000
```

Check health (also reports whether Ollama is reachable):

```bash
curl localhost:8000/health
```

Filter:

```bash
curl -X POST localhost:8000/ai/candidates/filter \
  -H 'Content-Type: application/json' \
  -d '{"query":"Show Mumbai candidates with 3+ years of Python",
       "job_id":"123","session_id":"s1"}'
```

Remove a filter deterministically (chip ×) — send the remaining filters:

```bash
curl -X PATCH localhost:8000/ai/candidates/filter/state \
  -H 'Content-Type: application/json' \
  -d '{"job_id":"123","session_id":"s1","logic":"AND",
       "filters":[{"field":"location","operator":"equals","value":"Mumbai"}]}'
```

Clear all filters:

```bash
curl -X DELETE 'localhost:8000/ai/candidates/filter/state?job_id=123&session_id=s1'
```

## 5. Interactive CLI

```bash
python demo.py
```

## Response shapes

| status | meaning |
|---|---|
| `ok` | candidates returned, `chips` describe active filters |
| `clarify` | ambiguous — `question` + `options` |
| `unsupported` | attribute not available — `message` |
| `no_match` | zero results — `message` + `suggestions` |
| `error` | invalid filter — `message` |

## Wiring to your system

- **Candidates**: `app/core/candidates.py` currently joins two real DB exports
  (`rebee_client_rebeeai.parsedresumes.json` + `.jdmatchresults.json`, gitignored
  -- not sample data) into real per-job matched-candidate lists, keyed by
  either the internal `jdId` or the human-facing `jobId`. Replace it with a
  live call to `GET /api/v1/jd/:jdId/matching/results` (or direct DB access)
  once that's reachable. Keep the candidate dict shape (see the docstring in
  `app/core/engine.py`).
- **Session store**: swap `InMemorySessionStore` for a Redis implementation of
  the same interface in `app/core/session.py` for multi-instance deployments.
- **Vocabulary**: add fields in `app/core/vocabulary.py` — the prompt, JSON
  schema, and validator all derive from it automatically.

## Config (env vars)

| var | default |
|---|---|
| `OLLAMA_URL` | `http://localhost:11434` |
| `MODEL` | `qwen2.5:1.5b` (dev-only; design target is `qwen3:8b` on GPU/high-RAM hardware) |
| `LLM_TIMEOUT` | `30` (bump to ~60 when running the 8B target) |
| `LLM_MAX_RETRIES` | `2` |
| `NUM_CTX` | `4096` |
| `PARSED_RESUMES_PATH` | `rebee_client_rebeeai.parsedresumes.json` (repo root) |
| `JD_MATCH_RESULTS_PATH` | `rebee_client_rebeeai.jdmatchresults.json` (repo root) |
| `MASTER_UNIVERSITIES_PATH` | `master_universities.csv` (repo root) |
| `COMPANY_RANKS_PATH` | `company_ranks.json` (repo root) |
| `LOCATION_JSON_PATH` | `Location.json` (repo root) |
| `SESSION_TTL` | `3600` |
