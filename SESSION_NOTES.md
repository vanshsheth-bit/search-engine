# Session Notes — Candidate Search Engine (NL Filter Pipeline)

Handoff document for continuing this work on a different machine. Covers everything done, found, and still open from this session. Repo: `vanshsheth-bit/search-engine`, branch `feature/nl-filter-hardening` (checked out locally, tracks `origin/feature/nl-filter-hardening`).

## 0. Current repo state — READ THIS FIRST

- **Uncommitted changes** in 10 files (see list below) — nothing from this session has been committed yet. Review and commit before doing anything destructive.
- **A git stash exists** (`stash@{0}`: "think:false fix + Untitled design.png before pulling feature branch") — `git stash drop` was blocked by a permission classifier during this session and was never cleared. It's harmless sitting there but should be reviewed/dropped manually.
- **`Untitled design.png`** is an untracked file at repo root, unrelated to this work — probably safe to ignore/delete, not investigated.
- Files modified this session:
  - `app/core/candidates.py` — industry-lookup bug fix, evidence-gathering functions
  - `app/core/company_type.py` — evidence-based classification
  - `app/core/service.py` — conversation memory, deterministic Yes/No clarify resolution
  - `app/llm/client.py` — `think` toggle, history param
  - `app/llm/json_schema.py` — `clarify_value`/`clarify_unit` schema fields
  - `app/llm/prompt.py` — rule 0 (history), rule 6-clarify-value, new few-shot example
  - `app/models/schemas.py` — `ChatTurn`, `SessionState.history`, `PendingClarify.value/unit`, `LLMOutput.clarify_value/clarify_unit`
  - `scripts/warm_company_types.py` — `--with-evidence` flag
  - `search-ui/index.html` — `JOB_ID` changed to `"SYN-BACKEND-01"` (was a real prod job ID that doesn't exist in synthetic test data)
  - `tests/test_service.py` — `FakeLLM.translate` signature updated for new `history` param

All 97 tests pass (`.venv/Scripts/python -m pytest -q`).

## 1. Environment setup (what it took to get this running at all)

- **Python version matters a lot**: the system default was Python 3.8.10, which cannot run this codebase — `list[Any]`-style builtin generics in `app/models/schemas.py` need Python 3.9+. Had to rebuild the venv against **Python 3.13** (`C:\Users\riyaj\AppData\Local\Programs\Python\Python313\python.exe`).
- **`requirements.txt` has a bad pin**: `python-dotenv==1.2.3` does not exist (latest real release is 1.0.1 as of this session). Installed `python-dotenv==1.0.1` instead; consider fixing the pin in the repo.
- **Ollama models needed**: `qwen3:8b` (design target) and `qwen2.5:1.5b` (repo's documented CPU fallback) both had to be `ollama pull`ed — neither was present initially.
- Install sequence that worked:
  ```
  python -m venv .venv   # using Python 3.13, NOT the system 3.8
  .venv/Scripts/pip install -q fastapi==0.115.0 "uvicorn[standard]==0.30.6" pydantic==2.9.2 requests==2.32.3 pytest==8.3.3 httpx==0.27.2 python-dotenv==1.0.1
  ollama pull qwen3:8b
  ollama pull qwen2.5:1.5b
  ```

## 2. Data files

Two categories exist:

- **Real production data** (gitignored, appeared at repo root partway through this session — `rebee_client_rebeeai.parsedresumes.json`, `rebee_client_rebeeai.jdmatchresults.json`, `company_ranks.json` (~900MB / 7M lines), `companyDetection.json`, `university_abbs.json`, `Location.json`, `master_universities.csv`). These are the real thing, not samples.
- **Synthetic test data** (`test_data/synthetic_*.json`, committed as part of the pulled branch) — small, self-contained, known-ground-truth. Job IDs in this dataset: `SYN-BACKEND-01`, `SYN-ML-02` (NOT the real prod job ID `6a872a1378da36fe1ceea53f` the UI originally hardcoded — this mismatch caused a real bug, see §5).

To run against synthetic data (recommended for dev/testing), set these env vars before starting the server:
```
PARSED_RESUMES_PATH=<repo>/test_data/synthetic_parsedresumes.json
JD_MATCH_RESULTS_PATH=<repo>/test_data/synthetic_jdmatchresults.json
MASTER_UNIVERSITIES_PATH=<repo>/test_data/synthetic_master_universities.csv
COMPANY_RANKS_PATH=<repo>/test_data/synthetic_company_ranks.json
LOCATION_JSON_PATH=<repo>/test_data/synthetic_location.json
```
Omit these to use the real data files at repo root instead (if present).

## 3. Model choice: qwen3:8b vs qwen2.5:1.5b — the core tradeoff

This machine is **CPU-only, no GPU** (`ollama ps` / `/api/ps` shows `size_vram: 0`). That single fact drives almost everything else in this document.

| | `qwen3:8b` (design target) | `qwen2.5:1.5b` (repo's dev fallback) |
|---|---|---|
| Speed (think off) | ~150-200s cold prompt / ~10-20s warm-cache | ~5-8s |
| Speed (think on) | Multi-minute (2-8 min), even for trivial input like "hi" | not tested |
| Accuracy (tested) | Correct on every real query tried | **Wrong** — hallucinated a false "operator not supported" claim, wrong extracted values |
| Verdict | Keep this as the live model | Not accurate enough to use as-is |

**Concrete finding on thinking mode**: tested `"say hi in one word"` with `think:true` — took 2m22s, generated 682 tokens of rambling internal reasoning about greeting words in different languages, for a two-letter reply. Thinking imposes a large, **fixed** latency tax on every single query regardless of complexity — it does not scale with how hard the query actually is.

**Decision made**: `think: False` in `app/llm/client.py`'s `translate()` payload. Thinking was tried ON specifically to fix the "Yes" confirmation bug (§5) but didn't actually fix it (see below) — it just made the failure mode more polite (an honest re-ask instead of a hallucination) at a huge latency cost across all queries. The deterministic fix in §5 solves the actual bug without needing thinking at all.

## 4. Prompt caching — a real, measured 530x win, but fragile

**Ollama does cache the KV state for a repeated exact prefix.** Measured directly:
- Cold call (first time system prompt is processed): `prompt_eval` ≈ 150,000-200,000ms for the ~2050-token system prompt.
- Warm call (same session, cache hit): `prompt_eval` ≈ 300-2000ms. **That's a ~100-500x speedup on the dominant cost.**
- This works even across **different final queries** sharing the same system-prompt prefix, not just byte-identical repeats — confirmed with two different back-to-back queries both landing at ~1-2s prompt_eval.

**But it's fragile in practice.** Real server usage during this session showed the cache NOT holding between real user requests (every request paying the full ~150-200s cost) even though clean back-to-back test scripts got the fast path reliably. Root causes identified:
1. **`OLLAMA_NUM_PARALLEL` was unset** (defaults to Ollama's auto-selected value, likely >1) — concurrent "slots" each keep their own separate cache; requests round-robining between slots defeats the appearance of a warm single-slot cache. **Fixed**: relaunched Ollama with `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_KEEP_ALIVE=30m` (see §8 for the exact commands — **this was NOT set persistently, only for the current process this session; must be redone on the new machine**).
2. **Any gap where a different prompt/model gets processed in between** evicts the cache (confirmed: interleaving test scripts with different prompt content broke real-server caching repeatedly during this session).
3. **Thinking mode may itself break cache-prefix matching** — a "hi" test with `think:True` still showed cold `prompt_eval` (~165-174s) on both of two consecutive identical calls even with `OLLAMA_NUM_PARALLEL=1` set, unlike the `think:False` case which cached correctly under the same config. Not fully root-caused; deprioritized once thinking was turned off anyway.

**Practical implication for production**: don't rely on Ollama's default single-instance caching for concurrent multi-user traffic. The real production answer discussed (§9) is **vLLM with Automatic Prefix Caching**, which is purpose-built for "many concurrent users sharing one system prompt" — Ollama's caching here is best-effort/single-slot, not designed for that.

## 5. The "Yes" confirmation bug — full root cause and fix

### The bug (as originally reported)
1. User filters `Experience ≥ 5 yrs`.
2. User types "actually 6 years" → correctly applies `Experience ≥ 6 yrs` (deterministic numeric extraction already handled this case, via `PendingClarify` + `_extract_number`).
3. User types "actually make it 7 years instead" → **incorrectly** triggered a confirm-style clarify: *"Should the experience be at least 7 years?"* with Yes/No buttons, instead of just applying it.
4. User clicks "Yes" → the literal string `"Yes"` gets sent as a fresh query to the LLM. The LLM has **no memory of what it just asked** (each call was stateless — no conversation history was ever sent, only `CURRENT FILTERS` + the new query text) and cannot recover "7" from "yes" alone. Result: a wrong/generic response 60-190+ seconds later.

### Root cause #1: no conversation memory at all
`app/llm/client.py`'s `translate()` only ever sent `CURRENT FILTERS` + the new query — the LLM never saw its own prior questions or the user's prior answers. **Fixed generally** (not just for this one case):
- Added `ChatTurn` (`role`, `content`) and `SessionState.history: list[ChatTurn]` in `app/models/schemas.py`.
- `service.py` now appends a user/assistant turn pair on **every** response path (pending-lookup answers, clarify questions, unsupported-filter messages, ok/no_match applies, error rejections) via a `_append_history()` helper, bounded to the last 8 messages (`_MAX_HISTORY_TURNS`).
- `LLMClient.translate()` now takes a `history` param and replays it as real prior chat messages in the Ollama request (`messages: [system, *history, final_user_turn]`) — not a hand-parsed summary, the actual conversation.
- Added prompt rule 0 in `app/llm/prompt.py` instructing the model to resolve short replies against this history instead of re-asking.
- **Verified this plumbing works correctly** two ways: (a) a scripted fake-LLM replay of the exact bug scenario, and (b) direct inspection of the real HTTP payload sent to Ollama — confirmed all history messages present and correct.

### Root cause #2: the model still got it wrong even with correct history
Despite the fully correct history reaching it, `qwen3:8b` (think off) hallucinated: it claimed *"The 'experience' field is not supported for filtering"* — false, `experience` is a real supported field. With thinking on, it didn't hallucinate but also didn't resolve it — it honestly re-asked "Which filter are you confirming?", at a cost of 6+ minutes.

**This is a genuine model reliability limit, not a plumbing bug** — confirmed by directly testing `LLMClient` against live Ollama with the exact real history and inspecting the raw response.

### The actual fix: make it deterministic, don't rely on the LLM at all
Added `clarify_value` (and `clarify_unit`) to `LLMOutput` and `PendingClarify`:
- When the LLM emits a **confirm-style** clarify (a specific value already known, just asking yes/no — e.g. "Should the experience be at least 7 years?"), it must now also emit `clarify_value: 7` in the structured output (schema updated in `app/llm/json_schema.py`; prompt rule 6-clarify-value added with a concrete few-shot example).
- `service.py` now checks, before ever calling the LLM: if there's a pending confirm-clarify with a known `value`, and the reply is a recognizable "yes" or "no" word (see `_extract_yes_no()`), resolve it **entirely in code** — apply the value on "yes", ask what it should be instead on "no". **Zero LLM calls, zero chance of hallucination**, for this pattern.
- Verified via scripted test: only 2 LLM calls logged across a 4-turn conversation that should have needed 3 — the "Yes" resolution never touched the LLM and correctly applied `Experience ≥ 7 yrs`.

### Root cause #3 (found and fixed too): the confirm-clarify shouldn't have fired in the first place
Per the prompt's own existing rule 6, `"actually make it 7 years instead"` **already contains an explicit number** — it should never have triggered a clarify at all (should go straight to applying the filter, same as "actually 6 years" did). This was a genuine model rule-violation, not an edge case needing Yes/No resolution machinery. **Added an explicit reminder in rule 6-clarify-value** telling the model confirm-style clarify is only for values *it* inferred/assumed, never for a number the recruiter already typed themselves.

**Result after all three fixes, verified live against real `qwen3:8b`**: the exact original 4-turn scenario now resolves correctly — turn 3 applies `Experience ≥ 7 yrs` directly (no needless clarify), and a subsequent out-of-context bare "Yes" (with nothing pending) correctly gets an honest clarify with real options instead of a wrong answer — filters preserved throughout, no hallucination anywhere in the trace.

## 6. Company product-vs-service classification

### Original problem
`app/core/company_type.py` classifies each company as Product/Service/Both/Unknown, two-tier (direct per-company cache, falling back to an industry-level cache). Measured on real data: **only ~15% of the 1209-company direct cache had a real classification — 1029/1209 were `Unknown`.** Root cause: the LLM is asked to classify a bare company name from its own training-data memory, which structurally fails for small/regional companies it was never trained on — no amount of retrying or more data volume fixes this, because the input (a bare name) never changes.

### Fix implemented: evidence-based classification
Real resumes contain `experience[].description` text — actual, detailed descriptions of what people did at each company (e.g. *"Provided data management... reporting services to clients for standard and customization ERP projects"* → clearly Service). This was completely unused.

- **`app/core/candidates.py`**: added `_load_company_evidence()` / `get_company_evidence()` — gathers real resume-description excerpts per company, **deduped by `processId`** (important: the same person's resume can appear multiple times across different job-match records; naive per-record counting overcounts one person's account as independent evidence — confirmed this actually happens in the real data, e.g. "Thirdware Solutions Ltd" had 15 raw matches that were all the literal same person/description).
- **`app/core/company_type.py`**: added `warm_cache_with_evidence()` — classifies companies still `Unknown` using up to 5 real excerpts per company instead of the bare name, using an index-based response schema (not name-echo, which is fragile once paragraphs of resume text are involved).
- **Verified working**: "Thirdware Solutions Ltd" flips from `Unknown` (name alone) to correctly `Service` (with evidence) — confirmed both cases directly against the real model.
- **Wired into `scripts/warm_company_types.py`** via a new `--with-evidence` flag (runs after the industry and per-company passes, targets whatever's still `Unknown`).

### Known limitation flagged (not yet mitigated in code)
Resume text can be misleading — not fraud, but role-level language doesn't always reflect company-level business model (e.g. someone at a services company staffed on a client's product team writes "our platform," "our roadmap"). Discussed mitigations, **not yet implemented**:
1. Require multiple independent resumes' evidence before trusting the signal for a given company (single-witness evidence stays weak).
2. Cross-check against the (also now-fixed, see §7) industry signal — agreement between resume-evidence and industry-based inference is much stronger than either alone; disagreement should stay `Unknown`/`Both` rather than picking one.
3. Weight explicit client/customer/engagement language over vague self-description language in the classification prompt.

### Bonus bug found and fixed along the way: industry-lookup normalization mismatch
While investigating, found `company_types_for()` (in `company_type.py`) does a flat `.strip().lower()` lookup into the industry map, but the industry map (`_load_company_industries_from_ranks()`) is keyed by `_normalize_company()`-normalized names (which strips corporate suffixes like "Ltd"/"Inc"). **These two normalizations disagree whenever a company name has a suffix** — confirmed **353 real mismatches** in the actual dataset, including "Tata Consultancy Services Limited" silently never resolving its real, present industry data ("information technology and services"). 

**Fixed**: added `_industry_for_company()` / `_industry_lookup_for()` in `candidates.py`, doing the same normalize+prefix-match resolution `_company_tier_for()` already correctly does for the same underlying data, then passing a pre-resolved, correctly-keyed lookup into `company_types_for()`. **Measured impact**: industry coverage went from effectively broken (only matching by normalization coincidence) to **851/1247 (68%) of real companies** now resolving a real industry.

## 7. Scaling to production / "millions of users" — discussion, not yet implemented

Established that current setup (CPU-only, single local Ollama process) is a dev/prototype configuration, explicitly flagged as such in the original README. For real scale:

1. **GPU-backed inference is not optional** — CPU is ~20-50x slower for this workload. Either self-host on GPU (vLLM/TGI) or use a hosted inference API (Groq, Together.ai, Fireworks).
2. **Concurrency**: current Ollama setup serves essentially one request at a time; production needs multiple GPU replicas behind a queue, or a serving engine that batches concurrent requests (vLLM does this natively).
3. **Cache the LLM translation step itself** (not just the prompt prefix) — many users will type overlapping queries; a cache keyed on normalized query text + current filters could skip the LLM entirely on repeats. Not implemented.
4. **vLLM's Automatic Prefix Caching** specifically solves the "one shared system prompt, many concurrent different users" shape of this workload — a much better fit than Ollama's single-slot caching for real concurrent load.
5. **Fine-tuning** (see §8) reduces prompt size requirements too, which helps here independent of GPU/scale questions.

## 8. The big pending idea: fine-tuning instead of endless prompt-patching

### The problem this solves
The current prompt (`app/llm/prompt.py`) is 48 few-shot examples + rules, ~30KB, and its own documented philosophy is *reactive*: rules/examples get added only after a reproduced failure, never preemptively. This is explicitly **not scalable** — every new pattern added makes the prompt bigger, which makes every single query slower (36% of the current 30KB prompt is few-shot examples alone), and it still only ever covers what's been specifically tested. Genuinely novel phrasing (typos, sarcasm, compound corrections, mixed language) stays untested territory until someone hits it in production.

### The proposed direction (discussed, NOT started)
Move coverage from the **runtime prompt** (paid on every request, bounded by what's been manually patched) into **training data** (paid once, offline, and generalizes past what's literally in it):

1. **Build a synthetic query-generation pipeline** — use a strong model to generate a large, diverse set of `(recruiter query → correct structured filter JSON)` pairs, deliberately covering typos, sarcasm, compound/multi-part corrections, varied phrasing of the same rule, validated against the existing `vocabulary.py` schema rules. This is NOT the same as hardcoding — it's building a broad enough dataset that a fine-tuned model generalizes past the literal examples, the way real training data works. **Not yet built** — checked, and the existing `test_data/generate_synthetic_data.py` only generates synthetic *candidate records* for matching-correctness tests, not query-phrasing training data. This would be a new pipeline.
2. **Human review a sample** for quality/correctness before training on it.
3. **Fine-tune a small model** (e.g. `qwen2.5:1.5b` via LoRA — cheap, fast to train and run) specifically on this narrow task. Goal: 8B-level reliability at 1.5B speed, which resolves the speed-vs-accuracy tradeoff this whole session kept running into, instead of trading one for the other.
4. **Build a held-out eval set**, separate from training data, to catch overfitting rather than just training-set memorization.
5. Requires GPU access for the training step (separate from, but possibly shared with, the inference-time GPU discussed in §7).

**Status**: fully discussed and agreed as the right direction; **no code written yet**. This is the natural next thing to pick up.

## 9. How to run the server (as configured at end of session)

```bash
cd <repo>
export PARSED_RESUMES_PATH="$(pwd)/test_data/synthetic_parsedresumes.json"
export JD_MATCH_RESULTS_PATH="$(pwd)/test_data/synthetic_jdmatchresults.json"
export MASTER_UNIVERSITIES_PATH="$(pwd)/test_data/synthetic_master_universities.csv"
export COMPANY_RANKS_PATH="$(pwd)/test_data/synthetic_company_ranks.json"
export LOCATION_JSON_PATH="$(pwd)/test_data/synthetic_location.json"
export MODEL=qwen3:8b
export LLM_TIMEOUT=300
export LLM_MAX_RETRIES=1
.venv/Scripts/python -m uvicorn app.api.main:app --port 8000
```
UI: `http://localhost:8000/ui/`

**Before starting Ollama**, for the caching fix to take effect (see §4), launch it with:
```powershell
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_KEEP_ALIVE = "30m"
# then start ollama serve (this was NOT made persistent — redo on new machine,
# ideally via a real persistent env var / service config, not a one-off launch)
```

`think: False` is hardcoded in `app/llm/client.py`'s `translate()` payload — this is the current, deliberate choice (see §3), not left over from debugging.

## 10. Open items / suggested next steps

- [ ] Commit the 10 modified files (review diffs first — nothing has been committed this session).
- [ ] Resolve the stranded git stash (`stash@{0}`) — review and drop manually if not needed.
- [ ] Make `OLLAMA_NUM_PARALLEL=1` / `OLLAMA_KEEP_ALIVE` persistent (system env var or service config), not a one-off process launch.
- [ ] Broaden `_extract_yes_no()`'s affirmative/negative word coverage in `service.py` (cheap, zero-risk, flagged as a quick win, not yet done).
- [ ] Decide on and start the fine-tuning initiative (§8) if pursuing the real long-term fix for open-ended query robustness.
- [ ] Consider mitigations for company-type evidence reliability (§6): multi-witness thresholds, cross-checking against industry signal, weighting explicit relationship language.
- [ ] Fix the bad `python-dotenv==1.2.3` pin in `requirements.txt`.
- [ ] If moving to GPU/production infra: revisit §7 (vLLM + Automatic Prefix Caching is the recommended direction over continuing with Ollama at scale).
