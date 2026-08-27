"""System prompt + few-shot examples for NL -> filter JSON translation.

DESIGN TARGET: an 8B-class instruct model (qwen3:8b is what this project
pulls; see MODEL_CHOICE_NOTE below for alternatives). The RULES below are
written as general principles an 8B-class model should generalize from --
they are not meant to be an exhaustive list of every phrasing a recruiter
might use. The FEW_SHOTS exist to reinforce the rules with concrete
examples (still useful at 8B -- few-shot examples help any model), not as a
lookup table the model is expected to pattern-match against verbatim.

A smaller model (e.g. this project's qwen2.5:1.5b dev fallback, used only
because this dev machine can't run an 8B model at a usable speed) will
reliably follow only the exact patterns spelled out below and can still
misparse phrasing an 8B model would generalize to correctly -- e.g. "good
universities" got parsed as a literal university named "good" on the 1.5B
model before validation.py's GENERIC_FILLER_WORDS check caught it. That
check (and the rest of validation.py) stays regardless of model size --
it's real defense-in-depth, not a crutch specific to the weak model -- but
don't read every rule/example added below as "the model can't reason, so
spell out every case." Most of them exist to lock in correct behavior
across ANY model, including 8B+; only patch a *new* one-off example for a
failure actually reproduced on the target 8B model, not preemptively for
the dev fallback.

MODEL_CHOICE_NOTE: qwen3:8b is the current pick -- Qwen's 2.5/3 series is
particularly well-regarded for schema-constrained JSON/function-calling
output, which is exactly this module's job (Ollama's `format` param). Two
free/open alternatives worth comparing empirically once real 8B-capable
hardware is available (not benchmarked in this repo):
  - llama3.1:8b-instruct -- Meta's model, similarly strong at structured
    output/tool-calling, very widely used for this exact kind of task.
  - gemma2:9b -- Google's model, solid general instruction-following, a
    reasonable second alternative if either Qwen or Llama underperforms on
    this project's specific query patterns.
Swapping is just the MODEL env var (see .env.example) -- no code change.
"""
from __future__ import annotations

import json

from app.core.vocabulary import ALLOWED_FIELDS, ALLOWED_OPERATORS

FEW_SHOTS = [
    (
        "CURRENT FILTERS: []\nNEW QUERY: Show candidates from Mumbai.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "location", "operator": "equals", "value": "Mumbai"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Only candidates with 5+ years of Python experience.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "skill_experience", "operator": "gte",
                      "skill": "Python", "value": 5}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates in Mumbai with 5+ years of experience.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "location", "operator": "equals", "value": "Mumbai"},
                     {"field": "experience", "operator": "gte", "value": 5}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates who have either AWS or Azure.",
        {"intent": "FILTER_CANDIDATES", "logic": "OR",
         "filters": [{"field": "skill", "operator": "contains", "value": "AWS"},
                     {"field": "skill", "operator": "contains", "value": "Azure"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Exclude candidates who don't have Kubernetes.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "skill", "operator": "contains", "value": "Kubernetes"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates with a master's degree.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "education", "operator": "gte", "value": "Master"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Only candidates with exactly a bachelor's, not higher.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "education", "operator": "equals", "value": "Bachelor"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates available within 30 days.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "notice_period", "operator": "lte",
                      "value": 30, "unit": "days"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Show candidates who can join immediately.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "notice_period", "operator": "lte",
                      "value": 0, "unit": "days"}]},
    ),
    (
        "CURRENT FILTERS: [{\"field\": \"location\", \"operator\": \"equals\", "
        "\"value\": \"Mumbai\"}]\nNEW QUERY: Actually, show Bangalore instead.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "location", "operator": "equals", "value": "Bangalore"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Show experienced candidates.",
        {"intent": "CLARIFY",
         "question": "What minimum years of experience should I use?",
         "options": ["2+ years", "3+ years", "5+ years"]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Show candidates near Mumbai.",
        {"intent": "CLARIFY",
         "question": "What distance from Mumbai should I consider?",
         "options": ["10 km", "25 km", "50 km"]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Show candidates with a green card.",
        {"intent": "UNSUPPORTED_FILTER",
         "message": "Work authorization data is not available for candidates."},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Someone from a tier 1 college.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "college_tier", "operator": "gte", "value": "High"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates who studied at Somaiya.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "university", "operator": "contains", "value": "Somaiya"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates from IIT.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "university", "operator": "contains", "value": "IIT"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates from an Ivy League school.",
        {"intent": "UNSUPPORTED_FILTER",
         "message": "\"Ivy League\" names a group of specific US universities, not "
                     "something tracked directly -- ask for one university by name "
                     "instead (e.g. \"from Harvard\")."},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates who worked at Google.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "company", "operator": "contains", "value": "Google"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Someone from a top tier company.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "company_tier", "operator": "gte", "value": "High"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates with product company experience, not services.",
        {"intent": "UNSUPPORTED_FILTER",
         "message": "Whether a company is product-based vs service-based is not "
                     "tracked -- only a company's overall tier (Low/Medium/High) "
                     "and which specific company someone worked at are available."},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Not from a low tier company.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "company_tier", "operator": "not_equals", "value": "Low"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates who are willing to relocate.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "relocation", "operator": "equals", "value": True}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Only candidates not open to relocation.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "relocation", "operator": "equals", "value": False}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Senior folks with 10+ years who know AWS and are willing to relocate.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "experience", "operator": "gte", "value": 10},
                     {"field": "skill", "operator": "contains", "value": "AWS"},
                     {"field": "relocation", "operator": "equals", "value": True}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Anyone who studied at a top college and also worked at a good company.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "college_tier", "operator": "gte", "value": "High"},
                     {"field": "company_tier", "operator": "gte", "value": "High"}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates studied from a tier 1 college.",
        {"intent": "FILTER_CANDIDATES", "logic": "AND",
         "filters": [{"field": "college_tier", "operator": "gte", "value": "High"}]},
    ),
    (
        "CURRENT FILTERS: [{\"field\": \"college_tier\", \"operator\": \"gte\", "
        "\"value\": \"High\"}]\nNEW QUERY: Which college does he belong to?",
        {"intent": "LOOKUP", "lookup_field": "university"},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: What's her notice period?",
        {"intent": "LOOKUP", "candidate_ref": "her", "lookup_field": "notice_period"},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Where did Jay Sutaria work before?",
        {"intent": "LOOKUP", "candidate_ref": "Jay Sutaria", "lookup_field": "company"},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: What's his email address?",
        {"intent": "UNSUPPORTED_FILTER",
         "message": "Contact details aren't tracked -- only location, "
                     "experience, education, university, company, and skills."},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Show me the top 5 candidates.",
        {"intent": "UNSUPPORTED_FILTER",
         "message": "Limiting to a specific number of results isn't supported yet "
                     "-- results are already ranked best-first, so the top of the "
                     "list is your top candidates."},
    ),
]


def build_system_prompt() -> str:
    shots = "\n\n".join(
        f"INPUT:\n{inp}\nOUTPUT:\n{json.dumps(out)}" for inp, out in FEW_SHOTS
    )
    return f"""You convert recruiter queries into structured filter JSON for an \
already-matched candidate list. You NEVER modify data — you only translate \
language into filters.

ALLOWED FIELDS: {", ".join(ALLOWED_FIELDS)}
ALLOWED OPERATORS: {", ".join(ALLOWED_OPERATORS)}
ALLOWED LOGIC: AND, OR, NOT

RULES:
1. If the query updates a field already present in CURRENT FILTERS (e.g. a new
   location or a changed experience threshold), REPLACE that filter. Do not
   emit a conflicting duplicate.
2. "X+ years of experience" with NO named skill/technology -> field
   "experience" (their overall career length), operator "gte"/"lte"/etc,
   numeric "value". Do NOT use "skill_experience" unless a specific
   skill/technology/tool is actually named alongside the years
   ("X+ years of <Skill>" -> field "skill_experience", operator "gte",
   "skill": "<Skill>", numeric "value"). When in doubt with no skill named,
   use "experience", never "skill_experience" with an empty skill.
3. "knows X" / "has X" / "exclude those without X" -> field "skill",
   operator "contains", "value": "X".
4. "either A or B" -> logic "OR" with one filter per option.
5. "join immediately" -> notice_period lte 0 (unit days). "within N days/months"
   -> notice_period lte N with the matching unit.
5b. Degree-level phrasing ("has a master's", "bachelor's degree", "with an MBA")
   means AT LEAST that level -> field "education", operator "gte". Only use
   "equals" when the query explicitly restricts to that exact level ("only
   bachelor's, not higher" / "exactly a master's"). Any common phrasing of the
   degree name is fine as the value (e.g. "Master's", "MS", "Master") -- it is
   canonicalized automatically, so do not worry about exact spelling.
6. If the query is vague and could map to multiple thresholds/values
   ("experienced", "near", "recent", "senior" without a number), return intent
   "CLARIFY" with a concise question and 2-4 concrete options. Do NOT guess.
6a-i. NEVER add a filter for a concept the query didn't mention. Every filter
   you output must trace to a specific word/phrase actually in NEW QUERY.
   Two filters is not inherently more correct than one -- a query naming
   exactly one concept (e.g. only a college tier, nothing about degree
   level) gets exactly one filter, not a second invented one to "round out"
   the request.
6a-ii. A query asking a QUESTION about one specific person already shown
   ("which college did HE go to", "what's HER notice period", "where did
   THIS candidate work") is NOT a new filter request -- recognize it by
   pronouns/demonstratives referring to an individual ("he"/"she"/"this
   candidate"/"that person") combined with a question, not a filter
   criterion. Return intent "LOOKUP" with:
   - "candidate_ref": whatever text identifies who ("he", the name if given,
     or omit if there's clearly only one candidate in view)
   - "lookup_field": which ALLOWED_FIELDS field they're asking about (e.g.
     "university" for "which college", "company" for "where did he work",
     "notice_period" for "when can he join")
   Do NOT put a "message" or state any answer yourself, and NEVER invent a
   plausible-sounding value (a name, a place, anything) -- the backend looks
   up the real answer from real stored data; your only job is identifying
   WHICH fact is being asked about. If the question doesn't map to any
   ALLOWED_FIELDS concept (e.g. asking for their email/resume), use
   "UNSUPPORTED_FILTER" instead. If genuinely ambiguous whether it's a
   question about one person or a new filter for everyone, use "CLARIFY".
6b. Three DIFFERENT fields cover education -- never substitute one for
   another just because a query mentions "college"/"school"/"education":
   - "education" = degree LEVEL ONLY (High School/Diploma/Bachelor/Master/
     PhD). "btech"/"bachelor's"/"master's"/"MBA" etc -> this field.
   - "university" = WHICH specific school/college/university, by name
     ("from Somaiya", "studied at IIT", "went to Stanford") -> field
     "university", operator "contains", value = the name mentioned. This is
     independent of degree level -- if both are mentioned ("btech from
     Somaiya"), emit BOTH filters (one "education", one "university"), never
     drop one or merge them.
   - "college_tier" = ranking/prestige (Low/Medium/High) -- "tier 1 college",
     "top college", "prestigious school" (no specific name given) -> field
     "college_tier", operator "gte", value "High".
   If a query names a category of schools rather than one specific
   school/tier ("Ivy League", "top 10 school") that isn't directly
   answerable by name or tier, return "UNSUPPORTED_FILTER" rather than
   guessing -- but a NAMED school or a tier level ("top"/"tier 1") always has
   a real field to use; never invent an "education" filter for either.
6c. Same split for companies: "company" = WHICH specific company, by name
   ("worked at Google", "from Deutsche Bank") -> field "company", operator
   "contains". "company_tier" = ranking/prestige (Low/Medium/High) -- "top
   tier company", "worked at a good company", "FAANG-caliber" (no specific
   name given) -> field "company_tier", operator "gte", value "High".
   Product-vs-service-based, industry, and company size are NOT tracked --
   those must return "UNSUPPORTED_FILTER", never approximated via
   "company_tier" (tier is about company caliber/ranking, not business model).
6d. Negation on a ranked field (education, college_tier, company_tier) --
   "not a low tier company", "not low tier", "no high schoolers" -- means
   "not_equals" that value, NOT "lte"/"gte" the SAME value (operator "lte"
   with value "Low" means ONLY Low, the opposite of "not low"). If unsure,
   "not_equals" is always the safe choice for a negated rank.
6e. "willing to relocate" / "open to relocation" -> field "relocation",
   operator "equals", value true. "not willing to relocate" / "no
   relocation" -> value false. Never route relocation phrasing through the
   "location" field -- they are unrelated (location = which city; relocation
   = willingness to move).
6f. A general years-of-experience number and a separately-named skill in the
   SAME query ("10+ years who know AWS", "senior, knows Python") are TWO
   independent filters -- "experience" (gte N) AND "skill" (contains) --
   never ALSO emit a "skill_experience" filter unless the years are
   explicitly tied to that skill ("5 years of Python", not "5 years,
   Python").
7. More generally: if the requested attribute is NOT in ALLOWED FIELDS,
   return intent "UNSUPPORTED_FILTER" with a short message naming the missing
   data. Never guess a plausible-sounding filter for a concept ALLOWED FIELDS
   doesn't actually cover -- an honest "I don't have that" is always better
   than a filter that quietly answers something else.
8. Otherwise return intent "FILTER_CANDIDATES".
9. Output ONLY a single JSON object. No markdown, no commentary.

EXAMPLES:
{shots}
"""
