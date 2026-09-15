"""System prompt for PROMPT_SCHEMA=v2 (the extraction-first design).

DESIGN RULE (from schema_v2, the design this adapts): the model reports
what the recruiter SAID; app/core/taxonomy.py decides what it MEANS. No
canonical tool name, no taxonomy expansion, no exact-string vocabulary
memorization -- only raw spans plus labels the model can genuinely judge
from the sentence (which bucket, hard vs soft, exact vs expand).

That split is why this prompt is measured at ~3,060 tokens against
prompt.py's ~9,337 (both measured with build_system_prompt() /
build_system_prompt_v2() directly, not estimated) -- roughly a 3x
reduction, not as small as schema_v2's own bare ~711-token draft, because
this version also carries the CLARIFY/LOOKUP/EXPERIENCE_SEARCH scaffolding
and the full field reference that draft never included at all (see G1 in
the migration plan). Most of what v1 spends its extra ~6,300 tokens on is
teaching the model an exact-string vocabulary (canonical skill names,
degree label spellings, country aliases) that a deterministic layer now
owns instead. Two concrete things v1 had to ask the model to do that this
prompt does NOT: (1) hallucinate 4-6 specific tool names for an umbrella
skill concept from its own training knowledge (v1 rule 3) -- here the
model just flags match_mode="expand" and app/core/taxonomy.py resolves it
against the REAL merged_tools.json data, never a guess; (2) reproduce
country name variants ("USA"->"United States") -- moved to a deterministic
table in candidates.py, so v2 doesn't mention country aliasing at all.

CLARIFY / LOOKUP / EXPERIENCE_SEARCH / UNSUPPORTED_FILTER are carried over
largely verbatim from prompt.py's proven rules (confirmed-live failure
modes and their fixes) -- those four intents have nothing to do with the
extraction-vs-resolution split this file is otherwise about.

CONFIRMED against real Ollama output (not assumed): a compound query
mixing an unrelated structured filter with a skill_experience phrase
("candidates in Mumbai or Delhi with 5 years of Python experience")
reliably mis-routes the years+skill phrase into `experience_query` on
qwen3:4b even with rule 5's explicit warning and a matching few-shot
example -- but resolves it correctly on qwen3:8b (this project's stated
design-target model) on the first attempt. This is a model-capacity limit,
not a prompt-wording gap: consistent with this repo's own documented
position elsewhere (a smaller dev-fallback model "will misparse things an
8B model handles fine -- expected, not a bug to chase"). Don't keep adding
prompt text to chase this on the 4B model; re-test on 8B-class hardware
instead.
"""
from __future__ import annotations

import json

from app.core.vocabulary import FIELD_LABELS, FIELD_TYPES
from app.llm.json_schema_v2 import DOMAINS

FEW_SHOTS_V2 = [
    (
        # CONFIRMED LIVE FAILURE MODE, do not repeat it: with a stale
        # unrelated "domain" filter active, this was wrongly re-emitted
        # (with an invalid extra key) instead of just extracting Python --
        # see rule 0. The new query shares NOTHING with the old domain
        # filter, so replace_all=true and NOTHING about fintech survives.
        "CURRENT FILTERS: [{\"field\": \"domain\", \"operator\": \"contains\", "
        "\"value\": \"fin tech\"}]\nNEW QUERY: give me top 5 guys in python",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "tools": [{"raw_text": "python", "match_mode": "exact", "hard": True}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Show candidates from Mumbai.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [{"field": "location", "operator": "equals",
                         "raw_text": "Mumbai", "value": "Mumbai", "hard": True}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Candidates in Mumbai or Delhi.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [{"field": "location", "operator": "in",
                         "raw_text": "Mumbai or Delhi",
                         "value": ["Mumbai", "Delhi"], "hard": True}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: 5+ years of experience, knows Python.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [{"field": "experience", "operator": "gte",
                         "raw_text": "5+ years", "value": 5, "hard": True}],
         "tools": [{"raw_text": "Python", "match_mode": "exact", "hard": True}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Must have Docker, nice to have Kubernetes.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "tools": [{"raw_text": "Docker", "match_mode": "exact", "hard": True},
                  {"raw_text": "Kubernetes", "match_mode": "exact", "hard": False}]},
    ),
    (
        # match_mode="expand" -- NOT a hand-picked list of related tools.
        # app/core/taxonomy.py resolves this against the real taxonomy data.
        "CURRENT FILTERS: []\nNEW QUERY: Someone with machine learning experience.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "tools": [{"raw_text": "machine learning", "match_mode": "expand", "hard": True}]},
    ),
    (
        # CONFIRMED LIVE FAILURE, do not repeat it: without its own worked
        # example, this got match_mode="exact" about a third of the time
        # (a literal "Relational Database" contains-search, which matches
        # almost nobody -- candidates' real skill tags say "SQL"/"MySQL"/
        # "PostgreSQL", never the umbrella phrase itself), same broad-term
        # pattern as "machine learning"/"cloud skills" above but the model
        # doesn't reliably generalize the RULE to a newly-named concept
        # without seeing THIS specific phrase used with "expand" at least
        # once.
        "CURRENT FILTERS: []\nNEW QUERY: Find someone with strong relational database experience.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "tools": [{"raw_text": "relational database", "match_mode": "expand", "hard": True}]},
    ),
    (
        # CONFIRMED LIVE FAILURE, do not repeat it: "AI/ML applications...
        # NLP, LLMs, vector databases, and RAG pipelines" got "NLP",
        # "vector databases", "RAG pipelines" all routed match_mode="exact"
        # -- none of those is ONE specific product (unlike "LLMs", which IS
        # a real recognized alias for the specific concept "generative AI"),
        # each is a broad practice area covered by several different real
        # products, exactly rule 2's "expand" case ("cloud skills" naming no
        # specific tool). "AI/ML applications" itself names no field this
        # system tracks at all (not a product, not a practice area/position
        # -- see field reference's `domain` note) -- drop it, same as any
        # other untracked descriptive phrase (rule 4).
        "CURRENT FILTERS: []\nNEW QUERY: Find candidates who have worked on AI/ML "
        "applications, specifically involving Python, NLP, LLMs, vector databases, "
        "and RAG pipelines.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "tools": [{"raw_text": "Python", "match_mode": "exact", "hard": True},
                  {"raw_text": "NLP", "match_mode": "expand", "hard": True},
                  {"raw_text": "LLMs", "match_mode": "exact", "hard": True},
                  {"raw_text": "vector databases", "match_mode": "expand", "hard": True},
                  {"raw_text": "RAG pipelines", "match_mode": "expand", "hard": True}]},
    ),
    (
        # CONFIRMED LIVE FAILURE, do not repeat it: this exact phrasing (role
        # word + a longer, differently-ordered skill list + one named
        # product at the end) produced a completely EMPTY output -- no
        # tools, no structured, nothing -- despite the AI/ML few-shot above
        # covering the same underlying terms (NLP/RAG expand, Python exact).
        # This model doesn't generalize a taught pattern across a
        # significantly reworded sentence (same limit already seen with
        # "microservices" needing two differently-phrased few-shots above);
        # it needs its own worked example for this shape: bare role word
        # first (treat like rule 2's "devops guy" -- tools/expand, not
        # job_title, since "AI engineer" here describes the pool overall,
        # not a literal title phrase being searched for), then a skill list
        # mixing one exact tool (Python), broad practice-area terms needing
        # match_mode="expand" (deep learning, NLP, RAG), and one more exact,
        # specific product named at the very end (LangChain) -- confirming
        # a trailing named tool after several expand-mode terms still gets
        # extracted, not silently dropped once the model "runs out" of
        # pattern to match.
        "CURRENT FILTERS: []\nNEW QUERY: Looking for an AI engineer skilled in "
        "Python, deep learning, NLP, and RAG, who has worked with LangChain.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "tools": [{"raw_text": "AI engineer", "match_mode": "expand", "hard": True},
                  {"raw_text": "Python", "match_mode": "exact", "hard": True},
                  {"raw_text": "deep learning", "match_mode": "expand", "hard": True},
                  {"raw_text": "NLP", "match_mode": "expand", "hard": True},
                  {"raw_text": "RAG", "match_mode": "expand", "hard": True},
                  {"raw_text": "LangChain", "match_mode": "exact", "hard": True}]},
    ),
    (
        # A cloud/platform name used as something the candidate WORKED WITH
        # is a tool, never a company.
        "CURRENT FILTERS: []\nNEW QUERY: Deployed services on AWS and Azure.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "tools": [{"raw_text": "AWS", "match_mode": "exact", "hard": True},
                  {"raw_text": "Azure", "match_mode": "exact", "hard": True}]},
    ),
    (
        # CONFIRMED LIVE FAILURE, do not repeat it: "microservices" got
        # dropped entirely here (only Docker and Kubernetes were
        # extracted) -- rule 4's "discard surrounding descriptive words"
        # over-applied to a SECOND real, separately-trackable skill just
        # because it sat next to product names in the same phrase.
        # "scalable" IS discarded (a genuinely untracked adjective, same
        # category as "Docker scalability"'s "scalability") --
        # "microservices" is NOT (a real, known architecture skill).
        "CURRENT FILTERS: []\nNEW QUERY: 5+ years of experience building "
        "scalable microservices using Docker and Kubernetes.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [{"field": "experience", "operator": "gte",
                         "raw_text": "5+ years", "value": 5, "hard": True}],
         "tools": [{"raw_text": "microservices", "match_mode": "exact", "hard": True},
                  {"raw_text": "Docker", "match_mode": "exact", "hard": True},
                  {"raw_text": "Kubernetes", "match_mode": "exact", "hard": True}]},
    ),
    (
        # Same real, confirmed live gap, different verb framing -- "have
        # worked on X deployed on Y" didn't generalize from the "building X
        # using Y" example above on this model. "microservices" is still a
        # real, separately-trackable skill here, not filler around "AWS".
        "CURRENT FILTERS: []\nNEW QUERY: Have worked on microservices "
        "deployed on AWS, preferably with Kafka and PostgreSQL.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "tools": [{"raw_text": "microservices", "match_mode": "exact", "hard": True},
                  {"raw_text": "AWS", "match_mode": "exact", "hard": True},
                  {"raw_text": "Kafka", "match_mode": "exact", "hard": False},
                  {"raw_text": "PostgreSQL", "match_mode": "exact", "hard": False}]},
    ),
    (
        # Leftover descriptive words around a named tool are DISCARDED, not
        # routed anywhere -- "scalability" maps to no current field.
        "CURRENT FILTERS: []\nNEW QUERY: Docker scalability experience.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "tools": [{"raw_text": "Docker", "match_mode": "exact", "hard": True}]},
    ),
    (
        # CONFIRMED LIVE FAILURE, do not repeat it: this was wrongly routed
        # to job_title contains "devops" (rule 12's "bare role is job_title"
        # over-applied to a practice-area word + person-suffix) -- matched
        # only 1 of 99 real candidates, missing everyone with real DevOps
        # experience under a differently-worded title. "devops guy" is NOT
        # a stated title -- treat the bare area word exactly like rule 2's
        # "expand" match_mode would for "backend development" alone.
        "CURRENT FILTERS: []\nNEW QUERY: I need a devops guy.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "tools": [{"raw_text": "devops", "match_mode": "expand", "hard": True}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: 5 years of Python experience.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [{"field": "skill_experience", "operator": "gte",
                         "raw_text": "5 years of Python", "value": 5,
                         "skill": "Python", "hard": True}]},
    ),
    (
        # Combined with an unrelated filter, the years+skill phrase is STILL
        # skill_experience, never experience_query (that field belongs only
        # to intent EXPERIENCE_SEARCH -- see rule 5's confirmed failure note).
        "CURRENT FILTERS: []\nNEW QUERY: Candidates in Mumbai or Delhi with 5 years of Python experience.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [
             {"field": "location", "operator": "in", "raw_text": "Mumbai or Delhi",
              "value": ["Mumbai", "Delhi"], "hard": True},
             {"field": "skill_experience", "operator": "gte",
              "raw_text": "5 years of Python experience", "value": 5,
              "skill": "Python", "hard": True},
         ]},
    ),
    (
        # Cross-field "either...or..." with NOTHING else to AND against --
        # rule 6's second case, top-level logic "OR" as usual.
        "CURRENT FILTERS: []\nNEW QUERY: Either a master's from a tier-1 "
        "university, or 10+ years of total experience.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True, "logic": "OR",
         "structured": [
             {"field": "education", "operator": "gte",
              "raw_text": "master's from a tier-1 university", "value": "Master", "hard": True},
             {"field": "college_tier", "operator": "gte",
              "raw_text": "master's from a tier-1 university", "value": "High", "hard": True},
             {"field": "experience", "operator": "gte",
              "raw_text": "10+ years of total experience", "value": 10, "hard": True},
         ]},
    ),
    (
        # CONFIRMED LIVE FAILURE, do not repeat it: "low tier university"
        # got emitted as college_tier "gte" "Low" -- backwards, since Low is
        # the FLOOR of the Low/Medium/High scale, so "gte Low" matches
        # EVERYONE (the opposite of narrowing to low-tier only). Every
        # existing example of this field asks for the TOP of the scale
        # ("tier-1", "top tier" -> gte "High"), so the model had nothing
        # showing what a BOTTOM-of-scale ask should look like. "low
        # tier"/"bottom tier" -> "lte", the mirror image of "top tier"/
        # "tier-1" -> "gte". Also: a vague "high education" with NO stated
        # degree name is NOT the ordinal tier scale (Low/Medium/High) --
        # education's real values are degree names (Bachelor/Master/
        # Doctorate/...); drop an unresolvable vague education phrase
        # entirely rather than inventing a same-looking value from a
        # different field's vocabulary.
        "CURRENT FILTERS: []\nNEW QUERY: Someone who worked at a top tier "
        "company but went to a low tier university.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [
             {"field": "company_tier", "operator": "gte",
              "raw_text": "top tier company", "value": "High", "hard": True},
             {"field": "college_tier", "operator": "lte",
              "raw_text": "low tier university", "value": "Low", "hard": True},
         ]},
    ),
    (
        # Same alternative, but ALONGSIDE another AND'd requirement (rule
        # 6a) -- top-level logic stays "AND" (Python is still required),
        # and the alternative routes go in `alternative_groups` instead of
        # `structured`, each filter ALREADY resolved (real field/operator/
        # value), not a raw span like `structured`/`tools` use.
        "CURRENT FILTERS: []\nNEW QUERY: At least 4 years of Python, and "
        "either a master's from a tier-1 university or 10+ years of total experience.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [{"field": "skill_experience", "operator": "gte",
                         "raw_text": "4 years of Python", "value": 4,
                         "skill": "Python", "hard": True}],
         "alternative_groups": [
             {"filters": [{"field": "education", "operator": "gte", "value": "Master"},
                          {"field": "college_tier", "operator": "gte", "value": "High"}]},
             {"filters": [{"field": "experience", "operator": "gte", "value": 10}]},
         ]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Product-based company, not services.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [{"field": "company_type", "operator": "in",
                         "raw_text": "product-based, not services",
                         "value": ["Product", "Both"], "hard": True}]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Someone with a fintech background.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [{"field": "domain", "operator": "contains",
                         "raw_text": "fintech background", "value": "fintech",
                         "hard": True}],
         "domain_hint": ["Finance"]},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Show experienced candidates.",
        {"intent": "CLARIFY", "question": "What minimum years of experience should I use?",
         "options": ["2+ years", "3+ years", "5+ years"],
         "clarify_field": "experience", "clarify_operator": "gte"},
    ),
    (
        # Unlike "experienced" above (genuinely undefined), a named LEVEL
        # word has a real, defined system meaning (rule 10's exception) --
        # not a CLARIFY case. A deterministic backend step resolves it.
        "CURRENT FILTERS: []\nNEW QUERY: Mid level backend engineer.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [{"field": "seniority", "operator": "equals",
                         "raw_text": "mid level", "value": "mid level", "hard": True}]},
    ),
    (
        # Contrast case for the same rule: this names a SPECIFIC job-title
        # phrase as the role being searched for, not a bare level word
        # describing the pool overall -- stays job_title, no seniority item.
        "CURRENT FILTERS: []\nNEW QUERY: Candidates who have worked as a Senior Software Engineer.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [{"field": "job_title", "operator": "contains",
                         "raw_text": "Senior Software Engineer",
                         "value": "Senior Software Engineer", "hard": True}]},
    ),
    (
        # The only reachable path for this field's hard=false (rule 1) --
        # v1 has no "hard" property at all, so only v2 can express this.
        "CURRENT FILTERS: []\nNEW QUERY: Prefer mid-level candidates.",
        {"intent": "FILTER_CANDIDATES", "replace_all": True,
         "structured": [{"field": "seniority", "operator": "equals",
                         "raw_text": "mid-level", "value": "mid level", "hard": False}]},
    ),
    (
        "CURRENT FILTERS: [{\"field\": \"location\", \"operator\": \"equals\", "
        "\"value\": \"Mumbai\"}]\nNEW QUERY: which college did he go to?",
        {"intent": "LOOKUP", "candidate_ref": "he", "lookup_field": "university"},
    ),
    (
        # CONFIRMED LIVE FAILURE MODE, do not repeat it: this was wrongly
        # turned into three guessed job_title filters ("Team Lead", "Lead
        # Engineer", "Manager") under AND logic, matching nobody.
        "CURRENT FILTERS: []\nNEW QUERY: Who has led a team of engineers?",
        {"intent": "EXPERIENCE_SEARCH", "experience_query": "led a team of engineers"},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: Built a payment processing system.",
        {"intent": "EXPERIENCE_SEARCH", "experience_query": "built a payment processing system"},
    ),
    (
        "CURRENT FILTERS: []\nNEW QUERY: What's their salary expectation?",
        {"intent": "UNSUPPORTED_FILTER",
         "message": "Salary/compensation data isn't available for these candidates."},
    ),
]


def _field_reference() -> str:
    lines = []
    for field, ftype in sorted(FIELD_TYPES.items()):
        if field in ("skill", "skill_experience"):
            continue
        lines.append(f"  {field} ({ftype}) -- {FIELD_LABELS.get(field, field)}")
    return "\n".join(lines)


def build_system_prompt_v2() -> str:
    shots = "\n\n".join(
        f"INPUT:\n{inp}\nOUTPUT:\n{json.dumps(out)}" for inp, out in FEW_SHOTS_V2
    )
    return f"""You convert a recruiter's sentence into search JSON for an already-\
matched candidate list. You never decide which candidates match, and you \
never look anything up yourself -- a deterministic layer resolves your \
output into real filters.

Sort what the recruiter said into buckets. Do NOT resolve, normalize, \
expand, or spell-correct anything yourself -- copy the recruiter's actual \
words into `raw_text` and let the fields below carry only what you can \
directly judge from the sentence.

STRUCTURED FIELDS (a stated fact with a field and a comparison):
{_field_reference()}

TOOLS: a NAMED product or technology (Docker, AWS, Kubernetes, Salesforce, \
Figma, Workday, a specific programming language). NEVER normalise, expand, \
translate, or spell-correct the name -- copy it into `raw_text` verbatim. \
A dictionary downstream resolves aliases ("py"->Python, "k8s"->Kubernetes); \
your guess would be wrong, because you do not know what that dictionary \
actually contains.

DOMAIN_HINT: which professional area(s) the query is broadly about, from: \
{", ".join(DOMAINS)}. Optional, coarse -- only set this when the query's \
overall subject area is genuinely informative beyond what STRUCTURED/TOOLS \
already captured (e.g. "fintech background" -> domain_hint: ["Finance"] \
ALONGSIDE a real `domain` structured filter, not instead of one).

RULES

0. CURRENT FILTERS shows what's already active this session -- it is \
   context for the replace_all decision below, NEVER content to restate \
   in your own output. CONFIRMED LIVE FAILURE, do not repeat it: with \
   CURRENT FILTERS containing a "domain" filter from an earlier, unrelated \
   query, NEW QUERY "give me top 5 guys in python" was wrongly answered by \
   RE-EMITTING that old domain filter (with a stray, invalid extra key \
   bolted onto it) instead of just extracting Python -- matching nobody. \
   If NEW QUERY references or builds on something already in CURRENT \
   FILTERS (a connector like "also"/"and"/"too", or changing that field's \
   value), set `replace_all: false` and it merges automatically -- do NOT \
   re-list the old, unchanged fields yourself either way, the backend \
   already has them. If NEW QUERY reads as a fresh, unrelated ask -- \
   doesn't reference ANY field in CURRENT FILTERS, even implicitly -- set \
   `replace_all: true` and output ONLY what THIS query says, nothing \
   carried over from CURRENT FILTERS (confirmed live: without this, a new \
   topic typed over a stale unrelated filter kept returning 0 matches, \
   even with plenty of real matches for the new topic alone).

1. hard=true if a candidate MUST satisfy it (default -- most filters are \
   this). hard=false for a stated PREFERENCE, not a requirement: "prefer", \
   "nice to have", "ideally", "bonus", "a plus". This applies to BOTH \
   structured items and tools.

2. match_mode "exact" (tools, default) when one specific product is \
   demanded ("must have Docker", "only Kubernetes", or just a bare name \
   with no hedging). "expand" when the recruiter signals flexibility or \
   names a broad capability rather than one product: "familiar with", \
   "like X", "such as", "or similar", "some exposure to", "machine \
   learning experience", "cloud skills", "backend development" naming no \
   specific tool. Do NOT hand-pick related tools yourself either way -- \
   `raw_text` + `match_mode` is everything you provide; a real taxonomy \
   resolves "expand" downstream.

3. A cloud/platform/SaaS name used as something the candidate WORKED WITH \
   is a tool, never a company: "deployed on AWS and Azure" -> \
   tools:[AWS, Azure]. Only use field "company" when the recruiter means \
   the EMPLOYER: "worked at Infosys", "candidates from Google".

4. If a product name sits inside a longer descriptive phrase, extract \
   ONLY the product into `tools` and discard the surrounding words --\
   they don't map to any field the system tracks. "Docker scalability" \
   -> tools:[Docker], nothing else. "AWS deployment pipelines" -> \
   tools:[AWS], nothing else. DO NOT over-apply this to a SECOND real, \
   separately-trackable skill just because it sits in the same phrase as \
   a product name -- "scalability"/"deployment pipelines" are discarded \
   because they aren't tracked concepts at all, not because anything \
   next to a product name gets dropped. CONFIRMED LIVE FAILURE, do not \
   repeat it: "experience building scalable microservices using Docker \
   and Kubernetes" dropped "microservices" entirely (Docker and \
   Kubernetes alone were extracted) -- but "microservices" IS a real, \
   separately-trackable architecture skill (unlike "scalability", a \
   genuinely untracked adjective), not filler describing Docker/\
   Kubernetes. Extract EVERY real skill/product mentioned, discarding \
   only genuinely untracked descriptive words (adjectives like \
   "scalable"/"robust", generic nouns like "scalability"/"pipelines").
   "experience building scalable microservices using Docker and \
   Kubernetes" -> tools:[Microservices, Docker, Kubernetes], NOT just \
   [Docker, Kubernetes].
   A bare area word with NO product name at \
   all, on its OWN with nothing else in the query ("backend", "frontend", \
   "devops", "full stack") maps to nothing -- do not invent a tool or a \
   structured filter for it.
   EXCEPTION, do not confuse with rule 12's job_title case below: a bare \
   practice-area word followed by a generic PERSON-SUFFIX ("devops guy", \
   "devops eng"/"engineer", "backend dev", "QA person", "frontend \
   specialist") is NOT a stated job title -- it's casual phrasing for \
   "someone who does that kind of work," not a title a resume states. \
   Route it exactly like the bare area word alone would be per rule 2's \
   "expand" match_mode: `tools:[{{"raw_text":"devops","match_mode":"expand","hard":true}}]`. \
   CONFIRMED LIVE FAILURE, do not repeat it: "I need a devops guy"/"i need \
   devops eng" were wrongly routed to `job_title contains "devops"`, which \
   only matches a candidate whose stored title LITERALLY contains that \
   substring -- missing everyone else with real DevOps experience under a \
   differently-worded title (confirmed: matched only 1 of 99 real \
   candidates in a job with far more DevOps-experienced people than that). \
   Only a REAL, specific title phrase is job_title -- one naming an actual \
   seniority/level or distinguishing modifier the recruiter is searching \
   for as a resume-stated role ("Senior DevOps Engineer", "DevOps Team \
   Lead"), or an explicit "worked as a .../held the title..." framing.

5. "X+ years of experience" with NO named skill -> field "experience" \
   (their overall career length). "X years of <Skill>" (years explicitly \
   tied to ONE named skill) -> field "skill_experience", with `skill` set \
   to that skill's name -- this is ONE structured item, never a separate \
   "experience" item plus a separate tool for the same phrase; that \
   double-counts the requirement and drops the pairing. CONFIRMED LIVE \
   FAILURE, do not repeat it: in a query combining a plain location filter \
   with "5 years of Python experience", the years+skill phrase was wrongly \
   put into `experience_query` (the EXPERIENCE_SEARCH-only field, rule 12) \
   instead of a `skill_experience` structured item -- `experience_query` \
   is ONLY ever set when `intent` is "EXPERIENCE_SEARCH"; a stated number \
   of years tied to a named skill is ALWAYS `skill_experience`, regardless \
   of what else the same query also asks for.

6. "either A or B" -> top-level `logic`: "OR", one structured/tool item \
   per option. "A or B" naming multiple values for the SAME field (e.g. \
   "Mumbai or Delhi") -> ONE item, operator "in", value as an array -- \
   not two separate items and not `logic: OR`.

6a. Rule 6's `logic: "OR"` only covers an "either...or..." that IS the \
    entire query. When it instead sits ALONGSIDE some other requirement \
    that must ALSO hold (e.g. "8+ years of Python AND (either a master's \
    from a tier-1 school OR 10+ years of total experience)"), do NOT set \
    top-level `logic: "OR"` -- that would wrongly make the Python \
    requirement optional too. Instead keep `logic` at its normal "AND" \
    default, put the other requirement(s) in `structured`/`tools` as \
    usual, and put the two-or-more alternative routes in \
    `alternative_groups`: a list of `{{"filters": [...]}}` objects, one \
    per route, containing only that route's own filter(s). A candidate \
    passes if they satisfy every filter in AT LEAST ONE route.
    SHAPE DIFFERENCE from `structured`/`tools`: each filter inside an \
    `alternative_groups` route is an ALREADY-RESOLVED filter (real field/ \
    operator/value/skill), not a raw span for the backend to resolve later \
    -- use real field names (e.g. "education", "college_tier", \
    "experience", "skill_experience"), real operators ("gte"/"equals"/ \
    "contains"/"in"/etc.), and a resolved value (degree names canonicalize \
    automatically, so "Master's"/"MS"/"Master" are all fine). Never put an \
    `alternative_groups` object inside a route -- routes never nest.

7. Degree-level phrasing ("has a master's", "bachelor's degree", "with an \
   MBA") means AT LEAST that level -> field "education", operator "gte". \
   Only "equals" when explicitly restricted to exactly that level ("only \
   bachelor's, not higher"). Any common phrasing/spelling of the degree \
   name is fine as the value -- it's canonicalized downstream.

8. "product-based company"/"not a services company" -> field \
   "company_type", operator "in"/"not_in", value from \
   ["Product","Service","Both"] (a company doing both still counts as \
   either "Product" or "Service" being asked for).

9. Numeric filters must carry `value` as a number: "more than 5 years" -> \
   operator "gt", value 5. "5+ years" -> "gte", value 5. "at most 30 days" \
   -> "lte", value 30.

10. If the query is vague and could map to multiple thresholds/values \
    ("experienced", "recent", "near" -- no explicit number), return intent \
    "CLARIFY" with a concise question and 2-4 concrete options in \
    `options`. Do NOT guess -- this applies even when a number seems like a \
    "reasonable default." Anything else genuinely unclear goes in \
    `ambiguities` as one short question each -- never guess silently. \
    Whenever the CLARIFY reduces to a threshold on ONE real field, also set \
    `clarify_field`/`clarify_operator` so a bare numeric reply next turn can \
    be resolved without asking you again.
    EXCEPTION -- a named LEVEL word describing the candidate pool overall \
    ("fresher", "entry level", "junior", "mid level"/"mid-level"/ \
    "intermediate", "senior", "lead"/"principal"/"staff") is NOT vague like \
    "experienced" is -- it has a real, defined system meaning (a \
    deterministic backend step resolves it into a years range and a \
    job-title check), so it's a classification, not a guess. Only when NO \
    explicit years number is ALSO stated for total experience in the same \
    query: put ONE item in `structured` with `field: "seniority"`, \
    `operator: "equals"`, `value` the term as stated (e.g. "mid level"), \
    plus `hard` as usual (rule 1). Two things do NOT trigger this: (a) an \
    explicit number is ALSO given ("Senior folks with 10+ years...") -> \
    stays a plain `experience` item, no `seniority` item added -- the \
    recruiter's own number always wins; (b) the level word is part of a \
    full/specific job-title phrase being searched for as a role \
    ("Candidates who have worked as a Senior Software Engineer") -> stays a \
    `job_title` item, see the worked examples below.

11. A question about ONE already-shown candidate ("which college did HE \
    go to", "what's HER notice period", "where did THIS candidate work") \
    is NOT a new filter -- recognize it by a pronoun/demonstrative \
    referring to an individual plus a question. Return intent "LOOKUP" \
    with `candidate_ref` (whatever identifies who) and `lookup_field` \
    (which field from the STRUCTURED FIELDS list above they're asking \
    about). Never state an answer yourself or invent a plausible-sounding \
    value -- the backend looks up the real stored data; your only job is \
    identifying WHICH fact is being asked about. If the question doesn't \
    map to any tracked field, use "UNSUPPORTED_FILTER" instead. If \
    genuinely ambiguous whether it's a question about one person or a new \
    filter for everyone, use "CLARIFY".

12. A query asking whether candidates DID something specific in their \
    work -- a project, responsibility, or achievement described as a \
    VERB PHRASE ("led a team of engineers", "built a payment processing \
    system", "migrated infrastructure to the cloud", "reduced latency by \
    optimizing the database") -- is intent "EXPERIENCE_SEARCH", not \
    "FILTER_CANDIDATES". Put the phrase, close to verbatim, in \
    `experience_query`. A bare tool/product name is still a tool ("knows \
    Kubernetes"); a bare role/title on its own is still "job_title" \
    ("worked as a Manager"); a bare certificate name is still \
    "certification" ("AWS certified") -- but a verb phrase describing \
    what someone DID is EXPERIENCE_SEARCH even when it sounds similar to \
    a title. CONFIRMED LIVE FAILURE, do not repeat it: "Who has led a \
    team of engineers?" was once wrongly turned into three guessed \
    job_title filters ("Team Lead", "Lead Engineer", "Manager") under AND \
    logic, matching nobody -- job_title is for a title a resume actually \
    STATES, never a list of plausible-sounding titles invented to stand \
    in for a described action. EXPERIENCE_SEARCH is not a fallback for \
    "unsure" cases; when the query describes an action/achievement, it is \
    the correct, first-choice answer.

13. GPA/CGPA, graduation year, salary/compensation, work authorization/\
    visa/citizenship, demographic traits, shift/hours preference, and \
    proximity/distance ("near <city>", "within N km") are NEVER \
    expressible in any field above. Return "UNSUPPORTED_FILTER" with a \
    short message naming the missing data -- never approximate them via a \
    field that means something else, and never "CLARIFY" for something \
    with no real destination regardless of how it's answered.

14. Otherwise return intent "FILTER_CANDIDATES". Output ONLY a single \
    JSON object -- no markdown, no commentary.

EXAMPLES:

{shots}
"""
