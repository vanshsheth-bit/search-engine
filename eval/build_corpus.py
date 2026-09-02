"""Generates eval/corpus.jsonl -- the labelled (query -> correct answer)
bank the eval harness scores against.

Written as a Python literal list, not hand-typed JSONL, for two reasons:
this data doubles as few-shot retrieval bank and fine-tune training data
later (see SESSION_NOTES.md Sec 8 and the design discussion that produced
this file), so it needs to grow by editing structured Python, not by
hand-balancing JSON braces; and re-running this script is how the corpus
stays in sync if ALLOWED_FIELDS/rules change, instead of silently rotting
the way SESSION_NOTES.md flagged the static prompt itself doing.

Cases are grouped by what they probe (see the `tags` on each). Most mirror
or extend prompt.py's own FEW_SHOTS and numbered rules -- scoring the
current qwen3:8b against its own documented rules is the first useful
number to have. A smaller set (tagged "aspirational_*") encodes what the
CORRECT answer is for a compound utterance the CURRENT single-intent
schema cannot represent at all (see the multi-intent/tool-call redesign
discussed in-session) -- these are expected to fail today by construction,
and exist so the redesign has something concrete to measure improvement
against, not to imply the current system is broken for failing them.

Run: .venv/Scripts/python.exe eval/build_corpus.py
"""
from __future__ import annotations

import json
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "corpus.jsonl")

Case = dict  # {id, query, expect, [history], [current_filters], tags, [note]}

CASES: list[Case] = []


def add(id_, query, expect, tags, *, history=None, current_filters=None, note=""):
    c = {"id": id_, "query": query, "expect": expect, "tags": tags}
    if history:
        c["history"] = history
    if current_filters:
        c["current_filters"] = current_filters
    if note:
        c["note"] = note
    CASES.append(c)


# --------------------------------------------------------------------------- #
# location vs country (rule 6a-ii-b)
# --------------------------------------------------------------------------- #
add("loc_001", "Show candidates from Mumbai.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "location", "operator": "equals", "value": "Mumbai"}]},
    ["location_country"])

add("loc_002", "Candidates in India.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "country", "operator": "equals", "value": "India"}]},
    ["location_country"],
    note="A country name must never land in 'location' -- candidate locations "
         "are stored city-level, so this could never match anyone.")

add("loc_003", "Candidates based in the US with Python experience.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "country", "operator": "equals", "value": "United States"},
         {"field": "skill", "operator": "contains", "value": "Python"},
     ]},
    ["location_country"],
    note="Colloquial 'US' must canonicalize to the gazetteer's exact spelling.")

add("loc_004", "candidates in the UK",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "country", "operator": "equals", "value": "United Kingdom"}]},
    ["location_country"])

add("loc_005", "engineers based in the UAE",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "country", "operator": "equals", "value": "United Arab Emirates"},
         {"field": "job_title", "operator": "contains", "value": "Engineer"},
     ]},
    ["location_country"])

add("loc_006", "Mumbai candidates with 3+ years of Python.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "location", "operator": "equals", "value": "Mumbai"},
         {"field": "skill_experience", "operator": "gte", "skill": "Python", "value": 3},
     ]},
    ["location_country", "skill_experience"],
    note="Contrast with loc_007: naming a specific skill switches the "
         "second filter from 'experience' to 'skill_experience'.")

add("loc_007", "Candidates in Mumbai with 5+ years of experience.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "location", "operator": "equals", "value": "Mumbai"},
         {"field": "experience", "operator": "gte", "value": 5},
     ]},
    ["location_country", "experience"])

add("loc_008", "Show candidates near Mumbai.",
    {"intent": "UNSUPPORTED_FILTER", "message_mentions": ["proximity", "distance"]},
    ["location_country", "unsupported"],
    note="No proximity/distance field exists at all -- honest UNSUPPORTED, "
         "not a CLARIFY with no real destination (rule 7 / the near-Mumbai "
         "few-shot).")


# --------------------------------------------------------------------------- #
# experience vs skill_experience (rule 2)
# --------------------------------------------------------------------------- #
add("exp_001", "Only candidates with 5+ years of Python experience.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "skill_experience", "operator": "gte", "skill": "Python", "value": 5}]},
    ["experience", "skill_experience"])

add("exp_002", "Senior folks with 10+ years who know AWS and are willing to relocate.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "experience", "operator": "gte", "value": 10},
         {"field": "skill", "operator": "contains", "value": "AWS"},
         {"field": "relocation", "operator": "equals", "value": True},
     ]},
    ["experience", "compound"],
    note="10+ years is general career length (no skill tied to the number) "
         "-- must stay 'experience', never fold into 'skill_experience' "
         "with AWS (rule 6g).")

add("exp_003", "4 years of Java, specifically.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "skill_experience", "operator": "gte", "skill": "Java", "value": 4}]},
    ["experience", "skill_experience"],
    note="Regression case: 'X years of <skill>' is exactly ONE filter, "
         "never a separate experience+skill pair, even with 'specifically' "
         "attached (rule 2's explicit wrong/right example).")

add("exp_004", "Backend engineer with 6 years experience and Kubernetes know-how.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "job_title", "operator": "contains", "value": "Backend Engineer"},
         {"field": "experience", "operator": "gte", "value": 6},
         {"field": "skill", "operator": "contains", "value": "Kubernetes"},
     ]},
    ["experience", "compound"],
    note="'6 years experience' with no skill named alongside it is general "
         "career length, not skill_experience, even though a skill is "
         "mentioned elsewhere in the same sentence (rule 6g).")


# --------------------------------------------------------------------------- #
# umbrella skill concepts vs one named tool (rule 3)
# --------------------------------------------------------------------------- #
add("skc_001", "Someone with machine learning experience.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "skill", "operator": "in",
                     "value_includes": ["machine learning", "TensorFlow", "PyTorch"]}]},
    ["skill_concept"])

add("skc_002", "Someone with devops experience.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "skill", "operator": "in",
                     "value_includes": ["devops", "Kubernetes", "Docker"]}]},
    ["skill_concept"],
    note="Confirmed live in SESSION_NOTES Sec.3: without its own worked "
         "example, this concept fell back to a bare literal-word 'contains' "
         "and matched nobody.")

add("skc_003", "candidates who know Python",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "skill", "operator": "contains", "value": "Python"}]},
    ["skill_concept"],
    note="Contrast with skc_001/002: a SPECIFIC named tool stays a plain "
         "'contains', never expanded into a list.")

add("skc_004", "cloud experience needed",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "skill", "operator": "in",
                     "value_includes": ["cloud"]}]},
    ["skill_concept"])

add("skc_005", "Candidates who have either AWS or Azure.",
    {"intent": "FILTER_CANDIDATES", "logic": "OR",
     "predicates": [
         {"field": "skill", "operator": "contains", "value": "AWS"},
         {"field": "skill", "operator": "contains", "value": "Azure"},
     ]},
    ["skill_concept", "logic"],
    note="Two specific named products -- plain OR of two contains, NOT an "
         "umbrella-concept expansion (contrast with skc_001/002/004).")

add("skc_006", "knows Python and React",
    {"intent": "FILTER_CANDIDATES", "logic": "AND",
     "predicates": [
         {"field": "skill", "operator": "contains", "value": "Python"},
         {"field": "skill", "operator": "contains", "value": "React"},
     ]},
    ["skill_concept", "logic"])

add("skc_007", "Exclude candidates who don't have Kubernetes.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "skill", "operator": "contains", "value": "Kubernetes"}]},
    ["skill_concept"],
    note="Matches prompt.py's own few-shot verbatim -- 'exclude ... without "
         "X' resolves to a positive contains on X, not not_contains.")


# --------------------------------------------------------------------------- #
# education level / rank (rule 5b)
# --------------------------------------------------------------------------- #
add("edu_001", "Candidates with a master's degree.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "education", "operator": "gte", "value": "Master"}]},
    ["education"])

add("edu_002", "Only candidates with exactly a bachelor's, not higher.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "education", "operator": "equals", "value": "Bachelor"}]},
    ["education"])

add("edu_003", "candidates who hold a Doctor of Philosophy",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "education", "operator": "gte", "value": "Doctorate"}]},
    ["education"],
    note="Regresses the vocabulary.py bug from SESSION_NOTES Sec.5b -- the "
         "full phrase 'doctor of philosophy' was previously unrecognized "
         "as any degree level at all.")

add("edu_004", "btech from Somaiya",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "education", "operator": "gte", "value": "Bachelor"},
         {"field": "university", "operator": "contains", "value": "Somaiya"},
     ]},
    ["education", "university"],
    note="Both mentioned -> BOTH filters, never dropped or merged (rule 6b).")


# --------------------------------------------------------------------------- #
# notice period (rule 5)
# --------------------------------------------------------------------------- #
add("np_001", "Candidates available within 30 days.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "notice_period", "operator": "lte", "value": 30, "unit": "days"}]},
    ["notice_period"])

add("np_002", "Show candidates who can join immediately.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "notice_period", "operator": "lte", "value": 0, "unit": "days"}]},
    ["notice_period"])

add("np_003", "someone with a reasonable notice period",
    {"intent": "CLARIFY", "clarify_field": "notice_period", "clarify_operator": "lte"},
    ["notice_period", "clarify_open"],
    note="Vague ('reasonable' has no fixed number) -- must CLARIFY with "
         "resolution metadata attached (rule 6-clarify-field), not guess.")

add("np_004", "exclude anyone with more than 90 days notice period",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "notice_period", "operator": "lte", "value": 90, "unit": "days"}]},
    ["notice_period", "negation"],
    note="Numeric-threshold negation describes who to KEEP -> lte, never "
         "gt (rule 6d-i).")


# --------------------------------------------------------------------------- #
# relocation (rule 6e)
# --------------------------------------------------------------------------- #
add("rel_001", "Candidates who are willing to relocate.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "relocation", "operator": "equals", "value": True}]},
    ["relocation"])

add("rel_002", "Only candidates not open to relocation.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "relocation", "operator": "equals", "value": False}]},
    ["relocation"])


# --------------------------------------------------------------------------- #
# college_tier / company_tier / company_type / university / company name
# (rules 6b, 6c, 6d)
# --------------------------------------------------------------------------- #
add("tier_001", "Someone from a tier 1 college.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "college_tier", "operator": "gte", "value": "High"}]},
    ["tier_name_type"])

add("tier_002", "Candidates who studied at Somaiya.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "university", "operator": "contains", "value": "Somaiya"}]},
    ["tier_name_type"])

add("tier_003", "Candidates from an Ivy League school.",
    {"intent": "UNSUPPORTED_FILTER", "message_mentions": ["Ivy League"]},
    ["tier_name_type", "unsupported"])

add("tier_004", "Candidates who worked at Google.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "company", "operator": "contains", "value": "Google"}]},
    ["tier_name_type"])

add("tier_005", "Someone from a top tier company.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "company_tier", "operator": "gte", "value": "High"}]},
    ["tier_name_type"])

add("tier_006", "Not from a low tier company.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "company_tier", "operator": "not_equals", "value": "Low"}]},
    ["tier_name_type", "negation"],
    note="Negation on a RANKED field means not_equals, never lte/gte the "
         "same value -- 'lte Low' would mean ONLY Low, the opposite of "
         "'not low' (rule 6d).")

add("tier_007", "No high schoolers, please.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "education", "operator": "not_equals", "value": "High School"}]},
    ["tier_name_type", "negation"])

add("tier_008", "Candidates with product company experience, not services.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "company_type", "operator": "in", "value_includes": ["Product", "Both"]}]},
    ["tier_name_type"])

add("tier_009", "Not a services company, please.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "company_type", "operator": "not_in", "value_includes": ["Service"]}]},
    ["tier_name_type", "negation"])

add("tier_010", "Software developer in Mumbai with product based company experience.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "job_title", "operator": "contains", "value": "Software Developer"},
         {"field": "location", "operator": "equals", "value": "Mumbai"},
         {"field": "company_type", "operator": "in", "value_includes": ["Product", "Both"]},
     ]},
    ["tier_name_type", "compound"],
    note="A concept that IS supported should behave the same combined as "
         "alone -- regression for the 'fabricated company_tier under "
         "compound load' bug in SESSION_NOTES Sec.6.")

add("tier_011", "Software developer in Mumbai at a large company.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "job_title", "operator": "contains", "value": "Software Developer"},
         {"field": "location", "operator": "equals", "value": "Mumbai"},
     ],
     "message_mentions": ["company size"],
     "forbid_fields": ["company_tier", "company_type"]},
    ["tier_name_type", "compound", "unsupported"],
    note="Company SIZE is genuinely untracked and must never be "
         "approximated via company_tier (a different axis: caliber, not "
         "size) or company_type (product/service, not size).")

add("tier_012", "Candidates from a top 10 school.",
    {"intent": "UNSUPPORTED_FILTER"},
    ["tier_name_type", "unsupported"],
    note="A category of schools with no direct name or tier answer -- "
         "must not guess college_tier=High as a stand-in.")

add("tier_013", "Candidates who did not study at IIT.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "university", "operator": "not_contains", "value": "IIT"}]},
    ["tier_name_type", "negation"])


# --------------------------------------------------------------------------- #
# job_title / certification / employment_gap_months (rules 6f-i..iv)
# --------------------------------------------------------------------------- #
add("field_001", "worked as Manager",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "job_title", "operator": "contains", "value": "Manager"}]},
    ["other_fields"])

add("field_002", "AWS certified candidates",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "certification", "operator": "contains", "value": "AWS"}]},
    ["other_fields"],
    note="Contrast with field_003: 'certified' language routes to "
         "certification, not skill.")

add("field_003", "candidates who know AWS",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "skill", "operator": "contains", "value": "AWS"}]},
    ["other_fields"])

add("field_004", "no career gaps over 6 months",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "employment_gap_months", "operator": "lte", "value": 6}]},
    ["other_fields"])

add("field_005", "avoid job hoppers with long gaps",
    {"intent": "CLARIFY", "clarify_field": "employment_gap_months", "clarify_operator": "lte"},
    ["other_fields", "clarify_open"],
    note="No number given -- vague gap request must CLARIFY, same as any "
         "other unnumbered threshold (rule 6f-iii).")

add("field_006", "No one with more than a year-long career break.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "employment_gap_months", "operator": "lte", "value": 12}]},
    ["other_fields", "negation"],
    note="Rule 6d-i's own worked example, verbatim.")


# --------------------------------------------------------------------------- #
# unsupported concepts (rule 7)
# --------------------------------------------------------------------------- #
add("unsup_001", "Show candidates with a green card.",
    {"intent": "UNSUPPORTED_FILTER", "message_mentions": ["work authorization"]},
    ["unsupported"])

add("unsup_002", "candidates with GPA above 8",
    {"intent": "UNSUPPORTED_FILTER"},
    ["unsupported"],
    note="GPA is never tracked -- must not be approximated via 'education' "
         "(degree LEVEL only, not a grade) (rule 6f-iv).")

add("unsup_003", "graduated after 2020",
    {"intent": "UNSUPPORTED_FILTER"},
    ["unsupported"],
    note="Graduation year is never tracked (rule 6f-iv).")

add("unsup_004", "female candidates only",
    {"intent": "UNSUPPORTED_FILTER"},
    ["unsupported"],
    note="Demographic traits are never in ALLOWED_FIELDS (rule 7).")

add("unsup_005", "candidates expecting salary below 20 LPA",
    {"intent": "UNSUPPORTED_FILTER"},
    ["unsupported"])

add("unsup_006", "night shift candidates only",
    {"intent": "UNSUPPORTED_FILTER"},
    ["unsupported"])


# --------------------------------------------------------------------------- #
# open CLARIFY -- vague terms with no explicit number (rule 6)
# --------------------------------------------------------------------------- #
add("clar_001", "Show experienced candidates.",
    {"intent": "CLARIFY", "clarify_field": "experience", "clarify_operator": "gte"},
    ["clarify_open"])

add("clar_002", "Mid level software developer.",
    {"intent": "CLARIFY", "clarify_field": "experience", "clarify_operator": "gte"},
    ["clarify_open", "aspirational_multi_intent"],
    note="Mirrors prompt.py's own few-shot, which asks about experience "
         "and DROPS the explicit 'software developer' job_title entirely -- "
         "a real gap in the current single-intent schema (it cannot both "
         "apply a filter AND ask a question in one turn). Tagged "
         "aspirational: the CORRECT redesigned answer keeps job_title.")


# --------------------------------------------------------------------------- #
# confirm-style CLARIFY / explicit-number regression (rule 6-clarify-value,
# root cause #3 in SESSION_NOTES Sec.5 -- the actual "Yes" bug's real cause)
# --------------------------------------------------------------------------- #
add("confirm_001", "Actually, I meant senior, more like 7+.",
    {"intent": "CLARIFY", "clarify_field": "experience", "clarify_operator": "gte",
     "clarify_value": 7},
    ["confirm_clarify"],
    current_filters=[{"field": "experience", "operator": "gte", "value": 5}],
    note="'senior' is inferred/assumed by the model, not a number the "
         "recruiter typed -- correctly a confirm-style clarify.")

add("confirm_002", "actually make it 7 years instead",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "experience", "operator": "gte", "value": 7}]},
    ["confirm_clarify", "regression"],
    current_filters=[{"field": "experience", "operator": "gte", "value": 5}],
    note="THE regression case for SESSION_NOTES Sec.5's root cause #3: an "
         "EXPLICIT number must apply directly, never trigger a confirm "
         "clarify -- confirm-clarify is only for values the model itself "
         "inferred, never one the recruiter already typed.")

add("confirm_003", "make it 6 instead of 3",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "skill_experience", "operator": "gte", "skill": "Python", "value": 6}]},
    ["confirm_clarify", "regression"],
    current_filters=[{"field": "skill_experience", "operator": "gte", "skill": "Python", "value": 3}],
    note="Same regression, for a skill-scoped field -- the skill context "
         "must be preserved across the replace.")


# --------------------------------------------------------------------------- #
# LOOKUP -- a question about one already-shown candidate (rule 6a-ii)
# --------------------------------------------------------------------------- #
add("lookup_001", "which college did he go to",
    {"intent": "LOOKUP", "lookup_field": "university"},
    ["lookup"])

add("lookup_002", "what's her notice period",
    {"intent": "LOOKUP", "lookup_field": "notice_period"},
    ["lookup"])

add("lookup_003", "where did this candidate work",
    {"intent": "LOOKUP", "lookup_field": "company"},
    ["lookup"])

add("lookup_004", "what's his email address",
    {"intent": "UNSUPPORTED_FILTER"},
    ["lookup", "unsupported"],
    note="A question about one candidate, but email maps to no "
         "ALLOWED_FIELDS concept at all -- UNSUPPORTED, not a fabricated "
         "LOOKUP.")


# --------------------------------------------------------------------------- #
# session merge / replace (rule 1)
# --------------------------------------------------------------------------- #
add("merge_001", "Actually, show Bangalore instead.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "location", "operator": "equals", "value": "Bangalore"}]},
    ["merge"],
    current_filters=[{"field": "location", "operator": "equals", "value": "Mumbai"}],
    note="Must REPLACE the existing location filter, not add a second, "
         "conflicting one.")

add("merge_002", "actually make it 5 years",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "skill_experience", "operator": "gte", "skill": "Python", "value": 5}]},
    ["merge"],
    current_filters=[{"field": "skill_experience", "operator": "gte", "skill": "Python", "value": 3}],
    note="Skill context ('Python') must survive the replace even though "
         "the reply only restates the number.")

add("merge_003", "actually the US instead",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "country", "operator": "equals", "value": "United States"}]},
    ["merge"],
    current_filters=[{"field": "country", "operator": "equals", "value": "India"}])


# --------------------------------------------------------------------------- #
# generic filler words routed to the wrong field (a documented 1.5B failure;
# included specifically to quantify the model-choice gap the harness exists
# to measure -- see prompt.py's MODEL_CHOICE_NOTE and README Sec.1)
# --------------------------------------------------------------------------- #
add("filler_001", "candidates from good universities",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "college_tier", "operator": "gte", "value": "High"}]},
    ["filler_guard", "model_choice_sensitive"],
    note="'good' is a quality cue -> college_tier, never a literal "
         "university named 'good' (validation.py's GENERIC_FILLER_WORDS "
         "catches the literal-name case defensively, but the model should "
         "route correctly in the first place).")

add("filler_002", "someone from a big company",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [{"field": "company_tier", "operator": "gte", "value": "High"}]},
    ["filler_guard", "model_choice_sensitive"])


# --------------------------------------------------------------------------- #
# compound queries: multiple valid filters, with or without one unsupported
# clause mixed in (validation.py's per-filter-independent design)
# --------------------------------------------------------------------------- #
add("comp_001", "8+ years, knows Kubernetes, and open to relocating.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "experience", "operator": "gte", "value": 8},
         {"field": "skill", "operator": "contains", "value": "Kubernetes"},
         {"field": "relocation", "operator": "equals", "value": True},
     ]},
    ["compound"])

add("comp_002", "8+ years, Kubernetes, and open to relocating, at a large company.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "experience", "operator": "gte", "value": 8},
         {"field": "skill", "operator": "contains", "value": "Kubernetes"},
         {"field": "relocation", "operator": "equals", "value": True},
     ],
     "message_mentions": ["company size"],
     "forbid_fields": ["company_tier"]},
    ["compound", "unsupported"],
    note="One genuinely unsupported clause mixed with three valid ones -- "
         "apply the real filters and say honestly what couldn't be "
         "applied, never fabricate a field for it and never drop the "
         "whole query to UNSUPPORTED_FILTER.")


# --------------------------------------------------------------------------- #
# aspirational multi-intent cases: what the CORRECT answer is for an
# utterance mixing an explicit filter with a genuinely vague clause. The
# CURRENT schema has exactly one intent per turn, so it cannot represent
# "apply X, and ask about Y" -- these are expected to fail today. They exist
# to give the redesigned tool-call/multi-op interface (see the in-session
# design discussion) a concrete, numeric target: this tag's exact-match
# rate should go from ~0% to high as that work lands.
# --------------------------------------------------------------------------- #
add("asp_001", "Bangalore fintech folks, senior, payments experience, can start soon.",
    {"intent": "FILTER_CANDIDATES",
     "predicates": [
         {"field": "location", "operator": "equals", "value": "Bangalore"},
         {"field": "skill", "operator": "contains", "value": "payments"},
     ],
     "message_mentions": ["industry"]},
    ["aspirational_multi_intent"],
    note="Ideal: apply location + skill, ask about seniority AND notice "
         "period as two separate queued questions, report industry as "
         "untracked -- all in one turn. Not representable today.")

add("asp_002", "backend engineers who know either Go or Rust, from a product company, not Infosys",
    {"intent": "FILTER_CANDIDATES", "logic": "AND",
     "predicates": [
         {"field": "job_title", "operator": "contains", "value": "Backend Engineer"},
         {"field": "company_type", "operator": "in", "value_includes": ["Product", "Both"]},
         {"field": "company", "operator": "not_contains", "value": "Infosys"},
     ]},
    ["aspirational_multi_intent", "logic"],
    note="AND-over-a-group-containing-OR ((Go OR Rust) AND product AND "
         "NOT Infosys) -- FilterSpec.logic is a single flat value across "
         "all filters today, so mixed boolean structure like this cannot "
         "be expressed at all, only approximated by dropping the OR.")


def main() -> None:
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for c in CASES:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(CASES)} cases to {OUT_PATH}")

    tags: dict[str, int] = {}
    for c in CASES:
        for t in c["tags"]:
            tags[t] = tags.get(t, 0) + 1
    for t, n in sorted(tags.items()):
        print(f"  {t:<28} {n}")


if __name__ == "__main__":
    main()
