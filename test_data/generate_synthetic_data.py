"""Generates a deliberately adversarial synthetic candidate dataset for
testing the NL-filter pipeline against KNOWN ground truth -- unlike the real
data (where verifying a result meant manually auditing it by hand), every
property here is authored, so a query's correct answer is known by
construction and can be asserted directly.

This is NOT curated to make the system look good. Every candidate below
exists to probe a specific real failure mode already found or suspected in
this session: city-name aliasing, concept-vs-literal skill matching, messy
company/certification text, name collisions for LOOKUP, boundary values,
missing data, multi-degree ranking, and so on. Company and university names
are real -- only the candidate identities are invented.

Also writes minimal synthetic Location.json / master_universities.csv /
company_ranks.json equivalents (scoped to exactly the places/schools/
companies referenced above), so the whole test suite is genuinely
self-contained and doesn't silently depend on the real, gitignored
reference datasets being present. That WAS a real bug here: the test
fixture patched resume/match-result paths but not these three, so on a
clean checkout with no real data, city normalization fell back to an empty
gazetteer and the hyphenated-city test failed.

Run: .venv/Scripts/python.exe test_data/generate_synthetic_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# jobId/jdId pair for each fake job. Two jobs so job-scoping is testable: a
# candidate matched to JOB_BACKEND should not appear when querying JOB_ML
# unless they also have a match record for it.
JOB_BACKEND = {"jdId": "aaaaaaaaaaaaaaaaaaaaaaaa", "jobId": "SYN-BACKEND-01"}
JOB_ML = {"jdId": "bbbbbbbbbbbbbbbbbbbbbbbb", "jobId": "SYN-ML-02"}


def edu(degree, university, year="2015", gpa=""):
    return {"degree": degree, "university": university, "year": year, "gpa": gpa}


def exp(position, company, start, end, duration_years, is_ongoing=False,
        location="", current_location=None, contractor=None):
    months = round(duration_years * 12)
    return {
        "position": position, "company": company,
        "date": f"{start} - {end}", "start_date": start, "end_date": end,
        "duration_years": duration_years, "duration_months": months,
        "duration_display": f"{duration_years} years",
        "is_ongoing": is_ongoing, "description": f"Worked as {position} at {company}.",
        "location": location, "current_location": current_location,
        "contractor": contractor, "engagements": None,
    }


def gap(months, after_company, after_position, after_end, before_company, before_position, before_start):
    return {
        "gap_months": months, "gap_years": round(months / 12, 1),
        "gap_display": f"{months} months",
        "from": after_end, "to": before_start,
        "after": {"company": after_company, "position": after_position, "end_date": after_end},
        "before": {"company": before_company, "position": before_position, "start_date": before_start},
    }


def cert(name):
    return {"name": name}


def resume(process_id, name, personal_location, total_experience,
           experience, education, skills, certifications=None, gaps=None,
           skills_key="skillsNormalized"):
    r = {
        "processId": process_id,
        "personalInfo": {
            "name": name, "email": f"{process_id}@example-test.invalid",
            "phone": "", "personal_location": personal_location,
            "github": "", "linkedin": "",
        },
        "totalExperience": total_experience,
        "experience": experience,
        "education": education,
        "gaps": gaps or [],
        "certifications": certifications or [],
    }
    r[skills_key] = skills
    return r


# --------------------------------------------------------------------------- #
# 25 candidates. Each comment states the specific thing it's designed to test.
# --------------------------------------------------------------------------- #
CANDIDATES = []

# 1. Concept-vs-literal skill gap (ML): only specific tools, never the phrase
# "machine learning". college: real IIT (should hit real tier data).
CANDIDATES.append(resume(
    "proc_synth_001", "Priya Nair", "Bengaluru", "6 years 2 months",
    [exp("Machine Learning Engineer", "Google", "Jun 2019", "Present", 6.2, is_ongoing=True)],
    [edu("Master of Technology (Computer Science)", "Indian Institute of Technology Bombay", "2019")],
    ["TensorFlow", "PyTorch", "Docker", "AWS", "Python", "Kubernetes"],
    certifications=[cert("AWS Certified Machine Learning - Specialty")],
))

# 2 & 3. EXACT full-name collision ("Priya Sharma" x2) -- LOOKUP ambiguity.
# #2: frontend-only skills, no literal "frontend"; Delhi.
CANDIDATES.append(resume(
    "proc_synth_002", "Priya Sharma", "Delhi", "3 years",
    [exp("UI Developer", "Zomato", "Jan 2022", "Present", 3.0, is_ongoing=True)],
    [edu("Bachelor of Engineering (Computer Science)", "Delhi Technological University", "2021")],
    ["Vue.js", "Tailwind CSS", "TypeScript", "Figma"],
))
# #3: backend-only skills, Mumbai, senior. Same full name, different person.
CANDIDATES.append(resume(
    "proc_synth_003", "Priya Sharma", "Mumbai", "8 years",
    [exp("Backend Engineer", "Flipkart", "Mar 2016", "Present", 8.0, is_ongoing=True)],
    [edu("Bachelor of Technology (Information Technology)", "University of Mumbai", "2015")],
    ["Java", "Spring Boot", "PostgreSQL", "Kafka"],
))

# 4. Historical city name "Bombay" (not "Mumbai") -- alias canonicalization.
# Also: devops concept via tools only, no literal "devops" (taxonomy doesn't
# model "devops" as its own entry -- this is a KNOWN gap, expected to still
# fail unless the LLM's own-knowledge fallback catches it).
CANDIDATES.append(resume(
    "proc_synth_004", "Arjun Verma", "Bombay", "5 years",
    [exp("Site Reliability Engineer", "Swiggy", "Jul 2020", "Present", 5.0, is_ongoing=True)],
    [edu("Bachelor of Engineering (Electronics)", "Mumbai University", "2019")],
    ["Kubernetes", "Terraform", "Ansible", "Jenkins", "Prometheus"],
))

# 5. Historical city name "Bangalore" (the OTHER direction of the alias).
CANDIDATES.append(resume(
    "proc_synth_005", "Rohan Iyer", "Bangalore", "4 years",
    [exp("Cloud Engineer", "Infosys", "Jun 2021", "Present", 4.0, is_ongoing=True)],
    [edu("Bachelor of Engineering (Computer Science)", "R.V. College of Engineering", "2020")],
    ["AWS", "Azure", "Google Cloud Platform", "Cloud Computing"],
))

# 6. Hyphenated compound city "Navi-Mumbai" -- hyphen-splitting fix.
CANDIDATES.append(resume(
    "proc_synth_006", "Kavya Reddy", "Navi-Mumbai", "2 years",
    [exp("Data Analyst", "Tata Consultancy Services", "Jan 2023", "Present", 2.0, is_ongoing=True)],
    [edu("Bachelor of Science (Statistics)", "Mumbai University", "2022")],
    ["SQL", "Power BI", "Excel", "Python"],
))

# 7. Employment gap of EXACTLY 6 months -- boundary-value test for
# gte/lte 6 queries (off-by-one risk).
CANDIDATES.append(resume(
    "proc_synth_007", "Sanjay Kulkarni", "Pune", "7 years",
    [exp("DevOps Lead", "Wipro", "Jan 2020", "Present", 4.5, is_ongoing=True),
     exp("Systems Engineer", "Persistent Systems", "Jan 2017", "Jun 2019", 2.5)],
    [edu("Bachelor of Engineering (Computer Engineering)", "Pune University", "2016")],
    ["Docker", "Jenkins", "AWS", "Linux"],
    gaps=[gap(6, "Persistent Systems", "Systems Engineer", "Jun 2019", "Wipro", "DevOps Lead", "Jan 2020")],
))

# 8. Zero employment gaps -- confirms employment_gap_months defaults to 0,
# not missing/None (would wrongly fail a "gap <= N" query otherwise).
CANDIDATES.append(resume(
    "proc_synth_008", "Meera Pillai", "Chennai", "5 years",
    [exp("Full Stack Developer", "Zoho", "Jun 2019", "Present", 5.0, is_ongoing=True)],
    [edu("Bachelor of Technology (Information Technology)", "Anna University", "2019")],
    ["React", "Node.js", "MongoDB", "Express"],
))

# 9. Job-hopper with MANY companies (mirrors real "Allan R Davis" pattern) --
# includes a company name with trailing noise ("Infosys BPM Division,
# Bangalore") to stress the prefix-fallback company-tier matcher.
CANDIDATES.append(resume(
    "proc_synth_009", "Vikram Nair", "Hyderabad", "12 years",
    [exp("Engineering Manager", "Infosys BPM Division, Bangalore", "Jan 2023", "Present", 1.5, is_ongoing=True),
     exp("Principal Engineer", "Cognizant Technology Solutions India", "Jun 2020", "Dec 2022", 2.5),
     exp("Senior Engineer", "Capgemini", "Mar 2018", "May 2020", 2.2),
     exp("Software Engineer", "Mindtree", "Jan 2016", "Feb 2018", 2.1),
     exp("Associate Engineer", "Mphasis", "Jul 2013", "Dec 2015", 2.5)],
    [edu("Bachelor of Engineering (Computer Science)", "Osmania University", "2013")],
    ["Java", "Microservices", "Spring", "Docker"],
))

# 10. Garbled/generic education -- university name too vague to resolve to
# any real tier ("Engineering College" alone, no name/city).
CANDIDATES.append(resume(
    "proc_synth_010", "Ananya Desai", "Ahmedabad", "3 years",
    [exp("QA Engineer", "L&T Infotech", "Jun 2021", "Present", 3.0, is_ongoing=True)],
    [edu("B.Tech (Hons.)", "Engineering College", "2021")],
    ["Selenium", "Java", "TestNG", "JIRA"],
))

# 11. Multiple degrees, PhD should win over Bachelor's (max-rank check).
CANDIDATES.append(resume(
    "proc_synth_011", "Karthik Subramaniam", "Chennai", "10 years",
    [exp("Research Scientist", "Tata Consultancy Services", "Jan 2018", "Present", 7.0, is_ongoing=True)],
    [edu("Bachelor of Science (Physics)", "University of Madras", "2011"),
     edu("Doctor of Philosophy (Computer Science)", "Indian Institute of Science", "2017")],
    ["Python", "Research", "Machine Learning", "Statistics"],
))

# 12. Inconsistent casing/spacing variants of the SAME tool as separate list
# entries -- must not double-count oddly, and a "React" query must still
# match via case-insensitive exact-equals per item.
CANDIDATES.append(resume(
    "proc_synth_012", "Fatima Khan", "Hyderabad", "4 years",
    [exp("Frontend Developer", "Myntra", "Jun 2021", "Present", 4.0, is_ongoing=True)],
    [edu("Bachelor of Technology (Computer Science)", "JNTU Hyderabad", "2020")],
    ["react.js", "ReactJS", "React JS", "JavaScript", "CSS3"],
))

# 13. Certification stored as one messy paragraph-blob mashing several certs
# together (mirrors the real Splunk-blob pattern) -- must still match via
# substring "contains", not require an exact atomic token.
CANDIDATES.append(resume(
    "proc_synth_013", "David Chen", "Bengaluru", "9 years",
    [exp("Data Engineer", "Oracle", "Jan 2016", "Present", 9.0, is_ongoing=True)],
    [edu("Bachelor of Engineering (Computer Science)", "PES University", "2015")],
    ["Spark", "Hadoop", "AWS", "Python"],
    certifications=[cert("Certifications: AWS Certified Data Analytics - Specialty, "
                         "Databricks Certified Data Engineer Associate, and "
                         "Cloudera Certified Administrator for Apache Hadoop")],
))

# 14. Company name with special punctuation/parenthetical abbreviation --
# stresses _normalize_company's regex handling.
CANDIDATES.append(resume(
    "proc_synth_014", "Neha Agarwal", "Gurgaon", "6 years",
    [exp("Senior Consultant", "Ernst & Young (EY)", "Jul 2018", "Present", 6.0, is_ongoing=True)],
    [edu("Master of Business Administration (Finance)", "Indian Institute of Management Ahmedabad", "2018")],
    ["Financial Modeling", "Excel", "SQL", "Tableau"],
))

# 15. Freelancer/self-employed -- company field isn't a real company name at
# all; company_tier lookup must gracefully return None, not crash.
CANDIDATES.append(resume(
    "proc_synth_015", "Rahul Mehta", "Jaipur", "5 years",
    [exp("Independent Consultant", "Self-employed / Freelance", "Jan 2021", "Present", 4.0, is_ongoing=True),
     exp("Software Engineer", "Startup (Confidential)", "Jun 2019", "Dec 2020", 1.5)],
    [edu("Bachelor of Computer Applications", "Rajasthan University", "2019")],
    ["WordPress", "PHP", "MySQL"],
))

# 16. No location at all anywhere (personal_location empty, no experience
# location either) -- must resolve to None without crashing, still show up
# for non-location queries.
CANDIDATES.append(resume(
    "proc_synth_016", "Sameer Khan", "", "3 years",
    [exp("Backend Developer", "Paytm", "Jan 2022", "Present", 3.0, is_ongoing=True, location="")],
    [edu("Bachelor of Technology (Computer Science)", "Aligarh Muslim University", "2021")],
    ["Node.js", "Express", "MongoDB"],
))

# 17. International location outside India/US -- gazetteer coverage check
# for a non-Indian, non-US city.
CANDIDATES.append(resume(
    "proc_synth_017", "Hana Tanaka", "Tokyo, Japan", "7 years",
    [exp("Software Architect", "Rakuten", "Apr 2017", "Present", 7.0, is_ongoing=True)],
    [edu("Master of Engineering (Computer Science)", "University of Tokyo", "2016")],
    ["Java", "Kubernetes", "Kafka", "AWS"],
))

# 18. Skill substring trap: bare "C" as a token, distinct from "C++"/"C#" --
# exact-match-per-item must not confuse them (mirrors java/javascript case).
CANDIDATES.append(resume(
    "proc_synth_018", "Aditya Rao", "Chennai", "8 years",
    [exp("Embedded Systems Engineer", "Bosch", "Jun 2016", "Present", 8.0, is_ongoing=True)],
    [edu("Bachelor of Engineering (Electronics and Communication)", "Anna University", "2015")],
    ["C", "Embedded C", "RTOS", "Microcontrollers"],
))

# 19. Rapid job-hopper: many SHORT tenures in a few years.
CANDIDATES.append(resume(
    "proc_synth_019", "Sneha Kapoor", "Noida", "4 years",
    [exp("Product Manager", "Paytm", "Jan 2024", "Present", 0.7, is_ongoing=True),
     exp("Associate PM", "Ola", "Jun 2023", "Dec 2023", 0.5),
     exp("Business Analyst", "Swiggy", "Jan 2022", "May 2023", 1.4),
     exp("Analyst", "Deloitte", "Jul 2020", "Dec 2021", 1.4)],
    [edu("Bachelor of Business Administration", "Delhi University", "2020")],
    ["Product Management", "SQL", "Analytics", "JIRA"],
))

# 20. GPA present but in the known-messy real-world format (mixed scale) --
# must load without crashing even though GPA isn't a filterable field.
CANDIDATES.append(resume(
    "proc_synth_020", "Tara Singh", "Kolkata", "2 years",
    [exp("Junior Developer", "Cognizant", "Jul 2022", "Present", 2.0, is_ongoing=True)],
    [edu("Bachelor of Technology (Computer Science)", "Jadavpur University", "2022", gpa="CGPA:8.9/10.0")],
    ["Python", "Django", "PostgreSQL"],
))

# 21. Fresher: zero experience, empty experience list entirely.
CANDIDATES.append(resume(
    "proc_synth_021", "Ishaan Bhatt", "Mumbai", "0 years",
    [],
    [edu("Bachelor of Engineering (Information Technology)", "University of Mumbai", "2024")],
    ["Python", "Java", "Data Structures"],
))

# 22. Senior "everything checks out" positive control: high experience, PhD,
# elite college, high-tier companies, broad relevant skills.
CANDIDATES.append(resume(
    "proc_synth_022", "Rajesh Iyer", "Bengaluru", "22 years",
    [exp("VP of Engineering", "Microsoft", "Jan 2015", "Present", 11.0, is_ongoing=True),
     exp("Principal Engineer", "Amazon", "Jun 2005", "Dec 2014", 9.5)],
    [edu("Bachelor of Technology (Computer Science)", "Indian Institute of Technology Bombay", "2003"),
     edu("Doctor of Philosophy (Computer Science)", "Stanford University", "2005")],
    ["Python", "Distributed Systems", "AWS", "Kubernetes", "Leadership", "Machine Learning"],
    certifications=[cert("AWS Certified Solutions Architect - Professional")],
))

# 23. Negative control: junior, low-tier college and company, minimal/narrow
# skill set -- should fail most "impressive candidate" queries.
CANDIDATES.append(resume(
    "proc_synth_023", "Priyanka Yadav", "Lucknow", "1 year",
    [exp("Trainee Developer", "Local IT Solutions Pvt Ltd", "Jun 2024", "Present", 1.0, is_ongoing=True)],
    [edu("Bachelor of Computer Applications", "Lucknow University", "2024")],
    ["HTML", "CSS", "Basic JavaScript"],
))

# 24. Second boundary-value gap case: exactly 12 months (1 year) -- pairs
# with #7's 6-month gap to test different thresholds independently.
CANDIDATES.append(resume(
    "proc_synth_024", "Farhan Sheikh", "Kochi", "9 years",
    [exp("Senior Backend Engineer", "Amazon", "Jan 2021", "Present", 4.0, is_ongoing=True),
     exp("Software Engineer", "Flipkart", "Jan 2016", "Jan 2020", 4.0)],
    [edu("Bachelor of Technology (Computer Science)", "National Institute of Technology Calicut", "2015")],
    ["Java", "AWS", "DynamoDB", "Microservices"],
    gaps=[gap(12, "Flipkart", "Software Engineer", "Jan 2020", "Amazon", "Senior Backend Engineer", "Jan 2021")],
))

# 25. Skills stored under the OTHER real key ("skills" not
# "skillsNormalized") -- confirms the `skillsNormalized or skills` fallback.
CANDIDATES.append(resume(
    "proc_synth_025", "Divya Menon", "Thiruvananthapuram", "5 years",
    [exp("Mobile App Developer", "PayPal", "Jun 2019", "Present", 5.0, is_ongoing=True)],
    [edu("Bachelor of Technology (Computer Science)", "College of Engineering Trivandrum", "2019")],
    ["Swift", "Kotlin", "Flutter", "Firebase"],
    skills_key="skills",
))

# --------------------------------------------------------------------------- #
# Match records: everyone gets matched to JOB_BACKEND (a general-purpose
# opening most of these profiles are plausible candidates for). A subset
# with genuinely ML/data-relevant backgrounds ALSO gets matched to JOB_ML,
# with an independent (different) score -- this is what makes job-scoping
# testable: someone visible under one job but not the other.
# --------------------------------------------------------------------------- #
ML_RELEVANT = {"proc_synth_001", "proc_synth_006", "proc_synth_011",
               "proc_synth_013", "proc_synth_020", "proc_synth_022"}

# A couple of "failed" matches (status != "completed") -- must be excluded
# from results entirely, same as the real data's status filtering.
FAILED_BACKEND = {"proc_synth_023"}

MATCHES = []
for i, r in enumerate(CANDIDATES):
    pid = r["processId"]
    status = "failed" if pid in FAILED_BACKEND else "completed"
    MATCHES.append({
        "processId": pid, "jdId": {"$oid": JOB_BACKEND["jdId"]},
        "jobId": JOB_BACKEND["jobId"], "status": status,
        "rankingScore": 95 - i * 2, "fileName": r["personalInfo"]["name"],
    })
    if pid in ML_RELEVANT:
        MATCHES.append({
            "processId": pid, "jdId": {"$oid": JOB_ML["jdId"]},
            "jobId": JOB_ML["jobId"], "status": "completed",
            "rankingScore": 90 - i, "fileName": r["personalInfo"]["name"],
        })


# --------------------------------------------------------------------------- #
# Reference datasets (Location.json / master_universities.csv /
# company_ranks.json equivalents), scoped to exactly the real-world places,
# universities, and companies the 25 candidates above reference. Without
# these, the test suite silently depended on the real, gitignored versions
# being present on whoever's machine ran it -- confirmed as a real bug: on a
# clean checkout with no real data, city normalization falls back to an
# empty gazetteer and the hyphenated-city test fails ("Navi-Mumbai" never
# resolves to "Navi Mumbai"). "Engineering College" (candidate #10, Ananya
# Desai) is DELIBERATELY left out of the university tiers below -- that
# candidate exists specifically to test the college_tier=None path for an
# unresolvably generic university name.
# --------------------------------------------------------------------------- #
_LOCATIONS = [
    ("Mumbai", "Maharashtra", "India"), ("Navi Mumbai", "Maharashtra", "India"),
    ("Bombay", "Maharashtra", "India"),  # historical name, own gazetteer row -- mirrors the real quirk that motivated the alias map
    ("Bengaluru", "Karnataka", "India"), ("Bangalore", "Karnataka", "India"),
    ("Delhi", "Delhi", "India"), ("Pune", "Maharashtra", "India"),
    ("Chennai", "Tamil Nadu", "India"), ("Hyderabad", "Telangana", "India"),
    ("Ahmedabad", "Gujarat", "India"), ("Gurugram", "Haryana", "India"),
    ("Gurgaon", "Haryana", "India"), ("Jaipur", "Rajasthan", "India"),
    ("Kolkata", "West Bengal", "India"), ("Noida", "Uttar Pradesh", "India"),
    ("Kochi", "Kerala", "India"), ("Thiruvananthapuram", "Kerala", "India"),
    ("Lucknow", "Uttar Pradesh", "India"), ("Tokyo", "Tokyo", "Japan"),
]

_UNIVERSITY_TIERS = {
    "Indian Institute of Technology Bombay": "High",
    "Indian Institute of Science": "High",
    "Indian Institute of Management Ahmedabad": "High",
    "Stanford University": "High",
    "Delhi Technological University": "Medium",
    "Anna University": "Medium",
    "Osmania University": "Medium",
    "PES University": "Medium",
    "JNTU Hyderabad": "Medium",
    "National Institute of Technology Calicut": "Medium",
    "University of Madras": "Medium",
    "R.V. College of Engineering": "Medium",
    "Mumbai University": "Low",
    "University of Mumbai": "Low",
    "Pune University": "Low",
    "Rajasthan University": "Low",
    "Aligarh Muslim University": "Low",
    "Delhi University": "Low",
    "Jadavpur University": "Low",
    "Lucknow University": "Low",
    "College of Engineering Trivandrum": "Low",
    "University of Tokyo": "Low",
    # "Engineering College" deliberately absent
}

_COMPANY_TIERS = {
    "Google": "HIGH", "Microsoft": "HIGH", "Amazon": "HIGH", "Oracle": "HIGH",
    "PayPal": "HIGH", "Infosys": "HIGH", "Tata Consultancy Services": "HIGH",
    "Wipro": "MEDIUM", "Cognizant Technology Solutions India": "MEDIUM",
    "Cognizant": "MEDIUM", "Capgemini": "MEDIUM", "Mindtree": "MEDIUM",
    "Mphasis": "MEDIUM", "Persistent Systems": "MEDIUM", "Zoho": "MEDIUM",
    "Deloitte": "MEDIUM", "Rakuten": "MEDIUM", "Bosch": "MEDIUM",
    "Myntra": "MEDIUM", "Flipkart": "MEDIUM", "Swiggy": "MEDIUM",
    "Zomato": "MEDIUM", "Paytm": "MEDIUM", "Ola": "MEDIUM",
    "Ernst Young": "MEDIUM", "L T Infotech": "MEDIUM",
    "Local IT Solutions": "LOW",
    # Deliberate junk rows -- a real, confirmed false positive: a scraped
    # company database contains its OWN placeholder rows for these exact
    # non-company phrases. Included here so the placeholder guard
    # (candidates.py's _is_company_placeholder) is PROVEN to matter -- without
    # it, "Self-employed / Freelance" and "Startup (Confidential)" would
    # match these junk rows and get a fake tier, exactly as happened for
    # real before the fix.
    "self employed": "LOW", "startup": "MEDIUM",
}


def _write_location_json(path: Path) -> None:
    lines = [
        json.dumps({"_id": {"$oid": f"synth{i:018x}"}, "name": name,
                    "state_name": state, "country_name": country})
        for i, (name, state, country) in enumerate(_LOCATIONS)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_university_csv(path: Path) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["institution_name", "location", "Tier"])
        for name, tier in _UNIVERSITY_TIERS.items():
            writer.writerow([name, "India", tier])


def _write_company_ranks_json(path: Path) -> None:
    lines = [
        json.dumps({"_id": {"$oid": f"synth{i:018x}"}, "name": name,
                    "industry": "various", "company_tier": tier})
        for i, (name, tier) in enumerate(_COMPANY_TIERS.items())
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    resumes_path = OUT_DIR / "synthetic_parsedresumes.json"
    matches_path = OUT_DIR / "synthetic_jdmatchresults.json"
    resumes_path.write_text(json.dumps(CANDIDATES, indent=2, ensure_ascii=False), encoding="utf-8")
    matches_path.write_text(json.dumps(MATCHES, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(CANDIDATES)} candidates -> {resumes_path}")

    location_path = OUT_DIR / "synthetic_location.json"
    university_path = OUT_DIR / "synthetic_master_universities.csv"
    company_path = OUT_DIR / "synthetic_company_ranks.json"
    _write_location_json(location_path)
    _write_university_csv(university_path)
    _write_company_ranks_json(company_path)
    print(f"wrote {len(_LOCATIONS)} locations -> {location_path}")
    print(f"wrote {len(_UNIVERSITY_TIERS)} universities -> {university_path}")
    print(f"wrote {len(_COMPANY_TIERS)} companies -> {company_path}")
    print(f"wrote {len(MATCHES)} match records -> {matches_path}")


if __name__ == "__main__":
    main()
