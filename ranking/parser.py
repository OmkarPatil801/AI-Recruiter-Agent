"""
parser.py
---------
Loads the candidate JSON dataset and converts each nested record into a
flat dict ready for embedding.  The key output is `candidate_text`, a single
string that concatenates every semantically meaningful field so the embedding
model sees the full candidate profile in one pass.

Nothing in here touches embeddings or scoring — pure data extraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default: prefer the full evaluation dataset if present, fall back to sample.
_JSONL_PATH = Path(__file__).parent.parent / "data" / "candidates.jsonl"
_JSON_PATH  = Path(__file__).parent.parent / "data" / "sample_candidates.json"
DATA_PATH   = _JSONL_PATH if _JSONL_PATH.exists() else _JSON_PATH

# Process-level cache: re-use parsed candidates across ranking requests without
# re-reading 475 MB from disk or re-building candidate_text strings.
# Key: str(path)  Value: list[dict] (full parsed candidates)
_candidates_cache: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _join_skills(skills: list[dict]) -> str:
    """
    Format: "Python (advanced, 36m)", "SQL (intermediate, 24m)", ...
    Puts advanced skills first so the embedding picks up the strongest signals.
    """
    ordered = sorted(skills, key=lambda s: (
        {"advanced": 0, "intermediate": 1, "beginner": 2}.get(s.get("proficiency", ""), 3),
        -s.get("endorsements", 0),
    ))
    return ", ".join(
        f"{s['name']} ({s.get('proficiency', '?')}, {s.get('duration_months', 0)}m)"
        for s in ordered
    )


def _join_career(career_history: list[dict]) -> str:
    """Concatenate role descriptions chronologically (most-recent first)."""
    sorted_roles = sorted(career_history, key=lambda r: r.get("start_date", ""), reverse=True)
    parts = []
    for role in sorted_roles:
        header = f"{role.get('title', '')} at {role.get('company', '')}"
        desc = role.get("description", "").strip()
        if desc:
            parts.append(f"{header}: {desc}")
        else:
            parts.append(header)
    return " | ".join(parts)


def _join_education(education: list[dict]) -> str:
    parts = []
    for edu in education:
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        inst = edu.get("institution", "")
        tier = edu.get("tier", "")
        parts.append(f"{degree} {field} at {inst} ({tier})")
    return "; ".join(parts)


def _join_certifications(certifications: list[dict]) -> str:
    names = [c.get("name", "") for c in certifications if c.get("name")]
    return ", ".join(names)


def _behavioral_fields(signals: dict) -> dict:
    """Extract and normalise the redrob behavioral signals."""
    github_raw = signals.get("github_activity_score", -1)
    github_norm = max(0.0, github_raw) / 100.0 if github_raw >= 0 else 0.0

    return {
        "open_to_work": bool(signals.get("open_to_work_flag", False)),
        "profile_completeness": signals.get("profile_completeness_score", 0.0) / 100.0,
        "interview_completion_rate": float(signals.get("interview_completion_rate", 0.0)),
        "github_score_norm": github_norm,
        "notice_period_days": signals.get("notice_period_days", None),
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "preferred_work_mode": signals.get("preferred_work_mode", ""),
        "offer_acceptance_rate": float(signals.get("offer_acceptance_rate", 0.0)),
        "verified_email": bool(signals.get("verified_email", False)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_candidate_text(record: dict[str, Any]) -> str:
    """
    Produce the single string that will be embedded.

    Structure (order matters — put most-important content first so the
    transformer's attention window is best utilised):
        <headline> | <summary>
        Role: <current_title> at <current_company>
        Career: <all role descriptions>
        Skills: <skill list>
        Education: <education entries>
        Certifications: <cert list>
    """
    profile = record.get("profile", {})
    sections: list[str] = []

    headline = profile.get("headline", "").strip()
    summary = profile.get("summary", "").strip()
    if headline or summary:
        sections.append(f"{headline} | {summary}")

    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    if title:
        sections.append(f"Role: {title} at {company}")

    career_text = _join_career(record.get("career_history", []))
    if career_text:
        sections.append(f"Career: {career_text}")

    skills_text = _join_skills(record.get("skills", []))
    if skills_text:
        sections.append(f"Skills: {skills_text}")

    edu_text = _join_education(record.get("education", []))
    if edu_text:
        sections.append(f"Education: {edu_text}")

    cert_text = _join_certifications(record.get("certifications", []))
    if cert_text:
        sections.append(f"Certifications: {cert_text}")

    return "\n".join(sections)


def _load_raw(path: Path) -> list[dict]:
    """Read candidate records from .json (array) or .jsonl (one object per line)."""
    if path.suffix.lower() == ".jsonl":
        records: list[dict] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidates(path: Path = DATA_PATH) -> list[dict]:
    """
    Load and parse the candidate dataset, with a process-level in-memory cache.

    Supports both .json (array) and .jsonl (newline-delimited) formats.
    After the first call for a given path, subsequent calls return the cached
    list instantly — no disk I/O, no string reconstruction.

    Each dict contains:
        candidate_id, name, headline, summary, location, country,
        years_of_experience, current_title, current_company,
        candidate_text (embedding blob), career_titles, career_text,
        + all behavioral fields flattened in.
    """
    cache_key = str(path)
    if cache_key in _candidates_cache:
        return _candidates_cache[cache_key]

    raw: list[dict] = _load_raw(path)

    candidates: list[dict] = []
    for record in raw:
        profile = record.get("profile", {})
        signals = record.get("redrob_signals", {})

        # career_titles: most-recent first, used by the role relevance scorer
        career_history = record.get("career_history", [])
        sorted_career = sorted(career_history, key=lambda r: r.get("start_date", ""), reverse=True)
        career_titles = [r.get("title", "") for r in sorted_career if r.get("title")]

        # career_text: raw description prose only, used by the delivery evidence scorer
        career_text = " ".join(
            r.get("description", "").strip()
            for r in sorted_career
            if r.get("description", "").strip()
        )

        candidate = {
            "candidate_id": record.get("candidate_id", ""),
            "name": profile.get("anonymized_name", ""),
            "headline": profile.get("headline", ""),
            "summary": profile.get("summary", ""),
            "location": profile.get("location", ""),
            "country": profile.get("country", ""),
            "years_of_experience": profile.get("years_of_experience", 0),
            "current_title": profile.get("current_title", ""),
            "current_company": profile.get("current_company", ""),
            "current_industry": profile.get("current_industry", ""),
            "skills_raw": [s["name"] for s in record.get("skills", [])],
            "career_titles": career_titles,
            "career_text": career_text,
            "candidate_text": build_candidate_text(record),
            **_behavioral_fields(signals),
        }
        candidates.append(candidate)

    _candidates_cache[cache_key] = candidates
    return candidates
