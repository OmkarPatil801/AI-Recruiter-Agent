"""
explainer.py
------------
Deterministic explanation generator for recruiter-facing output.

Takes scored candidate dicts produced by scorer.py / rank.py and derives
structured recruiter text: strengths, weaknesses, confidence scores, decision
labels, and narrative summaries.  No LLM API calls - every field is computed
from the numeric scoring signals via rule-based logic.

Public API
----------
    assign_decision(candidate, shortlist_size) -> str
    compute_confidence(final_score) -> float
    get_strengths(candidate, required_skills) -> list[str]
    get_weaknesses(candidate, required_skills) -> list[str]
    build_candidate_explanation(candidate, shortlist_size, required_skills) -> dict
    build_summary(all_explanations, required_skills) -> dict

Decision rules
--------------
    Recommended     - passed filter AND rank <= shortlist_size
    Requires Review - passed filter but rank > shortlist_size,
                      OR failed filter with skill_match in [0.10, 0.15)
                      and a technical career category (not NON_TECH)
    Not Recommended - failed filter with skill_match < 0.10,
                      OR non-technical domain with insufficient skills

Pass/fail is inferred from candidate.get("rank"):
    not None  → passed the filter gate (rank is the filtered position)
    None      → rejected by the filter gate (filter_reason key is set)
"""

from __future__ import annotations

import statistics
from collections import Counter


# ---------------------------------------------------------------------------
# Decision assignment
# ---------------------------------------------------------------------------

def assign_decision(candidate: dict, shortlist_size: int) -> str:
    """
    Assign one of three mutually exclusive recruiter decisions.

    Parameters
    ----------
    candidate      : scored + (optionally) ranked candidate dict from rank.py
    shortlist_size : number of candidates the recruiter requested in top-list

    Returns
    -------
    "Recommended" | "Requires Review" | "Not Recommended"
    """
    rank = candidate.get("rank")            # set on passing candidates only
    skl  = candidate.get("skill_match_score", 0.0)
    cat  = candidate.get("candidate_category", "NON_TECH")

    if rank is not None:
        # Candidate passed the filter gate
        return "Recommended" if rank <= shortlist_size else "Requires Review"

    # Candidate failed the filter gate - borderline vs clear failure
    if skl >= 0.10 and cat != "NON_TECH":
        return "Requires Review"
    return "Not Recommended"


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------

def compute_confidence(final_score: float) -> float:
    """
    Map final_score [0, 1] to a recruiter-readable confidence percentage [0, 100].

    Calibrated against observed v4 score distributions:
        >= 0.60  →  90–100%  (strong match)
        0.45–0.60 →  70–89%  (good match)
        0.35–0.45 →  50–69%  (moderate)
        0.25–0.35 →  30–49%  (weak)
        <  0.25  →   0–29%  (poor)
    """
    if final_score >= 0.60:
        pct = 90.0 + (final_score - 0.60) / 0.40 * 10.0
    elif final_score >= 0.45:
        pct = 70.0 + (final_score - 0.45) / 0.15 * 20.0
    elif final_score >= 0.35:
        pct = 50.0 + (final_score - 0.35) / 0.10 * 20.0
    elif final_score >= 0.25:
        pct = 30.0 + (final_score - 0.25) / 0.10 * 20.0
    else:
        pct = max(0.0, final_score / 0.25 * 30.0)
    return round(min(pct, 100.0), 1)


# ---------------------------------------------------------------------------
# Strengths
# ---------------------------------------------------------------------------

def get_strengths(candidate: dict, required_skills: set[str]) -> list[str]:
    """
    Rule-based strength detection from the six v4 scoring signals.

    Returns a list of recruiter-readable strength statements ordered from
    most important (domain identity) to least (engagement signals).
    """
    strengths: list[str] = []

    sem     = candidate.get("cosine_score",         0.0)
    skl     = candidate.get("skill_match_score",    0.0)
    rol     = candidate.get("role_relevance_score", 0.0)
    dlv     = candidate.get("delivery_score",       0.0)
    exp     = candidate.get("experience_score",     0.0)
    beh     = candidate.get("behavioural_score",    0.0)
    cat     = candidate.get("candidate_category",   "")
    matched = candidate.get("required_skills_found", [])
    yoe     = candidate.get("years_of_experience",  0)
    total   = len(required_skills)

    # Domain identity - the strongest hiring signal
    if cat == "CORE_ML":
        strengths.append(
            "Core ML/AI practitioner - current role is directly in the target domain"
        )
    elif cat == "ML_ADJ":
        strengths.append(
            "ML-adjacent background (Data Engineering / MLOps / AI Platform)"
        )

    # Production delivery
    if dlv >= 0.80:
        strengths.append("Strong evidence of shipping ML systems to production")
    elif dlv >= 0.50:
        strengths.append("Demonstrated production delivery experience in career history")
    elif dlv >= 0.20:
        strengths.append("Some production delivery signals present in work descriptions")

    # Semantic alignment
    if sem >= 0.65:
        strengths.append("Excellent semantic alignment with the role description")
    elif sem >= 0.55:
        strengths.append("Strong semantic alignment with the role requirements")
    elif sem >= 0.45:
        strengths.append("Good general alignment with the role description")

    # Skill coverage
    if total > 0 and matched:
        pct = round(skl * 100)
        n   = len(matched)
        if skl >= 0.50:
            strengths.append(
                f"High skill coverage: {n}/{total} required skills matched ({pct}%)"
            )
        elif skl >= 0.30:
            strengths.append(
                f"Solid skill coverage: {n}/{total} required skills matched ({pct}%)"
            )
        elif skl >= 0.15:
            strengths.append(
                f"Adequate skill coverage: {n}/{total} required skills ({pct}%)"
            )

    # Career trajectory
    if rol >= 0.85:
        strengths.append("Career trajectory strongly aligned with ML/AI engineering")
    elif rol >= 0.65:
        strengths.append("Career history shows meaningful ML/AI domain exposure")

    # Experience
    if exp >= 1.0:
        strengths.append(f"Meets or exceeds experience requirement ({yoe} years)")
    elif exp >= 0.90:
        strengths.append(f"Close to required experience level ({yoe} years)")

    # Availability
    if candidate.get("open_to_work"):
        strengths.append("Currently open to work - immediately available")

    # Engagement
    if beh >= 0.70:
        strengths.append("Highly engaged - strong platform activity and interview completion")
    elif beh >= 0.50:
        strengths.append("Good platform engagement signals")

    return strengths or ["No significant strengths identified for this role"]


# ---------------------------------------------------------------------------
# Weaknesses
# ---------------------------------------------------------------------------

def get_weaknesses(candidate: dict, required_skills: set[str]) -> list[str]:
    """
    Rule-based weakness detection from the six v4 scoring signals.

    Returns a list of recruiter-readable weakness statements ordered from
    most impactful (domain mismatch) to least (engagement gaps).
    """
    weaknesses: list[str] = []

    sem     = candidate.get("cosine_score",         0.0)
    skl     = candidate.get("skill_match_score",    0.0)
    rol     = candidate.get("role_relevance_score", 0.0)
    dlv     = candidate.get("delivery_score",       0.0)
    exp     = candidate.get("experience_score",     0.0)
    cat     = candidate.get("candidate_category",   "")
    matched = set(candidate.get("required_skills_found", []))
    yoe     = candidate.get("years_of_experience",  0)
    total   = len(required_skills)
    missing = sorted(required_skills - matched)

    # Domain mismatch - most critical weakness
    if cat == "NON_TECH":
        weaknesses.append(
            "Non-technical career background - no ML, engineering, or data science history"
        )
    elif cat == "ENG" and rol < 0.50:
        weaknesses.append(
            "Engineering background without clear ML/AI domain specialisation"
        )

    # Skill coverage
    if skl < 0.15:
        weaknesses.append(
            f"Insufficient skill coverage: {len(matched)}/{total} required skills "
            f"({round(skl * 100)}%) - below minimum 15% threshold"
        )
    elif skl < 0.30:
        weaknesses.append(
            f"Limited skill coverage: {len(matched)}/{total} required skills matched"
        )

    # Missing skills list (top 6)
    if missing:
        preview = missing[:6]
        suffix  = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
        weaknesses.append(f"Missing required skills: {', '.join(preview)}{suffix}")

    # No delivery evidence
    if dlv == 0.0:
        weaknesses.append(
            "No production delivery evidence - profile language suggests AI interest "
            "rather than demonstrated delivery"
        )
    elif dlv < 0.20:
        weaknesses.append("Weak production delivery signals in career descriptions")

    # Role relevance
    if rol < 0.20:
        weaknesses.append("Career trajectory shows no meaningful alignment with ML/AI roles")
    elif rol < 0.40:
        weaknesses.append("Career relevance is low - primarily non-technical history")

    # Semantic
    if sem < 0.40:
        weaknesses.append("Low semantic similarity with the job description")

    # Experience
    if exp < 0.75:
        weaknesses.append(
            f"Below minimum experience threshold - {yoe} years provided, "
            "more required for this seniority level"
        )

    return weaknesses or ["No significant weaknesses identified"]


# ---------------------------------------------------------------------------
# Reason statements  (one sentence per decision type)
# ---------------------------------------------------------------------------

def _recommendation_reason(candidate: dict, required_skills: set[str]) -> str:
    """One-sentence reason for a Recommended decision."""
    dlv     = candidate.get("delivery_score",       0.0)
    cat     = candidate.get("candidate_category",   "")
    matched = candidate.get("required_skills_found", [])
    total   = len(required_skills)

    parts: list[str] = []
    if cat == "CORE_ML":
        parts.append("core ML/AI career background")
    elif cat == "ML_ADJ":
        parts.append("ML-adjacent career background")
    if matched:
        parts.append(f"{len(matched)}/{total} required skills matched")
    if dlv >= 0.60:
        parts.append("strong production delivery evidence")
    elif dlv >= 0.20:
        parts.append("production delivery evidence")

    if parts:
        return f"Recommended based on {', '.join(parts)}."
    return "Candidate meets minimum qualification thresholds for this role."


def _rejection_reason(candidate: dict, required_skills: set[str]) -> str:
    """One-sentence reason for a Not Recommended decision."""
    skl     = candidate.get("skill_match_score",    0.0)
    exp     = candidate.get("experience_score",     0.0)
    cat     = candidate.get("candidate_category",   "")
    matched = candidate.get("required_skills_found", [])
    total   = len(required_skills)

    reasons: list[str] = []
    if skl < 0.15:
        reasons.append(
            f"insufficient skill overlap ({len(matched)}/{total} skills, "
            "minimum threshold is 15%)"
        )
    if exp < 0.75:
        reasons.append(f"below minimum experience threshold (score {round(exp, 2)})")
    if cat == "NON_TECH":
        reasons.append("non-technical career domain")

    if reasons:
        return f"Not recommended due to {' and '.join(reasons)}."
    return "Does not meet minimum qualification thresholds for this role."


def _review_reason(candidate: dict, required_skills: set[str]) -> str:
    """One-sentence reason for a Requires Review decision."""
    rank = candidate.get("rank")
    skl  = candidate.get("skill_match_score", 0.0)
    cat  = candidate.get("candidate_category", "")

    if rank is not None:
        return (
            f"Qualifies technically (filter rank #{rank}) but falls outside the "
            "requested shortlist size. Consider expanding shortlist."
        )
    if skl >= 0.10:
        total = len(required_skills)
        matched = candidate.get("required_skills_found", [])
        return (
            f"Borderline skill coverage ({len(matched)}/{total} skills, "
            f"{round(skl * 100, 1)}% - just below the 15% minimum threshold). "
            "Manual review recommended."
        )
    return (
        f"Relevant {cat} background with marginal qualification. "
        "Recruiter review advised before final decision."
    )


# ---------------------------------------------------------------------------
# Recruiter narrative summary
# ---------------------------------------------------------------------------

def generate_recruiter_summary(
    candidate: dict,
    decision: str,
    required_skills: set[str],
) -> str:
    """
    Template-based 2-3 sentence recruiter summary.

    Combines candidate identity, domain classification, skill coverage, and
    decision rationale into a professional plain-English paragraph without
    any LLM calls.
    """
    name    = candidate.get("name",            "This candidate")
    title   = candidate.get("current_title",   "unknown role")
    company = candidate.get("current_company", "")
    yoe     = candidate.get("years_of_experience", 0)
    cat     = candidate.get("candidate_category",  "")
    matched = candidate.get("required_skills_found", [])
    final   = candidate.get("final_score", 0.0)
    dlv     = candidate.get("delivery_score", 0.0)
    total   = len(required_skills)

    at_company = f" at {company}" if company else ""
    s1 = f"{name} is currently a {title}{at_company} with {yoe} years of experience."

    cat_phrases = {
        "CORE_ML":  "a core ML/AI practitioner",
        "ML_ADJ":   "an ML-adjacent engineer (Data/MLOps/Platform)",
        "ENG":      "a software engineering professional",
        "NON_TECH": "a non-technical professional",
    }
    cat_phrase = cat_phrases.get(cat, "a professional")

    if matched:
        preview = ", ".join(matched[:4])
        extra   = f" and {len(matched) - 4} more" if len(matched) > 4 else ""
        s2 = (
            f"Identified as {cat_phrase}, covering {len(matched)}/{total} required "
            f"skills ({preview}{extra})."
        )
    else:
        s2 = f"Identified as {cat_phrase} with no overlap with the required skill set."

    if decision == "Recommended":
        if dlv >= 0.60:
            s3 = (
                "Strong production delivery evidence in their career history supports "
                f"this recommendation (overall match: {round(final * 100, 1)}%)."
            )
        else:
            s3 = (
                f"Meets qualification thresholds and ranks within the requested shortlist "
                f"(overall match: {round(final * 100, 1)}%)."
            )
    elif decision == "Requires Review":
        s3 = "Borderline qualification - manual recruiter review is advised before a final decision."
    else:
        s3 = (
            f"Does not meet minimum qualification thresholds for this role "
            f"(overall match: {round(final * 100, 1)}%)."
        )

    return f"{s1} {s2} {s3}"


# ---------------------------------------------------------------------------
# Full candidate explanation builder
# ---------------------------------------------------------------------------

def build_candidate_explanation(
    candidate: dict,
    shortlist_size: int,
    required_skills: set[str],
) -> dict:
    """
    Build a complete recruiter-facing explanation dict for one candidate.

    All fields are derived deterministically from the scoring pipeline.
    No LLM calls.

    Parameters
    ----------
    candidate       : scored dict from rank.py (filtered or rejected)
                      Passing candidates have "rank" set.
                      Rejected candidates have "filter_reason" set.
    shortlist_size  : number of slots in the recommended shortlist
    required_skills : canonical required skill names from extract_required_skills()

    Returns
    -------
    dict with keys matching the CandidateResult Pydantic model.
    """
    decision   = assign_decision(candidate, shortlist_size)
    confidence = compute_confidence(candidate.get("final_score", 0.0))
    strengths  = get_strengths(candidate, required_skills)
    weaknesses = get_weaknesses(candidate, required_skills)

    matched = candidate.get("required_skills_found", [])
    missing = sorted(required_skills - set(matched))

    rec_reason = _recommendation_reason(candidate, required_skills) if decision == "Recommended"     else None
    rej_reason = _rejection_reason(candidate, required_skills)      if decision == "Not Recommended" else None
    rev_reason = _review_reason(candidate, required_skills)         if decision == "Requires Review"  else None

    summary = generate_recruiter_summary(candidate, decision, required_skills)

    return {
        "candidate_id":               candidate.get("candidate_id", ""),
        "name":                       candidate.get("name", ""),
        "current_title":              candidate.get("current_title", ""),
        "current_company":            candidate.get("current_company", ""),
        "location":                   candidate.get("location", ""),
        "country":                    candidate.get("country", ""),
        "years_of_experience":        float(candidate.get("years_of_experience", 0)),
        "candidate_category":         candidate.get("candidate_category", ""),
        "decision":                   decision,
        "confidence_score":           confidence,
        "matched_skills":             matched,
        "missing_skills":             missing,
        "strengths":                  strengths,
        "weaknesses":                 weaknesses,
        "reason_for_recommendation":  rec_reason,
        "reason_for_rejection":       rej_reason,
        "reason_for_review":          rev_reason,
        "recruiter_summary":          summary,
        "scores": {
            "semantic":          candidate.get("cosine_score",         0.0),
            "skill_match":       candidate.get("skill_match_score",    0.0),
            "role_relevance":    candidate.get("role_relevance_score", 0.0),
            "delivery_evidence": candidate.get("delivery_score",       0.0),
            "experience":        candidate.get("experience_score",     0.0),
            "behavioural":       candidate.get("behavioural_score",    0.0),
            "final":             candidate.get("final_score",          0.0),
        },
        "rank":          candidate.get("rank"),
        "filter_reason": candidate.get("filter_reason"),
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def build_summary(
    all_explanations: list[dict],
    required_skills: set[str],
) -> dict:
    """
    Compute aggregate statistics across all candidate explanations.

    Parameters
    ----------
    all_explanations : list of dicts from build_candidate_explanation()
    required_skills  : canonical required skills from the JD

    Returns
    -------
    dict matching the RankingSummary Pydantic model schema.
    """
    recommended = [e for e in all_explanations if e["decision"] == "Recommended"]
    review      = [e for e in all_explanations if e["decision"] == "Requires Review"]
    not_rec     = [e for e in all_explanations if e["decision"] == "Not Recommended"]

    final_scores = [e["scores"]["final"] for e in all_explanations]
    yoe_values   = [e["years_of_experience"] for e in all_explanations]

    matched_counter: Counter = Counter()
    for e in all_explanations:
        for s in e["matched_skills"]:
            matched_counter[s] += 1

    missing_counter: Counter = Counter()
    for e in all_explanations:
        for s in e["missing_skills"]:
            missing_counter[s] += 1

    return {
        "total_candidates":           len(all_explanations),
        "recommended_count":          len(recommended),
        "requires_review_count":      len(review),
        "not_recommended_count":      len(not_rec),
        "average_match_score":        round(statistics.mean(final_scores) if final_scores else 0.0, 4),
        "average_experience_years":   round(statistics.mean(yoe_values)   if yoe_values   else 0.0, 1),
        "most_common_matched_skills": [s for s, _ in matched_counter.most_common(8)],
        "most_common_missing_skills": [s for s, _ in missing_counter.most_common(8)],
        "required_skills_count":      len(required_skills),
    }
