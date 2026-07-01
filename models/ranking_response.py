# pyrefly: ignore [missing-import]
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    """Component-level scores for one candidate against one job description."""
    semantic:          float
    skill_match:       float
    role_relevance:    float
    delivery_evidence: float
    experience:        float
    behavioural:       float
    final:             float

    model_config = {"from_attributes": True}


class CandidateResult(BaseModel):
    """
    Full recruiter-facing record for one candidate.

    decision       — "Recommended" | "Requires Review" | "Not Recommended"
    confidence_score — 0–100 map of final_score onto recruiter confidence
    matched_skills — required skills this candidate covers
    missing_skills — required skills this candidate lacks
    strengths      — rule-based plain-English strength statements
    weaknesses     — rule-based plain-English weakness statements
    reason_for_*   — one-sentence rationale (only the relevant one is non-null)
    recruiter_summary — 2-3 sentence narrative for the dashboard card
    scores         — full component breakdown
    rank           — filtered rank (null if candidate failed filter)
    filter_reason  — why the candidate was rejected (null if passed)
    """
    candidate_id:  str
    name:          str
    current_title: str
    current_company: str
    location:      str
    country:       str
    years_of_experience: float
    candidate_category:  str

    decision:        str
    confidence_score: float

    matched_skills: list[str]
    missing_skills: list[str]
    strengths:      list[str]
    weaknesses:     list[str]

    reason_for_recommendation: Optional[str] = None
    reason_for_rejection:      Optional[str] = None
    reason_for_review:         Optional[str] = None

    recruiter_summary: str
    scores:            ScoreBreakdown

    rank:          Optional[int]   = None
    filter_reason: Optional[str]   = None

    model_config = {"from_attributes": True}


class RankingSummary(BaseModel):
    """Aggregate statistics across all candidates for the dashboard header."""
    total_candidates:        int
    recommended_count:       int
    requires_review_count:   int
    not_recommended_count:   int
    average_match_score:     float
    average_experience_years: float
    most_common_matched_skills: list[str]
    most_common_missing_skills: list[str]
    required_skills_count:   int

    model_config = {"from_attributes": True}


class RankingResponse(BaseModel):
    """
    Top-level response from POST /rank/.

    summary          — aggregate stats for the dashboard header
    recommended      — candidates in the shortlist, sorted by rank
    requires_review  — borderline candidates, sorted by final_score desc
    not_recommended  — rejected candidates, sorted by final_score desc
    """
    summary:         RankingSummary
    recommended:     list[CandidateResult]
    requires_review: list[CandidateResult]
    not_recommended: list[CandidateResult]

    model_config = {"from_attributes": True}


class RankRequest(BaseModel):
    """Request body for POST /rank/."""
    job_description: str = Field(
        ...,
        min_length=20,
        description="Full text of the job posting",
    )
    shortlist_size: int = Field(
        default=10,
        ge=1,
        le=200,
        description="How many candidates to include in the Recommended bucket (top-N)",
    )
    job_id: Optional[int] = Field(
        default=None,
        description="Optional jobs.id to link this run to a job record in the database",
    )
