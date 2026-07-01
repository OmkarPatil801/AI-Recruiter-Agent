"""
routers/ranking.py
------------------
Thin HTTP layer for the ranking feature.

All orchestration logic lives in services/ranking_service.py.
This module only handles HTTP concerns: request parsing, response serialisation.

POST /rank/
    Body  : RankRequest  (job_description, shortlist_size, optional job_id)
    Return: RankingResponse  (summary + three decision buckets)
"""

from __future__ import annotations

from fastapi import APIRouter

from models.ranking_response import RankRequest, RankingResponse
from services.ranking_service import run_ranking

router = APIRouter(prefix="/rank", tags=["Ranking"])


@router.post(
    "/",
    response_model=RankingResponse,
    summary="Rank all candidates against a job description",
    description=(
        "Runs the v4 ranking pipeline over the full candidate dataset. "
        "Returns every candidate assigned to exactly one of three buckets: "
        "Recommended (top-N shortlist), Requires Review (borderline), or "
        "Not Recommended (clear mismatch). All explanations are deterministic - "
        "no LLM calls. Results are persisted to ranking_runs and ranking_results."
    ),
)
def rank_candidates(payload: RankRequest) -> RankingResponse:
    """
    Full end-to-end recruiter ranking endpoint.

    Parameters (request body)
    -------------------------
    job_description : str         Full text of the job posting.
    shortlist_size  : int         How many candidates appear in Recommended (default 10).
    job_id          : int | null  Optional link to a jobs table record.

    Returns
    -------
    RankingResponse with summary, recommended, requires_review, not_recommended.
    Results are also written to ranking_runs and ranking_results in SQLite.
    """
    return run_ranking(
        job_description=payload.job_description,
        shortlist_size=payload.shortlist_size,
        job_id=payload.job_id,
    )
