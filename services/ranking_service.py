"""
services/ranking_service.py
---------------------------
Orchestration layer for the v4 ranking pipeline.

This module is the single entry point between the API layer (routers) and the
ranking engine (ranking/).  Routers call run_ranking(); everything else is
internal.

Responsibilities
----------------
1. Run the v4 scoring + filter pipeline via run_evaluation()
2. Extract required skills from the job description
3. Generate deterministic recruiter explanations via build_candidate_explanation()
4. Persist run metadata (ranking_runs) and per-candidate scores (ranking_results)
5. Assemble and return a RankingResponse

Persistence schema
------------------
    ranking_runs    — one row per /rank/ call (summary stats + job linkage)
    ranking_results — one row per candidate per run (full score breakdown)

Both tables are created by db/init.sql on startup.
"""

from __future__ import annotations

import sqlite3

from database import get_connection
from models.ranking_response import (
    CandidateResult,
    RankingResponse,
    RankingSummary,
    ScoreBreakdown,
)
from ranking.explainer import build_candidate_explanation, build_summary
from ranking.rank import run_evaluation
from ranking.scorer import extract_required_skills


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

# Large text fields used only during scoring.  Stripped from scored dicts
# before explanation building to free ~400 MB on 100K-candidate runs.
_LARGE_FIELDS = frozenset({"candidate_text", "career_text", "summary", "career_titles"})


def _strip_large_fields(candidates: list[dict]) -> None:
    """Remove large text fields in-place from a list of scored candidate dicts."""
    for c in candidates:
        for field in _LARGE_FIELDS:
            c.pop(field, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ranking(
    job_description: str,
    shortlist_size: int,
    job_id: int | None = None,
) -> RankingResponse:
    """
    Run the full v4 ranking pipeline, persist results, and return the response.

    Parameters
    ----------
    job_description : Full text of the job posting.
    shortlist_size  : Number of candidates to place in the Recommended bucket.
    job_id          : Optional jobs.id to link this run to a database job record.

    Returns
    -------
    RankingResponse — three decision buckets + summary stats.

    Response size policy (100K-candidate scale):
        recommended     : up to shortlist_size candidates
        requires_review : up to shortlist_size * 2 candidates (top by score)
        not_recommended : empty list — count is in summary.not_recommended_count
    """
    # --- 1. Score all candidates and apply the filter gate ---
    # top_n=9999 ensures filtered_top_n contains ALL passing candidates.
    data = run_evaluation(job_description=job_description, top_n=9999)

    # --- 2. Strip large text fields no longer needed (candidate_text, career_text,
    #        summary, career_titles).  Scored dicts are already copies — safe to mutate.
    #        For 100K candidates this frees ~400 MB before explanation building begins.
    _strip_large_fields(data["filtered_top_n"])
    _strip_large_fields(data["rejected_candidates"])

    # --- 3. Extract required skills (pure text scan, no model load) ---
    required_skills: set[str] = extract_required_skills(job_description)

    # --- 4. Unified candidate list: passing (rank set) + rejected (filter_reason set) ---
    all_candidates: list[dict] = data["filtered_top_n"] + data["rejected_candidates"]

    # --- 5. Generate deterministic recruiter explanations for every candidate ---
    all_explanations: list[dict] = [
        build_candidate_explanation(
            candidate=c,
            shortlist_size=shortlist_size,
            required_skills=required_skills,
        )
        for c in all_candidates
    ]

    # --- 6. Aggregate summary statistics (computed over ALL candidates) ---
    summary_data = build_summary(all_explanations, required_skills)

    # --- 7. Persist run metadata + individual results in a single transaction ---
    _persist(job_id, summary_data, all_explanations)

    # --- 8. Split into three decision buckets, capped for API response size ---
    recommended = sorted(
        [e for e in all_explanations if e["decision"] == "Recommended"],
        key=lambda x: (x["rank"] or 9999),
    )
    requires_review = sorted(
        [e for e in all_explanations if e["decision"] == "Requires Review"],
        key=lambda x: x["scores"]["final"],
        reverse=True,
    )
    # not_recommended can be 99K+ records at scale — omit from response body.
    # The count is in summary.not_recommended_count.
    review_cap = max(shortlist_size * 2, 20)

    return RankingResponse(
        summary=RankingSummary(**summary_data),
        recommended=[_to_result(e) for e in recommended],
        requires_review=[_to_result(e) for e in requires_review[:review_cap]],
        not_recommended=[],
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_result(explanation: dict) -> CandidateResult:
    """Convert a flat explanation dict to a CandidateResult Pydantic model."""
    scores = ScoreBreakdown(**explanation["scores"])
    fields = {k: v for k, v in explanation.items() if k != "scores"}
    return CandidateResult(**fields, scores=scores)


def _persist(
    job_id: int | None,
    summary_data: dict,
    all_explanations: list[dict],
) -> int:
    """
    Write one ranking_runs row and N ranking_results rows inside a transaction.

    Returns the auto-assigned run_id so callers can reference this run later.
    """
    with get_connection() as conn:
        run_id = _insert_run(conn, job_id, summary_data)
        _insert_results(conn, run_id, all_explanations)
    return run_id


def _insert_run(
    conn: sqlite3.Connection,
    job_id: int | None,
    summary: dict,
) -> int:
    """Insert one row into ranking_runs and return its new id.

    If job_id references a non-existent jobs row (FK violation), the run is
    persisted without a job linkage rather than aborting the ranking response.
    """
    values = (
        job_id,
        summary["total_candidates"],
        summary["recommended_count"],
        summary["requires_review_count"],
        summary["not_recommended_count"],
        summary["average_match_score"],
    )
    sql = """
        INSERT INTO ranking_runs (
            job_id,
            total_candidates,
            recommended_count,
            requires_review_count,
            not_recommended_count,
            average_match_score
        ) VALUES (?, ?, ?, ?, ?, ?)
    """
    try:
        cursor = conn.execute(sql, values)
    except sqlite3.IntegrityError:
        # job_id does not exist in the jobs table — persist without linkage.
        cursor = conn.execute(sql, (None,) + values[1:])
    return cursor.lastrowid  # type: ignore[return-value]


def _insert_results(
    conn: sqlite3.Connection,
    run_id: int,
    all_explanations: list[dict],
) -> None:
    """Bulk-insert one ranking_results row per candidate explanation."""
    rows = [
        (
            run_id,
            e["candidate_id"],
            e.get("rank"),
            e["decision"],
            e["candidate_category"],
            e["confidence_score"],
            e["scores"]["final"],
            e["scores"]["semantic"],
            e["scores"]["skill_match"],
            e["scores"]["experience"],
            e["scores"]["behavioural"],
            e["scores"]["role_relevance"],
            e["scores"]["delivery_evidence"],
            e.get("filter_reason"),
        )
        for e in all_explanations
    ]
    conn.executemany(
        """
        INSERT INTO ranking_results (
            run_id, candidate_id, rank, decision, candidate_category,
            confidence_score, final_score, semantic_score, skill_score,
            experience_score, behavioural_score, role_relevance_score,
            delivery_score, filter_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
