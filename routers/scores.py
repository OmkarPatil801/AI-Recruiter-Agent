from fastapi import APIRouter, HTTPException, status
from typing import List
from database import get_connection
from models.score import ScoreCreate, ScoreResponse, RankedCandidate

router = APIRouter(prefix="/scores", tags=["Scores"])


def _row_to_dict(row) -> dict:
    return dict(row)


@router.post(
    "/",
    response_model=ScoreResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a score entry (triggers async scoring)",
)
def create_score(payload: ScoreCreate):
    with get_connection() as conn:
        job = conn.execute("SELECT id FROM jobs WHERE id = ?", (payload.job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

        candidate = conn.execute(
            "SELECT id FROM candidates WHERE id = ?", (payload.candidate_id,)
        ).fetchone()
        if not candidate:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")

        try:
            cur = conn.execute(
                "INSERT INTO scores (job_id, candidate_id, status) VALUES (?, ?, 'pending')",
                (payload.job_id, payload.candidate_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM scores WHERE id = ?", (cur.lastrowid,)).fetchone()
        except Exception as exc:
            if "UNIQUE constraint failed" in str(exc):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Score entry already exists for this job/candidate pair",
                )
            raise
    return _row_to_dict(row)


@router.get("/{score_id}", response_model=ScoreResponse, summary="Get a score by ID")
def get_score(score_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM scores WHERE id = ?", (score_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Score not found")
    return _row_to_dict(row)


@router.get(
    "/job/{job_id}",
    response_model=List[ScoreResponse],
    summary="Get all scores for a job",
)
def get_scores_for_job(job_id: int):
    with get_connection() as conn:
        job = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        rows = conn.execute(
            "SELECT * FROM scores WHERE job_id = ? ORDER BY overall_score DESC", (job_id,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


@router.get(
    "/job/{job_id}/ranked",
    response_model=List[RankedCandidate],
    summary="Get ranked candidates for a job",
)
def get_ranked_candidates(job_id: int):
    with get_connection() as conn:
        job = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

        rows = conn.execute(
            """
            SELECT s.*, c.name AS candidate_name, c.email AS candidate_email
            FROM scores s
            JOIN candidates c ON c.id = s.candidate_id
            WHERE s.job_id = ?
            ORDER BY s.overall_score DESC
            """,
            (job_id,),
        ).fetchall()

    ranked = []
    for rank, row in enumerate(rows, start=1):
        d = _row_to_dict(row)
        ranked.append(
            RankedCandidate(
                rank=rank,
                candidate_id=d["candidate_id"],
                candidate_name=d["candidate_name"],
                candidate_email=d["candidate_email"],
                overall_score=d["overall_score"],
                skills_score=d["skills_score"],
                experience_score=d["experience_score"],
                education_score=d["education_score"],
                location_score=d["location_score"],
                reasoning=d.get("reasoning"),
                status=d["status"],
            )
        )
    return ranked


@router.delete("/{score_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a score")
def delete_score(score_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM scores WHERE id = ?", (score_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Score not found")
        conn.execute("DELETE FROM scores WHERE id = ?", (score_id,))
        conn.commit()
