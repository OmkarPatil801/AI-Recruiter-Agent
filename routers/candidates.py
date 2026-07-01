from fastapi import APIRouter, HTTPException, status
from typing import List
from database import get_connection
from models.candidate import CandidateCreate, CandidateUpdate, CandidateResponse

router = APIRouter(prefix="/candidates", tags=["Candidates"])


def _row_to_dict(row) -> dict:
    return dict(row)


@router.get("/", response_model=List[CandidateResponse], summary="List all candidates")
def list_candidates():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM candidates ORDER BY created_at DESC").fetchall()
    return [_row_to_dict(r) for r in rows]


@router.get("/{candidate_id}", response_model=CandidateResponse, summary="Get a candidate by ID")
def get_candidate(candidate_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
    return _row_to_dict(row)


@router.post("/", response_model=CandidateResponse, status_code=status.HTTP_201_CREATED, summary="Create a candidate")
def create_candidate(payload: CandidateCreate):
    sql = """
        INSERT INTO candidates (name, email, phone, location, resume_text, resume_filename,
                                skills, experience_years, education, linkedin_url, github_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        try:
            cur = conn.execute(sql, (
                payload.name, str(payload.email), payload.phone, payload.location,
                payload.resume_text, payload.resume_filename, payload.skills,
                payload.experience_years, payload.education,
                payload.linkedin_url, payload.github_url,
            ))
            conn.commit()
            row = conn.execute("SELECT * FROM candidates WHERE id = ?", (cur.lastrowid,)).fetchone()
        except Exception as exc:
            if "UNIQUE constraint failed" in str(exc):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A candidate with this email already exists",
                )
            raise
    return _row_to_dict(row)


@router.patch("/{candidate_id}", response_model=CandidateResponse, summary="Update a candidate")
def update_candidate(candidate_id: int, payload: CandidateUpdate):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")

    if "email" in updates:
        updates["email"] = str(updates["email"])

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [candidate_id]

    with get_connection() as conn:
        row_check = conn.execute("SELECT id FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if not row_check:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
        conn.execute(
            f"UPDATE candidates SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            values,
        )
        conn.commit()
        row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
    return _row_to_dict(row)


@router.delete("/{candidate_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a candidate")
def delete_candidate(candidate_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Candidate not found")
        conn.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
        conn.commit()
