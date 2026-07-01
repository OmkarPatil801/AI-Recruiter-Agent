from fastapi import APIRouter, HTTPException, status
from typing import List
from database import get_connection
from models.job import JobCreate, JobUpdate, JobResponse

router = APIRouter(prefix="/jobs", tags=["Jobs"])


def _row_to_dict(row) -> dict:
    return dict(row)


@router.get("/", response_model=List[JobResponse], summary="List all jobs")
def list_jobs():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [_row_to_dict(r) for r in rows]


@router.get("/{job_id}", response_model=JobResponse, summary="Get a job by ID")
def get_job(job_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _row_to_dict(row)


@router.post("/", response_model=JobResponse, status_code=status.HTTP_201_CREATED, summary="Create a job")
def create_job(payload: JobCreate):
    sql = """
        INSERT INTO jobs (title, company, description, requirements, location,
                          salary_min, salary_max, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cur = conn.execute(sql, (
            payload.title, payload.company, payload.description,
            payload.requirements, payload.location,
            payload.salary_min, payload.salary_max, payload.status,
        ))
        conn.commit()
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _row_to_dict(row)


@router.patch("/{job_id}", response_model=JobResponse, summary="Update a job")
def update_job(job_id: int, payload: JobUpdate):
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [job_id]

    with get_connection() as conn:
        row_check = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row_check:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        conn.execute(
            f"UPDATE jobs SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            values,
        )
        conn.commit()
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a job")
def delete_job(job_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
