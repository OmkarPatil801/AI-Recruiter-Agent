# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ScoreBase(BaseModel):
    job_id: int
    candidate_id: int
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    skills_score: float = Field(default=0.0, ge=0.0, le=100.0)
    experience_score: float = Field(default=0.0, ge=0.0, le=100.0)
    education_score: float = Field(default=0.0, ge=0.0, le=100.0)
    location_score: float = Field(default=0.0, ge=0.0, le=100.0)
    reasoning: Optional[str] = None
    status: str = Field(default="pending", pattern="^(pending|scored|failed)$")


class ScoreCreate(BaseModel):
    job_id: int
    candidate_id: int


class ScoreResponse(ScoreBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class RankedCandidate(BaseModel):
    rank: int
    candidate_id: int
    candidate_name: str
    candidate_email: str
    overall_score: float
    skills_score: float
    experience_score: float
    education_score: float
    location_score: float
    reasoning: Optional[str] = None
    status: str
