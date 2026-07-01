# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime


class CandidateBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    phone: Optional[str] = None
    location: Optional[str] = None
    skills: Optional[str] = None
    experience_years: Optional[int] = Field(None, ge=0, le=60)
    education: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None


class CandidateCreate(CandidateBase):
    resume_text: Optional[str] = None
    resume_filename: Optional[str] = None


class CandidateUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    resume_text: Optional[str] = None
    resume_filename: Optional[str] = None
    skills: Optional[str] = None
    experience_years: Optional[int] = Field(None, ge=0, le=60)
    education: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None


class CandidateResponse(CandidateBase):
    id: int
    resume_filename: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
