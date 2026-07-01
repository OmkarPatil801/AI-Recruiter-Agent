# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class JobBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    company: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    requirements: Optional[str] = None
    location: Optional[str] = None
    salary_min: Optional[int] = Field(None, ge=0)
    salary_max: Optional[int] = Field(None, ge=0)
    status: str = Field(default="active", pattern="^(active|closed|draft)$")


class JobCreate(JobBase):
    pass


class JobUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    company: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1)
    requirements: Optional[str] = None
    location: Optional[str] = None
    salary_min: Optional[int] = Field(None, ge=0)
    salary_max: Optional[int] = Field(None, ge=0)
    status: Optional[str] = Field(None, pattern="^(active|closed|draft)$")


class JobResponse(JobBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
