"""
Gemini AI service — placeholder for Day 1.
Actual Gemini integration will be implemented in a later phase.
"""
from typing import Any


async def analyze_resume(resume_text: str, job_description: str) -> dict[str, Any]:
    """Placeholder: will call Gemini to extract structured data and assess fit."""
    raise NotImplementedError("Gemini integration not yet implemented")


async def extract_skills(text: str) -> list[str]:
    """Placeholder: will call Gemini to extract skills from resume or job text."""
    raise NotImplementedError("Gemini integration not yet implemented")


async def generate_reasoning(candidate_data: dict, job_data: dict, scores: dict) -> str:
    """Placeholder: will call Gemini to generate human-readable scoring rationale."""
    raise NotImplementedError("Gemini integration not yet implemented")
