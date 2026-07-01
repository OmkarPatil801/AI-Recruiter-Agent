from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database import init_db
from routers import jobs, candidates, scores
from routers import ranking as ranking_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="AI Recruiter Agent",
    description="AI-powered candidate discovery and ranking system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(candidates.router)
app.include_router(scores.router)
app.include_router(ranking_router.router)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "version": app.version}
