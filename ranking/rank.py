"""
rank.py — v3
------------
Pipeline orchestrator: parse → score → filter → sort → top-N.

Two public entry points:

    run_ranking(job_description, ...)
        Standard path.  Returns a list[dict] of the top-N ranked candidates
        after pre-filtering.  Same return type as v2 — existing callers are
        unaffected.

    run_evaluation(job_description, ...)
        Diagnostic path.  Runs the full pipeline twice — once without the
        filter to capture the "before" state, once with — and returns a rich
        dict for printing comparison tables.  Used by test.py's eval mode
        and for hackathon judging demonstrations.

Filter design
-------------
Candidates are eliminated BEFORE scoring reaches the ranking stage if they
fail either hard threshold.  This mirrors how a human recruiter triages a
pile of CVs: domain-wrong candidates are rejected on sight, before detailed
evaluation of the remaining pool.

    MIN_SKILL_MATCH     — candidate must cover at least this fraction of the
                          required skills.  0.15 requires ~6 of 35 required
                          skills for the current JD, eliminating candidates
                          whose skills list has no meaningful overlap with the
                          role (Accountants, PMs with 4 AI buzzwords, etc.).

    MIN_EXPERIENCE_SCORE — candidate must have at least 75% of the required
                          years.  0.75 means a 4-year minimum requires at
                          least 3 years.  Eliminates career-switchers with
                          <1 year of experience from senior roles.

Both thresholds are configurable at the call site via filter_config dict.
Defaults live in FILTER_CONFIG — change them in one place only.
"""

from __future__ import annotations

from pathlib import Path

from ranking.cache import load_cache
from ranking.parser import DATA_PATH, load_candidates
from ranking.scorer import score_candidates, score_candidates_fast


# ---------------------------------------------------------------------------
# Filter thresholds — single source of truth
# ---------------------------------------------------------------------------
FILTER_CONFIG: dict[str, float] = {
    "MIN_SKILL_MATCH":      0.15,   # must cover >=15% of required skills (~6 of 35 for current JD)
    "MIN_EXPERIENCE_SCORE": 0.75,   # must have >=75% of required years
}


# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------

def _apply_filter(
    scored: list[dict],
    config: dict[str, float],
) -> tuple[list[dict], list[dict]]:
    """
    Split a scored candidate list into (passing, rejected).

    Each rejected candidate gets a 'filter_reason' key explaining which
    threshold it failed, useful for the evaluation report.

    Parameters
    ----------
    scored  : output of score_candidates() — candidates with all score keys set
    config  : dict with MIN_SKILL_MATCH and MIN_EXPERIENCE_SCORE keys

    Returns
    -------
    (passing, rejected) — two lists, no mutation of original dicts
    """
    min_skill = config.get("MIN_SKILL_MATCH", FILTER_CONFIG["MIN_SKILL_MATCH"])
    min_exp   = config.get("MIN_EXPERIENCE_SCORE", FILTER_CONFIG["MIN_EXPERIENCE_SCORE"])

    passing:  list[dict] = []
    rejected: list[dict] = []

    for c in scored:
        reasons: list[str] = []

        if c["skill_match_score"] < min_skill:
            reasons.append(
                f"skill_match={c['skill_match_score']:.3f} < {min_skill:.2f}"
            )
        if c["experience_score"] < min_exp:
            reasons.append(
                f"experience={c['experience_score']:.3f} < {min_exp:.2f}"
            )

        if reasons:
            rejected.append({**c, "filter_reason": "; ".join(reasons)})
        else:
            passing.append(c)

    return passing, rejected


# ---------------------------------------------------------------------------
# Shared pipeline core
# ---------------------------------------------------------------------------

def _run_pipeline(
    job_description: str,
    data_path: Path | None,
    batch_size: int,
) -> tuple[list[dict], int]:
    """
    Load candidates and compute all scores.  Returns (scored_list, total_loaded).

    Fast path (cache hit): loads candidate embeddings from disk, embeds only
    the job description (1 text), then runs all non-embedding signals.

    Slow path (cache miss): embeds candidates + JD together in one batched
    pass — identical to the original behaviour, used as a fallback.
    """
    source_path = data_path or DATA_PATH
    candidates = load_candidates(path=source_path)
    total = len(candidates)
    print(f"[rank] Loaded {total:,} candidates.")

    cached = load_cache(source_path)
    if cached is not None:
        embeddings, cached_ids = cached
        live_ids = [c["candidate_id"] for c in candidates]
        if live_ids == cached_ids:
            print(f"[rank] Cache hit — embedding only the job description.")
            scored = score_candidates_fast(job_description, candidates, embeddings)
        else:
            print(f"[rank] Cache ID mismatch — falling back to inline embedding.")
            scored = score_candidates(job_description, candidates, batch_size=batch_size)
    else:
        print(f"[rank] No embedding cache found — run `python -m ranking.preprocess` to speed up future requests.")
        scored = score_candidates(job_description, candidates, batch_size=batch_size)

    return scored, total


# ---------------------------------------------------------------------------
# Public API — standard ranking
# ---------------------------------------------------------------------------

def run_ranking(
    job_description: str,
    data_path: Path | None = None,
    top_n: int = 20,
    batch_size: int = 32,
    filter_config: dict[str, float] | None = None,
) -> list[dict]:
    """
    End-to-end ranking pipeline with pre-filtering.

    Parameters
    ----------
    job_description : str
        Full text of the job posting.
    data_path : Path | None
        Override the default dataset path (useful for tests).
    top_n : int
        Number of candidates to return.
    batch_size : int
        Embedding batch size.
    filter_config : dict | None
        Override filter thresholds.  Keys: MIN_SKILL_MATCH, MIN_EXPERIENCE_SCORE.
        Defaults to FILTER_CONFIG when None.

    Returns
    -------
    list[dict]
        Top-N ranked candidates, each dict enriched with:
            rank, cosine_score, skill_match_score, role_relevance_score,
            experience_score, behavioural_score, final_score,
            required_skills_found
    """
    config = filter_config or FILTER_CONFIG

    scored, total = _run_pipeline(job_description, data_path, batch_size)
    passing, rejected = _apply_filter(scored, config)

    print(
        f"[rank] Filter: {total} candidates -> "
        f"{len(passing)} pass, {len(rejected)} removed"
        f" (skill<{config['MIN_SKILL_MATCH']:.2f} or exp<{config['MIN_EXPERIENCE_SCORE']:.2f})"
    )

    # Tie-break by candidate_id ascending so equal-scored candidates
    # always appear in the same order regardless of iteration state.
    ranked = sorted(passing, key=lambda c: (-c["final_score"], c["candidate_id"]))
    for position, candidate in enumerate(ranked, start=1):
        candidate["rank"] = position

    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Public API — evaluation / diagnostic mode
# ---------------------------------------------------------------------------

def run_evaluation(
    job_description: str,
    data_path: Path | None = None,
    top_n: int = 20,
    batch_size: int = 32,
    filter_config: dict[str, float] | None = None,
) -> dict:
    """
    Full diagnostic run.  Computes scores once, then produces both
    the unfiltered and filtered top-N views for side-by-side comparison.

    Returns
    -------
    dict with keys:
        total_loaded       : int
        total_passing      : int
        total_rejected     : int
        filter_config      : dict
        unfiltered_top_n   : list[dict]  — top-N before filter (with rank)
        filtered_top_n     : list[dict]  — top-N after filter (with rank)
        rejected_candidates: list[dict]  — all removed candidates with filter_reason
    """
    config = filter_config or FILTER_CONFIG

    scored, total = _run_pipeline(job_description, data_path, batch_size)

    # --- unfiltered ranking (before) ---
    # Tie-break by candidate_id ascending for full determinism.
    unfiltered_ranked = sorted(scored, key=lambda c: (-c["final_score"], c["candidate_id"]))
    for pos, c in enumerate(unfiltered_ranked, start=1):
        c["unfiltered_rank"] = pos
    unfiltered_top_n = [dict(c, rank=c["unfiltered_rank"]) for c in unfiltered_ranked[:top_n]]

    # --- filtered ranking (after) ---
    passing, rejected = _apply_filter(scored, config)
    filtered_ranked = sorted(passing, key=lambda c: (-c["final_score"], c["candidate_id"]))
    for pos, c in enumerate(filtered_ranked, start=1):
        c["rank"] = pos
    filtered_top_n = filtered_ranked[:top_n]

    print(
        f"[eval] {total} loaded -> "
        f"{len(passing)} pass, {len(rejected)} rejected by filter"
    )

    return {
        "total_loaded":        total,
        "total_passing":       len(passing),
        "total_rejected":      len(rejected),
        "filter_config":       config,
        "unfiltered_top_n":    unfiltered_top_n,
        "filtered_top_n":      filtered_top_n,
        "rejected_candidates": sorted(rejected, key=lambda c: (-c["final_score"], c["candidate_id"])),
    }
