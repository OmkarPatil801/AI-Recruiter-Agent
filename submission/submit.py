"""
submission/submit.py
--------------------
Official hackathon submission generator.

Produces a uniquely named submission CSV under submission/ with exactly 100
ranked candidates, then validates the output with submission/validate_submission.py.

Workflow:
    1.  Load data/candidates.jsonl.
    2.  Load pre-computed candidate embeddings from cache.
        Aborts with a clear error if cache is missing.
    3.  Read the job description from a file path or inline text.
    4.  Score all candidates using the existing ranking engine.
    5.  Sort globally by (final_score DESC, candidate_id ASC).
        Tie-breaking by candidate_id ascending ensures a fully stable order
        when two candidates share the same rounded final_score.
    6.  Select top 100; assign ranks 1-100.
    7.  Generate 1-2 sentence reasoning per candidate from scorer outputs.
        No LLM is used.  All claims are grounded in numeric signals.
    8.  Write <JobTitle>_<YYYYMMDD_HHMMSS>.csv (UTF-8, columns:
        candidate_id, rank, score, reasoning).
    9.  Append one row to submission/submission_history.csv.
    10. Auto-run validate_submission.py.

Usage:
    # Read JD from a text file:
    python -m submission.submit --jd path/to/job_description.txt

    # Pass JD as an inline string:
    python -m submission.submit --jd "Senior ML Engineer required..."

    # Override output path (disables auto-naming):
    python -m submission.submit --jd jd.txt --output my_submission.csv

    # Override dataset path:
    python -m submission.submit --jd jd.txt --dataset data/candidates.jsonl
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Make repository root importable when run as a script or module.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ranking.cache import load_cache
from ranking.parser import load_candidates
from ranking.scorer import extract_required_skills, score_candidates_fast

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DATASET = _REPO_ROOT / "data" / "candidates.jsonl"
_SUBMISSION_DIR  = Path(__file__).parent
_HISTORY_FILE    = _SUBMISSION_DIR / "submission_history.csv"
_TOP_N           = 100


# ---------------------------------------------------------------------------
# Job title inference for auto-generated filenames
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "for", "with", "in", "on", "at", "to",
    "of", "from", "by", "is", "are", "we", "our", "your", "who", "that",
    "this", "have", "has", "will", "be", "as", "up", "its", "via", "are",
    "required", "needed", "looking", "seeking", "hiring",
})


def _infer_job_title(jd: str, max_words: int = 5) -> str:
    """
    Derive a short sanitized job title from the job description.

    Takes the first non-empty line (usually the job title), strips
    everything after the first comma/pipe/colon, removes stop words,
    and joins with underscores.  Falls back to "Submission" if nothing
    meaningful can be extracted.
    """
    first_line = ""
    for line in jd.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break
    if not first_line:
        first_line = jd[:120]

    # Cut at first comma, pipe, colon, or em dash.
    first_line = re.split(r"[,|:—]", first_line)[0].strip()

    # Remove characters that are not alphanumeric, space, or hyphen.
    cleaned = re.sub(r"[^a-zA-Z0-9 \-]", " ", first_line)

    words = [
        w.capitalize()
        for w in cleaned.split()
        if w.lower() not in _STOP_WORDS and len(w) > 1
    ]

    title = "_".join(words[:max_words]) if words else "Submission"
    return title or "Submission"


def _generate_output_path(jd: str) -> Path:
    """
    Build a unique output path under submission/ that embeds the job title
    and current timestamp.

    Format: submission/<JobTitle>_<YYYYMMDD_HHMMSS>.csv
    Appends _1, _2, ... if the file already exists (same-second collision).
    Filename is capped at ~80 characters (excluding directory).
    """
    title = _infer_job_title(jd)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    base  = f"{title}_{ts}"

    if len(base) > 72:
        base = base[:72]

    candidate = _SUBMISSION_DIR / f"{base}.csv"
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = _SUBMISSION_DIR / f"{base}_{suffix}.csv"
        if not candidate.exists():
            return candidate
        suffix += 1


# ---------------------------------------------------------------------------
# Submission history log
# ---------------------------------------------------------------------------

_HISTORY_COLUMNS = [
    "timestamp", "filename", "job_title",
    "candidates_processed", "shortlisted",
    "highest_score", "lowest_score",
    "runtime_seconds", "validation",
]


def _append_history(
    *,
    filename: str,
    job_title: str,
    candidates_processed: int,
    shortlisted: int,
    highest_score: float,
    lowest_score: float,
    runtime_seconds: float,
    validation: str,
) -> None:
    """Append one row to submission/submission_history.csv."""
    write_header = not _HISTORY_FILE.exists()
    with _HISTORY_FILE.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_HISTORY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename":             filename,
            "job_title":            job_title,
            "candidates_processed": candidates_processed,
            "shortlisted":          shortlisted,
            "highest_score":        f"{highest_score:.4f}",
            "lowest_score":         f"{lowest_score:.4f}",
            "runtime_seconds":      f"{runtime_seconds:.1f}",
            "validation":           validation,
        })


# ---------------------------------------------------------------------------
# Job description loader
# ---------------------------------------------------------------------------

def _load_jd(value: str) -> str:
    """
    Resolve a JD value: if it refers to a readable file, return its contents;
    otherwise treat the value itself as the job description text.
    """
    p = Path(value)
    if p.exists() and p.is_file():
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            print(f"[submit] ERROR: JD file is empty: {p}")
            sys.exit(1)
        print(f"[submit] Job description loaded from {p} ({len(text):,} chars)")
        return text
    if len(value) < 20:
        print(f"[submit] WARNING: --jd value looks very short ({len(value)} chars). "
              f"Pass a file path or a full job description string.")
    return value.strip()


# ---------------------------------------------------------------------------
# Reasoning builder
# ---------------------------------------------------------------------------

_CAT_PHRASES: dict[str, str] = {
    "CORE_ML":  "core ML/AI practitioner",
    "ML_ADJ":   "ML-adjacent professional (Data Engineering / MLOps / Platform)",
    "ENG":      "software engineering professional",
    "NON_TECH": "non-technical professional",
}


def _build_reasoning(candidate: dict, required_skills: set[str]) -> str:
    """
    Generate a 1-2 sentence, data-grounded reasoning string for the CSV.

    All claims derive directly from the scoring signals in `candidate`.
    No LLM, no paraphrasing, no hallucination.

    Sentence 1: domain identity, title, experience level, overall score.
    Sentence 2: top qualifying signals — skill coverage, production delivery,
                and semantic alignment.
    """
    name    = candidate.get("name", "This candidate")
    title   = candidate.get("current_title", "unknown role")
    yoe     = float(candidate.get("years_of_experience", 0))
    cat     = candidate.get("candidate_category", "")
    matched = candidate.get("required_skills_found", [])
    total   = len(required_skills)
    score   = candidate.get("final_score",          0.0)
    dlv     = candidate.get("delivery_score",       0.0)
    sem     = candidate.get("cosine_score",         0.0)
    rol     = candidate.get("role_relevance_score", 0.0)
    skl     = candidate.get("skill_match_score",    0.0)

    cat_phrase = _CAT_PHRASES.get(cat, "professional")
    s1 = (
        f"{name} is a {cat_phrase} ({title}, {yoe:.0f} yrs experience) "
        f"with an overall match score of {score:.4f}."
    )

    parts: list[str] = []

    if matched:
        preview = ", ".join(matched[:4])
        extra   = f" (+{len(matched) - 4} more)" if len(matched) > 4 else ""
        parts.append(f"{len(matched)}/{total} required skills matched ({preview}{extra})")

    if dlv >= 0.60:
        parts.append("strong production delivery evidence in career history")
    elif dlv >= 0.20:
        parts.append("some production delivery evidence present")

    if sem >= 0.60:
        parts.append(f"high semantic alignment with the role ({sem:.3f})")
    elif sem >= 0.50:
        parts.append(f"solid semantic alignment ({sem:.3f})")

    if parts:
        s2 = "; ".join(parts).capitalize() + "."
    elif skl > 0:
        s2 = (
            f"Partial skill coverage ({skl:.0%}) with role relevance score {rol:.3f}; "
            f"ranked on combined signal strength."
        )
    else:
        s2 = (
            f"Ranked on semantic similarity ({sem:.3f}) and role relevance ({rol:.3f}); "
            f"no overlap with required skills."
        )

    return f"{s1} {s2}"


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], output_path: Path) -> None:
    """Write the top-N candidate rows to a UTF-8 CSV with the required columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for c in rows:
            writer.writerow({
                "candidate_id": c["candidate_id"],
                "rank":         c["rank"],
                "score":        c["final_score"],
                "reasoning":    c["reasoning"],
            })


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    jd: str,
    dataset_path: Path,
    output_path: Path,
    top_n: int = _TOP_N,
) -> tuple[Path, dict]:
    """
    Execute the full submission pipeline.

    Parameters
    ----------
    jd           : Full job description text.
    dataset_path : Path to candidates.jsonl (or .json sample).
    output_path  : Destination for the submission CSV.
    top_n        : Number of candidates to include (default 100).

    Returns
    -------
    (written CSV path, stats dict)
    Stats keys: elapsed (float), num_candidates (int),
                top_score (float), bot_score (float).
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Load candidates (process-level cache handles repeat calls)
    # ------------------------------------------------------------------
    print(f"[submit] Loading candidates from {dataset_path.name} ...")
    candidates = load_candidates(path=dataset_path)
    print(f"[submit] {len(candidates):,} candidates loaded.")

    # ------------------------------------------------------------------
    # Step 2: Load embedding cache — abort if missing
    # ------------------------------------------------------------------
    cached = load_cache(dataset_path)
    if cached is None:
        print()
        print("[submit] ERROR: Candidate embedding cache not found.")
        print(f"         Expected: {dataset_path.stem}_embeddings.npz")
        print()
        print("         Run the preprocessing step first:")
        print(f"             python -m ranking.preprocess --input {dataset_path}")
        sys.exit(1)

    embeddings, cached_ids = cached
    live_ids = [c["candidate_id"] for c in candidates]
    if live_ids != cached_ids:
        print("[submit] ERROR: Embedding cache is misaligned with the dataset.")
        print("         Re-run: python -m ranking.preprocess")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Extract required skills from JD (pure text scan)
    # ------------------------------------------------------------------
    required_skills = extract_required_skills(jd)
    print(f"[submit] Required skills identified ({len(required_skills)}): "
          f"{', '.join(sorted(required_skills))}")

    # ------------------------------------------------------------------
    # Step 4: Enforce deterministic inference before embedding the JD.
    #
    # Candidate embeddings come from the deterministic .npz cache.
    # The JD embedding is computed fresh each run.  On multi-threaded CPUs,
    # BLAS parallel reductions can produce slightly different floating-point
    # results between runs (different thread interleaving -> different
    # summation order -> different rounding at the last bit).  Setting
    # manual seeds and requesting deterministic algorithms eliminates this.
    #
    # This does NOT change the model, weights, or scoring formula.
    # ------------------------------------------------------------------
    try:
        import torch
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        # warn_only=True avoids crashing if any op lacks a deterministic
        # implementation on the current hardware/driver combination.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (ImportError, TypeError):
        pass

    import numpy as np
    np.random.seed(0)

    print(f"[submit] Scoring {len(candidates):,} candidates ...")
    scored = score_candidates_fast(jd, candidates, embeddings)

    # ------------------------------------------------------------------
    # Steps 5-6: Global top-N selection with deterministic tie-breaking.
    #
    # Ranking is global (no filter gate) to guarantee 100 rows are always
    # produced even at small dataset sizes.
    #
    # Sort key:
    #   Primary   — final_score descending  (higher score = better rank)
    #   Secondary — candidate_id ascending  (lexicographic, always unique)
    #
    # The secondary key is the authoritative tie-breaker: two candidates
    # whose final_scores are identical after rounding to 4 d.p. are always
    # ordered by candidate_id, regardless of floating-point variance in the
    # JD embedding layer.  This is the only place rank order is decided.
    # ------------------------------------------------------------------
    ranked = sorted(scored, key=lambda c: (-c["final_score"], c["candidate_id"]))
    top = ranked[:top_n]
    for position, candidate in enumerate(top, start=1):
        candidate["rank"] = position

    # ------------------------------------------------------------------
    # Step 7: Generate reasoning (deterministic, grounded, no LLM)
    # ------------------------------------------------------------------
    for candidate in top:
        candidate["reasoning"] = _build_reasoning(candidate, required_skills)

    # ------------------------------------------------------------------
    # Step 8: Write CSV
    # ------------------------------------------------------------------
    _write_csv(top, output_path)
    elapsed = time.perf_counter() - t0
    print(f"[submit] Submission written to {output_path}  "
          f"({len(top)} rows, {elapsed:.1f}s total)")
    print(f"[submit] Score range: "
          f"{top[-1]['final_score']:.4f} - {top[0]['final_score']:.4f}")

    stats = {
        "elapsed":        elapsed,
        "num_candidates": len(candidates),
        "top_score":      top[0]["final_score"],
        "bot_score":      top[-1]["final_score"],
    }
    return output_path, stats


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------

def _run_validator(csv_path: Path) -> tuple[bool, str]:
    """
    Invoke validate_submission.py as a subprocess and relay its output.
    Returns (passed: bool, output_text: str).
    """
    validator = Path(__file__).parent / "validate_submission.py"
    result = subprocess.run(
        [sys.executable, str(validator), str(csv_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = (result.stdout or "") + (result.stderr if result.returncode != 0 else "")
    return result.returncode == 0, output.strip()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m submission.submit",
        description="Generate the hackathon submission CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--jd",
        metavar="PATH_OR_TEXT",
        required=True,
        help=(
            "Job description: a path to a .txt file, or an inline string. "
            "If the value resolves to an existing file, its contents are used; "
            "otherwise the value is used as the job description text directly."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        metavar="PATH",
        help=f"Candidate dataset file (.jsonl or .json).  Default: {_DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Output CSV path.  If omitted, a unique name is auto-generated as "
            "submission/<JobTitle>_<YYYYMMDD_HHMMSS>.csv and a row is appended "
            "to submission/submission_history.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    jd = _load_jd(args.jd)

    # Determine output path.
    explicit_output = args.output is not None
    if explicit_output:
        output_path = args.output
        job_title   = _infer_job_title(jd)
    else:
        output_path = _generate_output_path(jd)
        job_title   = _infer_job_title(jd)

    csv_path, stats = run(
        jd=jd,
        dataset_path=args.dataset,
        output_path=output_path,
    )

    # Run validation.
    passed, validator_output = _run_validator(csv_path)
    validation_result = "PASS" if passed else "FAIL"

    # Append to history log for every run (explicit or auto-named).
    _append_history(
        filename=csv_path.name,
        job_title=job_title,
        candidates_processed=stats["num_candidates"],
        shortlisted=_TOP_N,
        highest_score=stats["top_score"],
        lowest_score=stats["bot_score"],
        runtime_seconds=stats["elapsed"],
        validation=validation_result,
    )

    # Final summary block.
    print()
    print("Submission generated successfully.")
    print(f"Output:      {csv_path}")
    print(f"Rows:        {_TOP_N}")
    print(f"Validation:  {validation_result}")
    if validator_output:
        print(validator_output)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
