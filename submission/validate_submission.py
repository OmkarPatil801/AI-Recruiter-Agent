"""
submission/validate_submission.py
----------------------------------
Official submission validator for the AI Recruiter hackathon.

Checks:
    1. File exists and is UTF-8 readable.
    2. Columns are exactly: candidate_id, rank, score, reasoning (in any order,
       but all four must be present — no extra columns allowed).
    3. Exactly 100 data rows.
    4. Ranks are integers 1–100, each appearing exactly once.
    5. Scores are finite floats in the range [0.0, 1.0].
    6. candidate_id values are non-empty strings with no duplicates.
    7. reasoning values are non-empty strings (no blank or whitespace-only entries).

Usage:
    python submission/validate_submission.py submission/submission.csv

Exit codes:
    0  — all checks passed
    1  — one or more checks failed (errors printed to stdout)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REQUIRED_COLUMNS: set[str] = {"candidate_id", "rank", "score", "reasoning"}
EXPECTED_ROWS = 100


def validate(csv_path: Path) -> list[str]:
    """
    Validate a submission CSV file.

    Returns a list of error strings.  Empty list means the file is valid.
    """
    errors: list[str] = []

    # ------------------------------------------------------------------ #
    # Check 1: file exists and is UTF-8 readable
    # ------------------------------------------------------------------ #
    if not csv_path.exists():
        return [f"File not found: {csv_path}"]

    try:
        text = csv_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [f"File is not valid UTF-8: {exc}"]

    # ------------------------------------------------------------------ #
    # Parse CSV
    # ------------------------------------------------------------------ #
    try:
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames is None:
            return ["CSV file is empty or has no header row."]
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    except Exception as exc:
        return [f"Failed to parse CSV: {exc}"]

    # ------------------------------------------------------------------ #
    # Check 2: column names
    # ------------------------------------------------------------------ #
    actual_columns = set(fieldnames)
    missing = REQUIRED_COLUMNS - actual_columns
    extra   = actual_columns - REQUIRED_COLUMNS
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
    if extra:
        errors.append(f"Unexpected extra columns: {sorted(extra)}")
    if missing:
        return errors  # column errors make further checks meaningless

    # ------------------------------------------------------------------ #
    # Check 3: row count
    # ------------------------------------------------------------------ #
    if len(rows) != EXPECTED_ROWS:
        errors.append(
            f"Expected exactly {EXPECTED_ROWS} rows, found {len(rows)}."
        )

    # ------------------------------------------------------------------ #
    # Per-row checks
    # ------------------------------------------------------------------ #
    seen_ranks:        set[int] = set()
    seen_candidate_ids: set[str] = set()
    rank_errors:    list[str] = []
    score_errors:   list[str] = []
    id_errors:      list[str] = []
    reason_errors:  list[str] = []

    for row_num, row in enumerate(rows, start=2):  # row 1 is the header

        # Check 4: rank
        raw_rank = row.get("rank", "").strip()
        try:
            rank_val = int(raw_rank)
        except (ValueError, TypeError):
            rank_errors.append(f"Row {row_num}: rank={raw_rank!r} is not an integer.")
            rank_val = None

        if rank_val is not None:
            if rank_val < 1 or rank_val > EXPECTED_ROWS:
                rank_errors.append(
                    f"Row {row_num}: rank={rank_val} is out of range [1, {EXPECTED_ROWS}]."
                )
            elif rank_val in seen_ranks:
                rank_errors.append(f"Row {row_num}: duplicate rank={rank_val}.")
            else:
                seen_ranks.add(rank_val)

        # Check 5: score
        raw_score = row.get("score", "").strip()
        try:
            score_val = float(raw_score)
        except (ValueError, TypeError):
            score_errors.append(f"Row {row_num}: score={raw_score!r} is not a valid float.")
            score_val = None

        if score_val is not None:
            import math
            if not math.isfinite(score_val):
                score_errors.append(f"Row {row_num}: score={score_val} is not finite.")
            elif score_val < 0.0 or score_val > 1.0:
                score_errors.append(
                    f"Row {row_num}: score={score_val:.6f} is outside [0.0, 1.0]."
                )

        # Check 6: candidate_id
        cid = row.get("candidate_id", "").strip()
        if not cid:
            id_errors.append(f"Row {row_num}: candidate_id is empty.")
        elif cid in seen_candidate_ids:
            id_errors.append(f"Row {row_num}: duplicate candidate_id={cid!r}.")
        else:
            seen_candidate_ids.add(cid)

        # Check 7: reasoning
        reason = row.get("reasoning", "").strip()
        if not reason:
            reason_errors.append(f"Row {row_num}: reasoning is empty or whitespace-only.")

    # Check 4 (continued): all ranks 1-100 must appear
    if len(rows) == EXPECTED_ROWS and not rank_errors:
        expected_ranks = set(range(1, EXPECTED_ROWS + 1))
        missing_ranks = expected_ranks - seen_ranks
        if missing_ranks:
            sample = sorted(missing_ranks)[:10]
            rank_errors.append(
                f"Missing ranks: {sample}"
                + (f" (+{len(missing_ranks) - 10} more)" if len(missing_ranks) > 10 else "")
            )

    errors.extend(rank_errors)
    errors.extend(score_errors)
    errors.extend(id_errors)
    errors.extend(reason_errors)

    return errors


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python submission/validate_submission.py <path/to/submission.csv>")
        return 1

    csv_path = Path(args[0])
    errors = validate(csv_path)

    if errors:
        print(f"Submission validation FAILED  ({len(errors)} error(s)):")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Submission validation PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
