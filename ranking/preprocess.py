"""
ranking/preprocess.py
---------------------
One-time preprocessing step: embed all candidates and save to disk.

Run this once before starting the server on a new dataset.  Subsequent
ranking requests load cached embeddings instead of re-embedding, reducing
per-request embedding time from O(N) to O(1).

Usage:
    python -m ranking.preprocess                              # auto-detects dataset
    python -m ranking.preprocess --input data/candidates.jsonl
    python -m ranking.preprocess --input data/sample_candidates.json

Auto-detection order (picks the first file that exists):
    1. data/candidates.jsonl  (production evaluation dataset)
    2. data/sample_candidates.json  (development sample)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ranking.cache import save_cache
from ranking.parser import load_candidates
from ranking.scorer import embed_texts

_DEFAULT_PATHS = [
    Path(__file__).parent.parent / "data" / "candidates.jsonl",
    Path(__file__).parent.parent / "data" / "sample_candidates.json",
]


def _resolve_input(arg: Path | None) -> Path:
    if arg is not None:
        if not arg.exists():
            print(f"[preprocess] File not found: {arg}")
            sys.exit(1)
        return arg
    for p in _DEFAULT_PATHS:
        if p.exists():
            return p
    print("[preprocess] No candidate dataset found.  Pass --input <path>.")
    sys.exit(1)


def preprocess(input_path: Path, batch_size: int = 64) -> None:
    t0 = time.perf_counter()

    print(f"[preprocess] Dataset  : {input_path}")
    print(f"[preprocess] Loading candidates ...")
    candidates = load_candidates(path=input_path)
    n = len(candidates)
    print(f"[preprocess] Loaded   : {n:,} candidates")

    texts = [c["candidate_text"] for c in candidates]
    ids   = [c["candidate_id"]   for c in candidates]

    t_embed = time.perf_counter()
    print(f"[preprocess] Embedding {n:,} texts (batch={batch_size}) ...")
    embeddings = embed_texts(texts, batch_size=batch_size, show_progress=True)
    embed_secs = time.perf_counter() - t_embed

    cache_file = save_cache(input_path, embeddings, ids)

    total_secs = time.perf_counter() - t0
    print()
    print("[preprocess] Complete.")
    print(f"  Candidates  : {n:,}")
    print(f"  Embed time  : {embed_secs:.1f}s  ({n / embed_secs:,.0f} candidates/sec)")
    print(f"  Total time  : {total_secs:.1f}s")
    print(f"  Cache file  : {cache_file}")
    print(f"  Cache size  : {cache_file.stat().st_size / 1_048_576:.1f} MB")
    print()
    print("  Ranking requests will now use cached embeddings automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-embed candidate dataset for fast ranking."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to candidate dataset (.jsonl or .json).  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Embedding batch size (default: 64).",
    )
    args = parser.parse_args()
    preprocess(_resolve_input(args.input), batch_size=args.batch_size)
