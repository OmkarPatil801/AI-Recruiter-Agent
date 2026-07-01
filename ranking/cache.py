"""
ranking/cache.py
----------------
Disk-based embedding cache for the candidate dataset.

Strategy:
- Embeddings are stored as a compressed NumPy archive (.npz) alongside the
  source dataset file.  E.g. data/candidates.jsonl -> data/candidates_embeddings.npz
- Freshness is determined by the source file's mtime + size.  If either
  changes, the cache is treated as stale and the caller must re-embed.
- load_cache() returns None on any miss (absent, stale, corrupt) so callers
  can fall back to inline embedding without special-casing.

Public API:
    cache_path_for(source_path)             -> Path
    load_cache(source_path)                 -> (embeddings, ids) | None
    save_cache(source_path, embeddings, ids) -> Path
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def cache_path_for(source_path: Path) -> Path:
    """Return the .npz cache path for a given candidate dataset file."""
    return source_path.with_name(source_path.stem + "_embeddings.npz")


def _source_signature(path: Path) -> tuple[float, int]:
    """(mtime, size) pair used as a cheap staleness check."""
    stat = path.stat()
    return stat.st_mtime, stat.st_size


def load_cache(source_path: Path) -> tuple[np.ndarray, list[str]] | None:
    """
    Load cached candidate embeddings.

    Returns (embeddings, candidate_ids) when cache is fresh, else None.
    embeddings    : float32 array (N, D) — L2-normalised, ready for dot products
    candidate_ids : list[str] of length N, in the same order as the dataset
    """
    cache = cache_path_for(source_path)
    if not cache.exists():
        return None

    try:
        data = np.load(str(cache), allow_pickle=True)

        cached_mtime = float(data["source_mtime"][0])
        cached_size  = int(data["source_size"][0])
        mtime, size  = _source_signature(source_path)

        if cached_mtime != mtime or cached_size != size:
            print(f"[cache] Stale — source changed since last preprocess: {cache.name}")
            return None

        embeddings    = data["embeddings"].astype(np.float32)
        candidate_ids = data["candidate_ids"].tolist()
        print(f"[cache] Loaded {len(candidate_ids):,} cached embeddings ({cache.name})")
        return embeddings, candidate_ids

    except Exception as exc:
        print(f"[cache] Load failed ({exc}) — will fall back to inline embedding.")
        return None


def save_cache(
    source_path: Path,
    embeddings: np.ndarray,
    candidate_ids: list[str],
) -> Path:
    """
    Persist embeddings to a compressed .npz alongside the source dataset.

    Returns the path of the written file.
    """
    cache = cache_path_for(source_path)
    mtime, size = _source_signature(source_path)

    np.savez_compressed(
        str(cache),
        embeddings=embeddings.astype(np.float32),
        candidate_ids=np.array(candidate_ids, dtype=object),
        source_mtime=np.array([mtime]),
        source_size=np.array([size], dtype=np.int64),
    )
    size_mb = cache.stat().st_size / 1_048_576
    print(f"[cache] Saved to {cache.name}  ({len(candidate_ids):,} candidates, {size_mb:.1f} MB)")
    return cache
