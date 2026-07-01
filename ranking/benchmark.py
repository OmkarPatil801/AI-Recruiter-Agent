"""
ranking/benchmark.py
--------------------
Measures the performance impact of the embedding cache optimisation.

Compares two execution paths against the same dataset:
    INLINE  — original behaviour: embed candidates + JD in every request
    CACHED  — optimised: load cached embeddings, embed only the JD

Metrics reported:
    Total wall time        (seconds)
    Embedding time         (seconds)
    Non-embedding time     (seconds)
    Peak memory delta      (MB, via tracemalloc)
    Throughput             (candidates scored per second)

Usage:
    python -m ranking.benchmark                         # auto-detects dataset
    python -m ranking.benchmark --input data/candidates.jsonl
    python -m ranking.benchmark --input data/sample_candidates.json
    python -m ranking.benchmark --no-cache              # inline path only

Notes:
- Run `python -m ranking.preprocess` first to warm the cache.
- The process-level candidate cache (parser.py) is shared between both runs
  in the same process, so I/O is only measured once.  This is representative
  of steady-state server behaviour (candidates loaded once per process).
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ranking.cache import load_cache
from ranking.parser import DATA_PATH, load_candidates
from ranking.scorer import embed_texts, extract_required_skills, score_candidates, score_candidates_fast

JOB_DESCRIPTION = """
Senior Machine Learning Engineer - Conversational AI Platform

We are building a next-generation conversational AI platform and are looking
for a Senior ML Engineer to lead model development and deployment.

Requirements:
- 4+ years of experience in machine learning engineering or applied NLP.
- Strong Python skills; proficiency in PyTorch or TensorFlow.
- Hands-on experience fine-tuning transformer models (BERT, GPT-style, T5,
  Llama, Mistral) using LoRA, QLoRA, or full fine-tuning.
- Experience with MLOps tooling: MLflow, Weights & Biases, or equivalent.
- Solid understanding of NLP fundamentals: tokenisation, embeddings,
  attention mechanisms, RLHF.
- Cloud platform experience (AWS, GCP, or Azure).
""".strip()


def _fmt(secs: float) -> str:
    return f"{secs:.3f}s"


def _run_inline(candidates: list[dict], batch_size: int) -> dict:
    """Benchmark the original inline embedding path."""
    tracemalloc.start()
    t0 = time.perf_counter()

    t_embed_start = time.perf_counter()
    all_texts = [JOB_DESCRIPTION] + [c["candidate_text"] for c in candidates]
    embeddings = embed_texts(all_texts, batch_size=batch_size, show_progress=False)
    embed_secs = time.perf_counter() - t_embed_start

    job_emb  = embeddings[0:1]
    cand_emb = embeddings[1:]
    _ = cand_emb @ job_emb[0]  # cosine scores (simulate scoring)
    required_skills = extract_required_skills(JOB_DESCRIPTION)
    scored = score_candidates(JOB_DESCRIPTION, candidates, batch_size=batch_size)

    total_secs = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "path":         "INLINE (no cache)",
        "n":            len(candidates),
        "total_secs":   total_secs,
        "embed_secs":   embed_secs,
        "other_secs":   total_secs - embed_secs,
        "peak_mb":      peak / 1_048_576,
        "throughput":   len(candidates) / total_secs,
    }


def _run_cached(candidates: list[dict], source_path: Path) -> dict | None:
    """Benchmark the optimised cached-embedding path."""
    result = load_cache(source_path)
    if result is None:
        print("\n[benchmark] No cache found for cached path.  Run `python -m ranking.preprocess` first.")
        return None

    cached_embeddings, cached_ids = result
    live_ids = [c["candidate_id"] for c in candidates]
    if live_ids != cached_ids:
        print("\n[benchmark] Cache ID mismatch — cannot benchmark cached path.")
        return None

    tracemalloc.start()
    t0 = time.perf_counter()

    t_embed_start = time.perf_counter()
    job_emb = embed_texts([JOB_DESCRIPTION], batch_size=1, show_progress=False)[0]
    embed_secs = time.perf_counter() - t_embed_start

    _ = cached_embeddings @ job_emb  # cosine scores
    scored = score_candidates_fast(JOB_DESCRIPTION, candidates, cached_embeddings)

    total_secs = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "path":         "CACHED (precomputed embeddings)",
        "n":            len(candidates),
        "total_secs":   total_secs,
        "embed_secs":   embed_secs,
        "other_secs":   total_secs - embed_secs,
        "peak_mb":      peak / 1_048_576,
        "throughput":   len(candidates) / total_secs,
    }


def _print_result(r: dict) -> None:
    print(f"\n  Path       : {r['path']}")
    print(f"  Candidates : {r['n']:,}")
    print(f"  Total      : {_fmt(r['total_secs'])}")
    print(f"  Embedding  : {_fmt(r['embed_secs'])}  ({r['embed_secs'] / r['total_secs'] * 100:.0f}% of total)")
    print(f"  Other      : {_fmt(r['other_secs'])}")
    print(f"  Peak mem   : {r['peak_mb']:.1f} MB")
    print(f"  Throughput : {r['throughput']:,.0f} candidates/sec")


def _print_comparison(inline: dict, cached: dict) -> None:
    speedup     = inline["total_secs"]  / cached["total_secs"]
    embed_ratio = inline["embed_secs"]  / max(cached["embed_secs"], 1e-6)
    mem_saved   = inline["peak_mb"]     - cached["peak_mb"]
    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print(f"  Total speedup       : {speedup:.1f}x faster")
    print(f"  Embedding speedup   : {embed_ratio:.1f}x faster  ({inline['n']+1} texts -> 1)")
    print(f"  Peak memory saved   : {mem_saved:.1f} MB")
    print(f"  Throughput gain     : {cached['throughput'] / inline['throughput']:.1f}x")
    print()


def main(source_path: Path, run_cache: bool, batch_size: int) -> None:
    sep = "=" * 60
    print(sep)
    print("  RANKING ENGINE BENCHMARK")
    print(sep)
    print(f"  Dataset : {source_path.name}")

    print(f"\n[benchmark] Loading {source_path.name} (populates process-level cache) ...")
    t_load = time.perf_counter()
    candidates = load_candidates(path=source_path)
    load_secs = time.perf_counter() - t_load
    print(f"[benchmark] Loaded {len(candidates):,} candidates in {load_secs:.2f}s")

    print(f"\n{sep}")
    print("  PATH 1: INLINE (baseline)")
    print(sep)
    inline = _run_inline(candidates, batch_size=batch_size)
    _print_result(inline)

    if run_cache:
        print(f"\n{sep}")
        print("  PATH 2: CACHED (optimised)")
        print(sep)
        cached = _run_cached(candidates, source_path)
        if cached:
            _print_result(cached)
            _print_comparison(inline, cached)

    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ranking engine performance.")
    parser.add_argument(
        "--input", type=Path, default=None, metavar="PATH",
        help="Candidate dataset (.jsonl or .json).  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Benchmark only the inline (no-cache) path.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N",
        help="Embedding batch size for inline path (default: 64).",
    )
    args = parser.parse_args()

    if args.input is not None:
        path = args.input
    elif DATA_PATH.exists():
        path = DATA_PATH
    else:
        print("[benchmark] No dataset found.  Pass --input <path>.")
        sys.exit(1)

    main(source_path=path, run_cache=not args.no_cache, batch_size=args.batch_size)
