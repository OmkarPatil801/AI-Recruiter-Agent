"""
test.py — v4
------------
End-to-end smoke test with evaluation mode.

Usage:
    python -m ranking.test            # standard top-20 leaderboard
    python -m ranking.test --eval     # full evaluation: before vs after filter
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ranking.rank import FILTER_CONFIG, run_evaluation, run_ranking


# ---------------------------------------------------------------------------
# Job description
# ---------------------------------------------------------------------------

JOB_DESCRIPTION = """
Senior Machine Learning Engineer - Conversational AI Platform

We are building a next-generation conversational AI platform and are looking
for a Senior ML Engineer to lead model development and deployment.

Responsibilities:
- Design, train, and fine-tune large language models (LLMs) and smaller
  task-specific models for dialogue, intent classification, and entity extraction.
- Build and maintain scalable ML pipelines using Python, Apache Spark, and
  Airflow for data ingestion, feature engineering, and model training.
- Deploy models to production via REST APIs (FastAPI / Flask), containerised
  with Docker and orchestrated on Kubernetes.
- Work with vector databases (Pinecone, Milvus, Weaviate) and embedding models
  for semantic search and retrieval-augmented generation (RAG).
- Collaborate closely with data engineers to ensure training data quality,
  schema consistency, and feature store reliability.
- Evaluate model performance using standard NLP metrics (BLEU, ROUGE, F1)
  and design A/B testing frameworks for production rollout.

Requirements:
- 4+ years of experience in machine learning engineering or applied NLP.
- Strong Python skills; proficiency in PyTorch or TensorFlow.
- Hands-on experience fine-tuning transformer models (BERT, GPT-style, T5,
  Llama, Mistral) using LoRA, QLoRA, or full fine-tuning.
- Experience with MLOps tooling: MLflow, Weights & Biases, or equivalent.
- Solid understanding of NLP fundamentals: tokenisation, embeddings,
  attention mechanisms, RLHF.
- Cloud platform experience (AWS, GCP, or Azure).
- Bonus: experience with speech recognition (ASR), TTS, or multimodal models.
- Bonus: familiarity with BentoML, vLLM, or TGI for model serving.
""".strip()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

W = 165  # total table width (wider in v4 to fit Del + Cat columns)


def _t(text: str, width: int) -> str:
    return textwrap.shorten(str(text), width=width, placeholder="...")


def _bar(score: float, width: int = 12) -> str:
    filled = round(score * width)
    return "#" * filled + "-" * (width - filled)


def _sep(char: str = "-") -> str:
    return char * W


def _header_row() -> str:
    return (
        f"{'#':>3}  {'ID':<15} {'Name':<17} {'Title':<22} "
        f"{'YoE':>4}  {'Sem':>6}  {'Skl':>5}  {'Rol':>5}  "
        f"{'Del':>5}  {'Exp':>5}  {'Beh':>5}  "
        f"{'Final':>6}  {'Cat':<8}  {'Bar':^12}  Matched skills"
    )


def _data_row(c: dict) -> str:
    matched = ", ".join(c.get("required_skills_found", [])[:4]) or "-"
    cat = c.get("candidate_category", "")
    return (
        f"{c['rank']:>3}  {c['candidate_id']:<15} {_t(c['name'], 17):<17} "
        f"{_t(c['current_title'], 22):<22} "
        f"{c['years_of_experience']:>4.1f}  "
        f"{c['cosine_score']:>6.4f}  "
        f"{c['skill_match_score']:>5.3f}  "
        f"{c['role_relevance_score']:>5.3f}  "
        f"{c['delivery_score']:>5.3f}  "
        f"{c['experience_score']:>5.3f}  "
        f"{c['behavioural_score']:>5.3f}  "
        f"{c['final_score']:>6.4f}  "
        f"{cat:<8}  "
        f"{_bar(c['final_score']):^12}  "
        f"{_t(matched, 32)}"
    )


def _print_leaderboard(title: str, results: list[dict]) -> None:
    print("\n" + _sep("="))
    print(f"  {title}")
    print(_sep("="))
    print(_header_row())
    print(_sep())
    for c in results:
        print(_data_row(c))
    print(_sep())


def _print_legend() -> None:
    print("\n--- SCORE COMPONENTS (v4) ---")
    print("  Sem   = Semantic cosine similarity                    weight 0.35")
    print("  Skl   = Skill match (required skills covered)         weight 0.20")
    print("  Rol   = Role relevance (domain / career tier)         weight 0.15")
    print("  Del   = Delivery evidence (production vs learning)    weight 0.15")
    print("  Exp   = Experience (normalised to required years)     weight 0.10")
    print("  Beh   = Behavioural boost (engagement signals)        weight 0.05")
    print("  Final = 0.35*Sem + 0.20*Skl + 0.15*Rol + 0.15*Del + 0.10*Exp + 0.05*Beh")
    print("  Cat   = CORE_ML | ML_ADJ | ENG | NON_TECH  (from current job title tier)")


def _print_profiles(results: list[dict], count: int = 3) -> None:
    for c in results[:count]:
        print("\n" + "=" * 80)
        print(f"  #{c['rank']} {c['name']}  [{c['candidate_id']}]")
        print("=" * 80)
        print(f"  Title         : {c['current_title']} @ {c['current_company']}")
        print(f"  Location      : {c['location']}, {c['country']}")
        print(f"  Category      : {c.get('candidate_category', '')}  |  "
              f"Open to work: {c['open_to_work']}")
        print(f"  Experience    : {c['years_of_experience']} yrs")
        print(f"  Career titles : {' > '.join(c.get('career_titles', [])[:4])}")
        print(f"  Skills        : {', '.join(c['skills_raw'])}")
        print(f"  Req. matched  : {', '.join(c['required_skills_found']) or 'none'}")
        print(
            f"  Scores        : sem={c['cosine_score']}  skl={c['skill_match_score']}  "
            f"rol={c['role_relevance_score']}  del={c['delivery_score']}  "
            f"exp={c['experience_score']}  beh={c['behavioural_score']}  "
            f"final={c['final_score']}"
        )
        print(f"\n  Summary:")
        for line in textwrap.wrap(c["summary"][:480], width=74):
            print(f"    {line}")


# ---------------------------------------------------------------------------
# Category distribution helper
# ---------------------------------------------------------------------------

def _print_category_dist(label: str, results: list[dict]) -> None:
    from collections import Counter
    counts = Counter(c.get("candidate_category", "UNKNOWN") for c in results)
    order = ["CORE_ML", "ML_ADJ", "ENG", "NON_TECH", "UNKNOWN"]
    parts = [f"{cat}: {counts.get(cat, 0)}" for cat in order if counts.get(cat, 0) > 0]
    print(f"  {label:<35} {' | '.join(parts)}")


# ---------------------------------------------------------------------------
# Standard mode
# ---------------------------------------------------------------------------

def run_standard() -> None:
    print("=" * 80)
    print("  AI RECRUITER - CANDIDATE RANKING ENGINE  (v4)")
    print("=" * 80)
    print("\nJob: Senior Machine Learning Engineer - Conversational AI Platform")
    print(
        f"Filter: skill_match >= {FILTER_CONFIG['MIN_SKILL_MATCH']:.2f}  |  "
        f"experience >= {FILTER_CONFIG['MIN_EXPERIENCE_SCORE']:.2f}\n"
    )

    results = run_ranking(job_description=JOB_DESCRIPTION, top_n=20)

    _print_leaderboard(f"TOP {len(results)} CANDIDATES (post-filter)", results)
    _print_legend()
    _print_profiles(results, count=min(3, len(results)))


# ---------------------------------------------------------------------------
# Evaluation mode  (--eval)
# ---------------------------------------------------------------------------

def run_eval_mode() -> None:
    print("=" * 80)
    print("  AI RECRUITER - EVALUATION MODE  (v4)")
    print("=" * 80)
    print("\nJob: Senior Machine Learning Engineer - Conversational AI Platform\n")

    data = run_evaluation(job_description=JOB_DESCRIPTION, top_n=20)

    # ---- filter summary ---------------------------------------------------
    cfg = data["filter_config"]
    print("\n" + _sep("="))
    print("  FILTER SUMMARY")
    print(_sep("="))
    print(f"  Total candidates loaded : {data['total_loaded']}")
    print(f"  Passed filter           : {data['total_passing']}")
    print(f"  Rejected by filter      : {data['total_rejected']}")
    print(f"  Thresholds              : skill_match >= {cfg['MIN_SKILL_MATCH']:.2f}  |  "
          f"experience >= {cfg['MIN_EXPERIENCE_SCORE']:.2f}")

    # ---- category distribution --------------------------------------------
    print("\n  Category distribution:")
    _print_category_dist("Before filter (all 50):", data["unfiltered_top_n"] + data["rejected_candidates"])
    _print_category_dist("After filter  (passing):", data["filtered_top_n"])

    # ---- rejected candidates table ----------------------------------------
    print("\n" + _sep("="))
    print("  CANDIDATES REMOVED BY FILTER")
    print(_sep("="))
    rej_hdr = (
        f"{'ID':<15} {'Name':<18} {'Title':<24} {'Cat':<8} "
        f"{'YoE':>4}  {'Skl':>5}  {'Del':>5}  {'Exp':>5}  Reason"
    )
    print(rej_hdr)
    print(_sep())
    for c in data["rejected_candidates"]:
        print(
            f"{c['candidate_id']:<15} {_t(c['name'], 18):<18} "
            f"{_t(c['current_title'], 24):<24} "
            f"{c.get('candidate_category', ''):<8} "
            f"{c['years_of_experience']:>4.1f}  "
            f"{c['skill_match_score']:>5.3f}  "
            f"{c['delivery_score']:>5.3f}  "
            f"{c['experience_score']:>5.3f}  "
            f"{c.get('filter_reason', '')}"
        )
    print(_sep())

    # ---- before / after leaderboards --------------------------------------
    _print_leaderboard(
        "TOP 20 BEFORE FILTER  (all candidates ranked by v4 final_score)",
        data["unfiltered_top_n"],
    )

    _print_leaderboard(
        "TOP 20 AFTER FILTER  (candidates passing skill & experience gates)",
        data["filtered_top_n"],
    )

    _print_legend()

    # ---- improvement summary -----------------------------------------------
    print("\n" + _sep("="))
    print("  RANKING IMPROVEMENT SUMMARY  (v3 -> v4)")
    print(_sep("="))

    before_ids = [c["candidate_id"] for c in data["unfiltered_top_n"]]
    after_ids  = [c["candidate_id"] for c in data["filtered_top_n"]]
    rejected_reasons = {
        c["candidate_id"]: c.get("filter_reason", "") for c in data["rejected_candidates"]
    }

    print("\n  Candidates that entered top-20 after filtering (new entries):")
    new_entries = [c for c in data["filtered_top_n"] if c["candidate_id"] not in before_ids]
    if new_entries:
        for c in new_entries:
            print(f"    {c['rank']:>3}. {c['candidate_id']}  {c['name']}  "
                  f"({c['current_title']})  [{c.get('candidate_category','')}]")
    else:
        print("    None - post-filter shortlist is a subset of pre-filter top-20.")

    print("\n  Candidates removed from top-20 by filter:")
    removed = [c for c in data["unfiltered_top_n"] if c["candidate_id"] not in after_ids]
    if removed:
        for c in removed:
            reason = rejected_reasons.get(c["candidate_id"], "passed filter")
            print(
                f"    prev #{c['rank']:>2}. {c['candidate_id']}  "
                f"{c['name']}  ({c['current_title']})  "
                f"[{c.get('candidate_category','')}]  |  {reason}"
            )
    else:
        print("    None.")

    # ---- top profiles from filtered results --------------------------------
    if data["filtered_top_n"]:
        _print_profiles(data["filtered_top_n"], count=min(3, len(data["filtered_top_n"])))
    else:
        print("\n  No candidates passed the filter for this job description.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--eval" in sys.argv:
        run_eval_mode()
    else:
        run_standard()
