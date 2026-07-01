"""
scorer.py — v4
--------------
Six scoring signals blended into a final ranking score.

  1. Semantic (cosine)     — embedding similarity between JD and candidate text.
                             Captures holistic fit but is susceptible to boilerplate
                             "AI-curious" language; reduced weight vs v2.

  2. Skill match           — fraction of required JD skills the candidate covers.
                             Hard vocabulary signal: directly penalises candidates
                             who mention AI topics but list no ML-specific skills.

  3. Role relevance        — keyword/category tier lookup over current title, career
                             history, and headline.  Primary domain gate.

  4. Delivery evidence     — NEW in v4.  Distinguishes "built and deployed production
                             ML systems" from "interested in AI / taking online
                             courses."  Scans summary + career description prose for
                             delivery language vs aspiration language.

  5. Experience            — normalised years vs minimum required.  Reduced weight
                             because the filter stage already enforces the hard gate,
                             and most candidates passing the filter score 1.0 here.

  6. Behavioural           — engagement signals from redrob_signals.  Pure tiebreaker;
                             reduced to 0.05 so high engagement cannot rescue a
                             domain-wrong candidate.

Formula (v4):
    final = 0.35 * semantic
          + 0.20 * skill_match
          + 0.15 * role_relevance
          + 0.15 * delivery
          + 0.10 * experience
          + 0.05 * behavioural

Weight rationale
----------------
  Semantic  0.35  Reduced from 0.40. Boilerplate AI-curiosity text inflates this for
                  non-practitioners; delivery now absorbs some of its purpose.

  Skill     0.20  Unchanged. Direct vocabulary verification — the cleanest hard signal.

  Role      0.15  Unchanged. Domain gate that collapses PM / Accountant scores.

  Delivery  0.15  New. Equal weight to role — "shipped in production" is as
                  important as the job title when distinguishing senior practitioners
                  from career-changers with buzzword-heavy CVs.

  Exp       0.10  Reduced from 0.15. Once candidates clear the filter (>=75% of
                  required years) this component differentiates very little.

  Beh       0.05  Reduced from 0.10. High open-to-work / interview rate should
                  not compensate for weak domain fit.
"""

from __future__ import annotations

import re
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[scorer] Loading model '{MODEL_NAME}' ...")
        _model = SentenceTransformer(MODEL_NAME)
        print("[scorer] Model ready.")
    return _model


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode texts -> L2-normalised float32 array (N, D). Dot-product == cosine."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Skill match
# ---------------------------------------------------------------------------

# Alias -> canonical name map.  Add new terms here when the ML ecosystem
# introduces new frameworks or techniques.  Longest aliases are matched first
# (sorted by -len in extract_required_skills) so multi-word entries take
# priority over single-word substrings.
_TECH_NORMALISATION: dict[str, str] = {
    # --- Core ML frameworks ---
    "pytorch": "pytorch", "torch": "pytorch",
    "pytorch lightning": "pytorch", "lightning ai": "pytorch",
    "tensorflow": "tensorflow", "tf": "tensorflow",
    "scikit-learn": "scikit-learn", "sklearn": "scikit-learn",
    "hugging face": "hugging face", "huggingface": "hugging face",
    "hugging face transformers": "hugging face",
    "transformers": "hugging face",      # HF Transformers library (plural)

    # --- Fine-tuning techniques ---
    "fine-tuning": "fine-tuning", "fine tuning": "fine-tuning",
    "finetuning": "fine-tuning", "fine-tuning llms": "fine-tuning",
    "lora": "lora", "qlora": "lora",
    "peft": "peft",                      # Parameter-Efficient Fine-Tuning library
    "parameter-efficient fine-tuning": "peft",
    "parameter efficient fine-tuning": "peft",
    "sft": "fine-tuning",               # Supervised Fine-Tuning -> fine-tuning canonical
    "supervised fine-tuning": "fine-tuning",
    "rlhf": "rlhf",
    "reinforcement learning from human feedback": "rlhf",
    "dpo": "dpo",
    "direct preference optimization": "dpo",

    # --- LLM identifiers ---
    "llm": "llm", "large language model": "llm", "llms": "llm",
    "bert": "bert", "gpt": "gpt", "t5": "t5",
    "llama": "llama", "mistral": "mistral",

    # --- MLOps / experiment tracking ---
    "mlflow": "mlflow",
    "weights & biases": "weights & biases", "wandb": "weights & biases",
    "weights and biases": "weights & biases",
    "kubeflow": "kubeflow",
    "dvc": "dvc", "data version control": "dvc",

    # --- Model serving ---
    "bentoml": "bentoml", "vllm": "vllm",
    "torchserve": "torchserve", "torch serve": "torchserve",
    "tgi": "tgi", "text generation inference": "tgi",
    "ray serve": "ray", "ray tune": "ray",

    # --- RAG / orchestration ---
    "rag": "rag", "retrieval-augmented generation": "rag",
    "haystack": "haystack",
    "langchain": "langchain", "lang chain": "langchain",
    "langgraph": "langgraph", "lang graph": "langgraph",
    "llamaindex": "llamaindex", "llama index": "llamaindex", "llama_index": "llamaindex",
    "dspy": "dspy",

    # --- Vector databases ---
    "pinecone": "pinecone", "milvus": "milvus", "weaviate": "weaviate",
    "faiss": "faiss",
    "chroma": "chroma", "chromadb": "chroma",

    # --- NLP ---
    "nlp": "nlp", "natural language processing": "nlp",
    "embeddings": "embeddings", "sentence transformers": "embeddings",
    "speech recognition": "speech recognition", "asr": "speech recognition",
    "tts": "tts",

    # --- Infrastructure ---
    "docker": "docker",
    "kubernetes": "kubernetes", "k8s": "kubernetes",
    "fastapi": "fastapi", "flask": "flask",
    "spark": "spark", "pyspark": "spark", "apache spark": "spark",
    "airflow": "airflow", "apache airflow": "airflow",
    "mlops": "mlops",

    # --- Cloud ---
    "aws": "aws", "gcp": "gcp", "azure": "azure",

    # --- Language ---
    "python": "python",
}


def _normalise_skill(skill: str) -> str:
    return _TECH_NORMALISATION.get(skill.lower().strip(), skill.lower().strip())


def extract_required_skills(job_description: str) -> set[str]:
    """Scan JD for known tech terms.  Returns a set of normalised skill names."""
    jd_lower = job_description.lower()
    found: set[str] = set()
    for alias, canonical in sorted(_TECH_NORMALISATION.items(), key=lambda x: -len(x[0])):
        if alias in jd_lower:
            found.add(canonical)
    return found


def skill_match_score(required_skills: set[str], candidate_skills: list[str]) -> float:
    """
    Fraction of required skills covered by the candidate.

    One-sided (job-perspective precision): extra candidate skills don't help
    or hurt.  Returns 0.0 when required_skills is empty.
    """
    if not required_skills:
        return 0.0
    candidate_normalised = {_normalise_skill(s) for s in candidate_skills}
    overlap = required_skills & candidate_normalised
    return len(overlap) / len(required_skills)


# ---------------------------------------------------------------------------
# Experience score
# ---------------------------------------------------------------------------

def extract_required_years(job_description: str) -> float:
    """Parse minimum years of experience from the JD.  Returns 0.0 if absent."""
    patterns = [
        r"(\d+)\+?\s*years?\s+of\s+experience",
        r"minimum\s+(\d+)\s+years?",
        r"at\s+least\s+(\d+)\s+years?",
    ]
    for pattern in patterns:
        match = re.search(pattern, job_description, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return 0.0


def experience_score(candidate_years: float, required_years: float) -> float:
    """
    Normalised experience score in [0, 1].

    Candidates meeting or exceeding the minimum get 1.0.
    Below-minimum candidates receive proportional partial credit.
    The hard gate lives in rank.py's filter stage.
    """
    if required_years <= 0:
        return 1.0
    if candidate_years >= required_years:
        return 1.0
    return float(np.clip(candidate_years / required_years, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Role relevance score
# ---------------------------------------------------------------------------

# Four tiers of domain relevance.  Each tier is a list of keyword fragments.
# A title/headline matches a tier if any keyword is found as a substring
# (case-insensitive).  Tiers are checked from highest to lowest; first match
# wins.  Adding a new role to a tier is a one-line change here.

_ROLE_TIERS: list[tuple[float, list[str]]] = [
    # score 1.0 — core ML/AI practitioners
    (1.0, [
        "machine learning", "ml engineer", "ai engineer", "nlp engineer",
        "nlp researcher", "deep learning", "data scientist", "applied scientist",
        "research scientist", "ml researcher", "ai researcher",
        "recommendation", "computer vision", "speech recognition",
        "conversational ai", "generative ai", "llm engineer",
        "natural language", "multimodal",
    ]),
    # score 0.70 — ML-adjacent technical roles
    (0.70, [
        "data engineer", "analytics engineer", "ml platform", "mlops",
        "ai platform", "platform engineer", "ml infra", "ai infra",
    ]),
    # score 0.40 — technical generalists (software, cloud, DevOps)
    (0.40, [
        "software engineer", "software developer", "backend engineer",
        "backend developer", "cloud engineer", "devops", "site reliability",
        "sre", "full stack", "fullstack", "java developer", "python developer",
        "mobile engineer", "android", "ios developer",
        "net developer", "dotnet",
    ]),
    # score 0.20 — technical-adjacent (data analysts, PMs with tech exposure)
    (0.20, [
        "data analyst", "business analyst", "product manager", "technical pm",
        "solutions architect", "cloud architect", "security engineer",
    ]),
    # score 0.05 — non-technical (domain-wrong)
    (0.05, [
        "hr manager", "human resources", "recruiter", "talent",
        "accountant", "finance manager", "financial analyst", "auditor",
        "graphic designer", "ux designer", "ui designer", "visual designer",
        "operations manager", "operations", "project manager",
        "marketing manager", "marketing", "content writer", "copywriter",
        "sales", "account manager", "customer success", "customer support",
        "civil engineer", "mechanical engineer", "electrical engineer",
        "architect",
    ]),
]


def _score_single_title(title: str) -> float:
    """Return the relevance score for a single job title string."""
    title_lower = title.lower()
    for score, keywords in _ROLE_TIERS:
        if any(kw in title_lower for kw in keywords):
            return score
    return 0.05


def role_relevance_score(
    current_title: str,
    career_titles: list[str],
    headline: str = "",
) -> float:
    """
    Compute a domain-relevance score in [0, 1].

    Weights:
        Current title   60%  — most recent role is the strongest signal
        Career history  30%  — last 3 roles, recency-weighted (1.0, 0.6, 0.3)
        Headline        10%  — explicit self-description of identity

    Recency weighting ensures someone who recently pivoted to ML gets partial
    credit, while a long-term PM who took one ML course doesn't get a free pass.
    """
    title_score = _score_single_title(current_title) * 0.60

    recency_weights = [1.0, 0.6, 0.3]
    career_score = 0.0
    weight_sum = 0.0
    for i, ctitle in enumerate(career_titles[:3]):
        w = recency_weights[i]
        career_score += _score_single_title(ctitle) * w
        weight_sum += w
    career_component = (career_score / weight_sum * 0.30) if weight_sum > 0 else 0.0

    headline_score = _score_single_title(headline) * 0.10

    return float(np.clip(title_score + career_component + headline_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Delivery evidence score  (NEW in v4)
# ---------------------------------------------------------------------------

# Phrases that appear in descriptions of practitioners who have BUILT and
# SHIPPED ML systems in production environments.
_DELIVERY_STRONG: list[str] = [
    "in production",
    "to production",
    "production system",
    "production ml",
    "production-grade",
    "production environment",
    "deployed",
    "shipped",
    "launched",
    "led the team",
    "led a team",
    "led migration",
    "led the migration",
    "fine-tuned",
    "fine-tuning",
    "trained models",
    "training models",
    "designed and implemented",
    "built and deployed",
    "implemented retrieval",
    "migrated our",
    "migrated the",
    "at scale",
    "real-time",
    "end-to-end",
]

# Phrases that appear in descriptions of people who are INTERESTED IN or
# LEARNING about AI/ML but have not shipped production ML systems.
_DELIVERY_WEAK: list[str] = [
    "interested in ai",
    "interested in ml",
    "curious about how",
    "curious about ai",
    "learning ml",
    "learning ai",
    "experimenting with",
    "taking online courses",
    "taking courses",
    "online course",
    "ai enthusiast",
    "excited about how",
    "excited about ai",
    "augment my work",
    "side project",
    "kaggle",
    "self-learner",
    "self-directed",
    "played with",
    "chatgpt",
    "ai tools could",
    "small rag",
    "keeping up with",
    "been curious",
    "been keeping up",
]


def delivery_evidence_score(summary: str, career_text: str) -> float:
    """
    Score [0, 1] that distinguishes "built production ML systems" from
    "interested in AI."

    Operates on summary + career prose (NOT the skills list — skill names
    alone don't indicate whether work was delivered in production).

    Each strong-evidence phrase hit adds 0.20.
    Each weak-evidence phrase hit subtracts 0.10.
    Result is clipped to [0.0, 1.0].

    A candidate with 5 strong signals scores 1.0.
    A candidate with 0 strong and 5 weak signals scores 0.0.
    A candidate with 2 strong and 3 weak signals scores 0.10.
    """
    text = (summary + " " + career_text).lower()
    strong = sum(1 for phrase in _DELIVERY_STRONG if phrase in text)
    weak   = sum(1 for phrase in _DELIVERY_WEAK   if phrase in text)
    return float(np.clip(strong * 0.20 - weak * 0.10, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Candidate category  (NEW in v4)
# ---------------------------------------------------------------------------

def candidate_category(current_title: str) -> str:
    """
    Map a job title to a recruiter-readable category.

    CORE_ML      — ML Engineers, NLP Engineers, Data Scientists, AI researchers
    ML_ADJ       — Data Engineers, Analytics Engineers, MLOps Engineers
    ENG          — Software/Backend/Cloud/DevOps Engineers
    NON_TECH     — Project Managers, HR, Analysts, Sales, non-software roles
    """
    score = _score_single_title(current_title)
    if score >= 1.0:
        return "CORE_ML"
    if score >= 0.70:
        return "ML_ADJ"
    if score >= 0.40:
        return "ENG"
    # Fallback: generic "developer" or "programmer" titles that fall through
    # the tier keyword list (e.g. ".NET Developer") still belong in ENG.
    t = current_title.lower()
    if any(w in t for w in ("developer", "programmer")):
        return "ENG"
    return "NON_TECH"


# ---------------------------------------------------------------------------
# Behavioural boost
# ---------------------------------------------------------------------------

def _behavioural_boost(candidate: dict) -> float:
    """
    Engagement signals from the platform.  Weight reduced to 0.05 in v4
    because high engagement cannot compensate for domain mismatch.
    """
    score = (
        0.30 * float(candidate.get("open_to_work", False))
        + 0.30 * float(candidate.get("interview_completion_rate", 0.0))
        + 0.20 * float(candidate.get("github_score_norm", 0.0))
        + 0.20 * float(candidate.get("profile_completeness", 0.0))
    )
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Formula weights  (v4)
# ---------------------------------------------------------------------------
SEMANTIC_WEIGHT       = 0.35
SKILL_MATCH_WEIGHT    = 0.20
ROLE_RELEVANCE_WEIGHT = 0.15
DELIVERY_WEIGHT       = 0.15
EXPERIENCE_WEIGHT     = 0.10
BEHAVIOURAL_WEIGHT    = 0.05


# ---------------------------------------------------------------------------
# Shared per-candidate scoring kernel
# ---------------------------------------------------------------------------

def _score_one(
    candidate: dict,
    cosine: float,
    required_skills: set[str],
    required_years: float,
) -> dict:
    """
    Apply all non-embedding scoring signals to one candidate and return
    a new enriched dict.  Called by both score_candidates() and
    score_candidates_fast() to guarantee identical output.
    """
    sm  = skill_match_score(required_skills, candidate.get("skills_raw", []))
    exp = experience_score(candidate.get("years_of_experience", 0), required_years)
    beh = _behavioural_boost(candidate)
    rol = role_relevance_score(
        current_title=candidate.get("current_title", ""),
        career_titles=candidate.get("career_titles", []),
        headline=candidate.get("headline", ""),
    )
    dlv = delivery_evidence_score(
        summary=candidate.get("summary", ""),
        career_text=candidate.get("career_text", ""),
    )
    cat = candidate_category(candidate.get("current_title", ""))

    final = (
        SEMANTIC_WEIGHT         * cosine
        + SKILL_MATCH_WEIGHT    * sm
        + ROLE_RELEVANCE_WEIGHT * rol
        + DELIVERY_WEIGHT       * dlv
        + EXPERIENCE_WEIGHT     * exp
        + BEHAVIOURAL_WEIGHT    * beh
    )

    candidate_norm = {_normalise_skill(s) for s in candidate.get("skills_raw", [])}
    matched = sorted(required_skills & candidate_norm)

    return {
        **candidate,
        "cosine_score":          round(cosine, 4),
        "skill_match_score":     round(sm,     4),
        "role_relevance_score":  round(rol,    4),
        "delivery_score":        round(dlv,    4),
        "experience_score":      round(exp,    4),
        "behavioural_score":     round(beh,    4),
        "final_score":           round(final,  4),
        "required_skills_found": matched,
        "candidate_category":    cat,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidates(
    job_description: str,
    candidates: list[dict],
    batch_size: int = 32,
) -> list[dict]:
    """
    Score all candidates against the job description.

    Embeds the job description AND all candidate texts in one batched pass.
    Use score_candidates_fast() instead when cached candidate embeddings
    are available — it embeds only the job description (1 text vs N+1).

    Returns the input list with these keys added to each dict:
        cosine_score, skill_match_score, role_relevance_score,
        delivery_score, experience_score, behavioural_score,
        final_score, required_skills_found, candidate_category

    The list is NOT sorted — rank.py handles ordering and filtering.
    """
    required_skills = extract_required_skills(job_description)
    required_years  = extract_required_years(job_description)
    print(f"[scorer] Required skills ({len(required_skills)}): {sorted(required_skills)}")
    print(f"[scorer] Required experience: {required_years} yrs")

    all_texts = [job_description] + [c["candidate_text"] for c in candidates]
    print(f"[scorer] Embedding {len(all_texts)} texts (1 job + {len(candidates)} candidates) ...")
    all_embeddings = embed_texts(all_texts, batch_size=batch_size)

    job_embedding        = all_embeddings[0]
    candidate_embeddings = all_embeddings[1:]
    cosine_scores: np.ndarray = candidate_embeddings @ job_embedding

    return [
        _score_one(c, float(cosine_scores[i]), required_skills, required_years)
        for i, c in enumerate(candidates)
    ]


def score_candidates_fast(
    job_description: str,
    candidates: list[dict],
    candidate_embeddings: np.ndarray,
) -> list[dict]:
    """
    Score candidates using pre-loaded candidate embeddings from the disk cache.

    Only embeds the job description (1 text instead of N+1).  All other
    scoring signals are identical to score_candidates().

    Parameters
    ----------
    candidate_embeddings : float32 (N, D) array loaded by ranking.cache.load_cache().
                           Must be in the same order as `candidates`.

    Returns the same enriched dict structure as score_candidates().
    """
    required_skills = extract_required_skills(job_description)
    required_years  = extract_required_years(job_description)
    print(f"[scorer] Required skills ({len(required_skills)}): {sorted(required_skills)}")
    print(f"[scorer] Required experience: {required_years} yrs")
    print(f"[scorer] Embedding job description only (candidate embeddings from cache) ...")

    job_embedding = embed_texts([job_description], batch_size=1, show_progress=False)[0]
    cosine_scores: np.ndarray = candidate_embeddings @ job_embedding

    return [
        _score_one(c, float(cosine_scores[i]), required_skills, required_years)
        for i, c in enumerate(candidates)
    ]
