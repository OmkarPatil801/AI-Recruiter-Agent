CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    description TEXT NOT NULL,
    requirements TEXT,
    location TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    location TEXT,
    resume_text TEXT,
    resume_filename TEXT,
    skills TEXT,
    experience_years INTEGER,
    education TEXT,
    linkedin_url TEXT,
    github_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    overall_score REAL NOT NULL DEFAULT 0.0,
    skills_score REAL NOT NULL DEFAULT 0.0,
    experience_score REAL NOT NULL DEFAULT 0.0,
    education_score REAL NOT NULL DEFAULT 0.0,
    location_score REAL NOT NULL DEFAULT 0.0,
    reasoning TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE,
    UNIQUE (job_id, candidate_id)
);

CREATE INDEX IF NOT EXISTS idx_scores_job_id ON scores(job_id);
CREATE INDEX IF NOT EXISTS idx_scores_candidate_id ON scores(candidate_id);
CREATE INDEX IF NOT EXISTS idx_scores_overall ON scores(overall_score DESC);

-- Ranking run metadata: one row per POST /rank/ call
CREATE TABLE IF NOT EXISTS ranking_runs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id                INTEGER,           -- nullable: links to jobs.id if supplied
    total_candidates      INTEGER NOT NULL,
    recommended_count     INTEGER NOT NULL,
    requires_review_count INTEGER NOT NULL,
    not_recommended_count INTEGER NOT NULL,
    average_match_score   REAL    NOT NULL,
    executed_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE SET NULL
);

-- Per-candidate results for each ranking run
CREATE TABLE IF NOT EXISTS ranking_results (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL,
    candidate_id         TEXT    NOT NULL,  -- external id from JSON (e.g. CAND_0000031)
    rank                 INTEGER,           -- filtered rank; NULL if candidate failed filter
    decision             TEXT    NOT NULL,  -- Recommended | Requires Review | Not Recommended
    candidate_category   TEXT    NOT NULL,  -- CORE_ML | ML_ADJ | ENG | NON_TECH
    confidence_score     REAL    NOT NULL,
    final_score          REAL    NOT NULL,
    semantic_score       REAL    NOT NULL,
    skill_score          REAL    NOT NULL,
    experience_score     REAL    NOT NULL,
    behavioural_score    REAL    NOT NULL,
    role_relevance_score REAL    NOT NULL,
    delivery_score       REAL    NOT NULL,
    filter_reason        TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES ranking_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ranking_runs_job_id          ON ranking_runs(job_id);
CREATE INDEX IF NOT EXISTS idx_ranking_results_run_id       ON ranking_results(run_id);
CREATE INDEX IF NOT EXISTS idx_ranking_results_candidate_id ON ranking_results(candidate_id);
CREATE INDEX IF NOT EXISTS idx_ranking_results_decision     ON ranking_results(decision);
