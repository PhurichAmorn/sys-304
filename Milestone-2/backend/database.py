import os

import psycopg

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/predictions",
)


CREATE_REQUEST_LOGS = """
CREATE TABLE IF NOT EXISTS request_logs (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    request_text TEXT,
    prediction SMALLINT,
    probability DOUBLE PRECISION,
    cache_hit BOOLEAN,
    status_code INTEGER NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    model_version TEXT,
    error_message TEXT
)
"""

# Every monitoring query filters or groups on time and path, and the retraining
# and drift readers take the most recent window. Without these indexes each
# dashboard refresh is a sequential scan that grows with the table.
CREATE_REQUEST_LOGS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS request_logs_created_at_idx ON request_logs (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS request_logs_path_created_at_idx ON request_logs (path, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS request_logs_probability_idx ON request_logs (probability) WHERE probability IS NOT NULL",
]

CREATE_RETRAINING_CYCLES = """
CREATE TABLE IF NOT EXISTS retraining_cycles (
    id BIGSERIAL PRIMARY KEY,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sample_count INTEGER NOT NULL,
    average_confidence DOUBLE PRECISION,
    confidence_threshold DOUBLE PRECISION NOT NULL,
    retraining_needed BOOLEAN NOT NULL,
    reason TEXT NOT NULL,
    accepted_labels INTEGER,
    candidate_f1 DOUBLE PRECISION,
    baseline_f1 DOUBLE PRECISION,
    candidate_recall DOUBLE PRECISION,
    baseline_recall DOUBLE PRECISION,
    deployed BOOLEAN NOT NULL DEFAULT FALSE,
    model_version TEXT,
    error_message TEXT
)
"""

CREATE_MODEL_VERSIONS = """
CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    promoted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source TEXT NOT NULL,
    model_version_before TEXT,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    rolled_back_at TIMESTAMPTZ,
    accuracy DOUBLE PRECISION,
    f1 DOUBLE PRECISION,
    disaster_recall DOUBLE PRECISION,
    accepted_labels INTEGER
)
"""

CREATE_DRIFT_CHECKS = """
CREATE TABLE IF NOT EXISTS drift_checks (
    id BIGSERIAL PRIMARY KEY,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_size INTEGER NOT NULL,
    psi_overall DOUBLE PRECISION,
    psi_verdict TEXT,
    js_divergence DOUBLE PRECISION,
    oov_rate DOUBLE PRECISION,
    length_ks_statistic DOUBLE PRECISION,
    length_ks_pvalue DOUBLE PRECISION,
    mean_confidence DOUBLE PRECISION,
    live_positive_rate DOUBLE PRECISION,
    reference_positive_rate DOUBLE PRECISION,
    top_features JSONB,
    error_message TEXT
)
"""

CREATE_DRIFT_CHECKS_INDEX = (
    "CREATE INDEX IF NOT EXISTS drift_checks_checked_at_idx ON drift_checks (checked_at DESC)"
)

# Volumes outlive the code that created them, so tables that already exist need
# explicit column migrations. Every statement is idempotent and runs on boot.
CREATE_MIGRATIONS = [
    "ALTER TABLE retraining_cycles ADD COLUMN IF NOT EXISTS model_version TEXT",
]


def initialize_database() -> None:
    try:
        with psycopg.connect(DATABASE_URL) as connection:
            connection.execute(CREATE_REQUEST_LOGS)
            for statement in CREATE_REQUEST_LOGS_INDEXES:
                connection.execute(statement)
            connection.execute(CREATE_RETRAINING_CYCLES)
            connection.execute(CREATE_MODEL_VERSIONS)
            connection.execute(CREATE_DRIFT_CHECKS)
            connection.execute(CREATE_DRIFT_CHECKS_INDEX)
            for statement in CREATE_MIGRATIONS:
                connection.execute(statement)
    except psycopg.Error as error:
        print(f"warning: database schema initialization failed: {error}", flush=True)


def set_active_version(version: str) -> None:
    """Mark ``version`` live again, used when a rollback restores an old version.

    ``log_model_version`` cannot express this: it inserts with
    ``ON CONFLICT DO NOTHING``, and a rolled-back-to version already has a row,
    so its ``active`` flag would never flip back to TRUE and the Grafana
    version panel would keep showing the reverted model as live.
    """
    try:
        with psycopg.connect(DATABASE_URL) as connection:
            connection.execute(
                "UPDATE model_versions SET active = FALSE WHERE active AND version <> %s",
                (version,),
            )
            connection.execute(
                "UPDATE model_versions SET active = TRUE WHERE version = %s", (version,)
            )
    except psycopg.Error as error:
        print(f"warning: could not record rollback in model_versions: {error}")


def log_model_version(
    *,
    version: str,
    source: str,
    model_version_before: str | None = None,
    accuracy: float | None = None,
    f1: float | None = None,
    disaster_recall: float | None = None,
    accepted_labels: int | None = None,
) -> None:
    """Record a promotion and deactivate the version it replaced."""
    try:
        with psycopg.connect(DATABASE_URL) as connection:
            if model_version_before:
                connection.execute(
                    "UPDATE model_versions SET active = FALSE WHERE version = %s",
                    (model_version_before,),
                )
            connection.execute(
                """
                INSERT INTO model_versions (
                    version, source, model_version_before, active,
                    accuracy, f1, disaster_recall, accepted_labels
                ) VALUES (%s, %s, %s, TRUE, %s, %s, %s, %s)
                ON CONFLICT (version) DO NOTHING
                """,
                (version, source, model_version_before, accuracy, f1, disaster_recall,
                 accepted_labels),
            )
    except psycopg.Error:
        pass


def log_drift_check(
    *,
    window_size: int,
    psi_overall: float | None = None,
    psi_verdict: str | None = None,
    js_divergence: float | None = None,
    oov_rate: float | None = None,
    length_ks_statistic: float | None = None,
    length_ks_pvalue: float | None = None,
    mean_confidence: float | None = None,
    live_positive_rate: float | None = None,
    reference_positive_rate: float | None = None,
    top_features: str | None = None,
    error_message: str | None = None,
    database_url: str | None = None,
) -> None:
    try:
        with psycopg.connect(database_url or DATABASE_URL) as connection:
            connection.execute(
                """
                INSERT INTO drift_checks (
                    window_size, psi_overall, psi_verdict, js_divergence,
                    oov_rate, length_ks_statistic, length_ks_pvalue,
                    mean_confidence, live_positive_rate, reference_positive_rate,
                    top_features, error_message
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (window_size, psi_overall, psi_verdict, js_divergence, oov_rate,
                 length_ks_statistic, length_ks_pvalue, mean_confidence,
                 live_positive_rate, reference_positive_rate, top_features,
                 error_message),
            )
    except psycopg.Error:
        pass


def log_retraining_cycle(
    *,
    sample_count: int,
    average_confidence: float | None,
    confidence_threshold: float,
    retraining_needed: bool,
    reason: str,
    accepted_labels: int | None = None,
    candidate_f1: float | None = None,
    baseline_f1: float | None = None,
    candidate_recall: float | None = None,
    baseline_recall: float | None = None,
    deployed: bool = False,
    model_version: str | None = None,
    error_message: str | None = None,
) -> None:
    try:
        with psycopg.connect(DATABASE_URL) as connection:
            connection.execute(
                """
                INSERT INTO retraining_cycles (
                    sample_count, average_confidence, confidence_threshold,
                    retraining_needed, reason, accepted_labels, candidate_f1,
                    baseline_f1, candidate_recall, baseline_recall, deployed,
                    model_version, error_message
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (sample_count, average_confidence, confidence_threshold,
                 retraining_needed, reason, accepted_labels, candidate_f1,
                 baseline_f1, candidate_recall, baseline_recall, deployed,
                 model_version, error_message),
            )
    except psycopg.Error:
        pass


def log_request(
    *,
    request_id: str,
    method: str,
    path: str,
    request_text: str | None,
    prediction: int | None,
    probability: float | None,
    cache_hit: bool | None,
    status_code: int,
    latency_ms: float,
    model_version: str,
    error_message: str | None,
) -> None:
    try:
        with psycopg.connect(DATABASE_URL) as connection:
            connection.execute(
                """
                INSERT INTO request_logs (
                    request_id, method, path, request_text, prediction,
                    probability, cache_hit, status_code, latency_ms,
                    model_version, error_message
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    request_id,
                    method,
                    path,
                    request_text,
                    prediction,
                    probability,
                    cache_hit,
                    status_code,
                    latency_ms,
                    model_version,
                    error_message,
                ),
            )
    except psycopg.Error:
        pass