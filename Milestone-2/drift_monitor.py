"""Periodically compare logged live traffic against the frozen training baseline.

Runs as its own Compose service so drift is measured continuously, independent of
whether the confidence trigger has fired. Each check reads the most recent
window of successful predictions, computes the drift report, and appends it to
``drift_checks`` for the Grafana dashboard.
"""

import json
import os
import sys
import time
from pathlib import Path

import joblib
import psycopg

PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_DIR / "backend"
for candidate in (PROJECT_DIR, BACKEND_DIR):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from database import initialize_database, log_drift_check

from drift import compute_drift

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:15432/predictions",
)
REFERENCE_PATH = Path(
    os.environ.get("DRIFT_REFERENCE_PATH", PROJECT_DIR / "artifacts/drift_reference.joblib")
)

# PSI is estimated from the window, so small windows are noisy enough to
# manufacture false positives. Measured on in-domain test.csv, a 50-row window
# peaked at 0.139 ("moderate") while windows of 100 and above stayed below
# 0.064. The default floor is the smallest size that behaves.
MINIMUM_ROWS = 100


def recent_window(database_url, window_size):
    """Most recent successful predictions with text, oldest first."""
    query = """
        SELECT request_text, probability
        FROM request_logs
        WHERE path = '/predict'
          AND status_code = 200
          AND request_text IS NOT NULL
          AND request_text <> ''
          AND probability IS NOT NULL
        ORDER BY created_at DESC, id DESC
        LIMIT %s
    """
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(query, (window_size,)).fetchall()
    return [(row[0], row[1]) for row in reversed(rows)]


def run_check(reference, window, database_url):
    texts = [row[0] for row in window]
    probabilities = [row[1] for row in window]
    report = compute_drift(reference, texts, probabilities)
    log_drift_check(
        database_url=database_url,
        window_size=report["window_size"],
        psi_overall=report["psi_overall"],
        psi_verdict=report["psi_verdict"],
        oov_rate=report["oov_rate"],
        length_ks_statistic=report["length_ks_statistic"],
        length_ks_pvalue=report["length_ks_pvalue"],
        mean_confidence=report["mean_confidence"],
        live_positive_rate=report["live_positive_rate"],
        reference_positive_rate=report["reference_positive_rate"],
        top_features=json.dumps(report["top_features"]),
    )
    return report


def main():
    poll_seconds = float(os.environ.get("DRIFT_POLL_SECONDS", "30"))
    window_size = int(os.environ.get("DRIFT_WINDOW_SIZE", "500"))
    min_rows = int(os.environ.get("DRIFT_MINIMUM_ROWS", str(MINIMUM_ROWS)))

    if not REFERENCE_PATH.exists():
        raise SystemExit(
            f"drift reference missing at {REFERENCE_PATH}; run: python build_drift_reference.py"
        )

    reference = joblib.load(REFERENCE_PATH)
    initialize_database()
    print(
        f"drift monitor started: reference={REFERENCE_PATH} rows={reference['rows']} "
        f"window={window_size}",
        flush=True,
    )

    while True:
        try:
            window = recent_window(DATABASE_URL, window_size)
            if len(window) >= min_rows:
                report = run_check(reference, window, DATABASE_URL)
                print(
                    f"drift psi={report['psi_overall']} ({report['psi_verdict']}) "
                    f"oov={report['oov_rate']} n={report['window_size']}",
                    flush=True,
                )
        except (OSError, ValueError, psycopg.Error) as error:
            print(f"drift check failed: {error}", flush=True)
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
