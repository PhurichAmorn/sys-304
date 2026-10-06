"""Autonomously monitor PostgreSQL and invoke retraining cycles.

The monitor uses request IDs as a simulated event stream. It does not require
wall-clock time: once enough new requests arrive, it evaluates the confidence
of that batch and invokes retrain_cycle.py when the threshold is crossed.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_DIR / "backend"
if BACKEND_DIR.is_dir():
    sys.path.insert(0, str(BACKEND_DIR))

import psycopg
from database import initialize_database, log_retraining_cycle

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:15432/predictions",
)
STATE_FILE = Path(os.environ.get("RETRAINING_STATE_FILE", "/state/retraining-monitor.json"))


def load_state():
    if not STATE_FILE.exists():
        return {"last_processed_id": 0}
    return json.loads(STATE_FILE.read_text())


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def latest_batch(last_processed_id, batch_size):
    query = """
        SELECT id, request_text, probability, model_version
        FROM request_logs
        WHERE id > %s
          AND path = '/predict'
          AND status_code = 200
          AND request_text IS NOT NULL
          AND request_text <> ''
          AND probability IS NOT NULL
        ORDER BY id ASC
        LIMIT %s
    """
    with psycopg.connect(DATABASE_URL) as connection:
        return connection.execute(query, (last_processed_id, batch_size)).fetchall()


def run_cycle(batch_size, minimum_rows, confidence_threshold, label_confidence,
              model_root, auto_deploy, rows):
    command = [
        sys.executable,
        str(PROJECT_DIR / "retrain_cycle.py"),
        "--database-url",
        DATABASE_URL,
        "--training-data",
        str(PROJECT_DIR / "train.csv"),
        "--model-root",
        model_root,
        "--batch-size",
        str(batch_size),
        "--minimum-rows",
        str(minimum_rows),
        "--confidence-threshold",
        str(confidence_threshold),
        "--minimum-label-confidence",
        str(label_confidence),
        "--request-ids",
        ",".join(str(row[0]) for row in rows),
    ]
    if auto_deploy:
        command.append("--deploy")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        print(f"retraining cycle failed: {completed.stderr.strip()}", flush=True)
        return {"error_message": completed.stderr.strip()}
    print(completed.stdout, flush=True)
    return json.loads(completed.stdout)


def main():
    poll_seconds = float(os.environ.get("RETRAINING_POLL_SECONDS", "10"))
    batch_size = int(os.environ.get("RETRAINING_BATCH_SIZE", "100"))
    minimum_rows = int(os.environ.get("RETRAINING_MINIMUM_ROWS", "100"))
    confidence_threshold = float(os.environ.get("RETRAINING_CONFIDENCE_THRESHOLD", "0.65"))
    label_confidence = float(os.environ.get("RETRAINING_LABEL_CONFIDENCE", "0.75"))
    model_root = os.environ.get("MODEL_ROOT", "/models")
    auto_deploy = os.environ.get("RETRAINING_AUTO_DEPLOY", "true").lower() == "true"
    state = load_state()
    initialize_database()
    print(
        f"retraining monitor started (auto_deploy={auto_deploy}, model_root={model_root})",
        flush=True,
    )

    while True:
        try:
            rows = latest_batch(state["last_processed_id"], batch_size)
            if len(rows) >= minimum_rows:
                batch_confidence = sum(
                    max(float(row[2]), 1 - float(row[2])) for row in rows
                ) / len(rows)
                report = run_cycle(
                    batch_size,
                    minimum_rows,
                    confidence_threshold,
                    label_confidence,
                    model_root,
                    auto_deploy,
                    rows,
                )
                if not report.get("error_message"):
                    state["last_processed_id"] = rows[-1][0]
                    save_state(state)
                trigger = report.get("trigger", {})
                baseline = report.get("baseline", {})
                candidate = report.get("candidate", {})
                average_confidence = trigger.get("average_confidence", batch_confidence)
                retraining_needed = trigger.get(
                    "triggered", batch_confidence < confidence_threshold
                )
                log_retraining_cycle(
                    sample_count=trigger.get("sample_count", len(rows)),
                    average_confidence=average_confidence,
                    confidence_threshold=confidence_threshold,
                    retraining_needed=retraining_needed,
                    reason=trigger.get("reason", "cycle_error"),
                    accepted_labels=report.get("accepted_llm_rows"),
                    candidate_f1=candidate.get("f1"),
                    baseline_f1=baseline.get("f1"),
                    candidate_recall=candidate.get("disaster_recall"),
                    baseline_recall=baseline.get("disaster_recall"),
                    deployed=report.get("deployed", False),
                    model_version=report.get("model_version"),
                    error_message=report.get("error_message"),
                )
        except (OSError, ValueError, psycopg.Error) as error:
            print(f"monitor check failed: {error}", flush=True)
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
