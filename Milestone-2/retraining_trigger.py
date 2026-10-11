"""Decide whether the continual-learning pipeline should start retraining.

This is Week 8, Part 1 only. It does not label data, train a model, or deploy
weights. It checks recent successful predictions for two trigger conditions:

- time: the configured interval has elapsed since the previous model update;
- confidence: average model confidence in the recent window is too low.

Examples:
    python retraining_trigger.py --interval-hours 24
    python retraining_trigger.py --confidence-threshold 0.65 --window-size 100
"""

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/predictions"


def parse_timestamp(value):
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp


def model_updated_at(model_root, fallback_model_path):
    pointer_path = Path(model_root) / "current.json"
    if pointer_path.exists():
        try:
            pointer = json.loads(pointer_path.read_text())
            promoted_at = pointer.get("promoted_at")
            if promoted_at:
                return parse_timestamp(promoted_at)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return datetime.fromtimestamp(Path(fallback_model_path).stat().st_mtime, tz=UTC)


def recent_confidence(database_url, window_size):
    query = """
        SELECT probability
        FROM request_logs
        WHERE path = '/predict'
          AND status_code = 200
          AND probability IS NOT NULL
        ORDER BY created_at DESC
        LIMIT %s
    """
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(query, (window_size,)).fetchall()
    confidences = [max(probability, 1 - probability) for (probability,) in rows]
    average = sum(confidences) / len(confidences) if confidences else None
    return average, len(confidences)


def evaluate_trigger(
    *,
    database_url,
    last_trained_at,
    interval_hours,
    confidence_threshold,
    window_size,
):
    now = datetime.now(UTC)
    average_confidence, sample_count = recent_confidence(database_url, window_size)
    time_due = now - last_trained_at >= timedelta(hours=interval_hours)
    confidence_drop = (
        average_confidence is not None and average_confidence < confidence_threshold
    )
    reasons = []
    if time_due:
        reasons.append("scheduled_interval_elapsed")
    if confidence_drop:
        reasons.append("recent_confidence_below_threshold")

    return {
        "retrain": bool(reasons),
        "reasons": reasons,
        "checked_at": now.isoformat(),
        "last_trained_at": last_trained_at.isoformat(),
        "interval_hours": interval_hours,
        "confidence_threshold": confidence_threshold,
        "window_size": window_size,
        "recent_sample_count": sample_count,
        "recent_average_confidence": (
            round(average_confidence, 6) if average_confidence is not None else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--last-trained-at",
        help="Previous model update as ISO-8601; defaults to the active model manifest",
    )
    parser.add_argument(
        "--model-path",
        default="backend/model.joblib",
        help="Legacy fallback model used when the versioned store is not initialized",
    )
    parser.add_argument(
        "--model-root",
        default=os.environ.get("MODEL_ROOT", "models"),
        help="Versioned model store containing current.json",
    )
    parser.add_argument("--interval-hours", type=float, default=24)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--window-size", type=int, default=100)
    args = parser.parse_args()

    if args.interval_hours <= 0:
        parser.error("--interval-hours must be positive")
    if not 0 < args.confidence_threshold < 1:
        parser.error("--confidence-threshold must be between 0 and 1")
    if args.window_size <= 0:
        parser.error("--window-size must be positive")

    last_trained_at = (
        parse_timestamp(args.last_trained_at)
        if args.last_trained_at
        else model_updated_at(args.model_root, args.model_path)
    )
    decision = evaluate_trigger(
        database_url=args.database_url,
        last_trained_at=last_trained_at,
        interval_hours=args.interval_hours,
        confidence_threshold=args.confidence_threshold,
        window_size=args.window_size,
    )
    print(json.dumps(decision, indent=2))
    raise SystemExit(0 if decision["retrain"] else 1)


if __name__ == "__main__":
    main()
