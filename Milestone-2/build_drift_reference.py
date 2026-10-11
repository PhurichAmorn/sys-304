"""Freeze the training distribution used as the drift baseline.

Run once before starting the stack:

    python build_drift_reference.py \
        --training-data ../Milestone-1/nlp-getting-started/train.csv \
        --output artifacts/drift_reference.joblib

The reference is deliberately independent of any served model. Drift must be
measured against the distribution the model was *trained* on, so that a
promotion cannot quietly redefine the baseline and hide ongoing drift.
"""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from drift import build_reference

DEFAULT_TRAINING_DATA = Path("../Milestone-1/nlp-getting-started/train.csv")
DEFAULT_OUTPUT = Path("artifacts/drift_reference.joblib")
MAX_FEATURES = 5000


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    texts = [row["text"] for row in rows if row.get("text", "").strip()]
    labels = [int(row["target"]) for row in rows if row.get("text", "").strip()]
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    texts, labels = load_rows(args.training_data)
    if not texts:
        raise SystemExit(f"no usable rows in {args.training_data}")

    # Same analyzer settings as the served model so both distributions live in
    # one comparable feature space.
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES, stop_words="english", ngram_range=(1, 2)
    )
    vectorizer.fit(texts)
    reference = build_reference(texts, labels, vectorizer)
    reference["built_at"] = datetime.now(UTC).isoformat()
    reference["source"] = str(args.training_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reference, args.output)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": reference["rows"],
                "features": len(reference["feature_names"]),
                "analyzer_vocabulary_size": reference["analyzer_vocabulary_size"],
                "reference_positive_rate": round(reference["positive_rate"], 6),
                "built_at": reference["built_at"],
                "source": reference["source"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
