"""Label unlabeled disaster tweets with the local Laya decision model.

Laya runs locally and returns a typed choice with probabilities. This script
uses neutral choices instead of Laya's ``noul`` mode because the model card
warns that ``noul`` can sometimes follow the option words rather than the
state text.

Example:
    python label_with_laya.py \
        --input ../Milestone-1/nlp-getting-started/test.csv \
        --output data/test_laya_labels.csv
"""

import argparse
import csv
import json
from pathlib import Path

QUESTION_KEY = "disaster"
QUESTIONS = {
    QUESTION_KEY: {
        "type": "choice",
        "instructions": "Does this tweet describe a real disaster event?",
        "criteria": {
            "A": "yes: an earthquake, flood, wildfire, storm, accident, damage, rescue, or other disaster event",
            "B": "no: ordinary conversation, opinion, entertainment, food, sports, or unrelated content",
        },
    }
}


def read_rows(input_path):
    with Path(input_path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'text' column")
        return list(reader), reader.fieldnames


def answer_from_result(result):
    answer = result.get("answers", {}).get(QUESTION_KEY, {})
    choice = answer.get("choice")
    confidence = answer.get("confidence")
    if choice not in {"A", "B"}:
        raise ValueError(f"Laya returned an invalid choice: {choice!r}")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError(f"Laya returned an invalid confidence: {confidence!r}")
    return {
        "laya_label": 1 if choice == "A" else 0,
        "laya_choice": choice,
        "laya_confidence": float(confidence),
        "laya_model": "convaiinnovations/laya",
        "label_source": "laya_local",
    }


def label_rows(rows, agent, min_confidence):
    labeled = []
    for row_number, row in enumerate(rows, start=1):
        text = row.get("text", "").strip()
        output = dict(row)
        output["laya_label"] = ""
        output["laya_choice"] = ""
        output["laya_confidence"] = ""
        output["laya_model"] = "convaiinnovations/laya"
        output["label_source"] = "laya_local"
        output["needs_review"] = "true"
        output["label_error"] = ""

        if not text:
            output["label_error"] = "empty text"
            labeled.append(output)
            continue

        try:
            result = agent.predict({"text": text}, QUESTIONS)
            label = answer_from_result(result)
            output.update({key: str(value) for key, value in label.items()})
            output["needs_review"] = str(label["laya_confidence"] < min_confidence).lower()
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            output["label_error"] = f"row {row_number}: {error}"

        labeled.append(output)
    return labeled


def write_rows(output_path, rows, fieldnames):
    extra_fields = [
        "laya_label",
        "laya_choice",
        "laya_confidence",
        "laya_model",
        "label_source",
        "needs_review",
        "label_error",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Unlabeled CSV containing a text column")
    parser.add_argument("--output", required=True, help="Output CSV for Laya labels")
    parser.add_argument(
        "--limit",
        type=int,
        help="Only label the first N rows; useful for a quick local smoke test",
    )
    parser.add_argument(
        "--model",
        default="convaiinnovations/laya",
        help="Hugging Face Laya checkpoint or local model path",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.75,
        help="Below this confidence, keep the label but mark needs_review=true",
    )
    args = parser.parse_args()
    if not 0 < args.min_confidence <= 1:
        parser.error("--min-confidence must be greater than 0 and at most 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    rows, fieldnames = read_rows(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    try:
        import laya
    except ImportError as error:
        raise SystemExit(
            "Laya is not installed. Run: pip install -r requirements-laya.txt"
        ) from error
    print(f"Loading local Laya model: {args.model}")
    agent = laya.load(args.model)
    labeled_rows = label_rows(rows, agent, args.min_confidence)
    write_rows(args.output, labeled_rows, fieldnames)

    summary = {
        "input_rows": len(rows),
        "labeled_rows": sum(bool(row["laya_label"]) for row in labeled_rows),
        "review_rows": sum(row["needs_review"] == "true" for row in labeled_rows),
        "error_rows": sum(bool(row["label_error"]) for row in labeled_rows),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
