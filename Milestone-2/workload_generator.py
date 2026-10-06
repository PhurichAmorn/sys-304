"""Generate synthetic prediction traffic for API and Grafana testing.

Example:
    python workload_generator.py --requests 300 --concurrency 30 --pattern mixed
"""

import argparse
import concurrent.futures
import csv
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

DISASTER_TEXTS = [
    "Wildfire forces mass evacuation across the city",
    "Massive earthquake destroys downtown buildings",
    "Flood waters rise after torrential rain",
    "Hurricane damages homes along the coast",
    "Rescue teams search for survivors after the landslide",
]
NON_DISASTER_TEXTS = [
    "Just had a great coffee with my friends",
    "The new restaurant downtown has excellent noodles",
    "Watching a movie at home tonight",
    "My team won the football match yesterday",
    "Taking a walk through the park this afternoon",
]


def make_inputs(count, pattern, seed):
    randomizer = random.Random(seed)
    corpus = DISASTER_TEXTS + NON_DISASTER_TEXTS

    if pattern == "repeated":
        return [DISASTER_TEXTS[0]] * count
    if pattern == "unique":
        return [f"{randomizer.choice(corpus)} request-{index}" for index in range(count)]
    return [randomizer.choice(corpus) for _ in range(count)]


def read_csv_inputs(input_path, limit, seed):
    with open(input_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'text' column")
        texts = [row["text"].strip() for row in reader if row.get("text", "").strip()]
    randomizer = random.Random(seed)
    randomizer.shuffle(texts)
    return texts if limit is None else texts[:limit]


def submit_request(base_url, text, timeout):
    payload = json.dumps({"text": text}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter()
    status_code = None
    prediction = None
    error = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.status
            body = json.loads(response.read())
            prediction = body.get("prediction")
    except urllib.error.HTTPError as exception:
        status_code = exception.code
        error = exception.read().decode(errors="replace")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exception:
        error = str(exception)

    return {
        "latency_ms": (time.perf_counter() - started_at) * 1000,
        "status_code": status_code,
        "prediction": prediction,
        "error": error,
    }


def percentile(values, percentile_rank):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * percentile_rank))
    return round(ordered[index], 2)


def run_workload(base_url, texts, concurrency, timeout):
    started_at = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda text: submit_request(base_url, text, timeout), texts))
    wall_seconds = time.perf_counter() - started_at

    latencies = [result["latency_ms"] for result in results]
    successful = [result for result in results if result["status_code"] == 200]
    status_counts = {}
    prediction_counts = {"0": 0, "1": 0}
    for result in results:
        key = str(result["status_code"]) if result["status_code"] is not None else "client_error"
        status_counts[key] = status_counts.get(key, 0) + 1
        if result["prediction"] in (0, 1):
            prediction_counts[str(result["prediction"])] += 1

    return {
        "requests": len(results),
        "concurrency": concurrency,
        "wall_seconds": round(wall_seconds, 3),
        "throughput_rps": round(len(results) / wall_seconds, 2) if wall_seconds else 0,
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "status_counts": status_counts,
        "prediction_counts": prediction_counts,
        "latency_ms": {
            "average": round(statistics.mean(latencies), 2) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": round(max(latencies), 2) if latencies else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="Backend base URL")
    default_test_data = Path(__file__).resolve().parent.parent / "Milestone-1" / "nlp-getting-started" / "test.csv"
    parser.add_argument(
        "--requests",
        type=int,
        help="Number of requests; defaults to all CSV rows or 300 synthetic inputs",
    )
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent workers")
    parser.add_argument(
        "--pattern",
        choices=("mixed", "unique", "repeated"),
        default=None,
        help="Traffic pattern; repeated traffic demonstrates Redis caching",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable traffic")
    parser.add_argument("--timeout", type=float, default=10, help="Request timeout in seconds")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="CSV containing text (defaults to Milestone-1/nlp-getting-started/test.csv)",
    )
    args = parser.parse_args()

    if args.requests is not None and args.requests < 1:
        parser.error("--requests must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.pattern and args.input_csv:
        parser.error("choose --pattern or --input-csv, not both")

    input_csv = args.input_csv or str(default_test_data)
    texts = (
        make_inputs(args.requests or 300, args.pattern, args.seed)
        if args.pattern
        else read_csv_inputs(input_csv, args.requests, args.seed)
    )
    if not texts:
        parser.error("--input-csv contains no non-empty text rows")
    source = f"synthetic {args.pattern} inputs" if args.pattern else input_csv
    print(f"Sending {len(texts)} requests from {source} to {args.url}...")
    summary = run_workload(args.url, texts, args.concurrency, args.timeout)
    if args.pattern:
        summary["pattern"] = args.pattern
    else:
        summary["source_csv"] = input_csv
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
