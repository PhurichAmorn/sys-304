"""Benchmark: sends a batch of requests through the naive (joblib) model and
the optimized (ONNX FP32 / ONNX INT8) models, comparing inference time,
memory footprint, on-disk size, and accuracy.

Each mode runs in its own subprocess so memory measurements aren't polluted
by other modes' imports/models sharing the same process.

Run from Milestone-2/backend/ (after optimize/export_onnx.py has been run):
    python optimize/benchmark.py [--n N]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd
import psutil
from sklearn.model_selection import train_test_split

MODES = ["naive", "onnx_fp32", "onnx_int8"]
TRAIN_CSV = "../../Milestone-1/nlp-getting-started/train.csv"


def load_eval_split(n):
    df = pd.read_csv(TRAIN_CSV)
    # Same 80/20 stratified split used in Milestone-1's notebook (seed 42),
    # so this is a fixed, reproducible batch of "requests".
    _, X_val, _, y_val = train_test_split(
        df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )
    return X_val.tolist()[:n], y_val.tolist()[:n]


def run_worker(mode, texts_path):
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", mode, "--texts", texts_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{mode} worker failed:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def worker_main(mode, texts_path):
    with open(texts_path) as f:
        payload = json.load(f)
    texts, labels = payload["texts"], payload["labels"]
    process = psutil.Process(os.getpid())

    t0 = time.perf_counter()
    if mode == "naive":
        import joblib

        vectorizer = joblib.load("vectorizer.joblib")
        model = joblib.load("model.joblib")
        size_bytes = os.path.getsize("vectorizer.joblib") + os.path.getsize("model.joblib")

        def predict_one(text):
            X = vectorizer.transform([text])
            proba = model.predict_proba(X)[0, 1]
            return int(proba >= 0.5)

    else:
        import numpy as np
        import onnxruntime as rt

        onnx_path = f"optimize/model_{'fp32' if mode == 'onnx_fp32' else 'int8'}.onnx"
        sess = rt.InferenceSession(onnx_path)
        size_bytes = os.path.getsize(onnx_path)

        def predict_one(text):
            out = sess.run(None, {"text_input": np.array([[text]], dtype=object)})
            return int(out[0][0])

    load_time_s = time.perf_counter() - t0
    rss_after_load_mb = process.memory_info().rss / (1024 * 1024)

    preds = []
    t0 = time.perf_counter()
    for text in texts:
        preds.append(predict_one(text))
    total_time_s = time.perf_counter() - t0

    accuracy = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)

    print(
        json.dumps(
            {
                "mode": mode,
                "n_requests": len(texts),
                "load_time_s": round(load_time_s, 4),
                "total_time_s": round(total_time_s, 4),
                "avg_latency_ms": round(total_time_s / len(texts) * 1000, 4),
                "throughput_qps": round(len(texts) / total_time_s, 1),
                "rss_after_load_mb": round(rss_after_load_mb, 2),
                "size_kb": round(size_bytes / 1024, 1),
                "accuracy": round(accuracy, 4),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of requests to send")
    parser.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--texts", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker, args.texts)
        return

    for path in ("optimize/model_fp32.onnx", "optimize/model_int8.onnx"):
        if not os.path.exists(path):
            sys.exit(f"missing {path} — run optimize/export_onnx.py first")

    texts, labels = load_eval_split(args.n)
    texts_path = "optimize/_bench_input.json"
    with open(texts_path, "w") as f:
        json.dump({"texts": texts, "labels": labels}, f)

    results = [run_worker(mode, texts_path) for mode in MODES]
    os.remove(texts_path)

    header = ["mode", "n_requests", "avg_latency_ms", "throughput_qps", "rss_after_load_mb", "size_kb", "accuracy"]
    col_w = {h: max(len(h), *(len(str(r[h])) for r in results)) for h in header}
    print(" | ".join(h.ljust(col_w[h]) for h in header))
    print("-|-".join("-" * col_w[h] for h in header))
    for r in results:
        print(" | ".join(str(r[h]).ljust(col_w[h]) for h in header))

    with open("optimize/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote optimize/results.json")


if __name__ == "__main__":
    main()
