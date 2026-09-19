"""System-level benchmark: Phase 2 (naive, single uvicorn process, no cache)
vs Phase 3 (optimized, gunicorn + 4 uvicorn workers, Redis exact-match cache).

Hits a running backend's /predict over HTTP with concurrent requests and
measures wall-clock latency/throughput as an external client would see it
(unlike optimize/benchmark.py, which measures in-process model latency).

Two traffic patterns:
  - unique:   every request has different text -> always a cache miss.
              Isolates the concurrency win (multi-worker vs single process).
  - repeated: every request has the same text -> cache hit after the first.
              Isolates the caching win.

Run from Milestone-2/ against already-running containers:
    python benchmark_system.py --naive-url http://localhost:8001 \
                                --optimized-url http://localhost:8000
"""

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request

N_REQUESTS = 300
CONCURRENCY = 30


def predict(base_url, text):
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{base_url}/predict", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    return time.perf_counter() - t0


def load_test(base_url, texts):
    latencies = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        for lat in pool.map(lambda t: predict(base_url, t), texts):
            latencies.append(lat)
    wall_s = time.perf_counter() - t0

    latencies_ms = sorted(l * 1000 for l in latencies)
    return {
        "n_requests": len(texts),
        "concurrency": CONCURRENCY,
        "wall_s": round(wall_s, 3),
        "throughput_rps": round(len(texts) / wall_s, 1),
        "avg_latency_ms": round(statistics.mean(latencies_ms), 2),
        "p50_latency_ms": round(statistics.median(latencies_ms), 2),
        "p95_latency_ms": round(latencies_ms[int(len(latencies_ms) * 0.95) - 1], 2),
    }


def container_mem_mb(container_name):
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_name],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # e.g. "45.2MiB / 7.667GiB" -> take the used side, normalize to MB
    used = out.split("/")[0].strip()
    if used.endswith("MiB"):
        return round(float(used[:-3]), 1)
    if used.endswith("GiB"):
        return round(float(used[:-3]) * 1024, 1)
    if used.endswith("KiB"):
        return round(float(used[:-3]) / 1024, 1)
    return used


def warm_up(base_url, n=5):
    for _ in range(n):
        try:
            predict(base_url, "warmup request")
        except urllib.error.URLError:
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive-url", default="http://localhost:8001")
    parser.add_argument("--optimized-url", default="http://localhost:8000")
    parser.add_argument("--naive-container", default="m2-naive")
    parser.add_argument("--optimized-container", default="milestone-2-backend-1")
    args = parser.parse_args()

    unique_texts = [f"wildfire forces mass evacuation request number {i}" for i in range(N_REQUESTS)]
    repeated_texts = ["wildfire forces mass evacuation"] * N_REQUESTS

    print("warming up...")
    warm_up(args.naive_url)
    warm_up(args.optimized_url)

    results = {}
    print("naive, unique requests (cache would always miss)...")
    results["naive_unique"] = load_test(args.naive_url, unique_texts)
    print("optimized, unique requests (cache always misses, isolates concurrency win)...")
    results["optimized_unique"] = load_test(args.optimized_url, unique_texts)
    print("optimized, repeated requests (cache hits after first, isolates caching win)...")
    results["optimized_repeated"] = load_test(args.optimized_url, repeated_texts)

    results["naive_mem_mb"] = container_mem_mb(args.naive_container)
    results["optimized_mem_mb"] = container_mem_mb(args.optimized_container)

    print("\n" + json.dumps(results, indent=2))
    with open("benchmark_system_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote benchmark_system_results.json")


if __name__ == "__main__":
    main()
