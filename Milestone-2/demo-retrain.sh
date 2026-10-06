#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

requests=${1:-100}
timeout=${RETRAINING_DEMO_TIMEOUT_SECONDS:-7200}
started_at=$(date -u '+%Y-%m-%d %H:%M:%S+00')

restore_threshold() {
  RETRAINING_CONFIDENCE_THRESHOLD=0.65 docker compose up -d --force-recreate retrainer >/dev/null
}
trap restore_threshold EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

RETRAINING_CONFIDENCE_THRESHOLD=0.9 docker compose up -d --build --force-recreate retrainer
python workload_generator.py --requests "$requests" --concurrency 10 --pattern unique

echo "Waiting for the retraining attempt; the retrainer will return to threshold 0.65 afterward."
for ((elapsed=0; elapsed<timeout; elapsed+=5)); do
  attempts=$(docker compose exec -T postgres psql -U postgres -d predictions -Atc \
    "SELECT COUNT(*) FROM retraining_cycles WHERE checked_at >= '$started_at' AND confidence_threshold = 0.9")
  if [[ "$attempts" -gt 0 ]]; then
    docker compose exec -T postgres psql -U postgres -d predictions -c \
      "SELECT checked_at, reason, accepted_labels, deployed, error_message FROM retraining_cycles WHERE checked_at >= '$started_at' AND confidence_threshold = 0.9 ORDER BY checked_at DESC LIMIT 1"
    exit 0
  fi
  sleep 5
done

echo "No retraining attempt appeared within ${timeout}s. The threshold will be restored." >&2
exit 1
