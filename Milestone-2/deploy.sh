#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v docker &>/dev/null; then
  echo "Docker is required: https://docs.docker.com/get-docker/" >&2
  exit 1
fi

docker compose up -d --build

echo "Waiting for backend to become healthy..."
for _ in $(seq 1 30); do
  status=$(docker compose ps --format '{{.Health}}' backend 2>/dev/null || true)
  [ "$status" = "healthy" ] && break
  sleep 2
done

if [ "$status" != "healthy" ]; then
  echo "Backend did not become healthy in time. Logs:" >&2
  docker compose logs backend >&2
  exit 1
fi

echo "Up: frontend http://localhost:8080  backend http://localhost:8000/docs"
