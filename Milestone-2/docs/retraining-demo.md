# Temporary Retraining Demo

This runbook demonstrates one automatic retraining attempt with a temporary confidence threshold of `0.9`. The helper restores the normal `0.65` threshold after the attempt, including on timeout or interruption.

## Prerequisites

From the repository's `Milestone-2` directory:

```bash
cd Milestone-2
```

Start Ollama and make sure the small local model is available:

```bash
ollama serve
ollama pull qwen2.5:0.5b
```

The retrainer uses these defaults:

```text
LLM_PROVIDER=ollama
LLM_BASE_URL=http://host.docker.internal:11434
LLM_MODEL=qwen2.5:0.5b
```

The LLM must return JSON containing only a binary `label` (`0` or `1`) and a confidence between `0` and `1`.

## Run The Demo

Start the stack with `./deploy.sh`, start Ollama, then run:

```bash
./demo-retrain.sh 150  # sends 150 synthetic requests to /predict
```

The script temporarily recreates the retrainer with threshold `0.9`, submits 150 fresh unique requests to `/predict`, waits for the cycle result, and recreates the retrainer at the normal `0.65` threshold. If the script times out, fails, or is interrupted, it still restores `0.65`. Set `RETRAINING_DEMO_TIMEOUT_SECONDS` to change its wait limit (default: 7200 seconds).

The generator needs at least 100 requests because that is the configured minimum batch size. The script sends requests through the same API used by the frontend; it does not invoke the training script directly.

## Watch The Automatic Cycle

```bash
docker compose logs -f retrainer
```

The expected flow is:

1. The monitor finds at least 100 new requests.
2. It calculates average confidence.
3. Average confidence is below `0.9`.
4. The monitor triggers `retrain_cycle.py`.
5. Ollama labels the new examples.
6. The candidate model is trained and exported to ONNX.
7. The ONNX candidate is evaluated against the baseline.
8. The candidate is promoted only if it passes the deployment gate.

## Verify The Database Result

```bash
docker compose exec postgres psql -P pager=off -U postgres -d predictions -c "
SELECT checked_at,
       sample_count,
       average_confidence,
       confidence_threshold,
       retraining_needed,
       reason,
       accepted_labels,
       deployed,
       model_version,
       LEFT(error_message, 240) AS error
FROM retraining_cycles
ORDER BY checked_at DESC
LIMIT 5;
"
```

A triggered cycle should show:

```text
retraining_needed = t
reason             = confidence_drop_and_minimum_batch_reached
average_confidence < 0.9
```

`deployed = false` is still a valid outcome if the candidate fails the quality gate. The gate requires the candidate ONNX F1 to improve, disaster recall not to regress, and conversion skew to remain within tolerance.

## Verify The Model Version

```bash
curl http://localhost:8000/model/info

docker compose exec retrainer python model_control.py list
```

If the candidate passes, a new version such as `v0002` appears and the API reloads it automatically. If it fails, the API continues serving the previous version.

## Verify Grafana

Open:

```text
http://localhost:3000/d/milestone-2-monitoring/milestone-2-request-monitoring
```

Login:

```text
username: admin
password: admin
```

Set the time range to **Last 1 hour** and refresh. Check:

- Retraining confidence checks
- Retraining history
- Live model version
- Model version history
- Requests by serving model version

## Rollback Demonstration

Only run rollback after a second model version exists:

```bash
docker compose exec retrainer \
  python model_control.py rollback \
  --reason "temporary demo rollback"
```

Confirm the API returns to the previous version:

```bash
curl http://localhost:8000/model/info
```

The script restores `0.65` automatically. Verify the running value with:

```bash
docker compose exec retrainer env | grep RETRAINING_CONFIDENCE_THRESHOLD
```

The normal threshold is `0.65`; `0.9` is used only during this demo attempt.

## Troubleshooting

Show the latest retrainer error:

```bash
docker compose logs --no-color --tail=150 retrainer
```

Test Ollama from inside the container:

```bash
docker compose exec retrainer python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:11434/api/tags', timeout=10).read().decode())"
```

If the monitor already consumed the batch and you need to repeat the demo, reset only its cursor:

```bash
docker compose exec retrainer sh -c 'rm -f /state/retraining-monitor.json'
docker compose restart retrainer
```

Do not run `docker compose down -v`; that deletes the PostgreSQL, Grafana, model-store, and retraining-state volumes.
