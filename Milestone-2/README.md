# Milestone 2: API, Frontend, Containerization, CI/CD

Serves the Milestone 1 disaster-tweet classifier (TF-IDF + Logistic
Regression, exported from the notebook) behind a REST API with a web UI.

## Structure

```
Milestone-2/
  backend/
    app.py              FastAPI app: /predict, /health
    vectorizer.joblib    exported from Milestone-1's notebook
    model.joblib
    requirements.txt
    requirements-dev.txt      pytest, httpx, ruff
    requirements-optimize.txt skl2onnx, onnx, onnxruntime, psutil, pandas
    optimize/
      export_onnx.py    exports the pipeline to ONNX, then quantizes it to INT8
      benchmark.py      naive (joblib) vs ONNX FP32 vs ONNX INT8: latency, memory, size, accuracy
      model_fp32.onnx   generated
      model_int8.onnx   generated
      results.json      generated
    tests/               unit + integration tests
    Dockerfile
  frontend/
    index.html           form UI
    app.js                fetch/formatting logic (testable, no DOM)
    tests/                node:test integration tests
    Dockerfile
  docker-compose.yml
  deploy.sh              one-command launch
```

## Run it

```bash
./deploy.sh
```

Builds both images, starts them, waits for the backend health check, then
prints the URLs:
- Frontend: http://localhost:8080
- Backend docs: http://localhost:8000/docs

Or manually: `docker compose up --build`.

## API

`POST /predict`
```json
{"text": "wildfire forces mass evacuation"}
```
→
```json
{"prediction": 1, "probability": 0.85}
```

`GET /health` → `{"status": "ok"}` (used by the Docker health check).

## Tests

Backend (`cd backend && pytest`): unit tests on the classification
threshold, integration tests hitting `/predict` through FastAPI's
`TestClient` (200 on valid input, correct disaster/non-disaster split,
422 on missing field).

Frontend (`cd frontend && node --test`): integration tests on the
fetch/format logic with a mocked `fetch` (success path, non-200 error,
label formatting).

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`: installs
dependencies, lints the backend with `ruff`, runs both test suites.

## Model-level optimization

The deployed model is a `TfidfVectorizer` (5000 features) + `LogisticRegression`
(sklearn) — a few hundred KB total, not a deep net. Straight FP32→FP16/INT8
weight casting and distillation don't apply to a linear model like this, so
we used two techniques that do:

1. **Model format optimization (ONNX).** `optimize/export_onnx.py` converts
   the fitted `Pipeline(vectorizer, model)` to ONNX via `skl2onnx`, so
   inference runs through `onnxruntime` instead of sklearn's Python path.
2. **Quantization (dynamic INT8, post-training).** The same script then runs
   `onnxruntime.quantization.quantize_dynamic` on the ONNX graph, converting
   the classifier's weight matrix from FP32 to INT8.

Run it:
```bash
pip install -r requirements-optimize.txt
python optimize/export_onnx.py     # writes model_fp32.onnx, model_int8.onnx
python optimize/benchmark.py --n 500
```

`benchmark.py` sends the same batch of requests (500 held-out tweets, the
same 80/20 stratified split Milestone-1 used) through all three model forms —
naive joblib, ONNX FP32, ONNX INT8 — one request at a time, each in its own
subprocess so memory readings aren't cross-contaminated. It reports average
latency, throughput, process RSS after model load, on-disk size, and
accuracy.

### Results (n=500 requests)

| mode      | avg latency (ms) | throughput (qps) | RSS after load (MB) | size (KB) | accuracy |
|-----------|-------------------|-------------------|-----------------------|-----------|----------|
| naive (joblib)  | 0.117 | 8,538  | 173.4 | 227.5 | 0.856 |
| ONNX FP32       | 0.061 | 16,484 | 183.3 | 160.1 | 0.858 |
| ONNX INT8       | 0.057 | 17,507 | 183.1 | 160.4 | 0.858 |

Full numbers in `optimize/results.json`.

**Takeaways:**
- The ONNX export is the bigger win here: ~2x higher throughput than the
  sklearn/joblib path, no accuracy loss.
- INT8 dynamic quantization gives a further small latency edge with no
  accuracy change, but barely moves file size or RSS at this scale: most of
  the ~160KB ONNX graph is the frozen TF-IDF vocabulary/IDF table (strings +
  float64 array for 5000 terms), not the classifier's weight matrix — dynamic
  quantization only touches the `Gemm` op's weights, which is a small slice
  of the total. On a model with a large dense weight matrix (not a 5000×1
  logistic regression), INT8 quantization would show a much bigger size/RSS
  delta.
- RSS after load is slightly *higher* for the ONNX modes than naive — that's
  fixed overhead from importing `onnxruntime` itself (numpy, protobuf, the
  runtime's own allocator warmup), not the model. It dominates at this model
  size; it wouldn't for a model large enough to actually stress memory.

## Demo


https://github.com/user-attachments/assets/f627b451-f586-49e9-860b-9dafcece695e


