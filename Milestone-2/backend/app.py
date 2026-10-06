import asyncio
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from uuid import uuid4

import redis
from database import initialize_database, log_model_version, log_request
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from model_store import BACKEND_DIR, MODEL_ROOT, ModelRegistry, positive_probability
from pydantic import BaseModel

IMAGE_FALLBACK = Path(BACKEND_DIR) / "model" / "model_fp32.onnx"
if not IMAGE_FALLBACK.exists():
    IMAGE_FALLBACK = Path(BACKEND_DIR).parent / "model" / "model_fp32.onnx"
MODEL_POLL_SECONDS = float(os.environ.get("MODEL_POLL_SECONDS", "5"))

artifacts = {}
CACHE_TTL_SECONDS = 3600


async def watch_for_model_changes(registry: ModelRegistry):
    """Swap in newly promoted models without restarting or dropping requests.

    Polling the shared pointer keeps the API decoupled from the retrainer: if the
    retrainer is mid-retrain or has died, the API keeps serving the last good
    model instead of failing requests.
    """
    while True:
        await asyncio.sleep(MODEL_POLL_SECONDS)
        try:
            if await asyncio.to_thread(registry.poll):
                print(f"model promoted: now serving {registry.info().get('version')}", flush=True)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - the watcher must never die
            print(f"model reload failed, keeping current version: {error}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry(MODEL_ROOT)
    registry.start(fallback=IMAGE_FALLBACK)
    artifacts["registry"] = registry
    initialize_database()
    model = registry.info()
    metrics = model.get("metrics", {})
    log_model_version(
        version=model["version"],
        source=model.get("source") or "api-startup",
        model_version_before=model.get("previous_version"),
        accuracy=metrics.get("accuracy"),
        f1=metrics.get("f1"),
        disaster_recall=metrics.get("disaster_recall"),
        accepted_labels=metrics.get("accepted_labels"),
    )
    artifacts["cache"] = redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    watcher = asyncio.create_task(watch_for_model_changes(registry))
    try:
        yield
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid4())
    started_at = time.perf_counter()
    response = None
    error_message = None
    try:
        response = await call_next(request)
        return response
    except Exception as error:
        error_message = str(error)
        raise
    finally:
        log_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            request_text=getattr(request.state, "request_text", None),
            prediction=getattr(request.state, "prediction", None),
            probability=getattr(request.state, "probability", None),
            cache_hit=getattr(request.state, "cache_hit", None),
            status_code=response.status_code if response else 500,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            model_version=getattr(request.state, "model_version", None),
            error_message=error_message,
        )


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    """Live model version, promotion provenance, and rollback target."""
    registry = artifacts.get("registry")
    if registry is None:
        return {"status": "not_loaded"}
    return registry.info()


def label_from_probability(proba: float) -> int:
    return int(proba >= 0.5)


def cache_key(text: str, model_version: str) -> str:
    """Namespace cached predictions by model version.

    Without the version in the key, a promoted model would keep serving the
    previous model's cached predictions until the TTL expired.
    """
    digest = hashlib.sha256(f"{model_version}\x00{text}".encode()).hexdigest()
    return f"predict:{model_version}:{digest}"


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, http_request: Request) -> PredictResponse:
    http_request.state.request_text = request.text
    registry = artifacts["registry"]
    session = registry.session()
    version = registry.info().get("version")
    http_request.state.model_version = version
    cache = artifacts["cache"]
    key = cache_key(request.text, version)

    try:
        cached = cache.get(key)
    except redis.RedisError:
        cached = None
    if cached is not None:
        result = PredictResponse(**json.loads(cached))
        http_request.state.prediction = result.prediction
        http_request.state.probability = result.probability
        http_request.state.cache_hit = True
        return result

    proba = positive_probability(session, request.text)
    result = PredictResponse(prediction=label_from_probability(proba), probability=proba)
    http_request.state.prediction = result.prediction
    http_request.state.probability = result.probability
    http_request.state.cache_hit = False

    try:
        cache.setex(key, CACHE_TTL_SECONDS, result.model_dump_json())
    except redis.RedisError:
        pass
    return result
