import hashlib
import json
import os
from contextlib import asynccontextmanager

import joblib
import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

artifacts = {}
CACHE_TTL_SECONDS = 3600


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts["vectorizer"] = joblib.load("vectorizer.joblib")
    artifacts["model"] = joblib.load("model.joblib")
    artifacts["cache"] = redis.from_url(
        os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


def label_from_probability(proba: float) -> int:
    return int(proba >= 0.5)


def cache_key(text: str) -> str:
    return "predict:" + hashlib.sha256(text.encode()).hexdigest()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    cache = artifacts["cache"]
    key = cache_key(request.text)

    try:
        cached = cache.get(key)
    except redis.RedisError:
        cached = None
    if cached is not None:
        return PredictResponse(**json.loads(cached))

    X = artifacts["vectorizer"].transform([request.text])
    proba = artifacts["model"].predict_proba(X)[0, 1]
    result = PredictResponse(prediction=label_from_probability(proba), probability=float(proba))

    try:
        cache.setex(key, CACHE_TTL_SECONDS, result.model_dump_json())
    except redis.RedisError:
        pass
    return result
