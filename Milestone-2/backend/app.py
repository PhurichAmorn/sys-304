from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts["vectorizer"] = joblib.load("vectorizer.joblib")
    artifacts["model"] = joblib.load("model.joblib")
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


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    X = artifacts["vectorizer"].transform([request.text])
    proba = artifacts["model"].predict_proba(X)[0, 1]
    return PredictResponse(prediction=label_from_probability(proba), probability=float(proba))
