from fastapi.testclient import TestClient

from app import app


def test_predict_returns_200_on_valid_input():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Massive earthquake destroys downtown buildings"})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_classifies_disaster_and_non_disaster_text_differently():
    with TestClient(app) as client:
        disaster = client.post("/predict", json={"text": "Wildfire forces mass evacuation, homes destroyed"})
        casual = client.post("/predict", json={"text": "Just had a great coffee with my friends"})
    assert disaster.status_code == 200
    assert casual.status_code == 200
    assert disaster.json()["prediction"] == 1
    assert casual.json()["prediction"] == 0


def test_predict_rejects_missing_text_field():
    with TestClient(app) as client:
        response = client.post("/predict", json={})
    assert response.status_code == 422
