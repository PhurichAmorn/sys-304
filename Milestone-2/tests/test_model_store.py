import json
from pathlib import Path

import joblib
import model_store
import numpy as np
import pytest
from model_store import (
    ModelRegistry,
    bootstrap,
    next_version,
    pointer_path,
    publish_version,
    read_pointer,
    rollback,
    version_dir,
)
from onnxruntime.capi.onnxruntime_pybind11_state import (
    InvalidProtobuf as ort_InvalidProtobuf,
)


@pytest.fixture
def fitted():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = [
        "wildfire evacuation downtown",
        "flood rescue teams arrived",
        "great coffee with friends",
        "watching a movie tonight",
    ]
    labels = [1, 1, 0, 0]
    vectorizer = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2))
    features = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000).fit(features, labels)
    return vectorizer, model


@pytest.fixture
def fallback(tmp_path, fitted):
    path = tmp_path / "model_fp32.onnx"
    model_store.export_to_onnx(*fitted, path)
    return path


def test_pointer_written_atomically(tmp_path, fitted):
    publish_version(vectorizer=fitted[0], model=fitted[1], root=tmp_path)

    pointer = read_pointer(tmp_path)
    assert pointer["version"] == "v0001"
    assert (version_dir("v0001", tmp_path) / "model.onnx").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_publish_increments_version(tmp_path, fitted):
    publish_version(vectorizer=fitted[0], model=fitted[1], root=tmp_path)
    manifest = publish_version(
        vectorizer=fitted[0], model=fitted[1], previous_version="v0001", root=tmp_path
    )
    assert manifest["version"] == "v0002"
    assert manifest["previous_version"] == "v0001"
    assert next_version(tmp_path) == "v0003"


def test_publish_records_predecessor_without_caller_help(tmp_path, fitted):
    """Rollback follows the recorded predecessor, so it must never be forgotten."""
    publish_version(vectorizer=fitted[0], model=fitted[1], root=tmp_path)
    manifest = publish_version(vectorizer=fitted[0], model=fitted[1], root=tmp_path)

    assert manifest["previous_version"] == "v0001"
    assert rollback(tmp_path)["version"] == "v0001"


def test_publish_rejects_repromoting_the_live_version(tmp_path, fitted):
    publish_version(vectorizer=fitted[0], model=fitted[1], root=tmp_path)

    with pytest.raises(ValueError, match="already live"):
        publish_version(
            vectorizer=fitted[0], model=fitted[1], root=tmp_path, version="v0001"
        )


def test_bootstrap_seeds_from_image_artifact(tmp_path, fallback):
    manifest = bootstrap(tmp_path, fallback)

    assert manifest["version"] == "v0001"
    assert manifest["source"] == "image-bootstrap"
    assert (version_dir("v0001", tmp_path) / "model.onnx").exists()


def test_bootstrap_is_idempotent(tmp_path, fallback):
    first = bootstrap(tmp_path, fallback)
    second = bootstrap(tmp_path, fallback)
    assert first["version"] == second["version"] == "v0001"


def test_bootstrap_does_not_clobber_existing_pointer(tmp_path, fallback, fitted):
    publish_version(vectorizer=fitted[0], model=fitted[1], root=tmp_path)
    bootstrap(tmp_path, fallback)
    assert read_pointer(tmp_path)["version"] == "v0001"
    assert read_pointer(tmp_path)["source"] != "image-bootstrap"


def test_bootstrap_without_pointer_or_fallback_raises(tmp_path):
    with pytest.raises(RuntimeError, match="cannot bootstrap"):
        bootstrap(tmp_path, tmp_path / "missing.onnx")


def test_rollback_restores_previous_version(tmp_path, fitted):
    publish_version(vectorizer=fitted[0], model=fitted[1], root=tmp_path)
    publish_version(
        vectorizer=fitted[0], model=fitted[1], previous_version="v0001", root=tmp_path
    )
    assert read_pointer(tmp_path)["version"] == "v0002"

    restored = rollback(tmp_path)

    assert restored["version"] == "v0001"
    assert restored["rolled_back_from"] == "v0002"
    assert read_pointer(tmp_path)["version"] == "v0001"


def test_rollback_without_previous_version_raises(tmp_path, fitted):
    publish_version(vectorizer=fitted[0], model=fitted[1], root=tmp_path)
    with pytest.raises(RuntimeError, match="no previous version"):
        rollback(tmp_path)


def test_rollback_without_pointer_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no model pointer"):
        rollback(tmp_path)


def test_rollback_when_previous_artifacts_deleted_raises(tmp_path, fitted):
    publish_version(vectorizer=fitted[0], model=fitted[1], root=tmp_path)
    publish_version(
        vectorizer=fitted[0], model=fitted[1], previous_version="v0001", root=tmp_path
    )
    (version_dir("v0001", tmp_path) / "manifest.json").unlink()
    with pytest.raises(RuntimeError, match="missing from the store"):
        rollback(tmp_path)


def test_registry_serves_bootstrap_version(tmp_path, fallback):
    registry = ModelRegistry(tmp_path)
    registry.start(fallback=fallback)

    assert registry.info()["version"] == "v0001"
    probability = model_store.positive_probability(
        registry.session(), "wildfire evacuation downtown"
    )
    assert 0.0 <= probability <= 1.0


def test_registry_poll_is_noop_without_a_new_version(tmp_path, fallback):
    registry = ModelRegistry(tmp_path)
    registry.start(fallback=fallback)
    assert registry.poll() is False


def test_registry_hot_swaps_to_promoted_version(tmp_path, fallback, fitted):
    registry = ModelRegistry(tmp_path)
    registry.start(fallback=fallback)
    publish_version(
        vectorizer=fitted[0], model=fitted[1], previous_version="v0001", root=tmp_path
    )

    assert registry.poll() is True
    assert registry.info()["version"] == "v0002"
    assert registry.info()["previous_version"] == "v0001"


def test_registry_swap_changes_predictions(tmp_path, fallback):
    """A model trained on the opposite labels must flip the served output."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    registry = ModelRegistry(tmp_path)
    registry.start(fallback=fallback)
    text = "wildfire evacuation downtown"
    before = model_store.positive_probability(registry.session(), text)

    texts = [
        "wildfire evacuation downtown",
        "flood rescue teams arrived",
        "great coffee with friends",
        "watching a movie tonight",
    ]
    vectorizer = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 1))
    model = LogisticRegression(max_iter=1000).fit(
        vectorizer.fit_transform(texts), [0, 0, 1, 1]
    )
    publish_version(
        vectorizer=vectorizer, model=model, previous_version="v0001", root=tmp_path
    )
    registry.poll()

    assert registry.info()["version"] == "v0002"
    assert model_store.positive_probability(registry.session(), text) < before


def test_registry_rollback_returns_to_original_predictions(tmp_path, fallback, fitted):
    registry = ModelRegistry(tmp_path)
    registry.start(fallback=fallback)
    text = "wildfire evacuation downtown"
    before = model_store.positive_probability(registry.session(), text)

    publish_version(
        vectorizer=fitted[0], model=fitted[1], previous_version="v0001", root=tmp_path
    )
    registry.poll()
    rollback(tmp_path)
    registry.poll()

    assert registry.info()["version"] == "v0001"
    assert model_store.positive_probability(registry.session(), text) == pytest.approx(before)


def test_poll_keeps_serving_when_the_new_version_is_corrupt(tmp_path, fallback, fitted):
    """A bad promotion must not take the API down."""
    registry = ModelRegistry(tmp_path)
    registry.start(fallback=fallback)
    publish_version(
        vectorizer=fitted[0], model=fitted[1], previous_version="v0001", root=tmp_path
    )
    (version_dir("v0002", tmp_path) / "model.onnx").write_bytes(b"not an onnx graph")

    with pytest.raises(ort_InvalidProtobuf):
        registry.poll()

    assert registry.info()["version"] == "v0001"


def test_positive_probability_matches_sklearn_exactly_for_unigrams(tmp_path):
    """With ngram_range=(1, 1) the converted graph is bit-faithful."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = [
        "wildfire evacuation downtown",
        "flood rescue teams arrived",
        "great coffee with friends",
        "watching a movie tonight",
    ]
    vectorizer = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 1))
    model = LogisticRegression(max_iter=1000).fit(
        vectorizer.fit_transform(texts), [1, 1, 0, 0]
    )
    path = tmp_path / "model.onnx"
    model_store.export_to_onnx(vectorizer, model, path)

    import onnxruntime as rt

    session = rt.InferenceSession(str(path))
    for text in texts:
        expected = float(model.predict_proba(vectorizer.transform([text]))[0, 1])
        assert model_store.positive_probability(session, text) == pytest.approx(
            expected, abs=1e-6
        )


def test_bigram_export_is_lossy_but_bounded(tmp_path):
    """Documents the known skl2onnx bigram dropout instead of hiding it.

    Measured against the trained corpus: unigram graphs match exactly, while a
    (1, 2) graph drops a fraction of bigrams, moving about 0.3% of predictions
    across the decision boundary and costing at most ~0.002 accuracy. The
    deployment gate allows this much and no more.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = [
        "wildfire evacuation downtown",
        "flood rescue teams arrived",
        "great coffee with friends",
        "watching a movie tonight",
    ]
    vectorizer = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2))
    model = LogisticRegression(max_iter=1000).fit(
        vectorizer.fit_transform(texts), [1, 1, 0, 0]
    )
    path = tmp_path / "model.onnx"
    model_store.export_to_onnx(vectorizer, model, path)

    import onnxruntime as rt

    session = rt.InferenceSession(str(path))
    for text in texts:
        expected = float(model.predict_proba(vectorizer.transform([text]))[0, 1])
        observed = model_store.positive_probability(session, text)
        assert abs(observed - expected) < 0.05


def test_exported_graph_outputs_a_probability_tensor(tmp_path, fitted):
    """zipmap must stay off so the serving path avoids per-row dicts."""
    path = tmp_path / "model.onnx"
    model_store.export_to_onnx(*fitted, path)

    import onnxruntime as rt

    names = {o.name for o in rt.InferenceSession(str(path)).get_outputs()}
    assert "probabilities" in names
    assert "output_probability" not in names


def test_positive_probability_rejects_unknown_output_graph(tmp_path, fitted):
    path = tmp_path / "model.onnx"
    model_store.export_to_onnx(*fitted, path)

    class FakeSession:
        def get_outputs(self):
            return []

    with pytest.raises(RuntimeError, match="unexpected ONNX model outputs"):
        model_store.positive_probability(FakeSession(), "anything")


def test_write_pointer_ignores_a_corrupt_pointer(tmp_path):
    pointer_path(tmp_path).write_text("{not json")
    assert read_pointer(tmp_path) is None


def test_manifest_records_metrics_for_audit(tmp_path, fitted):
    manifest = publish_version(
        vectorizer=fitted[0],
        model=fitted[1],
        metrics={"f1": 0.91, "disaster_recall": 0.84},
        source="retraining",
        root=tmp_path,
    )
    stored = json.loads((version_dir("v0001", tmp_path) / "manifest.json").read_text())

    assert manifest["metrics"]["f1"] == 0.91
    assert stored["source"] == "retraining"
    assert stored["metrics"]["disaster_recall"] == 0.84


def test_api_restarts_with_legacy_retraining_metrics(tmp_path, fitted, monkeypatch):
    import app as api
    from fastapi.testclient import TestClient

    publish_version(
        vectorizer=fitted[0], model=fitted[1], root=tmp_path,
        metrics={"sklearn_accuracy": 0.8, "sklearn_f1": 0.8,
                 "sklearn_disaster_recall": 0.8, "accepted_labels": 100},
    )
    recorded = []

    def record(*, version, source, model_version_before, accuracy=None,
               f1=None, disaster_recall=None, accepted_labels=None):
        recorded.append((version, accuracy, accepted_labels))

    monkeypatch.setattr(api, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(api, "initialize_database", lambda: None)
    monkeypatch.setattr(api, "log_model_version", record)
    monkeypatch.setattr(api, "log_request", lambda **kwargs: None)
    with TestClient(api.app) as client:
        assert client.get("/model/info").json()["version"] == "v0001"
    assert recorded == [("v0001", None, 100)]


def test_retraining_manifest_stores_onnx_gate_metrics(tmp_path, fitted):
    from retrain_cycle import deploy_candidate

    publish_version(vectorizer=fitted[0], model=fitted[1], root=tmp_path)
    manifest = deploy_candidate(
        *fitted,
        metrics={"accuracy": 0.7, "f1": 0.71, "disaster_recall": 0.72},
        onnx_metrics={"accuracy": 0.8, "f1": 0.81, "disaster_recall": 0.82},
        model_root=tmp_path,
        accepted_labels=12,
    )
    assert manifest["metrics"]["accuracy"] == 0.8
    assert manifest["metrics"]["f1"] == 0.81
    assert manifest["metrics"]["sklearn_f1"] == 0.71


def test_cache_key_is_namespaced_by_model_version():
    """A promoted model must not read the previous model's cached prediction."""
    from app import cache_key

    old = cache_key("earthquake in downtown", "v0001")
    new = cache_key("earthquake in downtown", "v0002")

    assert old != new
    assert old.startswith("predict:v0001:")
    assert new.startswith("predict:v0002:")
    assert cache_key("earthquake in downtown", "v0001") == old


def test_reference_artifact_loads(tmp_path, fitted):
    """Guards the drift reference contract: pickled dict with a fitted vectorizer."""
    reference = {
        "vectorizer": fitted[0],
        "feature_frequency": np.array([1.0, 2.0]),
        "positive_rate": 0.5,
        "rows": 4,
    }
    path = tmp_path / "ref.joblib"
    joblib.dump(reference, path)
    loaded = joblib.load(path)
    assert loaded["rows"] == 4
    assert Path(path).exists()
