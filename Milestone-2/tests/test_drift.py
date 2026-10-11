import numpy as np
import pytest

import drift


def test_psi_is_near_zero_for_identical_distributions():
    reference = np.array([50.0, 30.0, 15.0, 5.0])
    assert drift.population_stability_index(reference, reference) < 0.01


def test_psi_grows_when_live_mass_moves_between_buckets():
    reference = np.array([50.0, 30.0, 15.0, 5.0])
    shifted = np.array([5.0, 15.0, 30.0, 50.0])
    assert drift.population_stability_index(reference, shifted) > 0.25


def test_psi_stays_flat_when_noise_is_added_to_every_feature():
    """Unbucketed PSI over sparse features reports false drift on noise alone."""
    rng = np.random.default_rng(0)
    reference = rng.random(5000) ** 4 + 0.001
    noisy = reference * rng.uniform(0.98, 1.02, size=5000)
    assert drift.population_stability_index(reference, noisy) < 0.1


def test_psi_returns_none_for_empty_input():
    assert drift.population_stability_index(np.array([]), np.array([])) is None
    assert drift.population_stability_index(np.array([1.0]), np.array([0.0])) is None


@pytest.mark.parametrize(
    "value,expected",
    [(0.0, "stable"), (0.05, "stable"), (0.1, "moderate"), (0.24, "moderate"),
     (0.25, "significant"), (3.0, "significant")],
)
def test_classify_psi_thresholds(value, expected):
    assert drift.classify_psi(value) == expected


def test_classify_psi_handles_unknown():
    assert drift.classify_psi(None) == "unknown"


def test_oov_rate_is_zero_for_known_words():
    assert drift.out_of_vocabulary_rate({"fire", "and", "flood"}, ["fire and flood"]) == 0.0


def test_oov_rate_counts_unseen_words():
    rate = drift.out_of_vocabulary_rate({"fire"}, ["fire and kubernetes"])
    assert rate == pytest.approx(2 / 3)


def test_oov_rate_is_none_for_empty_input():
    assert drift.out_of_vocabulary_rate({"fire"}, []) is None


def test_length_shift_detects_shorter_texts():
    statistic, pvalue = drift.length_shift([100] * 200, [10] * 200)
    assert statistic == pytest.approx(1.0)
    assert pvalue < 0.01


def test_length_shift_on_same_distribution_is_insignificant():
    _, pvalue = drift.length_shift([50] * 300, [50] * 300)
    assert pvalue > 0.05


def test_positive_rate_and_confidence():
    assert drift.positive_rate([0.9, 0.2, 0.6, 0.1]) == pytest.approx(0.5)
    assert drift.mean_confidence([0.9, 0.1]) == pytest.approx(0.9)
    assert drift.positive_rate([]) is None
    assert drift.mean_confidence([]) is None


def test_top_shifted_features_ranks_by_contribution():
    """Feature 'a' loses far more mass than 'd', so it must rank first."""
    reference = np.array([100.0, 1.0, 1.0, 50.0])
    live = np.array([1.0, 1.0, 1.0, 99.0])
    features = drift.top_shifted_features(reference, live, ["a", "b", "c", "d"], limit=2)
    assert [f["feature"] for f in features] == ["a", "d"]
    assert features[0]["live_share"] < features[0]["reference_share"]
    assert features[1]["live_share"] > features[1]["reference_share"]


def test_top_shifted_features_tolerates_equal_contributions():
    """A perfectly mirrored swap has symmetric PSI, so either order is valid."""
    reference = np.array([100.0, 1.0, 1.0, 1.0])
    live = np.array([1.0, 1.0, 1.0, 100.0])
    features = drift.top_shifted_features(reference, live, ["a", "b", "c", "d"], limit=2)
    assert {f["feature"] for f in features} == {"a", "d"}
    assert features[0]["psi_contribution"] == pytest.approx(features[1]["psi_contribution"])


def test_compute_drift_on_empty_window_is_unknown():
    reference = {
        "vectorizer": None,
        "feature_frequency": np.array([]),
        "feature_names": [],
        "positive_rate": 0.4,
    }
    report = drift.compute_drift(reference, [], [])
    assert report["window_size"] == 0
    assert report["psi_verdict"] == "unknown"
    assert report["psi_overall"] is None


def test_compute_drift_reports_every_view():
    from sklearn.feature_extraction.text import TfidfVectorizer

    training = ["wildfire evacuation", "flood rescue", "earthquake damage", "hurricane storm"]
    live = ["wildfire evacuation", "wildfire evacuation", "flood rescue", "hurricane storm"]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    vectorizer.fit(training)
    reference = drift.build_reference(training, [1, 1, 1, 0], vectorizer)

    report = drift.compute_drift(reference, live, [0.95, 0.93, 0.88, 0.11])

    assert report["window_size"] == 4
    assert report["psi_verdict"] in {"stable", "moderate", "significant"}
    assert 0.0 <= report["oov_rate"] <= 1.0
    assert report["reference_positive_rate"] == pytest.approx(0.75)
    assert report["live_positive_rate"] == pytest.approx(0.75)
    assert report["mean_confidence"] > 0.5
    assert len(report["top_features"]) == 5


def test_compute_drift_handles_live_text_without_known_features():
    from sklearn.feature_extraction.text import TfidfVectorizer

    training = ["wildfire evacuation", "flood rescue"]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(training)
    reference = drift.build_reference(training, [1, 0], vectorizer)

    report = drift.compute_drift(reference, ["the !!!", "unseen-token"], [0.5, 0.5])

    assert report["psi_overall"] is None
    assert report["psi_verdict"] == "unknown"
    assert "js_divergence" not in report
    assert report["oov_rate"] == 1.0


def test_build_reference_oov_baseline_is_zero_on_training_text():
    from sklearn.feature_extraction.text import TfidfVectorizer

    training = ["wildfire evacuation", "flood rescue teams"]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    vectorizer.fit(training)
    reference = drift.build_reference(training, [1, 0], vectorizer)

    assert drift.out_of_vocabulary_rate(reference["vocabulary"], training) == 0.0
