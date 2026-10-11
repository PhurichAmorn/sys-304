"""Covariate drift between logged live traffic and the original training data.

Live traffic is unlabeled, so drift can only be measured on the input side. That
is what the milestone asks for: comparing live inputs against the original
training distributions.

Four complementary views, cheapest first:

- **PSI** over per-feature TF-IDF mass, bucketed into equal-count groups by
  reference mass. The headline number and the only thresholded verdict:
  <0.1 stable, 0.1-0.25 moderate, >0.25 significant.
- **OOV rate**: share of live words absent from the training vocabulary. The
  most explainable signal, and the one to quote when someone asks why.
- **Length shift** via a two-sample Kolmogorov-Smirnov test on character counts.

Every function here is pure so it can be unit tested without a database.
"""

import math
import re

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
PSI_EPSILON = 1e-6
PSI_BINS = 10
PSI_STABLE = 0.1
PSI_SIGNIFICANT = 0.25


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def _as_proportions(values):
    total = float(values.sum())
    if total <= 0:
        return values.astype(float)
    return values / total


def population_stability_index(reference_freq, live_freq, epsilon=PSI_EPSILON, bins=PSI_BINS):
    """PSI = sum((live% - ref%) * ln(live% / ref%)), in nats.

    Features are grouped into ``bins`` equal-count buckets ordered by reference
    mass, and PSI is computed across the bucket totals. Comparing 5000 sparse
    features directly does not work: the long tail of near-zero shares produces
    enormous ratios, so data drawn from the very same distribution still scores
    as heavily drifted. Bucketing is what makes the number comparable to the
    standard 0.1 / 0.25 thresholds.
    """
    import numpy as np

    reference = np.asarray(reference_freq, dtype=float)
    live = np.asarray(live_freq, dtype=float)
    if reference.size == 0 or reference.sum() <= 0 or live.sum() <= 0:
        return None

    order = np.argsort(reference, kind="stable")
    groups = np.array_split(order, min(bins, order.size))
    reference_buckets = np.array([reference[group].sum() for group in groups])
    live_buckets = np.array([live[group].sum() for group in groups])

    reference_share = reference_buckets / reference_buckets.sum() + epsilon
    live_share = live_buckets / live_buckets.sum() + epsilon
    live_share = live_share / live_share.sum()
    return float(np.sum((live_share - reference_share) * np.log(live_share / reference_share)))


def classify_psi(value):
    if value is None:
        return "unknown"
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def out_of_vocabulary_rate(known_tokens, texts, analyzer=None):
    """Share of live words the model never saw in training.

    Measured on unigrams only. Including bigrams inflates the figure with novel
    word pairs that individually carry little signal, which makes the number
    harder to explain to whoever is looking at the dashboard.
    """
    analyze = analyzer or tokenize
    known = set(known_tokens)
    total = 0
    unknown = 0
    for text in texts:
        for token in analyze(text):
            total += 1
            if token not in known:
                unknown += 1
    if total == 0:
        return None
    return unknown / total


def length_shift(reference_lengths, live_lengths):
    """Two-sample KS test on text length. Returns (statistic, p_value)."""
    from scipy.stats import ks_2samp

    if len(reference_lengths) == 0 or len(live_lengths) == 0:
        return None, None
    result = ks_2samp(reference_lengths, live_lengths)
    return float(result.statistic), float(result.pvalue)


def top_shifted_features(reference_freq, live_freq, feature_names, limit=5):
    """Features with the largest PSI contribution, for the dashboard table."""
    reference = _as_proportions(reference_freq)
    live = _as_proportions(live_freq)
    scored = []
    for index, name in enumerate(feature_names):
        ref = max(float(reference[index]), PSI_EPSILON)
        cur = max(float(live[index]), PSI_EPSILON)
        contribution = (cur - ref) * math.log(cur / ref)
        scored.append((name, contribution, float(reference[index]), float(live[index])))
    scored.sort(key=lambda item: abs(item[1]), reverse=True)
    return [
        {
            "feature": name,
            "psi_contribution": round(abs(contribution), 6),
            "reference_share": round(ref_share, 8),
            "live_share": round(live_share, 8),
        }
        for name, contribution, ref_share, live_share in scored[:limit]
    ]


def mean_confidence(probabilities):
    values = [float(p) for p in probabilities]
    if not values:
        return None
    return sum(max(value, 1 - value) for value in values) / len(values)


def positive_rate(probabilities, threshold=0.5):
    values = [float(p) for p in probabilities]
    if not values:
        return None
    return sum(value >= threshold for value in values) / len(values)


def build_reference(texts, labels, vectorizer):
    """Freeze the training distribution. Called once by build_drift_reference."""
    import numpy as np

    matrix = vectorizer.transform(texts)
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    unigram_vocabulary = set()
    for text in texts:
        unigram_vocabulary.update(tokenize(text))
    return {
        "vectorizer": vectorizer,
        "feature_names": list(vectorizer.get_feature_names_out()),
        "feature_frequency": frequencies,
        "vocabulary": sorted(unigram_vocabulary),
        "char_lengths": [len(text) for text in texts],
        "positive_rate": positive_rate(labels, threshold=0.5),
        "rows": len(texts),
        "analyzer_vocabulary_size": len(vectorizer.vocabulary_),
    }


def compute_drift(reference, texts, probabilities):
    """Full drift report for one live window against the frozen reference."""
    import numpy as np

    if not texts:
        return {
            "window_size": 0,
            "psi_overall": None,
            "psi_verdict": "unknown",
            "oov_rate": None,
            "length_ks_statistic": None,
            "length_ks_pvalue": None,
            "mean_confidence": None,
            "live_positive_rate": None,
            "reference_positive_rate": reference.get("positive_rate"),
            "top_features": [],
        }

    vectorizer = reference["vectorizer"]
    live_frequency = np.asarray(vectorizer.transform(texts).sum(axis=0)).ravel()
    reference_frequency = reference["feature_frequency"]

    psi = population_stability_index(reference_frequency, live_frequency)
    oov_rate = out_of_vocabulary_rate(reference["vocabulary"], texts)
    live_lengths = [len(text) for text in texts]
    ks_statistic, ks_pvalue = length_shift(reference["char_lengths"], live_lengths)

    return {
        "window_size": len(texts),
        "psi_overall": round(psi, 6) if psi is not None else None,
        "psi_verdict": classify_psi(psi),
        "oov_rate": round(oov_rate, 6) if oov_rate is not None else None,
        "length_ks_statistic": round(ks_statistic, 6) if ks_statistic is not None else None,
        "length_ks_pvalue": round(ks_pvalue, 6) if ks_pvalue is not None else None,
        "mean_confidence": round(mean_confidence(probabilities), 6)
        if probabilities
        else None,
        "live_positive_rate": round(positive_rate(probabilities), 6)
        if probabilities
        else None,
        "reference_positive_rate": reference.get("positive_rate"),
        "top_features": top_shifted_features(
            reference_frequency, live_frequency, reference["feature_names"]
        ),
    }
