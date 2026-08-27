from app import label_from_probability


def test_label_from_probability_above_threshold():
    assert label_from_probability(0.6) == 1


def test_label_from_probability_at_threshold():
    assert label_from_probability(0.5) == 1


def test_label_from_probability_below_threshold():
    assert label_from_probability(0.4) == 0
