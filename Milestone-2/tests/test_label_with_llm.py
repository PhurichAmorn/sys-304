import json

import pytest

from label_with_llm import _parse_label


def test_parse_label_accepts_binary_labels():
    assert _parse_label('{"label": 0, "confidence": 0.91}') == (0, 0.91)
    assert _parse_label('{"label": 1, "confidence": 0.88}') == (1, 0.88)


@pytest.mark.parametrize("label", [2, -1, "1", True, None])
def test_parse_label_rejects_non_binary_labels(label):
    with pytest.raises(ValueError, match="invalid label"):
        _parse_label(json.dumps({"label": label, "confidence": 0.9}))


def test_parse_label_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence"):
        _parse_label('{"label": 1, "confidence": 2}')