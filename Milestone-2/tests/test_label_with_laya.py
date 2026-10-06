import csv

from label_with_laya import label_rows, read_rows, write_rows


class FakeAgent:
    def __init__(self, responses):
        self.responses = iter(responses)

    def predict(self, state, questions):
        return next(self.responses)


def test_label_rows_maps_choices_and_marks_low_confidence():
    rows = [{"id": "1", "text": "wildfire evacuation"}, {"id": "2", "text": "nice coffee"}]
    agent = FakeAgent(
        [
            {"answers": {"disaster": {"choice": "A", "confidence": 0.91}}},
            {"answers": {"disaster": {"choice": "B", "confidence": 0.61}}},
        ]
    )

    labeled = label_rows(rows, agent, min_confidence=0.75)

    assert labeled[0]["laya_label"] == "1"
    assert labeled[0]["needs_review"] == "false"
    assert labeled[1]["laya_label"] == "0"
    assert labeled[1]["needs_review"] == "true"


def test_invalid_laya_response_is_reviewed_and_not_labeled():
    rows = [{"id": "1", "text": "unclear"}]
    agent = FakeAgent([{"answers": {"disaster": {"choice": "C", "confidence": 0.9}}}])

    labeled = label_rows(rows, agent, min_confidence=0.75)

    assert labeled[0]["laya_label"] == ""
    assert labeled[0]["needs_review"] == "true"
    assert "invalid choice" in labeled[0]["label_error"]


def test_write_and_read_laya_csv(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    source.write_text("id,text\n1,hello\n", encoding="utf-8")

    rows, fields = read_rows(source)
    rows[0].update(
        {
            "laya_label": "0",
            "laya_choice": "B",
            "laya_confidence": "0.8",
            "laya_model": "test",
            "label_source": "test",
            "needs_review": "false",
            "label_error": "",
        }
    )
    write_rows(output, rows, fields)

    with output.open(newline="", encoding="utf-8") as file:
        result = list(csv.DictReader(file))
    assert result[0]["laya_label"] == "0"
