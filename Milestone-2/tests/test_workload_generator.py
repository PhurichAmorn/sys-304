import sys

import pytest

import workload_generator as workload


@pytest.mark.parametrize("pattern", ["mixed", "unique", "repeated"])
def test_cli_uses_requested_synthetic_pattern(pattern, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["workload_generator.py", "--requests", "12",
                                     "--pattern", pattern])
    captured = []
    monkeypatch.setattr(workload, "run_workload",
                        lambda url, texts, concurrency, timeout: captured.extend(texts) or {})
    workload.main()
    assert captured == workload.make_inputs(12, pattern, 42)
    if pattern == "unique":
        assert len(set(captured)) == 12
    if pattern == "repeated":
        assert len(set(captured)) == 1


def test_cli_defaults_to_original_test_csv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["workload_generator.py"])
    captured = []
    monkeypatch.setattr(workload, "read_csv_inputs",
                        lambda path, limit, seed: captured.append((path, limit)) or ["tweet"])
    monkeypatch.setattr(workload, "run_workload", lambda *args: {})
    workload.main()
    assert captured[0][0].endswith("Milestone-1/nlp-getting-started/test.csv")
    assert captured[0][1] is None


def test_cli_rejects_ambiguous_sources(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["workload_generator.py", "--pattern", "unique",
                                     "--input-csv", "tweets.csv"])
    with pytest.raises(SystemExit) as error:
        workload.main()
    assert error.value.code == 2
