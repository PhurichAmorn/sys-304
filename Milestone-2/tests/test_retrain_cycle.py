from retrain_cycle import confidence, trigger_decision


def test_confidence_is_distance_from_uncertain_boundary():
    assert confidence(0.9) == 0.9
    assert confidence(0.1) == 0.9
    assert confidence(0.5) == 0.5


def test_trigger_requires_confidence_drop_and_batch_size():
    rows = [(1, "one", 0.6, "model"), (2, "two", 0.7, "model")]

    decision = trigger_decision(rows, threshold=0.7, minimum_rows=2)

    assert decision["triggered"] is True
    assert decision["sample_count"] == 2


def test_trigger_does_not_fire_for_small_batch():
    rows = [(1, "one", 0.1, "model")]

    decision = trigger_decision(rows, threshold=0.9, minimum_rows=2)

    assert decision["triggered"] is False
    assert decision["reason"] == "conditions_not_met"


def test_fetch_recent_requests_uses_monitor_selected_ids(monkeypatch):
    import retrain_cycle

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, query, parameters):
            assert "id = ANY(%s)" in query
            assert parameters == ([11, 12],)
            return self

        def fetchall(self):
            return [(11, "one", 0.6, "v1"), (12, "two", 0.7, "v1")]

    monkeypatch.setattr(retrain_cycle.psycopg, "connect", lambda _: Connection())
    assert [row[0] for row in retrain_cycle.fetch_recent_requests("db", 100, [11, 12])] == [11, 12]
