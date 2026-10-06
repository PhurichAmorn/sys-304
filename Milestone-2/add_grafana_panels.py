"""Append the Week 7 drift panels and the model-version panels to the dashboard.

Kept as a script so the dashboard JSON stays reviewable in git instead of being
hand-edited, and so the panels can be regenerated after a schema change.

    python add_grafana_panels.py
"""

import json
from pathlib import Path

DASHBOARD_PATH = Path("grafana/dashboards/request-monitoring.json")
DATASOURCE = {"type": "postgres", "uid": "prediction-postgres"}


def panel(panel_id, title, panel_type, raw_sql, grid, unit="none", decimals=3, **extra):
    body = {
        "datasource": DATASOURCE,
        "fieldConfig": {
            "defaults": {"decimals": decimals, "unit": unit, **extra.pop("defaults", {})},
            "overrides": [],
        },
        "gridPos": grid,
        "id": panel_id,
        "options": {},
        "targets": [
            {"datasource": DATASOURCE, "format": "time_series", "rawSql": raw_sql, "refId": "A"}
        ],
        "title": title,
        "type": panel_type,
    }
    body.update(extra)
    return body


def build_panels():
    return [
        panel(
            20,
            "Data drift: PSI vs training baseline",
            "timeseries",
            "SELECT checked_at AS time, psi_overall AS \"Population Stability Index\" "
            "FROM drift_checks WHERE $__timeFilter(checked_at) AND psi_overall IS NOT NULL "
            "ORDER BY checked_at",
            {"h": 8, "w": 12, "x": 0, "y": 30},
            defaults={
                "custom": {
                    "thresholdsStyle": {"mode": "dashed"},
                    "thresholds": [
                        {"color": "green", "value": None},
                        {"color": "orange", "value": 0.1},
                        {"color": "red", "value": 0.25},
                    ],
                },
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "orange", "value": 0.1},
                        {"color": "red", "value": 0.25},
                    ],
                },
            },
        ),
        panel(
            21,
            "Latest drift verdict",
            "stat",
            "SELECT psi_verdict AS verdict FROM drift_checks "
            "ORDER BY checked_at DESC LIMIT 1",
            {"h": 4, "w": 6, "x": 12, "y": 30},
            options={
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                "graphMode": "none",
                "textMode": "value",
            },
        ),
        panel(
            22,
            "Jensen-Shannon distance (trend)",
            "timeseries",
            "SELECT checked_at AS time, js_divergence AS \"JS distance\" "
            "FROM drift_checks WHERE $__timeFilter(checked_at) AND js_divergence IS NOT NULL "
            "ORDER BY checked_at",
            {"h": 4, "w": 6, "x": 18, "y": 30},
        ),
        panel(
            23,
            "Out-of-vocabulary rate",
            "timeseries",
            "SELECT checked_at AS time, oov_rate AS \"Unseen word rate\" "
            "FROM drift_checks WHERE $__timeFilter(checked_at) AND oov_rate IS NOT NULL "
            "ORDER BY checked_at",
            {"h": 4, "w": 8, "x": 0, "y": 38},
            unit="percentunit",
        ),
        panel(
            24,
            "Text length shift (KS statistic)",
            "timeseries",
            "SELECT checked_at AS time, length_ks_statistic AS \"KS statistic\" "
            "FROM drift_checks WHERE $__timeFilter(checked_at) "
            "AND length_ks_statistic IS NOT NULL ORDER BY checked_at",
            {"h": 4, "w": 8, "x": 8, "y": 38},
        ),
        panel(
            25,
            "Live vs training positive rate",
            "timeseries",
            "SELECT checked_at AS time, live_positive_rate AS \"Live\", "
            "reference_positive_rate AS \"Training baseline\" "
            "FROM drift_checks WHERE $__timeFilter(checked_at) ORDER BY checked_at",
            {"h": 4, "w": 8, "x": 16, "y": 38},
            unit="percentunit",
        ),
        panel(
            26,
            "Live model version",
            "stat",
            "SELECT version AS \"Serving model\" FROM model_versions WHERE active = TRUE "
            "ORDER BY promoted_at DESC LIMIT 1",
            {"h": 6, "w": 8, "x": 0, "y": 42},
            decimals=0,
            options={
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                "graphMode": "none",
                "textMode": "value",
            },
        ),
        panel(
            27,
            "Model version history",
            "table",
            "SELECT version, promoted_at, source, active, rolled_back_at, accuracy, f1, "
            "disaster_recall, accepted_labels, model_version_before "
            "FROM model_versions ORDER BY promoted_at DESC LIMIT 20",
            {"h": 6, "w": 16, "x": 8, "y": 42},
            options={"showHeader": True},
        ),
        panel(
            28,
            "Top shifted features (latest check)",
            "table",
            "SELECT feature ->> 'feature' AS feature, "
            "ROUND((feature ->> 'psi_contribution')::numeric, 4) AS psi_contribution, "
            "ROUND((feature ->> 'reference_share')::numeric, 6) AS training_share, "
            "ROUND((feature ->> 'live_share')::numeric, 6) AS live_share "
            "FROM (SELECT top_features FROM drift_checks "
            "WHERE top_features IS NOT NULL ORDER BY checked_at DESC LIMIT 1) latest, "
            "LATERAL jsonb_array_elements(latest.top_features) AS feature "
            "ORDER BY (feature ->> 'psi_contribution')::numeric DESC",
            {"h": 6, "w": 12, "x": 0, "y": 48},
            options={"showHeader": True},
        ),
        panel(
            29,
            "Requests by serving model version",
            "timeseries",
            "SELECT $__timeGroupAlias(created_at, '1m'), model_version AS metric, "
            "COUNT(*) AS requests FROM request_logs "
            "WHERE $__timeFilter(created_at) AND model_version IS NOT NULL "
            "GROUP BY 1, 2 ORDER BY 1",
            {"h": 6, "w": 12, "x": 12, "y": 48},
            decimals=0,
        ),
    ]


def main():
    dashboard = json.loads(DASHBOARD_PATH.read_text())
    existing = {p.get("id") for p in dashboard.get("panels", [])}
    added = [p for p in build_panels() if p["id"] not in existing]
    dashboard["panels"].extend(added)
    DASHBOARD_PATH.write_text(json.dumps(dashboard, indent=2) + "\n")
    print(f"added {len(added)} panels; dashboard now has {len(dashboard['panels'])}")


if __name__ == "__main__":
    main()