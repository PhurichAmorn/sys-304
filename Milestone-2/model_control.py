"""Inspect, promote, or roll back the live model version.

The model store is a shared volume, so promotion and rollback are both pointer
writes and need no API call. Only the retraining pipeline writes new versions;
this tool is for inspection and for reversing a bad promotion.

Examples:
    python model_control.py status
    python model_control.py list
    python model_control.py rollback --reason "candidate regressed on holdout recall"
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_DIR / "backend"
if BACKEND_DIR.is_dir():
    sys.path.insert(0, str(BACKEND_DIR))

from model_store import MODEL_ROOT, read_pointer, rollback

try:
    from database import set_active_version
except ImportError:  # database.py needs psycopg; rollback must still work without it
    set_active_version = None


def list_versions(root):
    versions_dir = Path(root) / "versions"
    if not versions_dir.exists():
        return []
    pointer = read_pointer(root) or {}
    entries = []
    for entry in sorted(versions_dir.iterdir()):
        manifest_path = entry / "manifest.json"
        if not entry.is_dir() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        entries.append(
            {
                "version": entry.name,
                "active": entry.name == pointer.get("version"),
                "source": manifest.get("source"),
                "promoted_at": manifest.get("promoted_at"),
                "metrics": manifest.get("metrics", {}),
                "size_kb": round(
                    (entry / "model.onnx").stat().st_size / 1024, 1
                )
                if (entry / "model.onnx").exists()
                else None,
            }
        )
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default=os.environ.get("MODEL_ROOT", str(MODEL_ROOT)))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show the live version")
    subparsers.add_parser("list", help="List every stored version")
    rollback_parser = subparsers.add_parser("rollback", help="Restore the previous version")
    rollback_parser.add_argument("--reason", default="manual rollback")
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(read_pointer(args.model_root), indent=2))
        return
    if args.command == "list":
        print(json.dumps(list_versions(args.model_root), indent=2))
        return

    restored = rollback(args.model_root)

    # The pointer flip is what serves traffic; this only keeps the Grafana
    # version panel honest, so it must never block a rollback.
    recorded = False
    if set_active_version is not None and os.environ.get("DATABASE_URL"):
        set_active_version(restored["version"])
        recorded = True

    print(
        json.dumps(
            {
                "rolled_back_to": restored.get("version"),
                "rolled_back_from": restored.get("rolled_back_from"),
                "rolled_back_at": restored.get("rolled_back_at"),
                "reason": args.reason,
                "metrics": restored.get("metrics", {}),
                "history_recorded": recorded,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()