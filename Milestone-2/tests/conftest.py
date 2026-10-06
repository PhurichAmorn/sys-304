"""Make the pipeline modules importable no matter where pytest is invoked from.

``drift.py`` lives in Milestone-2/ while ``model_store.py`` lives in
Milestone-2/backend/, and both are imported by the retraining scripts through
the same path bootstrap. The tests mirror that layout.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"

for candidate in (PROJECT_DIR, BACKEND_DIR):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))