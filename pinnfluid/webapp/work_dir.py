"""Where the web app writes: PINN_WEBAPP_WORK_DIR.

When PINN_WEBAPP_WORK_DIR is set (an absolute path), everything the app writes
goes below it, so the source tree can stay read-only (a container with a
read-only root and one writable directory):

    <work dir>/dem/        terrain downloads and uploads (one directory per domain)
    <work dir>/workspace/  the inputs being built for a prediction
    <work dir>/results/    saved runs, unless PINN_WEBAPP_RESULTS_DIR names another place

Unset, the historical layout inside the checkout is kept (dem/ at the repository
root, workspace/ and results/ next to app.py). The values are read once, at import.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parent.parent


def _work_dir_from_env() -> Optional[Path]:
    raw = os.environ.get("PINN_WEBAPP_WORK_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"PINN_WEBAPP_WORK_DIR must be an absolute path, got {raw!r}")
    return path


def _results_dir(work_dir: Optional[Path]) -> Path:
    raw = os.environ.get("PINN_WEBAPP_RESULTS_DIR", "").strip()
    if raw:
        return Path(raw)
    return work_dir / "results" if work_dir else WEBAPP_DIR / "results"


WORK_DIR = _work_dir_from_env()
# DEM_ROOT.parent plays the role of the repository root for code that looks up
# dem/<case>/ (input_prep.common.find_case_sidecars).
DEM_ROOT = (WORK_DIR or REPO_ROOT) / "dem"
WORKSPACE = WORK_DIR / "workspace" if WORK_DIR else WEBAPP_DIR / "workspace"
RESULTS_DIR = _results_dir(WORK_DIR)
