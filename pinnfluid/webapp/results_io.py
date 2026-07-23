"""Persist + reload predict_web inputs and predictions.

After a prediction runs, the OF case skeleton (`case_dir`) and the binary CFD
inputs (`cfd_dir`) are normally cleaned up. We mirror the parts we need under
`RESULTS_DIR/<domain>/inputs/` BEFORE cleanup so:
  - the report and 3D viewer can render the terrain and structure meshes,
  - on-demand exports (VTK, NPZ, ...) can be produced any number of times
    without re-running the model,
  - the ``flow.npz`` files in the saved tree contain the model PREDICTION
    rather than the placeholder zeros that were used as inference inputs.

Layout (mirrors the cfd_dir layout so `_load_grid_bundle` works directly):

  RESULTS_DIR/<domain>/inputs/
    presentation.json                 # display-only pressure reference
    cfd/
      meta.json
      terrain.npz
      flow.npz                      # PREDICTED (Ux, Uy, Uz, p, is_fluid)
      [phi_wall.npy]
      roi/
        roi_NNN/
          meta.json
          terrain.npz
          flow.npz                  # PREDICTED
          phi_wall.npy
    case/triSurface/
      ground.stl
      [structure.stl]
      transform.json
      domain_info.json
    selection.json                  # (optional) original DEM selection
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Saved-run retention. Each run keeps 50-100 MB, so an unbounded results/ dir
# fills the disk on a hosted deployment. Keep the newest N runs and stay under a
# size budget, deleting oldest first. Override per deployment via env vars; set
# either to 0 to disable that limit.
RETENTION_MAX_RUNS = int(os.environ.get("PINN_WEBAPP_MAX_RUNS", "40"))
RETENTION_MAX_GB = float(os.environ.get("PINN_WEBAPP_MAX_GB", "15"))


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def enforce_results_retention(
    results_root: Path,
    *,
    keep_name: Optional[str] = None,
    max_runs: Optional[int] = None,
    max_gb: Optional[float] = None,
) -> list[str]:
    """Delete oldest saved runs beyond the run-count / size budget.

    A run is any immediate subdirectory of results_root holding an `inputs/`
    folder. `keep_name` (the run just saved) is never deleted. Best-effort:
    never raises into the caller. Returns the names removed.
    """
    root = Path(results_root)
    max_runs = RETENTION_MAX_RUNS if max_runs is None else max_runs
    max_gb = RETENTION_MAX_GB if max_gb is None else max_gb
    removed: list[str] = []
    if not root.is_dir():
        return removed
    try:
        runs = [
            d for d in root.iterdir()
            if d.is_dir() and (d / "inputs").exists() and d.name != keep_name
        ]
    except OSError:
        return removed
    runs.sort(key=lambda d: d.stat().st_mtime)

    def _drop(d: Path) -> None:
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d.name)

    if max_runs and max_runs > 0:
        overflow = len(runs) - max(max_runs - 1, 0)
        while overflow > 0 and runs:
            _drop(runs.pop(0))
            overflow -= 1
    if max_gb and max_gb > 0 and runs:
        budget = int(max_gb * (1024 ** 3))
        sizes = {d: _dir_size_bytes(d) for d in runs}
        total = sum(sizes.values())
        while total > budget and runs:
            victim = runs.pop(0)
            total -= sizes.get(victim, 0)
            _drop(victim)
    return removed


def _saved_root(results_root: Path, domain_name: str) -> Path:
    return Path(results_root) / domain_name / "inputs"


def _saved_cfd(results_root: Path, domain_name: str) -> Path:
    return _saved_root(results_root, domain_name) / "cfd"


def _saved_case_tri(results_root: Path, domain_name: str) -> Path:
    return _saved_root(results_root, domain_name) / "case" / "triSurface"


def _write_predicted_flow_npz(out_path: Path, pred_flow: np.ndarray, is_fluid: np.ndarray) -> None:
    out_path = Path(out_path)
    pf = np.where(np.isfinite(pred_flow), pred_flow, 0.0).astype(np.float32, copy=False)
    np.savez_compressed(
        out_path,
        Ux=pf[..., 0],
        Uy=pf[..., 1],
        Uz=pf[..., 2],
        p=pf[..., 3],
        is_fluid=np.asarray(is_fluid, dtype=np.float32),
    )


def save_inputs_and_predictions(
    results_root: Path,
    domain_name: str,
    *,
    case_dir: Path,
    cfd_dir: Path,
    predict_out: dict,
    pressure_reference_kinematic: float,
    selection_path: Optional[Path] = None,
) -> Path:
    """Mirror the bits of case_dir/cfd_dir we want to keep, then overwrite
    flow.npz with predictions. Returns the saved root path.
    """
    case_dir = Path(case_dir)
    cfd_dir = Path(cfd_dir)
    saved_root = _saved_root(results_root, domain_name)
    saved_cfd = _saved_cfd(results_root, domain_name)
    saved_case_tri = _saved_case_tri(results_root, domain_name)

    if saved_cfd.exists():
        shutil.rmtree(saved_cfd)
    if saved_cfd.parent != saved_cfd:
        saved_cfd.parent.mkdir(parents=True, exist_ok=True)
    # `cfd_dir` is small (npz + json + a few MB of phi_wall). Mirror it whole.
    shutil.copytree(cfd_dir, saved_cfd)

    g_bundle = predict_out["bundle"]
    g_pred = predict_out["pred_flow"]
    _write_predicted_flow_npz(saved_cfd / "flow.npz", g_pred, g_bundle.is_fluid)

    roi_bundles = predict_out.get("roi_bundles") or {}
    roi_preds = predict_out.get("roi_preds") or {}
    for label, rb in roi_bundles.items():
        rf = roi_preds.get(label)
        if rf is None:
            continue
        roi_flow_path = saved_cfd / "roi" / label / "flow.npz"
        if roi_flow_path.parent.exists():
            _write_predicted_flow_npz(roi_flow_path, rf, rb.is_fluid)

    saved_case_tri.mkdir(parents=True, exist_ok=True)
    src_tri = case_dir / "constant" / "triSurface"
    for fname in ("ground.stl", "structure.stl", "transform.json", "domain_info.json"):
        src = src_tri / fname
        if src.exists():
            shutil.copy2(src, saved_case_tri / fname)

    if selection_path and Path(selection_path).exists():
        shutil.copy2(selection_path, saved_root / "selection.json")

    from pressure_reference import (  # type: ignore
        PRESSURE_REFERENCE_CONVENTION,
        PRESSURE_REFERENCE_LABEL,
    )
    (saved_root / "presentation.json").write_text(
        json.dumps(
            {
                "pressure_reference_convention": PRESSURE_REFERENCE_CONVENTION,
                "pressure_reference_label": PRESSURE_REFERENCE_LABEL,
                "pressure_reference_kinematic": float(pressure_reference_kinematic),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Keep results/ within its retention budget (never removes the run we just
    # saved). Best-effort: a retention failure must not fail the prediction.
    try:
        enforce_results_retention(results_root, keep_name=domain_name)
    except Exception:
        pass

    return saved_root


def load_saved_inputs(results_root: Path, domain_name: str) -> dict:
    """Reload predict-shaped dict from the saved tree.

    Returns the same structure as `inference.predict_all()` (minus scalers)
    plus a `transform_meta` dict (or None if absent). The returned `bundle.flow`
    and per-ROI `flow` already contain the predicted values.
    """
    from data_loader import _load_grid_bundle  # type: ignore

    saved_cfd = _saved_cfd(results_root, domain_name)
    if not (saved_cfd / "meta.json").exists():
        raise FileNotFoundError(f"No saved inputs for '{domain_name}'")

    g_bundle = _load_grid_bundle(saved_cfd, kind="global", roi_name=None, parent_name=None)

    roi_bundles: dict = {}
    rois_dir = saved_cfd / "roi"
    if rois_dir.exists():
        for r_dir in sorted(p for p in rois_dir.iterdir() if p.is_dir()):
            if not (r_dir / "meta.json").exists():
                continue
            rb = _load_grid_bundle(r_dir, kind="roi", roi_name=r_dir.name, parent_name=domain_name)
            roi_bundles[r_dir.name] = rb

    transform_meta = None
    tform_path = _saved_case_tri(results_root, domain_name) / "transform.json"
    if tform_path.exists():
        try:
            transform_meta = json.loads(tform_path.read_text())
        except Exception:
            transform_meta = None

    presentation_meta = {}
    presentation_path = _saved_root(results_root, domain_name) / "presentation.json"
    if presentation_path.exists():
        try:
            presentation_meta = json.loads(presentation_path.read_text())
        except Exception:
            presentation_meta = {}
    pressure_reference = presentation_meta.get("pressure_reference_kinematic")
    if pressure_reference is None:
        from pressure_reference import global_pressure_reference_kinematic  # type: ignore
        pressure_reference = global_pressure_reference_kinematic(
            g_bundle.flow,
            g_bundle.is_fluid,
        )

    return {
        "bundle": g_bundle,
        "pred_flow": g_bundle.flow,
        "roi_bundles": roi_bundles,
        "roi_preds": {label: rb.flow for label, rb in roi_bundles.items()},
        "transform_meta": transform_meta,
        "pressure_reference_kinematic": float(pressure_reference),
        "pressure_reference_convention": presentation_meta.get(
            "pressure_reference_convention",
            "global_fluid_arithmetic_mean_zero",
        ),
    }


def has_saved_inputs(results_root: Path, domain_name: str) -> bool:
    return (_saved_cfd(results_root, domain_name) / "meta.json").exists()
