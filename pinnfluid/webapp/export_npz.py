"""Export the predicted 3D field as a Python-friendly compressed .npz.

A user reads it with:
    d = np.load('export.npz')
    U = d['U']            # (nx, ny, nz, 3)  velocity components
    p = d['p']             # (nx, ny, nz)     pressure
    is_fluid = d['is_fluid']
    x, y, z = d['x'], d['y'], d['z']  # cell-centred coordinates

One file per global, one per ROI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def write_npz_export(
    out_path: Path,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_levels: np.ndarray,
    flow: np.ndarray,
    is_fluid: np.ndarray,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = int(len(x_coords)), int(len(y_coords)), int(len(z_levels))
    if flow.shape[:3] != (nx, ny, nz) or flow.shape[-1] != 4:
        raise ValueError(f"flow shape {flow.shape} != ({nx},{ny},{nz},4)")
    np.savez_compressed(
        out_path,
        x=np.asarray(x_coords, dtype=np.float32),
        y=np.asarray(y_coords, dtype=np.float32),
        z=np.asarray(z_levels, dtype=np.float32),
        U=flow[..., :3].astype(np.float32),
        p=flow[..., 3].astype(np.float32),
        is_fluid=(np.asarray(is_fluid) > 0.5).astype(np.uint8),
    )
    return out_path


def export_prediction_npz(
    out_dir: Path,
    *,
    domain_name: str,
    global_bundle,
    global_pred_flow: np.ndarray,
    roi_bundles: Optional[dict] = None,
    roi_pred_flows: Optional[dict] = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    p_global = out_dir / f"{domain_name}_global.npz"
    write_npz_export(
        p_global,
        global_bundle.x_coords,
        global_bundle.y_coords,
        global_bundle.z_levels,
        global_pred_flow,
        global_bundle.is_fluid,
    )
    files.append(p_global)
    if roi_bundles and roi_pred_flows:
        for label, rb in roi_bundles.items():
            rf = roi_pred_flows.get(label)
            if rf is None:
                continue
            p_roi = out_dir / f"{domain_name}_{label}.npz"
            write_npz_export(
                p_roi,
                rb.x_coords,
                rb.y_coords,
                rb.z_levels,
                rf,
                rb.is_fluid,
            )
            files.append(p_roi)
    return {"files": files}
