"""Write VTK structured-grid files (legacy BINARY) for ParaView.

The predicted flow lives on a structured grid (uniform xy spacing, non-uniform z),
which maps cleanly to VTK's STRUCTURED_GRID dataset. Output files open directly
in ParaView (File -> Open) with U as a vector field and Umag/p/is_fluid as
scalar fields — streamlines, slices, glyphs, all native filters.

Legacy VTK BINARY is big-endian regardless of host, hence the explicit `>f4`.
Point ordering is i-fastest then j then k (VTK convention).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# Skip VTK export above this many points to avoid runaway file sizes.
# 32 M points * 32 bytes/point ~= 1 GB per file.
_MAX_POINTS_PER_FILE = 32_000_000


def _write_structured_grid_vtk(
    out_path: Path,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_levels: np.ndarray,
    flow: np.ndarray,
    is_fluid: np.ndarray,
    *,
    description: str,
) -> Path:
    nx, ny, nz = int(len(x_coords)), int(len(y_coords)), int(len(z_levels))
    if flow.shape[:3] != (nx, ny, nz) or flow.shape[-1] != 4:
        raise ValueError(
            f"flow shape {flow.shape} does not match (nx,ny,nz,4) = "
            f"({nx},{ny},{nz},4)"
        )
    if is_fluid.shape != (nx, ny, nz):
        raise ValueError(f"is_fluid shape {is_fluid.shape} != ({nx},{ny},{nz})")
    n = nx * ny * nz

    # VTK i-fastest order: an array indexed (i,j,k) flattens correctly via
    # transpose(2,1,0) before reshape(-1) (C-order).
    xx = np.broadcast_to(
        np.asarray(x_coords, dtype=np.float32)[:, None, None], (nx, ny, nz)
    )
    yy = np.broadcast_to(
        np.asarray(y_coords, dtype=np.float32)[None, :, None], (nx, ny, nz)
    )
    zz = np.broadcast_to(
        np.asarray(z_levels, dtype=np.float32)[None, None, :], (nx, ny, nz)
    )
    pts = np.stack([xx, yy, zz], axis=-1).transpose(2, 1, 0, 3).reshape(-1, 3)

    flow_clean = np.where(np.isfinite(flow), flow, 0.0).astype(np.float32, copy=False)
    U = flow_clean[..., :3].transpose(2, 1, 0, 3).reshape(-1, 3)
    p = flow_clean[..., 3].transpose(2, 1, 0).reshape(-1)
    Umag = np.sqrt((flow_clean[..., :3] ** 2).sum(axis=-1))
    Umag = Umag.transpose(2, 1, 0).reshape(-1).astype(np.float32)
    mask = (np.asarray(is_fluid) > 0.5).astype(np.uint8)
    mask = mask.transpose(2, 1, 0).reshape(-1)

    def be(arr: np.ndarray) -> bytes:
        return arr.astype(">f4", copy=False).tobytes()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(description.encode("utf-8")[:255] + b"\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n".encode())
        f.write(f"POINTS {n} float\n".encode())
        f.write(be(pts))
        f.write(b"\n")
        f.write(f"POINT_DATA {n}\n".encode())
        f.write(b"VECTORS U float\n")
        f.write(be(U))
        f.write(b"\n")
        f.write(b"SCALARS Umag float 1\nLOOKUP_TABLE default\n")
        f.write(be(Umag))
        f.write(b"\n")
        f.write(b"SCALARS p float 1\nLOOKUP_TABLE default\n")
        f.write(be(p))
        f.write(b"\n")
        f.write(b"SCALARS is_fluid unsigned_char 1\nLOOKUP_TABLE default\n")
        f.write(mask.tobytes())
        f.write(b"\n")
    return out_path


def export_prediction_vtk(
    out_dir: Path,
    *,
    domain_name: str,
    global_bundle,
    global_pred_flow: np.ndarray,
    roi_bundles: Optional[dict] = None,
    roi_pred_flows: Optional[dict] = None,
) -> dict:
    """Export VTK structured-grid files for the global + ROI predictions.

    Returns:
        {"files": [Path, ...], "skipped": [{"label": str, "n_points": int}, ...]}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    skipped: list[dict] = []

    n_global = int(np.prod(global_pred_flow.shape[:3], dtype=np.int64))
    if n_global > _MAX_POINTS_PER_FILE:
        skipped.append({"label": "global", "n_points": n_global})
    else:
        p_global = out_dir / f"{domain_name}_global.vtk"
        _write_structured_grid_vtk(
            p_global,
            global_bundle.x_coords,
            global_bundle.y_coords,
            global_bundle.z_levels,
            global_pred_flow,
            global_bundle.is_fluid,
            description=f"{domain_name} (global) - pinn_terr_struc",
        )
        files.append(p_global)

    if roi_bundles and roi_pred_flows:
        for label, rb in roi_bundles.items():
            rf = roi_pred_flows.get(label)
            if rf is None:
                continue
            n_roi = int(np.prod(rf.shape[:3], dtype=np.int64))
            if n_roi > _MAX_POINTS_PER_FILE:
                skipped.append({"label": str(label), "n_points": n_roi})
                continue
            p_roi = out_dir / f"{domain_name}_{label}.vtk"
            _write_structured_grid_vtk(
                p_roi,
                rb.x_coords,
                rb.y_coords,
                rb.z_levels,
                rf,
                rb.is_fluid,
                description=f"{domain_name} ({label}) - pinn_terr_struc",
            )
            files.append(p_roi)

    return {"files": files, "skipped": skipped}
