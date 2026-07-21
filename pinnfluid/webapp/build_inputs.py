"""Build model-ready inputs (meta.json, terrain.npz, flow.npz [+ ROI subdirs])
for an arbitrary user-selected domain, without running any CFD.

Pipeline:
  1. Reuse scripts/domain_prep/domain_builder.build_domain to generate the
     OF case skeleton (ground.stl + optional structure.stl + domain_info.json).
  2. Reuse scripts/input_prep/export_domain.export_empty_case for the global
     binary arrays (terrain.npz, flow.npz with zeros + is_fluid mask, meta.json).
  3. If structures are present, also build one ROI subdirectory per structure
     group (per-component for single structures, enclosing-cluster for grids).
     Each ROI gets its own terrain.npz, flow.npz (zeros), meta.json, and
     phi_wall.npy computed from structure.stl via signed-distance. The global
     meta.json is then patched with `roi_paths` so the downstream plotting code
     can discover the ROIs.

All intermediate artefacts live under scripts/predict_web/workspace/ and are
cleaned up after prediction finishes (see cleanup_case).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
ROOT = SCRIPTS_DIR.parent
WORKSPACE = Path(__file__).resolve().parent / "workspace"

for extra in (SCRIPTS_DIR, SCRIPTS_DIR / "domain_prep", SCRIPTS_DIR / "input_prep"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


# ---------------------------------------------------------------------------
# Paths / lifecycle
# ---------------------------------------------------------------------------
def _case_paths(domain_name: str) -> dict:
    safe = domain_name.strip().replace("/", "_").replace("\\", "_")
    if not safe:
        raise ValueError("domain_name is empty")
    dem_root = ROOT / "dem"
    case_root = WORKSPACE / "cases" / "singlestructures"
    cfd_root = WORKSPACE / "cfd" / "singlestructures"
    return {
        "name": safe,
        "dem_dir": dem_root / safe,
        "case_dir": case_root / safe,
        "cfd_dir": cfd_root / safe,
        "dem_root": dem_root,
        "case_root": case_root,
    }


def cleanup_case(domain_name: str, *, keep_dem: bool = False) -> None:
    """Remove intermediates for one prediction run.

    Keeps only scripts/predict_web/results/<name>/ (the plots the user saves).
    When `keep_dem=True`, dem/<name>/ stays on disk — used right before a
    build so Confirm terrain's download isn't wiped.
    """
    paths = _case_paths(domain_name)
    targets = [paths["case_dir"], paths["cfd_dir"]]
    if not keep_dem:
        targets.append(paths["dem_dir"])
    for p in targets:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# Step 1: STL skeleton via domain_builder
# ---------------------------------------------------------------------------
def build_case_skeleton(
    *,
    domain_name: str,
    domain_size: float,
    wind_from: float,
    flat_terrain: bool,
    structures: list[dict],
    center_latlng: Optional[dict] = None,
    uref: float = 10.0,
    zref: float = 20.0,
    z0: float = 0.1,
    flatten_ground: bool = True,
    grid: Optional[dict] = None,
    z_top_offset: Optional[float] = None,
) -> Path:
    from domain_builder import build_domain  # type: ignore

    paths = _case_paths(domain_name)
    paths["case_root"].mkdir(parents=True, exist_ok=True)
    build_domain(
        domain_name=paths["name"],
        domain_size=domain_size,
        wind_from=wind_from,
        flat_terrain=flat_terrain,
        struct_list=list(structures or []),
        center_latlng=center_latlng,
        dem_root=str(paths["dem_root"]),
        out_root=str(paths["case_root"]),
        uref=float(uref),
        zref=float(zref),
        z0=float(z0),
        flatten_ground=bool(flatten_ground),
        grid=grid,
        z_top_offset=z_top_offset,
    )
    return paths["case_dir"]


# ---------------------------------------------------------------------------
# Step 2a: global empty export (reuse input_prep.export_empty_case)
# ---------------------------------------------------------------------------
def export_binary_inputs(domain_name: str, *, dx: float = 30.0, dy: float = 30.0) -> Path:
    from export_domain import export_empty_case  # type: ignore

    paths = _case_paths(domain_name)
    if not (paths["case_dir"] / "constant" / "triSurface" / "domain_info.json").exists():
        raise FileNotFoundError(
            f"Case skeleton missing at {paths['case_dir']}; call build_case_skeleton first."
        )
    paths["cfd_dir"].parent.mkdir(parents=True, exist_ok=True)
    export_empty_case(
        paths["case_dir"],
        paths["cfd_dir"],
        dx=float(dx),
        dy=float(dy),
        edge_cells=1,
        overwrite=True,
    )
    return paths["cfd_dir"]


# ---------------------------------------------------------------------------
# Step 2b: ROI empty export (our own — input_prep.export_empty_case skips ROI)
# ---------------------------------------------------------------------------
def _roi_global_bounds(case_meta: dict, cfd_dir: Path) -> list[float]:
    """The ROI clipping uses the global-grid bounds, not the raw case."""
    meta = json.loads((cfd_dir / "meta.json").read_text())
    return list(meta["bounds"])


def _terrain_surface_for_view(case_meta: dict, *, xs, ys):
    """Sample the ground STL (post-flatten) where available, fall back to DEM."""
    from common import (  # type: ignore
        sample_dem_on_grid,
        sample_ground_stl_on_grid,
        terrain_surface_on_grid,
    )
    ground_stl = case_meta.get("ground_stl")
    if ground_stl is not None and Path(ground_stl).exists():
        try:
            surface = sample_ground_stl_on_grid(Path(ground_stl), xs=xs, ys=ys)
            if np.isfinite(surface).all():
                return surface
            dem_path = case_meta.get("dem_path")
            if dem_path is not None:
                dem_surface = sample_dem_on_grid(
                    Path(dem_path), xs=xs, ys=ys,
                    z_offset_applied=float(case_meta.get("z_offset_applied", 0.0)),
                )
                fill = ~np.isfinite(surface) & np.isfinite(dem_surface)
                surface[fill] = dem_surface[fill]
            return surface
        except Exception:
            pass
    return terrain_surface_on_grid(case_meta, xs=xs, ys=ys)


def export_roi_inputs(
    domain_name: str,
    *,
    roi_dx: float = 0.5,
    roi_dy: float = 0.5,
    roi_upstream_h: float = 5.0,
    roi_downstream_h: float = 15.0,
    roi_lateral_h: float = 5.0,
    roi_top_h: float = 5.0,
    phi_chunk: int = 200_000,
) -> list[dict]:
    """Build one ROI subdir per structure group. No-op if no structures."""
    from common import (  # type: ignore
        build_z_levels,
        choose_vertical_profile,
        compute_terrain_channels,
        find_case_sidecars,
        preferred_structure_metadata,
        regular_xy_grid,
        roi_bounds_from_group,
        signed_wall_distance,
        structure_roi_groups,
        terrain_surface_on_grid,
    )

    paths = _case_paths(domain_name)
    case_dir = paths["case_dir"]
    cfd_dir = paths["cfd_dir"]

    case_meta = find_case_sidecars(case_dir, ROOT)
    structure_bounds, grid_info, n_structures, source = preferred_structure_metadata(case_meta)
    if not structure_bounds or case_meta["structure_stl"] is None:
        return []

    category = case_meta["category"]
    # ROI grouping policy:
    #   - 1 structure                      → 1 ROI (per_component)
    #   - 2+ separately-placed structures  → N ROIs (per_component) + chooser
    #   - grid placement (rows × cols)     → 1 enclosing-cluster ROI
    # `grid_info` is non-None iff the user used the "grid of structures"
    # placement; in that case we treat the whole grid as one ROI cluster.
    is_grid = grid_info is not None
    effective_category = "multistructures" if is_grid else category
    groups = structure_roi_groups(effective_category, structure_bounds)
    if not groups:
        return []

    global_bounds = _roi_global_bounds(case_meta, cfd_dir)
    z_base_global = float(global_bounds[4])
    z_top_global = float(global_bounds[5])

    abl_di = case_meta["domain_info"].get("ABL", {}) or {}
    abl = {
        "Uref": float(abl_di.get("Uref", 10.0)),
        "Zref": float(abl_di.get("Zref", 20.0)),
        "z0": float(abl_di.get("z0", 0.1)),
        "flowDir": [1.0, 0.0, 0.0],  # case is pre-rotated: wind always +x
        "wind_from_deg": float(case_meta["domain_info"].get("wind_from", 270.0)),
    }

    roi_root = cfd_dir / "roi"
    roi_root.mkdir(parents=True, exist_ok=True)
    roi_paths: list[str] = []
    roi_mode = groups[0]["mode"]

    for idx, group in enumerate(groups):
        roi_xy_bounds, roi_info = roi_bounds_from_group(
            group["members"],
            domain_bounds=global_bounds,
            upstream_h=float(roi_upstream_h),
            downstream_h=float(roi_downstream_h),
            lateral_h=float(roi_lateral_h),
        )
        xs, ys, roi_xy_info = regular_xy_grid(roi_xy_bounds, dx=float(roi_dx), dy=float(roi_dy))

        # Prefer the ground STL over the raw DEM here: the STL has any
        # user-requested ground flattening applied, while the DEM is the
        # original topography. At ROI resolution (~0.5 m) the difference
        # is visible — sampling the DEM leaves the structures appearing
        # to float ~2 m above the rendered terrain.
        terrain_surface = _terrain_surface_for_view(case_meta, xs=xs, ys=ys)
        if not np.isfinite(terrain_surface).all():
            terrain_surface = np.nan_to_num(terrain_surface, nan=float(np.nanmin(terrain_surface)))

        z_base = max(z_base_global, float(np.nanmin(terrain_surface)))
        # The ROI z-top must cover BOTH the structure clearance AND the
        # tallest terrain inside the ROI footprint. Without the terrain
        # term, an ROI placed on a slope ends up with z-top below the
        # surrounding hillshelf — leaving 50–70% of xy columns with
        # "terrain above z-top" → no fluid cells → blank patches in the
        # 2D plots regardless of any prediction.
        terrain_max_local = float(np.nanmax(terrain_surface))
        struct_clearance = float(roi_info["structure_z_max"]) + float(roi_top_h) * float(roi_info["H"])
        terrain_clearance = terrain_max_local + float(roi_top_h) * float(roi_info["H"])
        z_top = min(z_top_global, max(struct_clearance, terrain_clearance))
        z_top = max(z_base + 1.0, z_top)
        vprofile = choose_vertical_profile(category, "auto", grid_kind="roi")
        z_levels = build_z_levels(z_base, z_top, profile=vprofile)

        nx, ny, nz = len(xs), len(ys), len(z_levels)
        Xg, Yg, Zg = np.meshgrid(xs, ys, z_levels, indexing="ij")
        points_xyz = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1).astype(np.float32)

        phi = signed_wall_distance(
            case_meta["structure_stl"],
            points_xyz=points_xyz,
            chunk_size=int(phi_chunk),
            signed=True,
        ).reshape((nx, ny, nz))

        terrain_ij = terrain_surface.T.astype(np.float32, copy=False)  # (nx, ny)
        z_grid = z_levels[None, None, :]
        above_ground = z_grid > (terrain_ij[:, :, None] + 1e-6)
        outside_structure = phi > 0.0
        is_fluid = (above_ground & outside_structure).astype(np.uint8)

        zeros = np.zeros((nx, ny, nz), dtype=np.float32)

        terrain_channels = compute_terrain_channels(
            terrain_surface,
            dx=float(roi_xy_info["actual_dx"]),
            dy=float(roi_xy_info["actual_dy"]),
            extra_channels=(),
        )

        roi_dir = roi_root / f"roi_{idx:03d}"
        roi_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(roi_dir / "terrain.npz", **terrain_channels)
        np.savez_compressed(
            roi_dir / "flow.npz",
            Ux=zeros, Uy=zeros, Uz=zeros, p=zeros,
            is_fluid=is_fluid,
        )
        np.save(roi_dir / "phi_wall.npy", phi.astype(np.float32, copy=False))

        roi_meta = {
            "source": "predict_web_roi_empty_export",
            "is_empty": True,
            "category": category,
            "case_name": case_meta["case_name"],
            "grid_kind": "roi",
            "time": None,
            "grid_shape": [int(nx), int(ny), int(nz)],
            "grid_spacing": [float(roi_xy_info["actual_dx"]), float(roi_xy_info["actual_dy"]), None],
            "z_levels": [float(v) for v in z_levels],
            "bounds": [
                float(xs[0]), float(xs[-1]),
                float(ys[0]), float(ys[-1]),
                float(z_levels[0]), float(z_levels[-1]),
            ],
            "domain_size": [
                float(xs[-1] - xs[0]),
                float(ys[-1] - ys[0]),
                float(z_levels[-1] - z_levels[0]),
            ],
            "ABL": abl,
            "flat_terrain": bool(case_meta.get("flat_terrain", False)),
            "n_structures": int(len(group["members"])),
            "structure_bounds": list(group["members"]),
            "grid_info": grid_info,
            "terrain_channels": list(terrain_channels.keys()),
            "terrain_layout": "[ny, nx]",
            "flow_layout": "[nx, ny, nz]",
            "phi_wall_signed": True,
            "roi_index": int(idx),
            "roi_label": group["label"],
            "roi_mode": group["mode"],
            "component_labels": group["component_labels"],
            "roi_bounds_requested": [float(v) for v in roi_xy_bounds],
            "roi_margins_H": {
                "upstream": float(roi_upstream_h),
                "downstream": float(roi_downstream_h),
                "lateral": float(roi_lateral_h),
                "top": float(roi_top_h),
            },
            "parent_case_meta": "../../meta.json",
            "stats": {
                "fluid_fraction": float(is_fluid.mean()),
                "terrain_z_min": round(float(np.nanmin(terrain_channels["elevation"])), 2),
                "terrain_z_max": round(float(np.nanmax(terrain_channels["elevation"])), 2),
            },
        }
        with open(roi_dir / "meta.json", "w") as f:
            json.dump(roi_meta, f, indent=2)
        roi_paths.append(str(roi_dir.relative_to(cfd_dir)))

    # Patch global meta.json so domain_report helpers can discover the ROIs.
    global_meta_path = cfd_dir / "meta.json"
    gmeta = json.loads(global_meta_path.read_text())
    gmeta["roi_count"] = int(len(roi_paths))
    gmeta["roi_mode"] = roi_mode
    gmeta["roi_paths"] = roi_paths
    with open(global_meta_path, "w") as f:
        json.dump(gmeta, f, indent=2)

    return [{"path": str(cfd_dir / rp), "label": rp} for rp in roi_paths]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_inputs(
    *,
    domain_name: str,
    domain_size: float,
    wind_from: float,
    flat_terrain: bool,
    structures: list[dict],
    center_latlng: Optional[dict] = None,
    uref: float = 10.0,
    zref: float = 20.0,
    z0: float = 0.1,
    flatten_ground: bool = True,
    grid: Optional[dict] = None,
    z_top_offset: Optional[float] = None,
    dx: float = 30.0,
    dy: float = 30.0,
    roi_dx: float = 0.5,
    roi_dy: float = 0.5,
) -> dict[str, Any]:
    """End-to-end input preparation for one prediction run.

    Returns:
      case_dir   OF skeleton dir (has ground.stl + optional structure.stl)
      cfd_dir    binary-array dir (terrain.npz/flow.npz/meta.json + optional roi/)
      rois       list of ROI subdirs that were written (empty if no structures)
    """
    case_dir = build_case_skeleton(
        domain_name=domain_name,
        domain_size=domain_size,
        wind_from=wind_from,
        flat_terrain=flat_terrain,
        structures=structures,
        center_latlng=center_latlng,
        uref=uref,
        zref=zref,
        z0=z0,
        flatten_ground=flatten_ground,
        grid=grid,
        z_top_offset=z_top_offset,
    )
    cfd_dir = export_binary_inputs(domain_name, dx=dx, dy=dy)
    rois = export_roi_inputs(domain_name, roi_dx=roi_dx, roi_dy=roi_dy)
    return {"case_dir": case_dir, "cfd_dir": cfd_dir, "rois": rois}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build model-ready inputs for a predict_web domain.")
    ap.add_argument("--domain-name", required=True)
    ap.add_argument("--domain-size", type=float, required=True)
    ap.add_argument("--wind-from", type=float, required=True)
    ap.add_argument("--flat-terrain", action="store_true")
    ap.add_argument("--uref", type=float, default=10.0)
    ap.add_argument("--zref", type=float, default=20.0)
    ap.add_argument("--z0", type=float, default=0.1)
    ap.add_argument("--dx", type=float, default=30.0)
    ap.add_argument("--dy", type=float, default=30.0)
    args = ap.parse_args()

    out = build_inputs(
        domain_name=args.domain_name,
        domain_size=args.domain_size,
        wind_from=args.wind_from,
        flat_terrain=bool(args.flat_terrain),
        structures=[],
        uref=args.uref,
        zref=args.zref,
        z0=args.z0,
        dx=args.dx,
        dy=args.dy,
    )
    print(json.dumps({"case_dir": str(out["case_dir"]), "cfd_dir": str(out["cfd_dir"]), "rois": out["rois"]}, indent=2))
