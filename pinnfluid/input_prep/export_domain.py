#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from common import (
    build_z_levels,
    choose_vertical_profile,
    compute_terrain_channels,
    find_case_sidecars,
    infer_xy_spacing,
    infer_z_cap_offset,
    log,
    openfoam_internal_mesh,
    read_abl_conditions,
    preferred_structure_metadata,
    regular_xy_grid,
    roi_bounds_from_group,
    sample_openfoam_fields,
    signed_wall_distance,
    structure_roi_groups,
    terrain_surface_on_grid,
    trimmed_xy_grid,
)

ROOT = Path(__file__).resolve().parents[2]
EXTRA_CHOICES = ["curvature", "dog_fine", "dog_coarse"]
DEFAULT_ROI_DX = 0.5
DEFAULT_ROI_DY = 0.5
DEFAULT_ROI_UPSTREAM_H = 5.0
DEFAULT_ROI_DOWNSTREAM_H = 15.0
DEFAULT_ROI_LATERAL_H = 5.0
DEFAULT_ROI_TOP_H = 5.0


def _resolved_roi_margins(case_name: str, category: str, *, upstream_h: float, downstream_h: float, lateral_h: float, top_h: float) -> tuple[float, float, float, float, str]:
    """Return effective ROI margins, keeping existing defaults unless a case-specific profile is needed.

    Large wind-turbine ROI exports were previously only tractable when using a much tighter
    wake box (1H upstream, 5H downstream, 1H lateral, 1H top). Encode that rule here so
    rebuilds remain reproducible without remembering special CLI flags. Explicit non-default
    margins still win.

    The legacy multistructure training set was also exported with the same compact 1H/5H/1H/1H
    ROI profile. Preserve that behavior here so new multistructure exports match the existing
    dataset geometry instead of silently growing to the much larger generic 5H/15H/5H/5H box.
    """
    margins = (float(upstream_h), float(downstream_h), float(lateral_h), float(top_h))
    default_margins = (
        float(DEFAULT_ROI_UPSTREAM_H),
        float(DEFAULT_ROI_DOWNSTREAM_H),
        float(DEFAULT_ROI_LATERAL_H),
        float(DEFAULT_ROI_TOP_H),
    )
    if margins != default_margins:
        return margins[0], margins[1], margins[2], margins[3], 'explicit'

    case_name_l = str(case_name).lower()
    if category == 'singlestructures' and 'largewt' in case_name_l:
        return 1.0, 5.0, 1.0, 1.0, 'largewt_compact'
    if category == 'multistructures':
        return 1.0, 5.0, 1.0, 1.0, 'multistructure_compact_legacy'

    return margins[0], margins[1], margins[2], margins[3], 'default'


def _shape_info(arr: np.ndarray) -> str:
    return "x".join(str(v) for v in arr.shape)


def _build_points_xyz(xs: np.ndarray, ys: np.ndarray, z_levels: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    z_levels = np.asarray(z_levels, dtype=np.float32)

    ny = len(ys)
    nz = len(z_levels)
    yz = np.empty((ny * nz, 2), dtype=np.float32)
    yz[:, 0] = np.repeat(ys, nz)
    yz[:, 1] = np.tile(z_levels, ny)

    block = ny * nz
    out = np.empty((len(xs) * block, 3), dtype=np.float32)
    for i, x in enumerate(xs):
        start = i * block
        end = start + block
        out[start:end, 0] = x
        out[start:end, 1:] = yz
    return out


def _export_grid_bundle(
    *,
    case_meta: dict,
    case_dir: Path,
    internal_mesh,
    output_dir: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    raw_mesh_bounds: Sequence[float],
    resolution_profile: str,
    vertical_profile_name: str,
    extra_channels: Sequence[str],
    sample_chunk: int,
    phi_chunk: int,
    phi_signed: bool,
    abl: dict,
    active_time: float | None,
    grid_kind: str,
    xy_info: dict,
    structure_bounds_meta: Sequence[dict],
    grid_info: dict | None,
    include_phi_wall: bool,
    z_cap_offset_m: float | None,
    z_top_limit: float | None,
    vertical_coordinate_mode: str = "absolute",
    extra_meta: dict | None = None,
) -> dict:
    terrain_surface = terrain_surface_on_grid(case_meta, xs=xs, ys=ys)
    if not np.isfinite(terrain_surface).all():
        bad = int(np.size(terrain_surface) - np.isfinite(terrain_surface).sum())
        raise ValueError(f"Terrain surface for {case_meta['case_name']} still has {bad} NaN values after fallback.")

    terrain_ij = terrain_surface.T.astype(np.float32, copy=False)
    vertical_coordinate_mode = str(vertical_coordinate_mode or "absolute").lower()
    if vertical_coordinate_mode not in {"absolute", "terrain_following"}:
        raise ValueError(f"Unsupported vertical_coordinate_mode={vertical_coordinate_mode!r}")

    if vertical_coordinate_mode == "absolute":
        z_base = max(float(raw_mesh_bounds[4]), float(np.nanmin(terrain_surface)))
        if z_top_limit is None:
            if z_cap_offset_m is None:
                raise ValueError("z_cap_offset_m is required when z_top_limit is not provided.")
            z_top = min(float(raw_mesh_bounds[5]), float(np.nanmax(terrain_surface)) + float(z_cap_offset_m))
        else:
            z_top = min(float(raw_mesh_bounds[5]), float(z_top_limit))
            z_top = max(float(z_base), float(z_top))
        z_levels = build_z_levels(z_base, z_top, profile=vertical_profile_name)

        nx, ny, nz = len(xs), len(ys), len(z_levels)
        points_xyz = _build_points_xyz(xs, ys, z_levels)
        log_z = z_top

        log(
            f"Exporting {grid_kind} {case_meta['category']}/{case_meta['case_name']}: "
            f"grid={nx}x{ny}x{nz} (dx={xy_info['actual_dx']:.3f}m, dy={xy_info['actual_dy']:.3f}m, z_top={log_z:.3f}m)"
        )

        sampled = sample_openfoam_fields(internal_mesh, points_xyz=points_xyz, chunk_size=sample_chunk)
        terrain_3d = terrain_ij[:, :, None]
        valid = sampled["valid"].reshape((nx, ny, nz))
        z_grid_abs = z_levels[None, None, :]
        is_fluid = valid & np.isfinite(terrain_3d) & (z_grid_abs > terrain_3d + 1e-6)

        flow_arrays = {}
        for key in ("Ux", "Uy", "Uz", "p"):
            arr = sampled[key].reshape((nx, ny, nz)).astype(np.float32, copy=False)
            arr[~is_fluid] = np.nan
            flow_arrays[key] = arr
        bounds_z0 = float(z_levels[0])
        bounds_z1 = float(z_levels[-1])
        domain_z = float(z_levels[-1] - z_levels[0])
        phi_points_xyz = points_xyz
        phi_fill_indices = None
    else:
        if z_top_limit is None:
            if z_cap_offset_m is None:
                raise ValueError("z_cap_offset_m is required for terrain-following global export.")
            z_rel_top = float(z_cap_offset_m)
        else:
            z_rel_top = max(1.0, float(z_top_limit) - float(np.nanmin(terrain_surface)))
        z_levels = build_z_levels(0.0, z_rel_top, profile=vertical_profile_name)
        z_levels = z_levels[z_levels > 1.0e-6]
        if z_levels.size == 0:
            z_levels = np.asarray([min(2.0, max(1.0, z_rel_top))], dtype=np.float32)

        nx, ny, nz = len(xs), len(ys), len(z_levels)
        z_abs_grid = terrain_ij[:, :, None] + z_levels[None, None, :]
        within_domain = (
            np.isfinite(z_abs_grid)
            & (z_abs_grid >= float(raw_mesh_bounds[4]) + 1.0e-6)
            & (z_abs_grid <= float(raw_mesh_bounds[5]) - 1.0e-6)
            & (z_levels[None, None, :] > 0.0)
        )
        ii, jj, kk = np.nonzero(within_domain)
        if len(ii) == 0:
            raise ValueError(f"No terrain-following sample points fit inside mesh bounds for {case_meta['case_name']}.")
        points_xyz = np.stack(
            [
                np.asarray(xs, dtype=np.float32)[ii],
                np.asarray(ys, dtype=np.float32)[jj],
                z_abs_grid[ii, jj, kk].astype(np.float32, copy=False),
            ],
            axis=1,
        )
        log_z = float(np.nanmax(z_abs_grid[within_domain]))

        log(
            f"Exporting {grid_kind} {case_meta['category']}/{case_meta['case_name']}: "
            f"grid={nx}x{ny}x{nz} terrain-following "
            f"(dx={xy_info['actual_dx']:.3f}m, dy={xy_info['actual_dy']:.3f}m, z_rel_top={float(z_levels[-1]):.3f}m)"
        )

        sampled = sample_openfoam_fields(internal_mesh, points_xyz=points_xyz, chunk_size=sample_chunk)
        valid = np.zeros((nx, ny, nz), dtype=bool)
        valid[ii, jj, kk] = sampled["valid"]
        is_fluid = valid & within_domain

        flow_arrays = {}
        for key in ("Ux", "Uy", "Uz", "p"):
            arr = np.full((nx, ny, nz), np.nan, dtype=np.float32)
            vals = sampled[key].astype(np.float32, copy=False)
            arr[ii, jj, kk] = vals
            arr[~is_fluid] = np.nan
            flow_arrays[key] = arr
        finite_z = z_abs_grid[within_domain]
        bounds_z0 = float(np.nanmin(finite_z))
        bounds_z1 = float(np.nanmax(finite_z))
        domain_z = float(bounds_z1 - bounds_z0)
        phi_points_xyz = points_xyz
        phi_fill_indices = (ii, jj, kk)

    terrain_channels = compute_terrain_channels(
        terrain_surface,
        dx=float(xy_info["actual_dx"]),
        dy=float(xy_info["actual_dy"]),
        extra_channels=list(extra_channels),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "terrain.npz", **terrain_channels)
    np.savez_compressed(
        output_dir / "flow.npz",
        Ux=flow_arrays["Ux"],
        Uy=flow_arrays["Uy"],
        Uz=flow_arrays["Uz"],
        p=flow_arrays["p"],
        is_fluid=is_fluid.astype(np.uint8),
    )

    phi_written = False
    if include_phi_wall and case_meta["structure_stl"] is not None:
        sampled_phi = signed_wall_distance(
            case_meta["structure_stl"],
            points_xyz=phi_points_xyz,
            chunk_size=phi_chunk,
            signed=phi_signed,
        )
        if phi_fill_indices is None:
            phi = sampled_phi.reshape((nx, ny, nz))
        else:
            phi = np.full((nx, ny, nz), np.nan, dtype=np.float32)
            phi[phi_fill_indices] = sampled_phi.astype(np.float32, copy=False)
        np.save(output_dir / "phi_wall.npy", phi.astype(np.float32, copy=False))
        phi_written = True

    meta = {
        "source": "openfoam_direct_binary_export",
        "source_case_dir": str(case_dir),
        "category": case_meta["category"],
        "case_name": case_meta["case_name"],
        "grid_kind": grid_kind,
        "time": active_time,
        "grid_shape": [int(nx), int(ny), int(nz)],
        "grid_spacing": [float(xy_info["actual_dx"]), float(xy_info["actual_dy"]), None],
        "z_levels": [float(v) for v in z_levels],
        "vertical_coordinate_mode": vertical_coordinate_mode,
        "z_levels_are": (
            "relative_to_local_terrain" if vertical_coordinate_mode == "terrain_following" else "absolute_elevation"
        ),
        "bounds": [
            float(xs[0]),
            float(xs[-1]),
            float(ys[0]),
            float(ys[-1]),
            bounds_z0,
            bounds_z1,
        ],
        "domain_size": [
            float(xs[-1] - xs[0]),
            float(ys[-1] - ys[0]),
            domain_z,
        ],
        "ABL": {
            "Uref": float(abl["Uref"]),
            "Zref": float(abl["Zref"]),
            "z0": float(abl["z0"]),
            "flowDir": [float(v) for v in abl["flowDir"]],
            "wind_from_deg": float(abl.get("wind_from_deg", np.nan)),
        },
        "flat_terrain": bool(case_meta["flat_terrain"]),
        "n_structures": int(len(structure_bounds_meta)),
        "structure_bounds": list(structure_bounds_meta),
        "grid_info": grid_info,
        "terrain_channels": list(terrain_channels.keys()),
        "terrain_layout": "[ny, nx]",
        "flow_layout": "[nx, ny, nz]",
        "phi_wall_signed": bool(phi_signed) if phi_written else None,
        "preprocessing": {
            "resolution_profile": resolution_profile,
            "xy": xy_info,
            "vertical_profile": vertical_profile_name,
            "vertical_coordinate_mode": vertical_coordinate_mode,
            "z_cap_offset_m": float(z_cap_offset_m) if z_cap_offset_m is not None else None,
            "z_top_limit_m": float(z_top_limit) if z_top_limit is not None else None,
            "raw_mesh_bounds": [float(v) for v in raw_mesh_bounds],
            "terrain_source": str(case_meta["dem_path"] or case_meta["ground_stl"]),
            "z_offset_applied": float(case_meta.get("z_offset_applied", 0.0)),
        },
        "stats": {
            "terrain_z_min": round(float(np.nanmin(terrain_channels["elevation"])), 2),
            "terrain_z_max": round(float(np.nanmax(terrain_channels["elevation"])), 2),
            "terrain_relief": round(float(np.nanmax(terrain_channels["elevation"]) - np.nanmin(terrain_channels["elevation"])), 2),
            "fluid_fraction": float(is_fluid.mean()),
            "nan_fraction_Ux": float(np.isnan(flow_arrays["Ux"]).mean()),
        },
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(
        f"Wrote {output_dir}: terrain={_shape_info(terrain_surface)}, flow={_shape_info(flow_arrays['Ux'])}, "
        f"phi_wall={'yes' if phi_written else 'no'}"
    )
    return meta


def export_case(
    case_dir: Path,
    output_dir: Path,
    *,
    dx: float | None,
    dy: float | None,
    edge_buffer_m: float | None,
    edge_cells: int,
    z_cap_offset: float | None,
    vertical_profile: str,
    roi_dx: float,
    roi_dy: float,
    roi_vertical_profile: str,
    roi_upstream_h: float,
    roi_downstream_h: float,
    roi_lateral_h: float,
    roi_top_h: float,
    extra_channels: Sequence[str],
    phi_signed: bool,
    time_value: float | None,
    sample_chunk: int,
    phi_chunk: int,
    overwrite: bool,
    global_z_mode: str = "absolute",
    skip_roi: bool = False,
) -> dict:
    case_dir = case_dir.resolve()
    case_meta = find_case_sidecars(case_dir, ROOT)
    category = case_meta["category"]
    case_name = case_meta["case_name"]

    if output_dir.exists():
        if not overwrite:
            if (output_dir / "meta.json").exists():
                raise FileExistsError(f"Output already exists: {output_dir}")
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    internal_mesh, active_time = openfoam_internal_mesh(case_dir, time_value=time_value)
    bounds = [float(v) for v in internal_mesh.bounds]
    abl = read_abl_conditions(case_dir, domain_info=case_meta["domain_info"], transform=case_meta["transform"])

    dx_req, dy_req, resolution_profile = infer_xy_spacing(category, bounds, dx, dy)
    xs, ys, trim = trimmed_xy_grid(
        bounds,
        dx=dx_req,
        dy=dy_req,
        edge_buffer_m=edge_buffer_m,
        edge_cells=edge_cells,
    )
    global_meta = _export_grid_bundle(
        case_meta=case_meta,
        case_dir=case_dir,
        internal_mesh=internal_mesh,
        output_dir=output_dir,
        xs=xs,
        ys=ys,
        raw_mesh_bounds=bounds,
        resolution_profile=resolution_profile,
        vertical_profile_name=choose_vertical_profile(category, vertical_profile, grid_kind="global"),
        extra_channels=extra_channels,
        sample_chunk=sample_chunk,
        phi_chunk=phi_chunk,
        phi_signed=phi_signed,
        abl=abl,
        active_time=active_time,
        grid_kind="global",
        xy_info=trim,
        structure_bounds_meta=[],
        grid_info=case_meta["domain_info"].get("grid"),
        include_phi_wall=False,
        z_cap_offset_m=infer_z_cap_offset(bounds, z_cap_offset),
        z_top_limit=None,
        vertical_coordinate_mode=global_z_mode,
        extra_meta=None,
    )

    structure_bounds, grid_info, n_structures, structure_meta_source = preferred_structure_metadata(case_meta)
    roi_paths: list[str] = []
    roi_mode = None

    global_meta["n_structures"] = int(n_structures)
    global_meta["structure_bounds"] = structure_bounds
    global_meta["grid_info"] = grid_info
    global_meta["structure_metadata_source"] = structure_meta_source

    if skip_roi:
        global_meta["roi_mode"] = None
        global_meta["roi_count"] = 0
        global_meta["roi_paths"] = []
        global_meta["preprocessing"]["skip_roi"] = True
        with open(output_dir / "meta.json", "w") as f:
            json.dump(global_meta, f, indent=2)
        return global_meta

    if case_meta["structure_stl"] is not None and structure_bounds:
        groups = structure_roi_groups(category, structure_bounds)
        roi_mode = groups[0]["mode"] if groups else None
        eff_upstream_h, eff_downstream_h, eff_lateral_h, eff_top_h, roi_margin_profile = _resolved_roi_margins(
            case_name,
            category,
            upstream_h=roi_upstream_h,
            downstream_h=roi_downstream_h,
            lateral_h=roi_lateral_h,
            top_h=roi_top_h,
        )
        if roi_margin_profile != 'default':
            log(
                f"ROI margin profile for {category}/{case_name}: {roi_margin_profile} "
                f"(up={eff_upstream_h:g}H, down={eff_downstream_h:g}H, lat={eff_lateral_h:g}H, top={eff_top_h:g}H)"
            )
        roi_root = output_dir / "roi"
        for idx, group in enumerate(groups):
            roi_xy_bounds, roi_info = roi_bounds_from_group(
                group["members"],
                domain_bounds=bounds,
                upstream_h=eff_upstream_h,
                downstream_h=eff_downstream_h,
                lateral_h=eff_lateral_h,
            )
            roi_xs, roi_ys, roi_grid = regular_xy_grid(roi_xy_bounds, dx=roi_dx, dy=roi_dy)
            roi_dir = roi_root / f"roi_{idx:03d}"
            roi_z_top = min(float(bounds[5]), float(roi_info["structure_z_max"]) + float(eff_top_h) * float(roi_info["H"]))
            roi_meta = _export_grid_bundle(
                case_meta=case_meta,
                case_dir=case_dir,
                internal_mesh=internal_mesh,
                output_dir=roi_dir,
                xs=roi_xs,
                ys=roi_ys,
                raw_mesh_bounds=bounds,
                resolution_profile=f"roi_{roi_dx:g}m",
                vertical_profile_name=choose_vertical_profile(category, roi_vertical_profile, grid_kind="roi"),
                extra_channels=extra_channels,
                sample_chunk=sample_chunk,
                phi_chunk=phi_chunk,
                phi_signed=phi_signed,
                abl=abl,
                active_time=active_time,
                grid_kind="roi",
                xy_info=roi_grid,
                structure_bounds_meta=group["members"],
                grid_info=grid_info,
                include_phi_wall=True,
                z_cap_offset_m=None,
                z_top_limit=roi_z_top,
                vertical_coordinate_mode="absolute",
                extra_meta={
                    "roi_index": int(idx),
                    "roi_label": group["label"],
                    "roi_mode": group["mode"],
                    "component_labels": group["component_labels"],
                    "roi_bounds_requested": [float(v) for v in roi_xy_bounds],
                    "roi_margin_profile": roi_margin_profile,
                    "roi_margins_H": {
                        "upstream": float(eff_upstream_h),
                        "downstream": float(eff_downstream_h),
                        "lateral": float(eff_lateral_h),
                        "top": float(eff_top_h),
                    },
                    "parent_case_meta": str(Path("..") / ".." / "meta.json"),
                },
            )
            roi_paths.append(str(roi_dir.relative_to(output_dir)))

        global_meta["n_structures"] = int(n_structures)
        global_meta["structure_bounds"] = structure_bounds
        global_meta["roi_mode"] = roi_mode
        global_meta["roi_count"] = int(len(roi_paths))
        global_meta["roi_paths"] = roi_paths
        with open(output_dir / "meta.json", "w") as f:
            json.dump(global_meta, f, indent=2)

    return global_meta


def export_empty_case(
    case_dir: Path,
    output_dir: Path,
    *,
    dx: float = 30.0,
    dy: float = 30.0,
    edge_cells: int = 1,
    z_cap_offset: float | None = None,
    overwrite: bool = False,
) -> dict:
    """Lite export for an empty domain (no CFD).

    Reads the OF case skeleton (domain_info.json + dem_final.tif + ground.stl)
    and writes terrain.npz, flow.npz (zeros + is_fluid), and meta.json with
    `is_empty=True` marker. Output is loadable by the existing
    data_loader.CaseRepository (just point it at data/empty_domains/).

    No ROI is exported. Empty cases are global-only.
    """
    case_dir = case_dir.resolve()
    case_meta = find_case_sidecars(case_dir, ROOT)
    category = case_meta["category"]
    case_name = case_meta["case_name"]

    if output_dir.exists():
        if not overwrite:
            if (output_dir / "meta.json").exists():
                raise FileExistsError(f"Output already exists: {output_dir}")
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine domain size from domain_info.json or transform.json
    domain_info = case_meta["domain_info"] or {}
    transform = case_meta["transform"] or {}
    dsz = domain_info.get("domain_size") or transform.get("final_W")
    if isinstance(dsz, list):
        Lx = float(dsz[0])
        Ly = float(dsz[1]) if len(dsz) > 1 else Lx
    elif dsz is not None:
        Lx = Ly = float(dsz)
    else:
        raise ValueError(f"Could not determine domain size for {case_name} (no domain_info.json or transform.json)")

    raw_bounds = [0.0, Lx, 0.0, Ly, 0.0, 0.0]
    xs, ys, trim = trimmed_xy_grid(
        raw_bounds,
        dx=float(dx),
        dy=float(dy),
        edge_buffer_m=None,
        edge_cells=int(edge_cells),
    )

    terrain_surface = terrain_surface_on_grid(case_meta, xs=xs, ys=ys)
    if not np.isfinite(terrain_surface).all():
        nan_count = int(np.size(terrain_surface) - np.isfinite(terrain_surface).sum())
        log(f"Filled {nan_count} NaN terrain cells with 0 for {case_name}")
        terrain_surface = np.nan_to_num(terrain_surface, nan=0.0)

    # ABL: read from domain_info.json (no ABLConditions file in empty cases)
    di_abl = domain_info.get("ABL", {})
    abl = {
        "Uref": float(di_abl.get("Uref", 10.0)),
        "Zref": float(di_abl.get("Zref", 20.0)),
        "z0": float(di_abl.get("z0", 0.1)),
        "flowDir": [1.0, 0.0, 0.0],  # canonical: domain is pre-rotated so wind always blows +x
        "wind_from_deg": float(domain_info.get("wind_from", 270.0)),
    }

    # Z levels: terrain profile with size-dependent cap (matches trained export)
    z_base = float(np.nanmin(terrain_surface))
    z_top_offset = float(z_cap_offset) if z_cap_offset is not None else infer_z_cap_offset(raw_bounds, None)
    z_top = float(np.nanmax(terrain_surface)) + z_top_offset
    z_top = max(z_base + 1.0, z_top)
    z_levels = build_z_levels(z_base, z_top, profile="terrain")

    # Terrain channels
    terrain_channels = compute_terrain_channels(
        terrain_surface,
        dx=float(trim["actual_dx"]),
        dy=float(trim["actual_dy"]),
        extra_channels=(),
    )

    # is_fluid mask: z > terrain at (i, j)
    nx, ny, nz = len(xs), len(ys), len(z_levels)
    terrain_ij = terrain_surface.T.astype(np.float32, copy=False)  # (nx, ny)
    z_grid = z_levels[None, None, :]  # (1, 1, nz)
    is_fluid = (z_grid > terrain_ij[:, :, None] + 1e-6).astype(np.uint8)

    # Zero flow (model never reads it for AL — only consults is_fluid + terrain)
    zeros = np.zeros((nx, ny, nz), dtype=np.float32)

    # Write arrays
    np.savez_compressed(output_dir / "terrain.npz", **terrain_channels)
    np.savez_compressed(
        output_dir / "flow.npz",
        Ux=zeros, Uy=zeros, Uz=zeros, p=zeros,
        is_fluid=is_fluid,
    )

    # Structure metadata (if present) — no phi_wall.npy for empty cases (skip ROI)
    structure_bounds, grid_info, n_structures, structure_meta_source = preferred_structure_metadata(case_meta)

    meta = {
        "source": "openfoam_lite_empty_export",
        "is_empty": True,
        "source_case_dir": str(case_dir),
        "category": category,
        "case_name": case_name,
        "grid_kind": "global",
        "time": None,
        "grid_shape": [int(nx), int(ny), int(nz)],
        "grid_spacing": [float(trim["actual_dx"]), float(trim["actual_dy"]), None],
        "z_levels": [float(v) for v in z_levels],
        "bounds": [
            float(xs[0]),
            float(xs[-1]),
            float(ys[0]),
            float(ys[-1]),
            float(z_levels[0]),
            float(z_levels[-1]),
        ],
        "domain_size": [
            float(xs[-1] - xs[0]),
            float(ys[-1] - ys[0]),
            float(z_levels[-1] - z_levels[0]),
        ],
        "ABL": abl,
        "flat_terrain": bool(case_meta.get("flat_terrain", False)),
        "n_structures": int(n_structures),
        "structure_bounds": list(structure_bounds),
        "grid_info": grid_info,
        "structure_metadata_source": structure_meta_source,
        "diversity_score": (
            float(domain_info["diversity_score"])
            if "diversity_score" in domain_info
            else None
        ),
        "terrain_channels": list(terrain_channels.keys()),
        "terrain_layout": "[ny, nx]",
        "flow_layout": "[nx, ny, nz]",
        "phi_wall_signed": None,
        "preprocessing": {
            "resolution_profile": "global_30m",
            "xy": trim,
            "vertical_profile": "terrain",
            "z_cap_offset_m": float(z_top_offset),
            "z_top_limit_m": float(z_top),
            "raw_mesh_bounds": [float(v) for v in raw_bounds],
            "terrain_source": str(case_meta["dem_path"] or case_meta["ground_stl"]),
            "z_offset_applied": float(case_meta.get("z_offset_applied", 0.0)),
        },
        "stats": {
            "terrain_z_min": round(float(np.nanmin(terrain_channels["elevation"])), 2),
            "terrain_z_max": round(float(np.nanmax(terrain_channels["elevation"])), 2),
            "terrain_relief": round(float(np.nanmax(terrain_channels["elevation"]) - np.nanmin(terrain_channels["elevation"])), 2),
            "fluid_fraction": float(is_fluid.mean()),
            "nan_fraction_Ux": 0.0,
        },
    }

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(
        f"Wrote empty {output_dir}: terrain={_shape_info(terrain_surface)}, "
        f"grid={nx}x{ny}x{nz}, fluid_frac={is_fluid.mean():.3f}"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Export one OpenFOAM case into unified binary arrays.")
    ap.add_argument("--case-dir", required=True, help="Path to one OpenFOAM case under data_preparation/<category>/<case>.")
    ap.add_argument("--output-dir", required=True, help="Output directory, typically data/cfd/<category>/<case>.")
    ap.add_argument("--dx", type=float, default=None, help="Override GLOBAL horizontal spacing in x (m). Default: 30 m for all categories.")
    ap.add_argument("--dy", type=float, default=None, help="Override GLOBAL horizontal spacing in y (m). Default: 30 m for all categories.")
    ap.add_argument("--edge-buffer", type=float, default=None, help="Minimum inward trim from x/y boundaries in metres for the global grid.")
    ap.add_argument("--edge-cells", type=int, default=1, help="Additional trim in units of 1*dx/1*dy cells for the global grid.")
    ap.add_argument("--z-cap-offset", type=float, default=None, help="Override GLOBAL z-cap above local terrain max (m).")
    ap.add_argument(
        "--global-z-mode",
        choices=["absolute", "terrain_following"],
        default="absolute",
        help=(
            "GLOBAL vertical coordinate convention. absolute preserves the historical export; "
            "terrain_following stores z_levels as metres above local terrain and samples OpenFOAM at terrain+z_rel."
        ),
    )
    ap.add_argument(
        "--vertical-profile",
        choices=["auto", "terrain", "structure"],
        default="auto",
        help="GLOBAL vertical spacing template. auto -> terrain.",
    )
    ap.add_argument("--roi-dx", type=float, default=DEFAULT_ROI_DX, help="ROI horizontal spacing in x (m). Default: 0.5.")
    ap.add_argument("--roi-dy", type=float, default=DEFAULT_ROI_DY, help="ROI horizontal spacing in y (m). Default: 0.5.")
    ap.add_argument(
        "--roi-vertical-profile",
        choices=["auto", "terrain", "structure"],
        default="auto",
        help="ROI vertical spacing template. auto -> structure.",
    )
    ap.add_argument("--roi-upstream-h", type=float, default=DEFAULT_ROI_UPSTREAM_H, help="ROI upstream margin in multiples of H.")
    ap.add_argument("--roi-downstream-h", type=float, default=DEFAULT_ROI_DOWNSTREAM_H, help="ROI downstream margin in multiples of H.")
    ap.add_argument("--roi-lateral-h", type=float, default=DEFAULT_ROI_LATERAL_H, help="ROI lateral margin in multiples of H.")
    ap.add_argument("--roi-top-h", type=float, default=DEFAULT_ROI_TOP_H, help="ROI top margin in multiples of H.")
    ap.add_argument(
        "--extra-channels",
        nargs="*",
        default=[],
        choices=EXTRA_CHOICES,
        help="Optional extra terrain channels to store in terrain.npz.",
    )
    ap.add_argument("--time", type=float, default=None, help="Optional OpenFOAM time to export. Default: latest.")
    ap.add_argument("--sample-chunk", type=int, default=500_000, help="Chunk size for OpenFOAM sampling.")
    ap.add_argument("--phi-chunk", type=int, default=200_000, help="Chunk size for phi_wall computation.")
    ap.add_argument("--phi-unsigned", action="store_true", help="Store |implicit_distance| instead of signed distance in ROI phi_wall.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory contents.")
    ap.add_argument("--skip-roi", action="store_true", help="Export only the global bundle, even for structure cases.")
    ap.add_argument(
        "--mode",
        choices=["full", "empty"],
        default="full",
        help="full = read CFD output (default); empty = lite export from OF skeleton only (no CFD, no ROI).",
    )
    args = ap.parse_args()

    if args.mode == "empty":
        export_empty_case(
            Path(args.case_dir),
            Path(args.output_dir),
            dx=float(args.dx) if args.dx is not None else 30.0,
            dy=float(args.dy) if args.dy is not None else 30.0,
            edge_cells=int(args.edge_cells),
            z_cap_offset=args.z_cap_offset,
            overwrite=bool(args.overwrite),
        )
        return

    export_case(
        Path(args.case_dir),
        Path(args.output_dir),
        dx=args.dx,
        dy=args.dy,
        edge_buffer_m=args.edge_buffer,
        edge_cells=args.edge_cells,
        z_cap_offset=args.z_cap_offset,
        vertical_profile=args.vertical_profile,
        roi_dx=float(args.roi_dx),
        roi_dy=float(args.roi_dy),
        roi_vertical_profile=args.roi_vertical_profile,
        roi_upstream_h=float(args.roi_upstream_h),
        roi_downstream_h=float(args.roi_downstream_h),
        roi_lateral_h=float(args.roi_lateral_h),
        roi_top_h=float(args.roi_top_h),
        extra_channels=args.extra_channels,
        phi_signed=not args.phi_unsigned,
        time_value=args.time,
        sample_chunk=max(1, int(args.sample_chunk)),
        phi_chunk=max(1, int(args.phi_chunk)),
        overwrite=bool(args.overwrite),
        global_z_mode=str(args.global_z_mode),
        skip_roi=bool(args.skip_roi),
    )


if __name__ == "__main__":
    main()
