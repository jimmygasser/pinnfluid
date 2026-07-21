#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
import rasterio
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

PREFERRED_VELOCITY = ["U", "velocity", "Velocity", "u", "U_mean"]
PREFERRED_PRESSURE = ["p", "pressure", "P", "p_rgh"]
PREFERRED_NUT = ["nut", "nuTilda", "nut_mean"]
PREFERRED_K = ["k", "tke", "K"]
PREFERRED_EPS = ["epsilon", "Epsilon", "eps"]


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def _pick_array(arrs: Dict, preferred: Iterable[str]) -> Optional[str]:
    keys = list(arrs.keys())
    lower = {k.lower(): k for k in keys}
    for name in preferred:
        if name in arrs:
            return name
    for name in preferred:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def ensure_foam_file(case_dir: Path) -> Path:
    candidates = [case_dir / "foam.foam"] + sorted(case_dir.glob("*.foam")) + [case_dir / "mesh.mesh"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No readable OpenFOAM marker file found in {case_dir} "
        f"(expected foam.foam, *.foam, or mesh.mesh)."
    )


def _extract_internal_mesh(root: pv.MultiBlock):
    if not isinstance(root, pv.MultiBlock):
        return root
    if "internalMesh" in root.keys():
        return root["internalMesh"]
    for key in root.keys():
        block = root[key]
        if block is not None and getattr(block, "n_points", 0) > 0:
            return block
    raise RuntimeError("Could not find internal mesh in OpenFOAM reader output.")


def maybe_cell_to_point(mesh):
    if not hasattr(mesh, "point_data") or not hasattr(mesh, "cell_data"):
        return mesh

    missing = []
    checks = [
        ("U", PREFERRED_VELOCITY),
        ("p", PREFERRED_PRESSURE),
        ("nut", PREFERRED_NUT),
        ("k", PREFERRED_K),
        ("epsilon", PREFERRED_EPS),
    ]
    for label, preferred in checks:
        if _pick_array(mesh.point_data, preferred) is None and _pick_array(mesh.cell_data, preferred) is not None:
            missing.append(label)

    if not missing:
        return mesh

    log(f"Point data missing arrays {missing}; applying cell_data_to_point_data() fallback.")
    try:
        return mesh.cell_data_to_point_data()
    except Exception as exc:  # pragma: no cover - defensive
        warn(f"cell_data_to_point_data() failed ({type(exc).__name__}: {exc}); using raw mesh.")
        return mesh


ABL_RE = {
    "Uref": re.compile(r"^\s*Uref\s+([-+0-9.eE]+)\s*;", re.MULTILINE),
    "Zref": re.compile(r"^\s*Zref\s+([-+0-9.eE]+)\s*;", re.MULTILINE),
    "z0": re.compile(r"^\s*z0\s+(?:uniform\s+)?([-+0-9.eE]+)\s*;", re.MULTILINE),
    "flowDir": re.compile(r"^\s*flowDir\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)\s*;", re.MULTILINE),
}


def read_abl_conditions(case_dir: Path, *, domain_info: Optional[dict] = None, transform: Optional[dict] = None) -> dict:
    abl_path = case_dir / "0" / "include" / "ABLConditions"
    if not abl_path.exists():
        raise FileNotFoundError(f"Missing ABLConditions file: {abl_path}")

    text = abl_path.read_text()
    out = {}
    for key, regex in ABL_RE.items():
        match = regex.search(text)
        if not match:
            continue
        if key == "flowDir":
            out[key] = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
        else:
            out[key] = float(match.group(1))

    if "flowDir" not in out:
        out["flowDir"] = [1.0, 0.0, 0.0]

    wind_from = None
    if transform is not None:
        wind_from = transform.get("wind_from_deg")
    if wind_from is None and domain_info is not None:
        wind_from = domain_info.get("wind_from")
    if wind_from is not None:
        out["wind_from_deg"] = float(wind_from)

    missing = [key for key in ("Uref", "Zref", "z0") if key not in out]
    if missing:
        raise ValueError(f"ABLConditions missing keys {missing} in {abl_path}")
    return out


def find_case_sidecars(case_dir: Path, repo_root: Path) -> dict:
    category = case_dir.parent.name
    case_name = case_dir.name
    tri_dir = case_dir / "constant" / "triSurface"

    transform = load_json(tri_dir / "transform.json") or {}
    domain_info = load_json(tri_dir / "domain_info.json") or {}
    placed_spec = load_json(repo_root / "dem" / case_name / "placed" / "domain_spec.json") or {}

    dem_case = tri_dir / "dem_final.tif"
    dem_repo = repo_root / "dem" / case_name / "prep" / "dem_final.tif"
    dem_placed = Path(placed_spec.get("terrain", {}).get("dem_tif", "")) if placed_spec else None
    if dem_case.exists():
        dem_path = dem_case
    elif dem_repo.exists():
        dem_path = dem_repo
    elif dem_placed is not None and dem_placed.exists():
        dem_path = dem_placed
    else:
        dem_path = None

    ground_stl = tri_dir / "ground.stl"
    if not ground_stl.exists():
        ground_stl = tri_dir / "terrain.stl"
    ground_stl = ground_stl if ground_stl.exists() else None

    structure_stl = tri_dir / "structure.stl"
    structure_stl = structure_stl if structure_stl.exists() else None

    flat_terrain = bool(domain_info.get("flat_terrain", placed_spec.get("terrain", {}).get("flat", False)))
    z_offset = float(transform.get("z_offset_applied", 0.0))

    return {
        "category": category,
        "case_name": case_name,
        "tri_dir": tri_dir,
        "transform": transform,
        "domain_info": domain_info,
        "placed_spec": placed_spec,
        "dem_path": dem_path,
        "ground_stl": ground_stl,
        "structure_stl": structure_stl,
        "flat_terrain": flat_terrain,
        "z_offset_applied": z_offset,
    }


def infer_xy_spacing(category: str, bounds: Sequence[float], dx: Optional[float], dy: Optional[float]) -> Tuple[float, float, str]:
    if dx is not None or dy is not None:
        if dx is None:
            dx = dy
        if dy is None:
            dy = dx
        return float(dx), float(dy), "manual"

    if category in {"complexterrain_only", "singlestructures", "multistructures"}:
        return 30.0, 30.0, "global_30m"

    x_min, x_max, y_min, y_max = bounds[:4]
    span = max(float(x_max - x_min), float(y_max - y_min))
    warn(f"Unknown category {category!r}; defaulting global export spacing from span {span:g}m.")
    return 30.0, 30.0, "default_30m"


def infer_edge_buffer(dx: float, dy: float, edge_buffer_m: Optional[float], edge_cells: int) -> float:
    base = max(dx, dy) * max(int(edge_cells), 0)
    if edge_buffer_m is None:
        return max(2.5, base)
    return max(float(edge_buffer_m), base)


def trimmed_xy_grid(
    bounds: Sequence[float],
    *,
    dx: float,
    dy: float,
    edge_buffer_m: Optional[float],
    edge_cells: int,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    x_min, x_max, y_min, y_max = map(float, bounds[:4])
    buffer_requested = infer_edge_buffer(dx, dy, edge_buffer_m, edge_cells)
    min_extent = max(float(dx), float(dy))
    max_buffer_x = max(0.0, 0.5 * ((x_max - x_min) - min_extent))
    max_buffer_y = max(0.0, 0.5 * ((y_max - y_min) - min_extent))
    buffer = min(buffer_requested, max_buffer_x, max_buffer_y)
    if buffer < buffer_requested - 1e-9:
        warn(
            f"Requested trim buffer {buffer_requested:g}m is too large for bounds {bounds[:4]}; "
            f"using relaxed buffer {buffer:g}m instead."
        )
    x0 = x_min + buffer
    x1 = x_max - buffer
    y0 = y_min + buffer
    y1 = y_max - buffer
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"Trim buffer {buffer:g}m is too large for bounds {bounds[:4]}."
        )

    nx = max(2, int(round((x1 - x0) / float(dx))) + 1)
    ny = max(2, int(round((y1 - y0) / float(dy))) + 1)
    xs = np.linspace(x0, x1, nx, dtype=np.float32)
    ys = np.linspace(y0, y1, ny, dtype=np.float32)

    actual_dx = float(xs[1] - xs[0]) if len(xs) > 1 else 0.0
    actual_dy = float(ys[1] - ys[0]) if len(ys) > 1 else 0.0
    trim = {
        "raw_bounds": [x_min, x_max, y_min, y_max],
        "clean_bounds": [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
        "buffer_m": float(buffer),
        "edge_cells": int(edge_cells),
        "requested_dx": float(dx),
        "requested_dy": float(dy),
        "actual_dx": actual_dx,
        "actual_dy": actual_dy,
    }
    return xs, ys, trim


def infer_z_cap_offset(bounds: Sequence[float], override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    x_min, x_max, y_min, y_max = map(float, bounds[:4])
    span = max(x_max - x_min, y_max - y_min)
    if span <= 600.0:
        return 200.0
    if span <= 1200.0:
        return 300.0
    return 500.0


def regular_xy_grid(bounds: Sequence[float], *, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, dict]:
    x_min, x_max, y_min, y_max = map(float, bounds[:4])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Invalid xy bounds {bounds[:4]} for regular grid export.")

    nx = max(2, int(round((x_max - x_min) / float(dx))) + 1)
    ny = max(2, int(round((y_max - y_min) / float(dy))) + 1)
    xs = np.linspace(x_min, x_max, nx, dtype=np.float32)
    ys = np.linspace(y_min, y_max, ny, dtype=np.float32)

    actual_dx = float(xs[1] - xs[0]) if len(xs) > 1 else 0.0
    actual_dy = float(ys[1] - ys[0]) if len(ys) > 1 else 0.0
    info = {
        "requested_bounds": [x_min, x_max, y_min, y_max],
        "clean_bounds": [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
        "requested_dx": float(dx),
        "requested_dy": float(dy),
        "actual_dx": actual_dx,
        "actual_dy": actual_dy,
    }
    return xs, ys, info


VERTICAL_BANDS = {
    "terrain": [
        (0.0, 50.0, 2.0),
        (50.0, 300.0, 5.0),
        (300.0, 1000.0, 20.0),
        (1000.0, 2000.0, 50.0),
    ],
    "structure": [
        (0.0, 10.0, 0.5),
        (10.0, 50.0, 1.0),
        (50.0, 150.0, 2.0),
        (150.0, 400.0, 5.0),
        (400.0, 1000.0, 20.0),
    ],
}


def build_z_levels(z_min: float, z_top: float, *, profile: str) -> np.ndarray:
    bands = VERTICAL_BANDS[profile]
    rel_max = max(0.0, float(z_top - z_min))
    vals: list[float] = []
    for z0, z1, dz in bands:
        if z0 >= rel_max:
            break
        z1 = min(z1, rel_max)
        n = int(math.floor((z1 - z0) / dz))
        for k in range(n):
            vals.append(float(z_min + z0 + k * dz))
    if not vals:
        vals = [float(z_min)]
    if vals[-1] < float(z_top):
        vals.append(float(z_top))
    return np.asarray(vals, dtype=np.float32)


def choose_vertical_profile(category: str, requested: str, *, grid_kind: str = "global") -> str:
    if requested != "auto":
        return requested
    return "structure" if grid_kind == "roi" else "terrain"


def load_dem_surface(dem_path: Path, *, z_offset_applied: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0
        if np.isfinite(nodata):
            z[z == nodata] = np.nan
        tf = src.transform
        ny, nx = z.shape
        px = float(tf.a)
        py = float(-tf.e)
        x0 = float(tf.c)
        y0_top = float(tf.f)
        xs_abs = x0 + px * (np.arange(nx, dtype=np.float32) + 0.5)
        ys_abs = y0_top - py * (np.arange(ny, dtype=np.float32) + 0.5)
        if ys_abs[0] > ys_abs[-1]:
            ys_abs = ys_abs[::-1]
            z = z[::-1, :]
        xs = xs_abs - float(xs_abs.min())
        ys = ys_abs - float(ys_abs.min())
        if z_offset_applied:
            z = z + float(z_offset_applied)
        return z.astype(np.float32), xs.astype(np.float32), ys.astype(np.float32)


def sample_dem_on_grid(
    dem_path: Path,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    z_offset_applied: float,
) -> np.ndarray:
    elev, xs_dem, ys_dem = load_dem_surface(dem_path, z_offset_applied=z_offset_applied)
    interp = RegularGridInterpolator((ys_dem, xs_dem), elev, bounds_error=False, fill_value=np.nan)
    x2d, y2d = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([y2d.ravel(), x2d.ravel()])
    sampled = interp(pts).reshape((len(ys), len(xs))).astype(np.float32)
    return sampled


def sample_ground_stl_on_grid(ground_stl: Path, *, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    mesh = pv.read(str(ground_stl)).triangulate()
    z_min, z_max = float(mesh.bounds[4]), float(mesh.bounds[5])
    if abs(z_max - z_min) < 1e-6:
        return np.full((len(ys), len(xs)), z_min, dtype=np.float32)

    out = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    start_z = z_max + 5.0
    end_z = z_min - 5.0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            pts, _ = mesh.ray_trace((float(x), float(y), start_z), (float(x), float(y), end_z))
            if pts is not None and len(pts) > 0:
                out[j, i] = float(np.max(pts[:, 2]))
    return out


def _fill_nan_nearest_2d(surface: np.ndarray) -> np.ndarray:
    """Fill NaNs from the nearest finite neighbour in 2D.

    This is intended for small DEM sampling gaps near raster boundaries or nodata
    seams. It is much cheaper than STL ray-tracing and preserves the local terrain
    much better than a global constant fill.
    """
    arr = np.asarray(surface, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.all() or not finite.any():
        return arr
    invalid = ~finite
    _, indices = distance_transform_edt(invalid, return_distances=True, return_indices=True)
    filled = arr.copy()
    filled[invalid] = arr[tuple(idx[invalid] for idx in indices)]
    return filled


def terrain_surface_on_grid(case_meta: dict, *, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    dem_path = case_meta["dem_path"]
    ground_stl = case_meta["ground_stl"]
    if dem_path is not None:
        surface = sample_dem_on_grid(
            dem_path,
            xs=xs,
            ys=ys,
            z_offset_applied=float(case_meta.get("z_offset_applied", 0.0)),
        )
        if np.isfinite(surface).all():
            return surface
        bad = int(np.size(surface) - np.isfinite(surface).sum())
        bad_frac = float(bad) / float(np.size(surface))
        if np.isfinite(surface).any() and bad_frac <= 0.05:
            warn(
                f"DEM sampling for {case_meta['case_name']} produced {bad} NaNs "
                f"({100.0 * bad_frac:.2f}%); filling from nearest DEM samples."
            )
            surface = _fill_nan_nearest_2d(surface)
            if np.isfinite(surface).all():
                return surface
        warn(f"DEM sampling for {case_meta['case_name']} produced NaNs; falling back to ground STL where possible.")
        if ground_stl is None:
            return surface
        stl_surface = sample_ground_stl_on_grid(ground_stl, xs=xs, ys=ys)
        fill = ~np.isfinite(surface) & np.isfinite(stl_surface)
        surface[fill] = stl_surface[fill]
        return surface

    if ground_stl is None:
        raise FileNotFoundError(f"No DEM or ground STL found for case {case_meta['case_name']}")
    return sample_ground_stl_on_grid(ground_stl, xs=xs, ys=ys)


def compute_terrain_channels(
    elevation: np.ndarray,
    *,
    dx: float,
    dy: float,
    extra_channels: Sequence[str],
) -> dict[str, np.ndarray]:
    dzdy, dzdx = np.gradient(elevation, float(dy), float(dx))
    slope = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))).astype(np.float32)
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) % 360.0).astype(np.float32)
    aspect[slope < 1e-6] = 0.0

    channels: dict[str, np.ndarray] = {
        "elevation": elevation.astype(np.float32),
        "slope": slope,
        "aspect": aspect,
    }

    extras = set(extra_channels)
    if "curvature" in extras:
        dyy, dyx = np.gradient(dzdy, float(dy), float(dx))
        dxy, dxx = np.gradient(dzdx, float(dy), float(dx))
        p = dzdx ** 2 + dzdy ** 2
        curvature = np.zeros_like(elevation, dtype=np.float32)
        mask = p > 1e-10
        curvature[mask] = (
            -(dxx[mask] * dzdx[mask] ** 2 + 2 * dxy[mask] * dzdx[mask] * dzdy[mask] + dyy[mask] * dzdy[mask] ** 2)
            / (p[mask] * np.sqrt(p[mask] + 1.0))
        ).astype(np.float32)
        channels["curvature"] = curvature
    if "dog_fine" in extras:
        channels["dog_fine"] = (gaussian_filter(elevation, sigma=3.0) - gaussian_filter(elevation, sigma=1.0)).astype(np.float32)
    if "dog_coarse" in extras:
        channels["dog_coarse"] = (gaussian_filter(elevation, sigma=15.0) - gaussian_filter(elevation, sigma=5.0)).astype(np.float32)
    return channels


def openfoam_internal_mesh(case_dir: Path, *, time_value: Optional[float]) -> tuple[pv.DataSet, Optional[float]]:
    foam_file = ensure_foam_file(case_dir)
    reader = pv.OpenFOAMReader(str(foam_file))
    times = list(reader.time_values) if hasattr(reader, "time_values") else []
    active_time = None
    if times:
        if time_value is None:
            active_time = float(times[-1])
        else:
            requested = float(time_value)
            active_time = min(times, key=lambda t: abs(float(t) - requested))
            if abs(float(active_time) - requested) > 1e-6:
                warn(
                    f"Requested OpenFOAM time {requested:g} not found exactly for {case_dir.name}; "
                    f"using closest available time {float(active_time):g}."
                )
        reader.set_active_time_value(float(active_time))
    root = reader.read()
    internal = maybe_cell_to_point(_extract_internal_mesh(root))
    return internal, active_time


def sample_openfoam_fields(
    internal_mesh: pv.DataSet,
    *,
    points_xyz: np.ndarray,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    n = len(points_xyz)
    ux = np.full(n, np.nan, dtype=np.float32)
    uy = np.full(n, np.nan, dtype=np.float32)
    uz = np.full(n, np.nan, dtype=np.float32)
    p = np.full(n, np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    warned_velocity_missing = False
    warned_pressure_missing = False

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        target = pv.PolyData(points_xyz[start:end])
        sampled = target.sample(
            internal_mesh,
            tolerance=1e-6,
            pass_cell_data=False,
            pass_field_data=False,
        )
        vel_name = _pick_array(sampled.point_data, PREFERRED_VELOCITY)
        p_name = _pick_array(sampled.point_data, PREFERRED_PRESSURE)
        valid_name = _pick_array(sampled.point_data, ["vtkValidPointMask", "ValidPointMask"])
        chunk_valid = (
            np.asarray(sampled.point_data[valid_name]).astype(bool)
            if valid_name is not None
            else np.ones(end - start, dtype=bool)
        )
        valid[start:end] = chunk_valid

        if vel_name is not None:
            vel = np.asarray(sampled.point_data[vel_name], dtype=np.float32)
            if vel.ndim == 2 and vel.shape[1] >= 3:
                ux[start:end] = vel[:, 0]
                uy[start:end] = vel[:, 1]
                uz[start:end] = vel[:, 2]
        elif not warned_velocity_missing:
            warn("Velocity field not found in sampled OpenFOAM data; Ux/Uy/Uz will remain NaN.")
            warned_velocity_missing = True

        if p_name is not None:
            p[start:end] = np.asarray(sampled.point_data[p_name], dtype=np.float32)
        elif not warned_pressure_missing:
            warn("Pressure field not found in sampled OpenFOAM data; p will remain NaN.")
            warned_pressure_missing = True

        outside = ~chunk_valid
        if outside.any():
            chunk_idx = np.arange(start, end)[outside]
            ux[chunk_idx] = np.nan
            uy[chunk_idx] = np.nan
            uz[chunk_idx] = np.nan
            p[chunk_idx] = np.nan

    return {"Ux": ux, "Uy": uy, "Uz": uz, "p": p, "valid": valid}


def structure_component_bounds(structure_stl: Path) -> list[dict]:
    mesh = pv.read(str(structure_stl)).triangulate()
    try:
        conn = mesh.connectivity()
    except Exception:
        conn = mesh
    region_name = "RegionId"
    if region_name in conn.cell_data:
        region_ids = np.unique(np.asarray(conn.cell_data[region_name]))
        bounds = []
        for rid in region_ids:
            sub = conn.threshold([float(rid), float(rid)], scalars=region_name)
            b = sub.bounds
            bounds.append({
                "min": [float(b[0]), float(b[2]), float(b[4])],
                "max": [float(b[1]), float(b[3]), float(b[5])],
                "label": f"component_{int(rid)}",
            })
        if bounds:
            return bounds
    b = mesh.bounds
    return [{
        "min": [float(b[0]), float(b[2]), float(b[4])],
        "max": [float(b[1]), float(b[3]), float(b[5])],
        "label": "component_0",
    }]


def structure_bounds_from_placed_spec(placed_spec: dict) -> list[dict]:
    out = []
    for idx, item in enumerate(placed_spec.get("structures", [])):
        pb = item.get("placed_bounds")
        if not pb or len(pb) != 2:
            continue
        out.append({
            "min": [float(v) for v in pb[0]],
            "max": [float(v) for v in pb[1]],
            "label": str(item.get("label") or item.get("id") or f"structure_{idx:03d}"),
        })
    return out


def preferred_structure_metadata(case_meta: dict) -> tuple[list[dict], dict | None, int, str | None]:
    domain_info = case_meta.get("domain_info") or {}
    placed_spec = case_meta.get("placed_spec") or {}

    structure_bounds = list(domain_info.get("structure_bounds", []))
    source = "domain_info" if structure_bounds else None

    if not structure_bounds and placed_spec.get("structures"):
        structure_bounds = structure_bounds_from_placed_spec(placed_spec)
        if structure_bounds:
            source = "placed_spec"

    if not structure_bounds and case_meta.get("structure_stl") is not None:
        structure_bounds = structure_component_bounds(case_meta["structure_stl"])
        if structure_bounds:
            source = "stl_connectivity"

    grid_info = domain_info.get("grid")
    if grid_info is None:
        grid_info = placed_spec.get("grid")

    n_structures = int(domain_info.get("n_structures", placed_spec.get("n_structures", len(structure_bounds))))
    return structure_bounds, grid_info, n_structures, source


def structure_roi_groups(category: str, structure_bounds: Sequence[dict]) -> list[dict]:
    bounds = [dict(b) for b in structure_bounds]
    if not bounds:
        return []
    if category == "multistructures":
        labels = [str(b.get("label", f"component_{idx:03d}")) for idx, b in enumerate(bounds)]
        return [{
            "label": "cluster_000",
            "mode": "enclosing_cluster",
            "members": bounds,
            "component_labels": labels,
        }]
    groups = []
    for idx, b in enumerate(bounds):
        label = str(b.get("label", f"component_{idx:03d}"))
        groups.append({
            "label": label,
            "mode": "per_component",
            "members": [b],
            "component_labels": [label],
        })
    return groups


def roi_bounds_from_group(
    members: Sequence[dict],
    *,
    domain_bounds: Sequence[float],
    upstream_h: float,
    downstream_h: float,
    lateral_h: float,
) -> Tuple[list[float], dict]:
    if not members:
        raise ValueError("ROI group has no structure bounds.")

    mins = np.asarray([m["min"] for m in members], dtype=np.float32)
    maxs = np.asarray([m["max"] for m in members], dtype=np.float32)
    x_min = float(np.min(mins[:, 0]))
    y_min = float(np.min(mins[:, 1]))
    z_min = float(np.min(mins[:, 2]))
    x_max = float(np.max(maxs[:, 0]))
    y_max = float(np.max(maxs[:, 1]))
    z_max = float(np.max(maxs[:, 2]))
    H = max(1e-6, float(np.max(maxs[:, 2] - mins[:, 2])))

    raw = [
        x_min - float(upstream_h) * H,
        x_max + float(downstream_h) * H,
        y_min - float(lateral_h) * H,
        y_max + float(lateral_h) * H,
    ]
    clipped = [
        max(float(domain_bounds[0]), raw[0]),
        min(float(domain_bounds[1]), raw[1]),
        max(float(domain_bounds[2]), raw[2]),
        min(float(domain_bounds[3]), raw[3]),
    ]
    if clipped[1] <= clipped[0] or clipped[3] <= clipped[2]:
        raise ValueError(f"Invalid ROI bounds after clipping: raw={raw}, clipped={clipped}")

    return clipped, {
        "raw_bounds": raw,
        "clipped_bounds": clipped,
        "H": float(H),
        "structure_z_min": float(z_min),
        "structure_z_max": float(z_max),
        "n_components": int(len(members)),
    }


def signed_wall_distance(
    structure_stl: Path,
    *,
    points_xyz: np.ndarray,
    chunk_size: int,
    signed: bool,
) -> np.ndarray:
    surface = pv.read(str(structure_stl)).triangulate().clean()
    out = np.empty(len(points_xyz), dtype=np.float32)
    for start in range(0, len(points_xyz), chunk_size):
        end = min(len(points_xyz), start + chunk_size)
        pts = pv.PolyData(points_xyz[start:end].astype(np.float32, copy=False))
        sampled = pts.compute_implicit_distance(surface, inplace=False)
        dist = np.asarray(sampled["implicit_distance"], dtype=np.float32)
        if not signed:
            dist = np.abs(dist)
        out[start:end] = dist
    return out
