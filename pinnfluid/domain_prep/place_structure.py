#!/usr/bin/env python3
"""
place_structure.py – Place one or more structure STLs on a prepared terrain DEM.

Given:
  - A prepared DEM directory (dem_final.tif + transform.json from dem_prep.py)
  - A structure STL file
  - Placement coordinates in Swiss LV95 (EPSG:2056) or local domain coords

Produces (in --out-dir):
  - structures_placed.stl   (all structures combined into one STL)
  - terrain.stl             (symlink or copy from DEM prep)
  - domain_spec.json        (full metadata for downstream pipeline)
  - placement_report.txt    (human-readable summary)

Placement modes:
  1) Single structure (default):
       --crs-xy X Y  or  --local-xy X Y

  2) Grid of identical structures:
       --crs-xy X Y --grid 5x5 --grid-spacing 2 2
       The reference coordinate is placed at grid center by default.
       Use --grid-ref I J (1-indexed) to specify which grid cell sits
       at the given coordinate.  --grid-yaw-deg rotates the whole grid.

  3) Explicit coordinate list (JSON file):
       --coords-file positions.json
       Format: [{"crs_x":..,"crs_y":..}, ...] or [{"local_x":..,"local_y":..}, ...]

Usage:
  # Single structure
  python3 scripts/place_structure.py \\
      --dem-dir dem/prafleuri/prep \\
      --stl single_stl/helioplant.stl \\
      --crs-xy 2594749 1103000 \\
      --out-dir domains/parc_solaire_prafleuri

  # 5x5 grid, 2m spacing, center of grid at the given CRS point
  python3 scripts/place_structure.py \\
      --dem-dir dem/prafleuri/prep \\
      --stl single_stl/helioplant.stl \\
      --crs-xy 2594749 1103000 \\
      --grid 5x5 --grid-spacing 2 2 \\
      --out-dir domains/parc_solaire_grid

  # 5x5 grid, structure at row 2 col 4 sits at the CRS point
  python3 scripts/place_structure.py \\
      --dem-dir dem/prafleuri/prep \\
      --stl single_stl/helioplant.stl \\
      --crs-xy 2594749 1103000 \\
      --grid 5x5 --grid-spacing 2 2 --grid-ref 2 4 \\
      --out-dir domains/parc_solaire_grid

  # Explicit coordinate list
  python3 scripts/place_structure.py \\
      --dem-dir dem/prafleuri/prep \\
      --stl single_stl/helioplant.stl \\
      --coords-file positions.json \\
      --out-dir domains/parc_solaire_custom
"""

import argparse
import json
import math
import os
import shutil
import sys
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import rowcol

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    from stl import mesh as stl_mesh
except ImportError:
    stl_mesh = None


INCLINED_ALIGNMENT_TOKENS = (
    "inclined",
    "flowerpanel",
    "advancedpanel",
    "concentrator",
    "highinc",
    "midinc",
    "lowinc",
)


def _wrap_deg(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def _to_dem_sample_crs(dem_tif: str, crs_x: float, crs_y: float,
                       transform_json: str = None) -> Tuple[float, float]:
    """Rotate original CRS coordinates into dem_final.tif sample coordinates."""
    if transform_json is None:
        tf_path = os.path.join(os.path.dirname(dem_tif), "transform.json")
    else:
        tf_path = transform_json

    if os.path.exists(tf_path):
        with open(tf_path) as f:
            meta = json.load(f)
        theta_math = float(meta.get("theta_math_deg", 0.0))
        pxy = meta.get("pivot_xy", [0, 0])
        if abs(theta_math) > 1e-6:
            theta_rad = math.radians(theta_math)
            dx = crs_x - float(pxy[0])
            dy = crs_y - float(pxy[1])
            crs_x = float(pxy[0]) + dx * math.cos(theta_rad) - dy * math.sin(theta_rad)
            crs_y = float(pxy[1]) + dx * math.sin(theta_rad) + dy * math.cos(theta_rad)
    return float(crs_x), float(crs_y)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def crs_to_local(crs_x: float, crs_y: float, dem_tif: str,
                  transform_json: str = None) -> Tuple[float, float]:
    """
    Convert CRS (EPSG:2056) coordinates to local domain coordinates.

    If the DEM was rotated by dem_prep.py, reads the rotation angle and pivot
    from transform.json and applies the same rotation to the CRS point before
    converting to local coords.
    """
    # Read rotation info from transform.json
    theta_rad = 0.0
    pivot_x = pivot_y = 0.0
    has_rotation = False

    if transform_json is None:
        # Try to find transform.json next to dem_tif
        tf_path = os.path.join(os.path.dirname(dem_tif), "transform.json")
    else:
        tf_path = transform_json

    if os.path.exists(tf_path):
        with open(tf_path) as f:
            meta = json.load(f)
        theta_math = float(meta.get("theta_math_deg", 0.0))
        pxy = meta.get("pivot_xy", [0, 0])
        pivot_x, pivot_y = float(pxy[0]), float(pxy[1])
        if abs(theta_math) > 1e-6:
            theta_rad = math.radians(theta_math)
            has_rotation = True

    # If rotated, apply the same rotation to the CRS point around the pivot
    if has_rotation:
        dx = crs_x - pivot_x
        dy = crs_y - pivot_y
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        crs_x = pivot_x + dx * cos_t - dy * sin_t
        crs_y = pivot_y + dx * sin_t + dy * cos_t

    # Now do simple offset conversion using dem_final.tif bounds
    with rasterio.open(dem_tif) as src:
        tf = src.transform
        ny, nx = src.height, src.width
        px = tf.a
        py = -tf.e
        x0 = tf.c
        y0_top = tf.f

        xs_min = x0 + px * 0.5
        ys = y0_top - py * (np.arange(ny) + 0.5)

        # Match dem_to_stl: if ys descending, flip so y=0 is at min(ys)
        ys_min = float(min(ys[0], ys[-1]))

        local_x = crs_x - xs_min
        local_y = crs_y - ys_min

    return local_x, local_y


def sample_terrain_elevation(dem_tif: str, crs_x: float, crs_y: float,
                              nodata: float = -9999.0,
                              transform_json: str = None) -> float:
    """Sample terrain elevation at a CRS point using bilinear interpolation.

    Handles rotated DEMs by rotating the CRS point before sampling.
    """
    crs_x, crs_y = _to_dem_sample_crs(dem_tif, crs_x, crs_y, transform_json=transform_json)

    with rasterio.open(dem_tif) as src:
        # rasterio row/col from (rotated) CRS coords
        r, c = rowcol(src.transform, crs_x, crs_y, op=float)
        r = float(r)
        c = float(c)

        # Bounds check
        if r < 0 or r >= src.height or c < 0 or c >= src.width:
            raise ValueError(
                f"Placement point ({crs_x}, {crs_y}) is outside DEM bounds.\n"
                f"DEM bounds: {src.bounds}"
            )

        Z = src.read(1).astype(np.float32)
        Z[Z == nodata] = np.nan

        # Bilinear interpolation
        r0 = int(math.floor(r))
        c0 = int(math.floor(c))
        r1 = min(r0 + 1, Z.shape[0] - 1)
        c1 = min(c0 + 1, Z.shape[1] - 1)
        dr = r - r0
        dc = c - c0

        z00 = Z[r0, c0]
        z01 = Z[r0, c1]
        z10 = Z[r1, c0]
        z11 = Z[r1, c1]

        vals = np.array([z00, z01, z10, z11])
        if np.any(np.isnan(vals)):
            # Fall back to nearest valid
            z_elev = vals[~np.isnan(vals)]
            if len(z_elev) == 0:
                raise ValueError(f"No valid elevation data at ({crs_x}, {crs_y}).")
            return float(np.mean(z_elev))

        z_interp = (
            z00 * (1 - dr) * (1 - dc)
            + z01 * (1 - dr) * dc
            + z10 * dr * (1 - dc)
            + z11 * dr * dc
        )
        return float(z_interp)


def local_to_crs(local_x: float, local_y: float, dem_tif: str,
                  transform_json: str = None) -> Tuple[float, float]:
    """Convert local domain coords back to CRS coords (inverse of crs_to_local).

    Handles rotated DEMs by applying the inverse rotation.
    """
    with rasterio.open(dem_tif) as src:
        tf = src.transform
        ny = src.height
        px = tf.a
        py = -tf.e
        x0 = tf.c

        xs_min = x0 + px * 0.5
        ys = tf.f - py * (np.arange(ny) + 0.5)
        ys_min = float(min(ys[0], ys[-1]))

        crs_x = local_x + xs_min
        crs_y = local_y + ys_min

    # Read rotation info and apply INVERSE rotation
    if transform_json is None:
        tf_path = os.path.join(os.path.dirname(dem_tif), "transform.json")
    else:
        tf_path = transform_json

    if os.path.exists(tf_path):
        with open(tf_path) as f:
            meta = json.load(f)
        theta_math = float(meta.get("theta_math_deg", 0.0))
        pxy = meta.get("pivot_xy", [0, 0])
        pivot_x, pivot_y = float(pxy[0]), float(pxy[1])
        if abs(theta_math) > 1e-6:
            theta_rad = math.radians(-theta_math)  # inverse rotation
            dx = crs_x - pivot_x
            dy = crs_y - pivot_y
            cos_t = math.cos(theta_rad)
            sin_t = math.sin(theta_rad)
            crs_x = pivot_x + dx * cos_t - dy * sin_t
            crs_y = pivot_y + dx * sin_t + dy * cos_t

    return float(crs_x), float(crs_y)


# ---------------------------------------------------------------------------
# STL placement
# ---------------------------------------------------------------------------

def load_and_place_stl_trimesh(
    stl_path: str,
    local_x: float,
    local_y: float,
    z_terrain: float,
    yaw_deg: float = 0.0,
    base_clearance: float = 0.0,
) -> "trimesh.Trimesh":
    """Load STL, center XY at origin, apply yaw, translate to terrain point."""
    mesh = trimesh.load_mesh(stl_path)

    # Center XY at origin, keep base at z=0
    bounds = mesh.bounds  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    cx = (bounds[0][0] + bounds[1][0]) / 2.0
    cy = (bounds[0][1] + bounds[1][1]) / 2.0
    z_base = bounds[0][2]
    mesh.apply_translation([-cx, -cy, -z_base])

    # Apply yaw rotation (around z-axis)
    if abs(yaw_deg) > 1e-6:
        angle_rad = math.radians(yaw_deg)
        rot = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
        mesh.apply_transform(rot)

    # Translate to placement point on terrain
    mesh.apply_translation([local_x, local_y, z_terrain + base_clearance])

    return mesh


def load_and_place_stl_numpystl(
    stl_path: str,
    local_x: float,
    local_y: float,
    z_terrain: float,
    yaw_deg: float = 0.0,
    base_clearance: float = 0.0,
) -> "stl_mesh.Mesh":
    """Fallback using numpy-stl if trimesh is not available."""
    m = stl_mesh.Mesh.from_file(stl_path)
    verts = m.vectors.reshape(-1, 3)

    # Center XY at origin, base at z=0
    cx = (verts[:, 0].min() + verts[:, 0].max()) / 2.0
    cy = (verts[:, 1].min() + verts[:, 1].max()) / 2.0
    z_base = verts[:, 2].min()
    verts[:, 0] -= cx
    verts[:, 1] -= cy
    verts[:, 2] -= z_base

    # Yaw rotation
    if abs(yaw_deg) > 1e-6:
        angle = math.radians(yaw_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x_rot = verts[:, 0] * cos_a - verts[:, 1] * sin_a
        y_rot = verts[:, 0] * sin_a + verts[:, 1] * cos_a
        verts[:, 0] = x_rot
        verts[:, 1] = y_rot

    # Translate
    verts[:, 0] += local_x
    verts[:, 1] += local_y
    verts[:, 2] += z_terrain + base_clearance

    m.vectors = verts.reshape(-1, 3, 3)
    return m


# ---------------------------------------------------------------------------
# Feasibility checks
# ---------------------------------------------------------------------------

def check_slope_at_point(dem_tif: str, crs_x: float, crs_y: float,
                          max_slope_deg: float = 30.0,
                          nodata: float = -9999.0) -> Tuple[float, bool]:
    """Estimate local terrain slope (degrees) at placement point."""
    crs_x, crs_y = _to_dem_sample_crs(dem_tif, crs_x, crs_y)
    with rasterio.open(dem_tif) as src:
        Z = src.read(1).astype(np.float32)
        Z[Z == nodata] = np.nan
        px = src.transform.a
        py = -src.transform.e
        r, c = rowcol(src.transform, crs_x, crs_y, op=float)
        ri, ci = int(round(r)), int(round(c))

        # Gradient from neighboring pixels
        if ri < 1 or ri >= Z.shape[0] - 1 or ci < 1 or ci >= Z.shape[1] - 1:
            return 0.0, True  # edge — skip check

        dz_dx = (Z[ri, ci + 1] - Z[ri, ci - 1]) / (2.0 * px)
        dz_dy = (Z[ri - 1, ci] - Z[ri + 1, ci]) / (2.0 * py)

        if np.isnan(dz_dx) or np.isnan(dz_dy):
            return 0.0, True

        slope_rad = math.atan(math.sqrt(float(dz_dx)**2 + float(dz_dy)**2))
        slope_deg = math.degrees(slope_rad)

        return slope_deg, slope_deg <= max_slope_deg


def terrain_gradient_at_point(dem_tif: str, crs_x: float, crs_y: float,
                              nodata: float = -9999.0) -> Tuple[float, float, float]:
    """Return dz/dx, dz/dy and slope angle in local DEM axes."""
    crs_x, crs_y = _to_dem_sample_crs(dem_tif, crs_x, crs_y)
    with rasterio.open(dem_tif) as src:
        Z = src.read(1).astype(np.float32)
        Z[Z == nodata] = np.nan
        px = src.transform.a
        py = -src.transform.e
        r, c = rowcol(src.transform, crs_x, crs_y, op=float)
        ri, ci = int(round(r)), int(round(c))

        if ri < 1 or ri >= Z.shape[0] - 1 or ci < 1 or ci >= Z.shape[1] - 1:
            return 0.0, 0.0, 0.0

        dz_dx = (Z[ri, ci + 1] - Z[ri, ci - 1]) / (2.0 * px)
        dz_dy = (Z[ri - 1, ci] - Z[ri + 1, ci]) / (2.0 * py)
        if np.isnan(dz_dx) or np.isnan(dz_dy):
            return 0.0, 0.0, 0.0
        slope_rad = math.atan(math.sqrt(float(dz_dx)**2 + float(dz_dy)**2))
        return float(dz_dx), float(dz_dy), float(math.degrees(slope_rad))


def _is_inclined_alignment_stl(stl_name: str) -> bool:
    lower = str(stl_name or "").lower()
    if "tableflat" in lower or "flatpanel" in lower:
        return False
    return any(token in lower for token in INCLINED_ALIGNMENT_TOKENS)


def _dominant_inclined_face_normal_azimuth(stl_path: str) -> Optional[float]:
    """Infer the unrotated azimuth of the main upward inclined face normal."""
    if trimesh is None:
        return None
    try:
        mesh = trimesh.load_mesh(stl_path)
        normals = np.asarray(mesh.face_normals, dtype=np.float64)
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        horiz = np.linalg.norm(normals[:, :2], axis=1)
        # Upward, non-horizontal/non-vertical faces are the panel planes we want.
        mask = (normals[:, 2] > 0.15) & (normals[:, 2] < 0.98) & (horiz > 0.08) & np.isfinite(areas)
        if not bool(np.any(mask)):
            return None
        score = areas[mask] * horiz[mask]
        idxs = np.where(mask)[0]
        normal = normals[idxs[int(np.argmax(score))]]
        return float(math.degrees(math.atan2(normal[1], normal[0])))
    except Exception:
        return None


def aligned_yaw_for_slope(
    *,
    dem_tif: str,
    crs_x: float,
    crs_y: float,
    stl_normal_azimuth_deg: Optional[float],
    requested_yaw_deg: float,
    min_slope_deg: float,
    jitter_limit_deg: float,
    nodata: float,
) -> tuple[float, dict]:
    """Yaw an inclined panel so its dominant panel plane follows the terrain slope.

    The STL's main inclined face normal is aligned with the terrain normal
    projection. The requested yaw is retained only as bounded jitter, preserving
    some orientation diversity without creating panels that face steep slopes.
    """
    dz_dx, dz_dy, slope_deg = terrain_gradient_at_point(dem_tif, crs_x, crs_y, nodata=nodata)
    info = {
        "enabled": False,
        "slope_deg": float(slope_deg),
        "terrain_normal_azimuth_deg": None,
        "stl_normal_azimuth_deg": stl_normal_azimuth_deg,
        "requested_yaw_deg": float(requested_yaw_deg),
    }
    if stl_normal_azimuth_deg is None or slope_deg < float(min_slope_deg):
        return float(requested_yaw_deg), info
    if abs(dz_dx) < 1e-12 and abs(dz_dy) < 1e-12:
        return float(requested_yaw_deg), info

    terrain_normal_azimuth = math.degrees(math.atan2(-dz_dy, -dz_dx))
    jitter = max(-float(jitter_limit_deg), min(float(jitter_limit_deg), _wrap_deg(requested_yaw_deg)))
    yaw = _wrap_deg(terrain_normal_azimuth - float(stl_normal_azimuth_deg) + jitter)
    info.update({
        "enabled": True,
        "terrain_normal_azimuth_deg": float(terrain_normal_azimuth),
        "jitter_deg": float(jitter),
        "yaw_deg": float(yaw),
    })
    return float(yaw), info


def check_domain_clearance(local_x: float, local_y: float,
                            domain_w: float, domain_h: float,
                            min_clearance: float = 100.0) -> Tuple[float, bool]:
    """Check that placement point is far enough from domain boundaries."""
    d_left = local_x
    d_right = domain_w - local_x
    d_bottom = local_y
    d_top = domain_h - local_y
    min_dist = min(d_left, d_right, d_bottom, d_top)
    return min_dist, min_dist >= min_clearance


# ---------------------------------------------------------------------------
# STL footprint measurement
# ---------------------------------------------------------------------------

def get_stl_footprint(stl_path: str, yaw_deg: float = 0.0) -> Tuple[float, float]:
    """
    Return the (width_x, width_y) footprint of a structure STL after centering
    at origin and applying yaw rotation.  Used to convert edge-to-edge spacing
    to center-to-center spacing.
    """
    if trimesh is not None:
        mesh = trimesh.load_mesh(stl_path)
        bounds = mesh.bounds
        cx = (bounds[0][0] + bounds[1][0]) / 2.0
        cy = (bounds[0][1] + bounds[1][1]) / 2.0
        z_base = bounds[0][2]
        mesh.apply_translation([-cx, -cy, -z_base])
        if abs(yaw_deg) > 1e-6:
            rot = trimesh.transformations.rotation_matrix(
                math.radians(yaw_deg), [0, 0, 1])
            mesh.apply_transform(rot)
        ext = mesh.extents  # [dx, dy, dz]
        return float(ext[0]), float(ext[1])
    elif stl_mesh is not None:
        m = stl_mesh.Mesh.from_file(stl_path)
        verts = m.vectors.reshape(-1, 3)
        cx = (verts[:, 0].min() + verts[:, 0].max()) / 2.0
        cy = (verts[:, 1].min() + verts[:, 1].max()) / 2.0
        verts[:, 0] -= cx
        verts[:, 1] -= cy
        if abs(yaw_deg) > 1e-6:
            a = math.radians(yaw_deg)
            x_rot = verts[:, 0] * math.cos(a) - verts[:, 1] * math.sin(a)
            y_rot = verts[:, 0] * math.sin(a) + verts[:, 1] * math.cos(a)
            verts[:, 0] = x_rot
            verts[:, 1] = y_rot
        wx = verts[:, 0].max() - verts[:, 0].min()
        wy = verts[:, 1].max() - verts[:, 1].min()
        return float(wx), float(wy)
    else:
        raise ImportError("Neither trimesh nor numpy-stl is installed.")


# ---------------------------------------------------------------------------
# Grid position computation
# ---------------------------------------------------------------------------

def compute_grid_positions(
    ref_local_x: float, ref_local_y: float,
    nx: int, ny: int,
    dx: float, dy: float,
    ref_i: int, ref_j: int,
    grid_yaw_deg: float = 0.0,
) -> list:
    """
    Compute local (x, y) positions for an nx x ny grid of structures.

    Parameters
    ----------
    ref_local_x, ref_local_y : reference position in local coords
    nx, ny : grid dimensions (nx = columns along x, ny = rows along y)
    dx, dy : spacing in meters
    ref_i, ref_j : which grid cell (1-indexed) sits at the reference position
    grid_yaw_deg : rotation of the entire grid around the reference point (degrees CCW)

    Returns
    -------
    List of (local_x, local_y, grid_label) tuples, row-major order.
    """
    cos_a = math.cos(math.radians(grid_yaw_deg))
    sin_a = math.sin(math.radians(grid_yaw_deg))

    positions = []
    for j in range(1, ny + 1):      # row (y direction)
        for i in range(1, nx + 1):   # col (x direction)
            # Offset relative to reference cell in grid-local frame
            ox = (i - ref_i) * dx
            oy = (j - ref_j) * dy
            # Rotate by grid yaw
            rx = ox * cos_a - oy * sin_a
            ry = ox * sin_a + oy * cos_a
            lx = ref_local_x + rx
            ly = ref_local_y + ry
            label = f"r{j}c{i}"
            positions.append((lx, ly, label))

    return positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Place structure STL(s) on prepared terrain DEM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dem-dir", required=True,
                   help="Directory with dem_final.tif + transform.json (from dem_prep.py)")
    p.add_argument("--stl", required=True, help="Path to structure STL file")
    p.add_argument("--out-dir", required=True, help="Output directory for placed domain")
    p.add_argument("--name", default=None, help="Domain name (defaults to out-dir basename)")

    # --- Placement coordinates (single point or file) ---
    # If none given, defaults to domain center from transform.json
    coord_group = p.add_mutually_exclusive_group(required=False)
    coord_group.add_argument("--crs-xy", nargs=2, type=float, metavar=("X", "Y"),
                             help="Reference coordinates in EPSG:2056 (easting, northing)")
    coord_group.add_argument("--local-xy", nargs=2, type=float, metavar=("X", "Y"),
                             help="Reference coordinates in local domain frame")
    coord_group.add_argument("--coords-file", type=str,
                             help="JSON file with list of coordinates for each structure. "
                                  'Format: [{"crs_x":..,"crs_y":..}, ...] or '
                                  '[{"local_x":..,"local_y":..}, ...]')

    # --- Grid mode ---
    p.add_argument("--grid", type=str, default=None, metavar="NxM",
                   help="Grid of structures, e.g. 5x5. Requires --crs-xy or --local-xy "
                        "as reference point, plus --grid-spacing.")
    p.add_argument("--grid-spacing", nargs=2, type=float, metavar=("DX", "DY"),
                   default=None,
                   help="Spacing between structures in x and y (meters)")
    p.add_argument("--grid-ref", nargs=2, type=int, metavar=("I", "J"),
                   default=None,
                   help="Which grid cell (1-indexed, col row) sits at the reference "
                        "coordinate. Default: center of grid.")
    p.add_argument("--grid-spacing-mode", choices=["center", "edge"], default="center",
                   help="How spacing is measured: 'center' = center-to-center (default), "
                        "'edge' = gap between closest points of adjacent structures")
    p.add_argument("--grid-yaw-deg", type=float, default=0.0,
                   help="Rotation of the grid around the reference point (degrees CCW)")

    # --- Per-structure options ---
    p.add_argument("--yaw-deg", type=float, default=0.0,
                   help="Yaw rotation of each structure (degrees, CCW positive from +x)")
    p.add_argument("--align-inclined-to-slope", action="store_true",
                   help="For inclined panel-like STLs, rotate yaw so the panel plane follows steep local terrain slope.")
    p.add_argument("--slope-align-min-deg", type=float, default=20.0,
                   help="Minimum local slope before --align-inclined-to-slope changes yaw.")
    p.add_argument("--slope-align-jitter-deg", type=float, default=15.0,
                   help="Maximum retained requested-yaw jitter around the slope-aligned yaw.")
    p.add_argument("--z-mode", choices=["sit_on_terrain", "fixed_z_offset"],
                   default="sit_on_terrain")
    p.add_argument("--base-clearance", type=float, default=0.0,
                   help="Extra vertical gap above terrain (meters)")
    p.add_argument("--z-offset", type=float, default=0.0,
                   help="Fixed z offset (only used with --z-mode fixed_z_offset)")

    # --- Checks ---
    p.add_argument("--max-slope-deg", type=float, default=30.0,
                   help="Maximum allowed terrain slope under structure")
    p.add_argument("--min-boundary-clearance", type=float, default=100.0,
                   help="Minimum distance from domain boundaries (meters)")
    p.add_argument("--skip-checks", action="store_true",
                   help="Skip feasibility checks")
    p.add_argument("--require-all-structures", action="store_true",
                   help="Fail if any requested structure cannot be placed or violates enabled checks.")

    args = p.parse_args()

    # ----- Resolve paths -----
    dem_dir = args.dem_dir
    dem_tif = os.path.join(dem_dir, "dem_final.tif")
    transform_json = os.path.join(dem_dir, "transform.json")
    terrain_stl = os.path.join(dem_dir, "terrain.stl")

    for fpath in [dem_tif, transform_json, args.stl]:
        if not os.path.exists(fpath):
            print(f"ERROR: File not found: {fpath}", file=sys.stderr)
            sys.exit(1)

    with open(transform_json) as f:
        tf_meta = json.load(f)

    domain_w = tf_meta["final_W"]
    domain_h = tf_meta["final_H"]
    nodata = tf_meta.get("nodata", -9999.0)
    domain_name = args.name or os.path.basename(os.path.normpath(args.out_dir))
    stl_basename = os.path.splitext(os.path.basename(args.stl))[0]
    stl_name = os.path.basename(args.stl)
    align_inclined = bool(args.align_inclined_to_slope and _is_inclined_alignment_stl(stl_name))
    stl_normal_azimuth = _dominant_inclined_face_normal_azimuth(args.stl) if align_inclined else None
    base_yaw_deg = float(args.yaw_deg)
    grid_alignment_info = None
    if args.align_inclined_to_slope and not align_inclined:
        print(f"Slope alignment requested but {stl_name} is treated as non-inclined; keeping yaw={base_yaw_deg:.1f}°")
    elif align_inclined and stl_normal_azimuth is None:
        print(f"Slope alignment requested but no inclined face normal was inferred for {stl_name}; keeping yaw={base_yaw_deg:.1f}°")

    if trimesh is None and stl_mesh is None:
        print("ERROR: Neither trimesh nor numpy-stl is installed.", file=sys.stderr)
        sys.exit(1)

    # ==================================================================
    # 1. Build list of placement positions: [(local_x, local_y, label)]
    # ==================================================================
    positions = []  # list of (local_x, local_y, label)

    if args.coords_file is not None:
        # --- Mode: explicit coordinate file ---
        with open(args.coords_file) as f:
            coord_list = json.load(f)
        if not isinstance(coord_list, list) or len(coord_list) == 0:
            print("ERROR: --coords-file must contain a non-empty JSON list.", file=sys.stderr)
            sys.exit(1)

        for idx, entry in enumerate(coord_list, 1):
            label = entry.get("label", f"{idx:03d}")
            if "crs_x" in entry and "crs_y" in entry:
                lx, ly = crs_to_local(entry["crs_x"], entry["crs_y"], dem_tif)
            elif "local_x" in entry and "local_y" in entry:
                lx, ly = entry["local_x"], entry["local_y"]
            else:
                print(f"ERROR: Entry {idx} in coords file must have "
                      "(crs_x, crs_y) or (local_x, local_y).", file=sys.stderr)
                sys.exit(1)
            positions.append((lx, ly, label))

        print(f"Loaded {len(positions)} positions from {args.coords_file}")

    else:
        # Resolve the single reference point
        if args.crs_xy is not None:
            ref_crs_x, ref_crs_y = args.crs_xy
            ref_local_x, ref_local_y = crs_to_local(ref_crs_x, ref_crs_y, dem_tif)
            print(f"Reference CRS:   ({ref_crs_x:.2f}, {ref_crs_y:.2f})")
            print(f"Reference local: ({ref_local_x:.2f}, {ref_local_y:.2f})")
        elif args.local_xy is not None:
            ref_local_x, ref_local_y = args.local_xy
            ref_crs_x, ref_crs_y = local_to_crs(ref_local_x, ref_local_y, dem_tif)
            print(f"Reference local: ({ref_local_x:.2f}, {ref_local_y:.2f})")
            print(f"Reference CRS:   ({ref_crs_x:.2f}, {ref_crs_y:.2f})")
        else:
            # Default: domain center from transform.json
            ref_local_x = domain_w / 2.0
            ref_local_y = domain_h / 2.0
            ref_crs_x, ref_crs_y = local_to_crs(ref_local_x, ref_local_y, dem_tif)
            print(f"No coordinates given — using domain center")
            print(f"Reference local: ({ref_local_x:.2f}, {ref_local_y:.2f})")
            print(f"Reference CRS:   ({ref_crs_x:.2f}, {ref_crs_y:.2f})")
            print(f"Reference local: ({ref_local_x:.2f}, {ref_local_y:.2f})")
            print(f"Reference CRS:   ({ref_crs_x:.2f}, {ref_crs_y:.2f})")

        if args.grid is not None:
            # --- Mode: grid ---
            try:
                parts = args.grid.lower().split("x")
                grid_nx, grid_ny = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                print("ERROR: --grid must be NxM, e.g. 5x5", file=sys.stderr)
                sys.exit(1)

            if args.grid_spacing is None:
                print("ERROR: --grid requires --grid-spacing DX DY", file=sys.stderr)
                sys.exit(1)
            gdx, gdy = args.grid_spacing

            if align_inclined and stl_normal_azimuth is not None:
                base_yaw_deg, grid_alignment_info = aligned_yaw_for_slope(
                    dem_tif=dem_tif,
                    crs_x=ref_crs_x,
                    crs_y=ref_crs_y,
                    stl_normal_azimuth_deg=stl_normal_azimuth,
                    requested_yaw_deg=float(args.yaw_deg),
                    min_slope_deg=float(args.slope_align_min_deg),
                    jitter_limit_deg=float(args.slope_align_jitter_deg),
                    nodata=nodata,
                )
                if grid_alignment_info.get("enabled"):
                    print(
                        f"Slope-aligned grid yaw: requested={args.yaw_deg:.1f}° -> "
                        f"{base_yaw_deg:.1f}° at slope={grid_alignment_info['slope_deg']:.1f}°"
                    )

            # Convert edge spacing to center-to-center if needed.
            # Spacing is applied in the GRID frame (gdx along grid-x, gdy along
            # grid-y), and the grid is then rotated by grid_yaw_deg to world.
            # The structure is rotated by yaw_deg in world. So in the grid
            # frame, the structure is rotated by (yaw_deg - grid_yaw_deg).
            # Using the world-frame AABB here would over- or under-spaces the
            # grid whenever struct_yaw != grid_yaw (e.g. inclined-panel grids
            # with mismatched yaws can end up with overlapping plates).
            if args.grid_spacing_mode == "edge":
                yaw_in_grid_frame = base_yaw_deg - args.grid_yaw_deg
                footprint_x, footprint_y = get_stl_footprint(
                    args.stl, yaw_deg=yaw_in_grid_frame)
                edge_dx, edge_dy = gdx, gdy
                gdx = edge_dx + footprint_x
                gdy = edge_dy + footprint_y
                print(f"STL footprint: {footprint_x:.3f} x {footprint_y:.3f} m  "
                      f"(in grid frame, yaw={yaw_in_grid_frame:+.1f}°)")
                print(f"Edge gap:      {edge_dx:.2f} x {edge_dy:.2f} m  "
                      f"-> center-to-center: {gdx:.3f} x {gdy:.3f} m")

            if args.grid_ref is not None:
                ref_i, ref_j = args.grid_ref
            else:
                # Default: center of grid (ceiling for even grids)
                ref_i = (grid_nx + 1) // 2
                ref_j = (grid_ny + 1) // 2

            if not (1 <= ref_i <= grid_nx and 1 <= ref_j <= grid_ny):
                print(f"ERROR: --grid-ref ({ref_i}, {ref_j}) out of range "
                      f"for {grid_nx}x{grid_ny} grid.", file=sys.stderr)
                sys.exit(1)

            positions = compute_grid_positions(
                ref_local_x, ref_local_y,
                grid_nx, grid_ny, gdx, gdy,
                ref_i, ref_j,
                grid_yaw_deg=args.grid_yaw_deg,
            )

            print(f"Grid: {grid_nx}x{grid_ny} = {len(positions)} structures")
            print(f"Spacing (center-to-center): {gdx:.3f} x {gdy:.3f} m")
            print(f"Spacing mode: {args.grid_spacing_mode}")
            print(f"Reference cell: ({ref_i}, {ref_j})")
            if abs(args.grid_yaw_deg) > 1e-6:
                print(f"Grid yaw: {args.grid_yaw_deg:.1f}°")

        else:
            # --- Mode: single structure ---
            positions = [(ref_local_x, ref_local_y, "001")]

    n_structures = len(positions)
    print(f"\nPlacing {n_structures} structure(s)...")

    # ==================================================================
    # 2. Place each structure
    # ==================================================================
    placed_meshes = []       # trimesh objects or numpy-stl meshes
    structure_specs = []     # metadata per structure

    for idx, (lx, ly, label) in enumerate(positions):
        # Get CRS coords for this position
        cx, cy = local_to_crs(lx, ly, dem_tif)

        # Sample terrain elevation
        try:
            z_t = sample_terrain_elevation(dem_tif, cx, cy, nodata=nodata)
        except ValueError as e:
            print(f"  [{label}] SKIP — {e}", file=sys.stderr)
            if args.require_all_structures:
                print("ERROR: Required structure placement failed.", file=sys.stderr)
                sys.exit(1)
            continue

        z_place = z_t if args.z_mode == "sit_on_terrain" else args.z_offset
        yaw_here = base_yaw_deg
        alignment_info = grid_alignment_info if grid_alignment_info is not None else {"enabled": False}
        if align_inclined and args.grid is None and stl_normal_azimuth is not None:
            yaw_here, alignment_info = aligned_yaw_for_slope(
                dem_tif=dem_tif,
                crs_x=cx,
                crs_y=cy,
                stl_normal_azimuth_deg=stl_normal_azimuth,
                requested_yaw_deg=float(args.yaw_deg),
                min_slope_deg=float(args.slope_align_min_deg),
                jitter_limit_deg=float(args.slope_align_jitter_deg),
                nodata=nodata,
            )

        # Feasibility checks
        if not args.skip_checks:
            slope_deg, slope_ok = check_slope_at_point(
                dem_tif, cx, cy,
                max_slope_deg=args.max_slope_deg, nodata=nodata,
            )
            clearance, clearance_ok = check_domain_clearance(
                lx, ly, domain_w, domain_h,
                min_clearance=args.min_boundary_clearance,
            )
            warnings = []
            if not slope_ok:
                warnings.append(f"slope {slope_deg:.1f}°")
            if not clearance_ok:
                warnings.append(f"clearance {clearance:.0f}m")
            warn_str = f" WARNING: {', '.join(warnings)}" if warnings else ""
            if warnings and args.require_all_structures:
                print(f"  [{label}] FAIL — {', '.join(warnings)}", file=sys.stderr)
                sys.exit(1)
        else:
            slope_deg = 0.0
            warn_str = ""

        # Place the STL
        if trimesh is not None:
            mesh = load_and_place_stl_trimesh(
                args.stl, lx, ly, z_place,
                yaw_deg=yaw_here,
                base_clearance=args.base_clearance,
            )
            bounds = mesh.bounds
            extents = mesh.extents
        else:
            mesh = load_and_place_stl_numpystl(
                args.stl, lx, ly, z_place,
                yaw_deg=yaw_here,
                base_clearance=args.base_clearance,
            )
            verts = mesh.vectors.reshape(-1, 3)
            bounds = np.array([verts.min(axis=0), verts.max(axis=0)])
            extents = bounds[1] - bounds[0]

        placed_meshes.append(mesh)

        structure_specs.append({
            "id": f"{stl_basename}_{label}",
            "label": label,
            "stl_source": os.path.abspath(args.stl),
            "stl_name": stl_name,
            "placement_crs": {"epsg": 2056, "x": cx, "y": cy},
            "placement_local": {"x": lx, "y": ly},
            "z_terrain": z_t,
            "z_mode": args.z_mode,
            "base_clearance_m": args.base_clearance,
            "yaw_deg": yaw_here,
            "yaw_requested_deg": args.yaw_deg,
            "slope_alignment": alignment_info,
            "terrain_slope_deg": slope_deg,
            "placed_bounds": bounds.tolist(),
            "placed_extents": extents.tolist(),
        })

        if n_structures <= 50 or (idx + 1) % 10 == 0 or idx == 0:
            align_str = ""
            if alignment_info and alignment_info.get("enabled"):
                align_str = f" yaw={yaw_here:.1f}°"
            print(f"  [{label}] local=({lx:.2f}, {ly:.2f})  z={z_t:.2f}m slope={slope_deg:.1f}°{align_str}{warn_str}")

    if not placed_meshes:
        print("ERROR: No structures could be placed.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSuccessfully placed {len(placed_meshes)}/{n_structures} structures.")

    # ==================================================================
    # 3. Write outputs
    # ==================================================================
    os.makedirs(args.out_dir, exist_ok=True)

    # Combine all meshes into one STL
    combined_path = os.path.join(args.out_dir, "structures_placed.stl")
    if trimesh is not None:
        combined = trimesh.util.concatenate(placed_meshes)
        combined.export(combined_path)
    else:
        # numpy-stl: concatenate mesh data
        all_data = np.concatenate([m.data for m in placed_meshes])
        combined_mesh = stl_mesh.Mesh(all_data)
        combined_mesh.save(combined_path)
    print(f"Combined STL: {combined_path}  ({len(placed_meshes)} structures)")

    # Copy terrain STL
    terrain_dst = os.path.join(args.out_dir, "terrain.stl")
    if os.path.exists(terrain_stl) and not os.path.exists(terrain_dst):
        shutil.copy2(terrain_stl, terrain_dst)
        print(f"Terrain STL:  {terrain_dst}")

    # Domain spec
    grid_info = None
    if args.grid is not None:
        parts = args.grid.lower().split("x")
        gi_nx, gi_ny = int(parts[0]), int(parts[1])
        grid_info = {
            "nx": gi_nx,
            "ny": gi_ny,
            "spacing_mode": args.grid_spacing_mode,
            "spacing_input_x": args.grid_spacing[0],
            "spacing_input_y": args.grid_spacing[1],
            "spacing_center_x": gdx,
            "spacing_center_y": gdy,
            "ref_i": args.grid_ref[0] if args.grid_ref else (gi_nx + 1) // 2,
            "ref_j": args.grid_ref[1] if args.grid_ref else (gi_ny + 1) // 2,
            "grid_yaw_deg": args.grid_yaw_deg,
        }

    domain_spec = {
        "domain_name": domain_name,
        "terrain": {
            "dem_dir": os.path.abspath(dem_dir),
            "dem_tif": os.path.abspath(dem_tif),
            "transform": tf_meta,
            "domain_size_m": [domain_w, domain_h],
            "z_range": tf_meta.get("z_range", []),
        },
        "abl": {
            "Uref": None,
            "Zref": None,
            "z0": None,
            "wind_from_deg": tf_meta.get("wind_from_deg"),
            "note": "Fill from abl.txt or set manually",
        },
        "n_structures": len(structure_specs),
        "stl_combined": os.path.abspath(combined_path),
        "grid": grid_info,
        "structures": structure_specs,
    }

    spec_path = os.path.join(args.out_dir, "domain_spec.json")
    with open(spec_path, "w") as f:
        json.dump(domain_spec, f, indent=2)
    print(f"Domain spec:  {spec_path}")

    # Placement report
    report_path = os.path.join(args.out_dir, "placement_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Domain: {domain_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Terrain DEM:  {dem_tif}\n")
        f.write(f"Domain size:  {domain_w:.0f} x {domain_h:.0f} m\n")
        f.write(f"Z range:      {tf_meta.get('z_range', ['?', '?'])}\n")
        f.write(f"Wind from:    {tf_meta.get('wind_from_deg', '?')}°\n\n")
        f.write(f"Structure:    {stl_name}\n")
        f.write(f"Count:        {len(structure_specs)}\n")
        f.write(f"Yaw:          {args.yaw_deg:.1f}°\n")
        f.write(f"Z mode:       {args.z_mode}\n")
        f.write(f"Clearance:    {args.base_clearance:.2f} m\n")

        if grid_info:
            f.write(f"\nGrid layout:  {grid_info['nx']}x{grid_info['ny']}\n")
            f.write(f"Spacing mode: {grid_info['spacing_mode']}\n")
            f.write(f"Spacing input:{grid_info['spacing_input_x']:.2f} x {grid_info['spacing_input_y']:.2f} m "
                    f"({grid_info['spacing_mode']})\n")
            f.write(f"Spacing c2c:  {grid_info['spacing_center_x']:.3f} x {grid_info['spacing_center_y']:.3f} m\n")
            f.write(f"Ref cell:     ({grid_info['ref_i']}, {grid_info['ref_j']})\n")
            if abs(args.grid_yaw_deg) > 1e-6:
                f.write(f"Grid yaw:     {args.grid_yaw_deg:.1f}°\n")

        f.write(f"\n{'─'*60}\n")
        f.write(f"{'Label':>8} {'Local X':>10} {'Local Y':>10} {'Z terrain':>10}\n")
        f.write(f"{'─'*60}\n")
        for s in structure_specs:
            f.write(f"{s['label']:>8} "
                    f"{s['placement_local']['x']:10.2f} "
                    f"{s['placement_local']['y']:10.2f} "
                    f"{s['z_terrain']:10.2f}\n")
        f.write(f"{'─'*60}\n")

    print(f"Report:       {report_path}")
    print(f"\nDone. {len(structure_specs)} structure(s) placed.")


if __name__ == "__main__":
    main()
