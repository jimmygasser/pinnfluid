#!/usr/bin/env python3
"""
dem_prep.py - Merge SwissALTI3D GeoTIFF tiles, rotate DEM to align wind with +x,
optionally crop to a target W×H, and export both GeoTIFF + STL.

Wind convention (Swiss wind atlas / meteorological):
- wind_from_deg = 0   => wind from North (towards South)
- wind_from_deg = 90  => wind from East  (towards West)
- wind_from_deg = 180 => wind from South (towards North)
- wind_from_deg = 270 => wind from West  (towards East)

Pipeline:
1) Merge input TIFs into one mosaic
2) Rotate (in pixel space) so wind "to" direction aligns with +x
3) Crop:
   - Mode A: if --width and --height given -> crop to W×H around pivot
   - Mode B: if no size given -> keep the largest possible valid rectangle
4) Write:
   - merged_dem.tif
   - dem_rotated.tif (intermediate)
   - dem_final.tif (final aligned DEM; may equal dem_rotated.tif in max mode)
   - transform.json
   - terrain.stl (default on)

Notes / assumptions:
- Input DEM in projected CRS with meters (e.g. EPSG:2056).
- Pixel spacing roughly square.
- No QGIS required.

Examples
--------
# Mode B (max possible from provided tiles)
python scripts/dem_prep.py --tifs dem/pass_nufenen/*.tif --wind-from 315 --out-dir nufenen_pass

# Mode A (explicit size in meters)
python scripts/dem_prep.py --tifs dem/pass_nufenen/*.tif --wind-from 315 --width 2000 --height 1500 --out-dir nufenen_pass
"""

import argparse
import json
import glob
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from rasterio.transform import rowcol

try:
    from scipy.ndimage import affine_transform
except ImportError as e:
    raise ImportError("scipy is required: pip install scipy") from e

# STL export: prefer numpy-stl (simple and robust)
try:
    from stl import mesh as stl_mesh  # pip install numpy-stl
except ImportError:
    stl_mesh = None


# -----------------------------
# Metadata
# -----------------------------
@dataclass
class TransformMeta:
    crs: str
    wind_from_deg: float
    wind_to_deg: float
    theta_deg: float                 # applied rotation (deg CCW) so wind_to aligns with +x
    pivot_xy: Tuple[float, float]    # pivot in CRS coords (meters)
    final_W: float                   # meters
    final_H: float                   # meters
    source_S: float                  # meters (sqrt(W^2+H^2) if Mode A, else 0)
    nodata: float
    mode: str                        # "explicit_size" or "max_possible"
    warning: Optional[str] = None


# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _meteo_to_rotation(wind_from_deg: float) -> Tuple[float, float]:
    """
    Convert meteorological wind direction (FROM, clockwise from North) to:
    - wind_to_deg: direction wind blows TO, clockwise from North
    - theta_deg: rotation (deg) in math x/y (CCW positive) so wind_to aligns with +x.
    """
    wind_from_deg = wind_from_deg % 360.0
    wind_to_deg = (wind_from_deg + 180.0) % 360.0
    # Using math convention in pixel/world plane: +x is 0°, CCW positive.
    # If we interpret wind_to_deg also as 0°=North, CW positive, then mapping to math angle is:
    #   math_angle = 90 - wind_to_deg
    # To align wind_to with +x (math_angle=0), rotate by -math_angle.
    # => theta = -(90 - wind_to_deg) = wind_to_deg - 90
    #
    # This is the correct bridge between meteo and x-axis alignment.
    theta_deg = wind_to_deg - 90.0
    return wind_to_deg, theta_deg


def _read_abl_raw(path: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                k, v = parts[0], parts[1]
            data[k.strip()] = float(v)
    return data


def _extract_indexed_series(data: Dict[str, float], prefix: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k, v in data.items():
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix):]
        if suffix.isdigit():
            out[int(suffix)] = float(v)
    return out


def _read_abl_thetas(path: str) -> Dict[int, float]:
    data = _read_abl_raw(path)
    t_series = _extract_indexed_series(data, "theta")
    if t_series:
        return t_series
    if "theta" in data:
        return {1: float(data["theta"])}
    raise ValueError("ABL file has no theta/thetaN entries.")


def _theta_slug(theta_deg: float) -> str:
    if abs(theta_deg - round(theta_deg)) < 1e-6:
        return f"theta_{int(round(theta_deg))}"
    return f"theta_{theta_deg:.1f}".replace(".", "p")


def merge_tifs(tif_paths: List[str], out_tif: str, input_nodata: Optional[float] = None) -> None:
    """
    Merge multiple GeoTIFFs into a single mosaic GeoTIFF.
    Assumes all have same CRS/resolution; rasterio.merge handles overlaps.
    """
    if len(tif_paths) == 0:
        raise ValueError("No input TIFs provided.")

    srcs = [rasterio.open(p) for p in tif_paths]

    # Determine nodata to use for merge
    if input_nodata is None:
        input_nodata = srcs[0].nodata

    mosaic, transform = merge(srcs, nodata=input_nodata)  # mosaic shape: (bands, H, W)

    # Use first raster profile as template
    profile = srcs[0].profile.copy()
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        count=1,
        nodata=input_nodata,
        compress="lzw",
        tiled=True,
    )

    # If merged array has multiple bands, take first
    data = mosaic[0].astype(np.float32)

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(data, 1)

    for s in srcs:
        s.close()


def _read_window(src, left, bottom, right, top) -> np.ndarray:
    win = from_bounds(left, bottom, right, top, transform=src.transform)
    arr = src.read(1, window=win).astype(np.float32)
    return arr, win


def _rotate_array_about_pivot(
    arr: np.ndarray,
    theta_deg: float,
    pivot_rc: Tuple[float, float],
    resample_order: int,
    nodata_out: float,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Rotate 2D array about an arbitrary pivot (row, col) by theta_deg (CCW positive).
    Returns rotated array and the (min_row, min_col) of the rotated grid in the
    original array coordinate system.
    """
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Rotation matrix for row/col (y/x) coordinates
    # row' =  cos*row + sin*col
    # col' = -sin*row + cos*col
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)

    h, w = arr.shape
    pivot_r, pivot_c = pivot_rc

    # Rotate corners to determine output bounds
    corners = np.array(
        [
            [0.0, 0.0],
            [0.0, float(w - 1)],
            [float(h - 1), 0.0],
            [float(h - 1), float(w - 1)],
        ],
        dtype=np.float64,
    )
    pivot_vec = np.array([pivot_r, pivot_c], dtype=np.float64)
    corners_rot = (R @ (corners - pivot_vec).T).T + pivot_vec
    min_row = math.floor(float(np.min(corners_rot[:, 0])))
    max_row = math.ceil(float(np.max(corners_rot[:, 0])))
    min_col = math.floor(float(np.min(corners_rot[:, 1])))
    max_col = math.ceil(float(np.max(corners_rot[:, 1])))

    out_h = int(max_row - min_row + 1)
    out_w = int(max_col - min_col + 1)

    # Inverse rotation for mapping output -> input
    # R_inv for row/col with -theta
    R_inv = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    offset = pivot_vec - (R_inv @ pivot_vec)
    offset = offset + (R_inv @ np.array([min_row, min_col], dtype=np.float64))

    rot_out = affine_transform(
        arr,
        matrix=R_inv,
        offset=offset,
        output_shape=(out_h, out_w),
        order=resample_order,
        mode="constant",
        cval=np.nan,
        prefilter=True,
    )

    rot_out = np.where(np.isfinite(rot_out), rot_out, nodata_out).astype(np.float32)
    return rot_out, (float(min_row), float(min_col))


def rotate_dem_to_tif(
    input_tif: str,
    output_rot_tif: str,
    wind_from_deg: float,
    pivot_xy: Optional[Tuple[float, float]] = None,
    resample_order: int = 1,
    nodata_out: float = -9999.0,
    pad_fraction: float = 0.0,
    input_nodata: Optional[float] = None,
) -> Tuple[str, TransformMeta]:
    """
    Rotate the entire DEM around pivot so that wind aligns with +x.
    Writes rotated DEM GeoTIFF with an affine transform centered at pivot.

    pad_fraction:
      optional extra padding as fraction of max(dem_width, dem_height) before rotating
      to reduce chance of corner nodata for later max-size cropping. Usually 0 is fine.
    """
    wind_to_deg, theta_math_deg = _meteo_to_rotation(wind_from_deg)
    # Image row/col coordinates have +row downward (south), so CCW in math x/y
    # becomes CW in row/col. Flip sign for the actual pixel-space rotation.
    theta_deg = -theta_math_deg

    with rasterio.open(input_tif) as src:
        if src.crs is None:
            raise ValueError("Input GeoTIFF has no CRS.")

        bounds = src.bounds
        if pivot_xy is None:
            pivot_xy = ((bounds.left + bounds.right) / 2.0, (bounds.bottom + bounds.top) / 2.0)
        else:
            if not (bounds.left <= pivot_xy[0] <= bounds.right and bounds.bottom <= pivot_xy[1] <= bounds.top):
                raise ValueError("Pivot must lie inside the input DEM bounds.")

        if abs(src.transform.b) > 1e-9 or abs(src.transform.d) > 1e-9:
            raise ValueError("Input GeoTIFF must be north-up (no rotation/shear in transform).")

        px = src.transform.a
        py = -src.transform.e
        if px <= 0 or py <= 0:
            raise ValueError("Unexpected transform; pixel size could not be inferred.")
        if abs(px - py) > 1e-6:
            raise ValueError("Input DEM must have square pixels for rotation to be geometrically correct.")

        # Read full raster
        arr, win = _read_window(src, bounds.left, bounds.bottom, bounds.right, bounds.top)

        # Handle nodata
        src_nodata = input_nodata if input_nodata is not None else src.nodata
        if src_nodata is not None:
            arr = arr.copy()
            arr[arr == src_nodata] = np.nan
        else:
            print("WARNING: Input nodata is not defined; rotation may smear nodata into valid data.", file=sys.stderr)

        # Optional padding in pixel space
        pad = 0
        if pad_fraction < 0:
            raise ValueError("--pad-fraction must be >= 0.")
        if pad_fraction > 0:
            pad = int(math.ceil(pad_fraction * max(arr.shape)))
            if pad > 0:
                arr = np.pad(arr, ((pad, pad), (pad, pad)), mode="constant", constant_values=np.nan)

        # Rotate array
        pivot_r, pivot_c = rowcol(src.transform, pivot_xy[0], pivot_xy[1], op=float)
        pivot_r = float(pivot_r) + float(pad)
        pivot_c = float(pivot_c) + float(pad)

        rot_out, (min_row, min_col) = _rotate_array_about_pivot(
            arr,
            theta_deg=theta_deg,
            pivot_rc=(pivot_r, pivot_c),
            resample_order=resample_order,
            nodata_out=nodata_out,
        )

        # Build new transform: keep pixel size, and center rotated raster at pivot_xy
        h1, w1 = rot_out.shape
        cx, cy = pivot_xy
        new_ulx = cx + (min_col - pivot_c) * px
        new_uly = cy - (min_row - pivot_r) * py
        rot_transform = Affine(px, 0, new_ulx, 0, -py, new_uly)

        out_profile = src.profile.copy()
        out_profile.update(
            driver="GTiff",
            dtype="float32",
            nodata=nodata_out,
            height=h1,
            width=w1,
            transform=rot_transform,
            compress="lzw",
            tiled=True,
            count=1,
        )

        with rasterio.open(output_rot_tif, "w", **out_profile) as dst:
            dst.write(rot_out, 1)

        meta = TransformMeta(
            crs=str(src.crs),
            wind_from_deg=float(wind_from_deg),
            wind_to_deg=float(wind_to_deg),
            theta_deg=float(theta_deg),
            pivot_xy=(float(cx), float(cy)),
            final_W=float(w1 * px),
            final_H=float(h1 * py),
            source_S=0.0,
            nodata=float(nodata_out),
            mode="rotation_only",
        )

    return output_rot_tif, meta


def crop_rotated_dem(
    rotated_tif: str,
    output_tif: str,
    pivot_xy: Tuple[float, float],
    W: float,
    H: float,
    nodata_out: float = -9999.0,
) -> Tuple[float, float]:
    """
    Crop rotated DEM to an axis-aligned W×H rectangle centered at pivot_xy.
    Returns actual (W_written, H_written) in meters.
    """
    cx, cy = pivot_xy
    halfW, halfH = 0.5 * W, 0.5 * H
    left = cx - halfW
    right = cx + halfW
    bottom = cy - halfH
    top = cy + halfH

    with rasterio.open(rotated_tif) as src:
        win = from_bounds(left, bottom, right, top, transform=src.transform)
        arr = src.read(1, window=win).astype(np.float32)
        out_transform = src.window_transform(win)

        out_profile = src.profile.copy()
        out_profile.update(
            driver="GTiff",
            dtype="float32",
            nodata=nodata_out,
            height=arr.shape[0],
            width=arr.shape[1],
            transform=out_transform,
            compress="lzw",
            tiled=True,
            count=1,
        )

        with rasterio.open(output_tif, "w", **out_profile) as dst:
            dst.write(arr, 1)

        # compute written size from bounds
        b = rasterio.windows.bounds(win, transform=src.transform)
        W_written = b[2] - b[0]
        H_written = b[3] - b[1]
        return float(W_written), float(H_written)


def max_valid_inner_bounds(src: rasterio.io.DatasetReader, nodata: float) -> Tuple[float, float, float, float]:
    """
    Find the largest axis-aligned rectangle inside the rotated DEM that contains no nodata.
    Returns bounds in CRS coords: (xmin, ymin, xmax, ymax)
    """
    Z = src.read(1)
    valid = (Z != nodata) & np.isfinite(Z)

    if not np.any(valid):
        raise ValueError("Rotated DEM contains no valid data (all NoData).")

    # Maximal rectangle in a binary matrix using histogram method
    heights = np.zeros(valid.shape[1], dtype=np.int32)
    best_area = 0
    best = None  # (top, bottom, left, right)

    for row in range(valid.shape[0]):
        heights = np.where(valid[row], heights + 1, 0)

        stack: List[Tuple[int, int]] = []
        for i in range(len(heights) + 1):
            h = heights[i] if i < len(heights) else 0
            start = i
            while stack and stack[-1][1] > h:
                idx, height = stack.pop()
                area = height * (i - idx)
                if area > best_area and height > 0:
                    best_area = area
                    left = idx
                    right = i - 1
                    bottom = row
                    top = row - height + 1
                    best = (top, bottom, left, right)
                start = idx
            if not stack or stack[-1][1] < h:
                stack.append((start, h))

    if best is None:
        raise ValueError("Could not find a valid rectangle without NoData.")

    top, bottom, left, right = best
    win = rasterio.windows.Window(
        col_off=left,
        row_off=top,
        width=(right - left + 1),
        height=(bottom - top + 1),
    )
    xmin, ymin, xmax, ymax = rasterio.windows.bounds(win, transform=src.transform)
    return xmin, ymin, xmax, ymax


def max_valid_inner_bounds_with_pivot(
    src: rasterio.io.DatasetReader,
    nodata: float,
    pivot_xy: Tuple[float, float],
    pivot_threshold_m: float,
) -> Tuple[float, float, float, float, bool]:
    """
    Find the largest axis-aligned rectangle inside the rotated DEM that:
    - contains the pivot_xy point
    - contains no nodata
    If pivot_threshold_m > 0, prefer rectangles where pivot is at least that
    distance from all boundaries (if possible).
    Returns bounds in CRS coords: (xmin, ymin, xmax, ymax)
    """
    Z = src.read(1)
    valid = (Z != nodata) & np.isfinite(Z)
    if not np.any(valid):
        raise ValueError("Rotated DEM contains no valid data (all NoData).")

    # Pivot in pixel coordinates
    pivot_c, pivot_r = rowcol(src.transform, pivot_xy[0], pivot_xy[1], op=float)
    pr = int(round(pivot_r))
    pc = int(round(pivot_c))
    if pr < 0 or pr >= valid.shape[0] or pc < 0 or pc >= valid.shape[1]:
        raise ValueError("Pivot is outside rotated DEM bounds.")
    if not valid[pr, pc]:
        raise ValueError("Pivot lies on NoData after rotation; choose a different pivot or add tiles.")

    nrows, ncols = valid.shape

    # For each row, find left/right bounds of contiguous valid segment containing pc
    left = np.full(nrows, -1, dtype=np.int32)
    right = np.full(nrows, -1, dtype=np.int32)
    for r in range(nrows):
        if not valid[r, pc]:
            continue
        # Expand left
        l = pc
        while l - 1 >= 0 and valid[r, l - 1]:
            l -= 1
        # Expand right
        rr = pc
        while rr + 1 < ncols and valid[r, rr + 1]:
            rr += 1
        left[r] = l
        right[r] = rr

    # Rows where pivot column is invalid cannot be part of rectangle
    valid_rows = np.where(left >= 0)[0]
    if len(valid_rows) == 0:
        raise ValueError("No valid rows through pivot column after rotation.")

    best_area = 0
    best = None  # (top, bottom, left, right, min_edge_dist_m, center_dist_m)
    best_thresh_area = 0
    best_thresh = None  # same as best, but for threshold-satisfying rectangles

    px = float(src.transform.a)
    py = float(-src.transform.e)
    if px <= 0 or py <= 0:
        px = py = max(px, py, 1.0)

    for top in range(pr, -1, -1):
        if left[top] < 0:
            break  # cannot include pivot row if this row invalid
        max_left = left[top]
        min_right = right[top]
        for bottom in range(pr, nrows):
            if left[bottom] < 0:
                break
            if bottom != top:
                if left[bottom] > max_left:
                    max_left = left[bottom]
                if right[bottom] < min_right:
                    min_right = right[bottom]
            width = min_right - max_left + 1
            if width <= 0:
                break
            height = bottom - top + 1
            area = width * height
            # distances to edges (meters)
            d_left = (pc - max_left) * px
            d_right = (min_right - pc) * px
            d_top = (pr - top) * py
            d_bottom = (bottom - pr) * py
            min_edge_dist = min(d_left, d_right, d_top, d_bottom)
            # distance to rectangle center
            center_c = 0.5 * (max_left + min_right)
            center_r = 0.5 * (top + bottom)
            center_dist = math.hypot((pc - center_c) * px, (pr - center_r) * py)

            # Overall best: max area, then max min-edge distance, then min center distance
            if (
                area > best_area
                or (area == best_area and best is not None and min_edge_dist > best[4])
                or (area == best_area and best is not None and min_edge_dist == best[4] and center_dist < best[5])
                or (area == best_area and best is None)
            ):
                best_area = area
                best = (top, bottom, max_left, min_right, min_edge_dist, center_dist)

            if pivot_threshold_m > 0 and min_edge_dist >= pivot_threshold_m:
                if (
                    area > best_thresh_area
                    or (area == best_thresh_area and best_thresh is not None and min_edge_dist > best_thresh[4])
                    or (area == best_thresh_area and best_thresh is not None and min_edge_dist == best_thresh[4] and center_dist < best_thresh[5])
                    or (area == best_thresh_area and best_thresh is None)
                ):
                    best_thresh_area = area
                    best_thresh = (top, bottom, max_left, min_right, min_edge_dist, center_dist)

    if best is None:
        raise ValueError("Could not find a valid rectangle containing pivot.")

    use_thresh = best_thresh is not None
    choice = best_thresh if use_thresh else best
    top, bottom, left_i, right_i = choice[0], choice[1], choice[2], choice[3]
    win = rasterio.windows.Window(
        col_off=left_i,
        row_off=top,
        width=(right_i - left_i + 1),
        height=(bottom - top + 1),
    )
    xmin, ymin, xmax, ymax = rasterio.windows.bounds(win, transform=src.transform)
    return xmin, ymin, xmax, ymax, use_thresh


def dem_to_stl(
    dem_tif: str,
    out_stl: str,
    nodata: float = -9999.0,
    local_xy: bool = True,
    z_base: Optional[float] = None,
    close_sides: bool = False,
) -> None:
    """
    Convert a DEM GeoTIFF to an STL terrain surface via regular grid triangulation.

    local_xy:
      - True  -> STL x,y start at (0,0) at DEM's lower-left (recommended for OpenFOAM).
      - False -> STL x,y are in CRS coordinates (meters), rarely needed.

    z_base:
      - If provided and close_sides=True, creates a closed volume down to z_base.
      - If None, only the top surface is written.

    close_sides:
      - If True, adds side walls and bottom (requires z_base).

    Requires:
      pip install numpy-stl
    """
    if stl_mesh is None:
        raise ImportError("numpy-stl is required for STL export: pip install numpy-stl")

    with rasterio.open(dem_tif) as src:
        Z = src.read(1).astype(np.float32)
        tf = src.transform

        # Create x/y coordinate vectors from transform
        ny, nx = Z.shape
        px = tf.a
        py = -tf.e
        x0 = tf.c
        y0_top = tf.f

        # CRS coords of pixel centers:
        xs = x0 + px * (np.arange(nx) + 0.5)
        ys = y0_top - py * (np.arange(ny) + 0.5)

        # Ensure ys is ascending (raster rows go top-to-bottom = north-to-south,
        # but STL/OpenFOAM expect y increasing = south-to-north).
        if ys[0] > ys[-1]:
            ys = ys[::-1]
            Z = Z[::-1, :]

        # Convert to local coords if requested
        if local_xy:
            xs = xs - xs.min()
            ys = ys - ys.min()

        # Mask nodata
        valid = np.isfinite(Z) & (Z != nodata)
        if not np.any(valid):
            raise ValueError("DEM has no valid data for STL export.")

        # Fill edge nodata by nearest valid propagation before STL export.
        #
        # Rotated/cropped DEMs can have thin nodata bands on one or more outer
        # edges. If those invalid edge cells are left as-is, the exported terrain
        # surface does not seal against the CFD box and snappyHexMesh can retain a
        # residual flat floor underneath the terrain.
        Z_fill = Z.copy()
        valid_fill = valid.copy()
        for _ in range(8):
            changed = False

            for i in range(1, ny):
                take = (~valid_fill[i]) & valid_fill[i - 1]
                if np.any(take):
                    Z_fill[i, take] = Z_fill[i - 1, take]
                    valid_fill[i, take] = True
                    changed = True
            for i in range(ny - 2, -1, -1):
                take = (~valid_fill[i]) & valid_fill[i + 1]
                if np.any(take):
                    Z_fill[i, take] = Z_fill[i + 1, take]
                    valid_fill[i, take] = True
                    changed = True
            for j in range(1, nx):
                take = (~valid_fill[:, j]) & valid_fill[:, j - 1]
                if np.any(take):
                    Z_fill[take, j] = Z_fill[take, j - 1]
                    valid_fill[take, j] = True
                    changed = True
            for j in range(nx - 2, -1, -1):
                take = (~valid_fill[:, j]) & valid_fill[:, j + 1]
                if np.any(take):
                    Z_fill[take, j] = Z_fill[take, j + 1]
                    valid_fill[take, j] = True
                    changed = True

            if np.all(valid_fill) or not changed:
                break

        # Build a corner-based surface so the STL reaches the DEM footprint edges.
        #
        # The previous implementation used DEM pixel centres directly as mesh
        # vertices. That systematically inset the terrain surface by half a cell
        # on each side, which can leave a connected volume underneath the
        # terrain after snappyHexMesh if the CFD box extends to the true DEM
        # footprint.
        #
        # Here we build a (ny+1) x (nx+1) corner grid. Corner elevations are a
        # simple average of the adjacent valid cell-centre values, with natural
        # edge/corner replication. This is sufficient for STL export and, most
        # importantly, makes the surface intersect the CFD side boundaries at the
        # full domain footprint.
        x_edges = px * np.arange(nx + 1, dtype=np.float32)
        y_edges = py * np.arange(ny + 1, dtype=np.float32)

        if not local_xy:
            x_edges = x0 + x_edges
            y_edges = (y0_top - py * ny) + y_edges

        Zc = np.zeros((ny + 1, nx + 1), dtype=np.float32)
        Vc = np.zeros((ny + 1, nx + 1), dtype=bool)
        for ii in range(ny + 1):
            i0 = max(ii - 1, 0)
            i1 = min(ii, ny - 1)
            for jj in range(nx + 1):
                j0 = max(jj - 1, 0)
                j1 = min(jj, nx - 1)
                vals = []
                for i in {i0, i1}:
                    for j in {j0, j1}:
                        if valid_fill[i, j]:
                            vals.append(float(Z_fill[i, j]))
                if vals:
                    Zc[ii, jj] = float(np.mean(vals))
                    Vc[ii, jj] = True

        Xg, Yg = np.meshgrid(x_edges, y_edges)
        V = np.stack([Xg, Yg, Zc], axis=-1)

        # Triangulate: two triangles per cell where all 4 corners are valid.
        tris = []
        for i in range(ny):
            for j in range(nx):
                if not (Vc[i, j] and Vc[i, j + 1] and Vc[i + 1, j] and Vc[i + 1, j + 1]):
                    continue
                v00 = V[i, j]
                v10 = V[i, j + 1]
                v01 = V[i + 1, j]
                v11 = V[i + 1, j + 1]
                tris.append([v00, v10, v01])
                tris.append([v10, v11, v01])

        if len(tris) == 0:
            raise ValueError("No valid triangles could be generated (nodata too large or crop too tight).")

        tris = np.asarray(tris, dtype=np.float32)

        # Optional: close sides and bottom to make watertight solid
        if close_sides:
            if z_base is None:
                raise ValueError("close_sides=True requires z_base to be set.")
            # Create a simple perimeter wall by using outer boundary indices
            # Only adds walls along the full raster edges; assumes edges are mostly valid.
            # For complex nodata edges, keep close_sides=False.
            z0 = float(z_base)

            # Helper to add quad as two tris
            def add_wall(a, b):
                a0 = a.copy(); a0[2] = z0
                b0 = b.copy(); b0[2] = z0
                # two triangles forming rectangle (a->b->b0->a0)
                return [[a, b, a0], [b, b0, a0]]

            # top edge (row 0)
            for j in range(nx - 1):
                if valid[0, j] and valid[0, j + 1]:
                    tris = np.concatenate([tris, np.array(add_wall(V[0, j], V[0, j + 1]), dtype=np.float32)], axis=0)
            # bottom edge (last row)
            for j in range(nx - 1):
                if valid[ny - 1, j] and valid[ny - 1, j + 1]:
                    tris = np.concatenate([tris, np.array(add_wall(V[ny - 1, j + 1], V[ny - 1, j]), dtype=np.float32)], axis=0)
            # left edge (col 0)
            for i in range(ny - 1):
                if valid[i, 0] and valid[i + 1, 0]:
                    tris = np.concatenate([tris, np.array(add_wall(V[i + 1, 0], V[i, 0]), dtype=np.float32)], axis=0)
            # right edge (last col)
            for i in range(ny - 1):
                if valid[i, nx - 1] and valid[i + 1, nx - 1]:
                    tris = np.concatenate([tris, np.array(add_wall(V[i, nx - 1], V[i + 1, nx - 1]), dtype=np.float32)], axis=0)

            # Bottom plate as a fan triangulation over bounding box (simple, not perfect for nodata holes)
            # Use the four corners in local xy at z_base.
            xmin, xmax = float(xs.min()), float(xs.max())
            ymin, ymax = float(ys.min()), float(ys.max())
            p0 = np.array([xmin, ymin, z0], dtype=np.float32)
            p1 = np.array([xmax, ymin, z0], dtype=np.float32)
            p2 = np.array([xmax, ymax, z0], dtype=np.float32)
            p3 = np.array([xmin, ymax, z0], dtype=np.float32)
            tris = np.concatenate([tris, np.array([[p0, p1, p2], [p0, p2, p3]], dtype=np.float32)], axis=0)

        m = stl_mesh.Mesh(np.zeros(tris.shape[0], dtype=stl_mesh.Mesh.dtype))
        m.vectors[:] = tris
        m.save(out_stl)


# -----------------------------
# Main program
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tifs", nargs="+", required=True, help="List of input GeoTIFFs (tiles or mosaics).")
    p.add_argument("--wind-from", type=float, default=None, help="Wind direction (meteorological FROM) in degrees.")
    p.add_argument("--abl", type=str, default=None, help="ABL file (abl.txt) to read theta/thetaN from.")
    p.add_argument("--abl-index", type=int, default=1, help="ABL variant index (1-based) when using theta1/theta2/...")
    p.add_argument("--all-thetas", action="store_true", help="If ABL has multiple thetas, generate one output per theta.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--width", type=float, default=None, help="Final domain width W in meters (optional).")
    p.add_argument("--height", type=float, default=None, help="Final domain height H in meters (optional).")
    p.add_argument("--pivot-x", type=float, default=None, help="Optional pivot X in CRS coords (meters).")
    p.add_argument("--pivot-y", type=float, default=None, help="Optional pivot Y in CRS coords (meters).")
    p.add_argument("--pivot-threshold-m", type=float, default=100.0, help="Preferred minimum distance (m) from pivot to domain boundaries when auto-cropping.")
    p.add_argument("--resample-order", type=int, default=1, choices=[0, 1, 3], help="Rotation resampling: 0=nearest, 1=bilinear, 3=cubic.")
    p.add_argument("--nodata", type=float, default=-9999.0, help="NoData value to write into outputs.")
    p.add_argument("--input-nodata", type=float, default=None, help="NoData value in input (overrides source metadata).")
    p.add_argument("--pad-fraction", type=float, default=0.0, help="Optional padding (fraction of max dimension) before rotation.")
    p.add_argument("--max-km-warning", type=float, default=5.0, help="Warn if final W or H exceed this (km).")
    p.add_argument("--stl", action="store_true", help="Export STL terrain mesh (default on).")
    p.add_argument("--no-stl", dest="stl", action="store_false", help="Disable STL export.")
    p.add_argument("--stl-close-sides", action="store_true", help="Close STL to a base (requires --stl-z-base).")
    p.add_argument("--stl-z-base", type=float, default=None, help="Base elevation for watertight STL if closing sides.")
    p.add_argument("--stl-crs-xy", action="store_true", help="Use CRS x/y in STL (default is local x/y from 0).")
    p.set_defaults(stl=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(args.out_dir)

    if (args.width is None) != (args.height is None):
        raise ValueError("Both --width and --height must be provided together.")

    if args.all_thetas and args.abl is None:
        raise ValueError("--all-thetas requires --abl.")
    if args.wind_from is None and args.abl is None:
        raise ValueError("Provide --wind-from or --abl.")
    if args.all_thetas and args.wind_from is not None:
        raise ValueError("Use either --all-thetas with --abl, or --wind-from (single).")

    # Expand wildcards so PowerShell/Windows works like bash
    tif_paths: List[str] = []
    for p in args.tifs:
        if any(ch in p for ch in ["*", "?", "["]):
            matches = glob.glob(p)
            tif_paths.extend(matches)
        else:
            tif_paths.append(p)
    if len(tif_paths) == 0:
        raise ValueError("No input TIFs matched. Check --tifs patterns.")
    merged_tif = os.path.join(args.out_dir, "merged_dem.tif")

    # Resolve wind_from list
    if args.abl:
        theta_map = _read_abl_thetas(args.abl)
        if args.all_thetas:
            wind_from_list = [theta_map[k] for k in sorted(theta_map.keys())]
        else:
            if args.abl_index not in theta_map:
                raise ValueError(f"ABL index {args.abl_index} not found in theta list.")
            wind_from_list = [theta_map[args.abl_index]]
    else:
        wind_from_list = [float(args.wind_from)]

    # Resolve pivot (center) from args, ABL, or selection.json (from dem_selector)
    pivot_xy_base = None
    pivot_threshold_m = float(args.pivot_threshold_m)
    if args.pivot_x is not None and args.pivot_y is not None:
        # Explicit CLI pivot takes priority
        pivot_xy_base = (args.pivot_x, args.pivot_y)
    elif args.abl:
        abl_data = _read_abl_raw(args.abl)
        if "pivot_x" in abl_data and "pivot_y" in abl_data:
            pivot_xy_base = (float(abl_data["pivot_x"]), float(abl_data["pivot_y"]))
        elif "center_x" in abl_data and "center_y" in abl_data:
            pivot_xy_base = (float(abl_data["center_x"]), float(abl_data["center_y"]))
        if "pivot_threshold_m" in abl_data:
            pivot_threshold_m = float(abl_data["pivot_threshold_m"])

    # Fallback: read center from dem_selector's selection.json
    if pivot_xy_base is None:
        for tif_path in tif_paths:
            sel_json = os.path.join(os.path.dirname(tif_path), "selection.json")
            if os.path.exists(sel_json):
                with open(sel_json) as f:
                    sel = json.load(f)
                center = sel.get("center_lv95", {})
                if "E" in center and "N" in center:
                    pivot_xy_base = (float(center["E"]), float(center["N"]))
                    print(f"[INFO] Pivot from selection.json: E={center['E']:.1f}, N={center['N']:.1f}")
                    break

    # 1) Merge once
    merge_tifs(tif_paths, merged_tif, input_nodata=args.input_nodata)

    def _compute_z_range(tif_path: str, nodata_val: float) -> Tuple[float, float]:
        with rasterio.open(tif_path) as src:
            Z = src.read(1).astype(np.float32)
        if np.isfinite(nodata_val):
            Z = np.where(Z == nodata_val, np.nan, Z)
        return float(np.nanmin(Z)), float(np.nanmax(Z))

    for wind_from in wind_from_list:
        out_dir = args.out_dir
        if len(wind_from_list) > 1:
            out_dir = os.path.join(args.out_dir, _theta_slug(wind_from))
            _ensure_dir(out_dir)

        rotated_tif = os.path.join(out_dir, "dem_rotated.tif")
        final_tif = os.path.join(out_dir, "dem_final.tif")
        meta_json = os.path.join(out_dir, "transform.json")
        stl_path = os.path.join(out_dir, "terrain.stl")

        # 2) Rotate
        pivot_xy = pivot_xy_base

        _, meta_rot = rotate_dem_to_tif(
            input_tif=merged_tif,
            output_rot_tif=rotated_tif,
            wind_from_deg=wind_from,
            pivot_xy=pivot_xy,
            resample_order=args.resample_order,
            nodata_out=args.nodata,
            pad_fraction=args.pad_fraction,
            input_nodata=args.input_nodata,
        )

        # Ensure pivot is known (from rotation step)
        pivot_xy = meta_rot.pivot_xy
        wind_to_deg, theta_math_deg = _meteo_to_rotation(wind_from)
        theta_deg = -theta_math_deg

        warning = None
        max_m = args.max_km_warning * 1000.0

        # 3) Crop: Mode A or Mode B
        if args.width is not None and args.height is not None:
            # Mode A: explicit size
            W = float(args.width)
            H = float(args.height)
            # source S is the minimal square required for rotation-safe crop (informational)
            S = float(math.sqrt(W * W + H * H))

            # Coverage check against merged bounds.
            # For exact multiples of 90° (0, 90, 180, 270), a WxH rectangle
            # rotated by 90° or 180° fits inside WxH (or HxW) — no diagonal
            # expansion needed.  Only non-axis-aligned rotations need the
            # full diagonal S = sqrt(W²+H²).
            theta_mod = abs(theta_deg) % 360
            is_axis_aligned = any(abs(theta_mod - a) < 1e-3 for a in (0, 90, 180, 270, 360))

            with rasterio.open(merged_tif) as src:
                b = src.bounds
                if is_axis_aligned:
                    # For 90°/270° rotation of WxH, the footprint becomes HxW
                    if any(abs(theta_mod - a) < 1e-3 for a in (90, 270)):
                        halfX, halfY = 0.5 * H, 0.5 * W  # swapped
                    else:
                        halfX, halfY = 0.5 * W, 0.5 * H
                    if (pivot_xy[0] - halfX) < b.left or (pivot_xy[0] + halfX) > b.right or \
                       (pivot_xy[1] - halfY) < b.bottom or (pivot_xy[1] + halfY) > b.top:
                        raise ValueError(
                            f"Input DEM mosaic is not large enough for {W:.0f}x{H:.0f} m crop around pivot.\n"
                            f"Pivot: {pivot_xy}\n"
                            f"Mosaic bounds: left={b.left:.1f}, bottom={b.bottom:.1f}, right={b.right:.1f}, top={b.top:.1f}\n"
                            "Provide more tiles or reduce W/H."
                        )
                else:
                    halfS = 0.5 * S
                    if (pivot_xy[0] - halfS) < b.left or (pivot_xy[0] + halfS) > b.right or \
                       (pivot_xy[1] - halfS) < b.bottom or (pivot_xy[1] + halfS) > b.top:
                        raise ValueError(
                            "Input DEM mosaic is not large enough to support the requested W×H after rotation.\n"
                            f"Need at least S=sqrt(W^2+H^2)={S:.1f} m square around pivot.\n"
                            f"Pivot: {pivot_xy}\n"
                            f"Mosaic bounds: left={b.left:.1f}, bottom={b.bottom:.1f}, right={b.right:.1f}, top={b.top:.1f}\n"
                            "Provide more tiles (bigger mosaic) or reduce W/H."
                        )

            W_written, H_written = crop_rotated_dem(
                rotated_tif=rotated_tif,
                output_tif=final_tif,
                pivot_xy=pivot_xy,
                W=W,
                H=H,
                nodata_out=args.nodata,
            )

            mode = "explicit_size"
            if W_written > max_m or H_written > max_m:
                warning = f"Domain size is {W_written/1000:.2f} km × {H_written/1000:.2f} km (> {args.max_km_warning:.1f} km). Accurate predictions not ensured."

            meta = TransformMeta(
                crs=meta_rot.crs,
                wind_from_deg=float(wind_from),
                wind_to_deg=float(wind_to_deg),
                theta_deg=float(theta_deg),
                pivot_xy=pivot_xy,
                final_W=float(W_written),
                final_H=float(H_written),
                source_S=float(S),
                nodata=float(args.nodata),
                mode=mode,
                warning=warning,
            )

        else:
            # Mode B: maximum possible within provided mosaic (after rotation)
            with rasterio.open(rotated_tif) as src:
                # shrink away nodata borders created by rotation
                if pivot_xy is not None:
                    try:
                        xmin, ymin, xmax, ymax, used_thresh = max_valid_inner_bounds_with_pivot(
                            src, nodata=args.nodata, pivot_xy=pivot_xy, pivot_threshold_m=pivot_threshold_m
                        )
                    except ValueError as e:
                        print(f"WARNING: {e}. Making DEM in default region.", file=sys.stderr)
                        xmin, ymin, xmax, ymax = max_valid_inner_bounds(src, nodata=args.nodata)
                        used_thresh = False
                else:
                    xmin, ymin, xmax, ymax = max_valid_inner_bounds(src, nodata=args.nodata)
                    used_thresh = False

            # Write this max-valid window as final_tif
            with rasterio.open(rotated_tif) as src:
                win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
                arr = src.read(1, window=win).astype(np.float32)
                out_transform = src.window_transform(win)

                out_profile = src.profile.copy()
                out_profile.update(
                    driver="GTiff",
                    dtype="float32",
                    nodata=args.nodata,
                    height=arr.shape[0],
                    width=arr.shape[1],
                    transform=out_transform,
                    compress="lzw",
                    tiled=True,
                    count=1,
                )

                with rasterio.open(final_tif, "w", **out_profile) as dst:
                    dst.write(arr, 1)

            W = float(xmax - xmin)
            H = float(ymax - ymin)

            mode = "max_possible"
            if W > max_m or H > max_m:
                warning = f"Domain size is {W/1000:.2f} km × {H/1000:.2f} km (> {args.max_km_warning:.1f} km). Accurate predictions not ensured."

            meta = TransformMeta(
                crs=meta_rot.crs,
                wind_from_deg=float(wind_from),
                wind_to_deg=float(wind_to_deg),
                theta_deg=float(theta_deg),
                pivot_xy=pivot_xy,
                final_W=W,
                final_H=H,
                source_S=0.0,
                nodata=float(args.nodata),
                mode=mode,
                warning=warning,
            )

        # 4) Save metadata (add domain size + z-range)
        z_min, z_max = _compute_z_range(final_tif, args.nodata)
        meta_dict = meta.__dict__.copy()
        meta_dict["theta_math_deg"] = float(theta_math_deg)
        if pivot_xy is not None and args.width is None and args.height is None:
            meta_dict["pivot_threshold_m"] = float(pivot_threshold_m)
            meta_dict["pivot_threshold_met"] = bool(used_thresh)
        meta_dict["domain_size"] = [float(meta.final_W), float(meta.final_H), float(z_max - z_min)]
        meta_dict["z_range"] = [float(z_min), float(z_max)]
        with open(meta_json, "w") as f:
            json.dump(meta_dict, f, indent=2)

        # 5) STL export (optional but recommended for your pipeline)
        if args.stl:
            dem_to_stl(
                dem_tif=final_tif,
                out_stl=stl_path,
                nodata=args.nodata,
                local_xy=(not args.stl_crs_xy),
                z_base=args.stl_z_base,
                close_sides=args.stl_close_sides,
            )

        # 6) Print summary
        print(f"merged:  {merged_tif}")
        print(f"rotated: {rotated_tif}")
        print(f"final:   {final_tif}")
        print(f"meta:    {meta_json}")
        if args.stl:
            print(f"stl:     {stl_path}")
        if meta.warning:
            print(f"WARNING: {meta.warning}")
        print(f"mode={meta.mode} wind_from={meta.wind_from_deg:.1f} wind_to={meta.wind_to_deg:.1f} theta={meta.theta_deg:.1f}deg")
        print(f"pivot={meta.pivot_xy} final_size={meta.final_W:.1f}m x {meta.final_H:.1f}m")


if __name__ == "__main__":
    main()
