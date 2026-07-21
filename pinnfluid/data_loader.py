
"""Binary-array data loading for the unified terrain-structure model."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import (
    ABL_UREF_MAX,
    BC_POINTS_INLET,
    BC_POINTS_OUTLET,
    BC_POINTS_SIDE,
    BC_POINTS_TOP,
    DATA_CFD_ROOT,
    FD_FLUID_MASK_THRESHOLD,
    GLOBAL_INPUT_COLS,
    GLOBAL_SUPERVISED_GROUND_K_FRAC,
    GLOBAL_SUPERVISED_NEAR_GROUND_FRAC,
    GRID_UNET_ROI_STRUCTURE_MODE,
    LATERAL_TRIM_M,
    LATERAL_TRIM_THRESHOLD,
    OUTPUT_COLS,
    PATCH_NEAR_GROUND_PROB,
    PHI_GROUND_H,
    PHI_WALL_H,
    ROI_INPUT_COLS,
    ROI_NEAR_WALL_DMAX,
    ROI_TARGET_GEOM_WAKE_FRAC,
    ROI_TARGET_HIGH_SPEED_FRAC,
    ROI_TARGET_HIGH_SPEED_RATIO_MIN,
    ROI_TARGET_HIGH_SPEED_ZREL_MAX,
    ROI_TARGET_LOW_SPEED_FRAC,
    ROI_TARGET_LOW_SPEED_RATIO_MAX,
    ROI_TARGET_LOW_SPEED_ZREL_MAX,
    ROI_TARGET_MAX_ABOVE_STRUCTURE_H,
    ROI_TARGET_NEAR_WALL_BACKFILL_DMAX,
    ROI_TARGET_NEAR_WALL_DMAX,
    ROI_TARGET_NEAR_WALL_FRAC,
    ROI_TARGET_RANDOM_FRAC,
    ROI_TARGET_VERY_NEAR_WALL_DMAX,
    ROI_TARGET_VERY_NEAR_WALL_FRAC,
    ROI_TARGET_VERY_NEAR_WALL_MAX_REPEAT,
    ROI_TARGET_WAKE_MIN_CONTEXT,
    ROI_TARGET_WAKE_ZREL_MAX,
    ROI_PATCH_HIGH_SPEED_PROB,
    ROI_PATCH_HIGH_SPEED_RATIO_MIN,
    ROI_PATCH_HIGH_SPEED_ZREL_MAX,
    SCALER_POINTS_PER_GRID,
    STRUCTURE_CONTEXT_DENSITY_SIGMA_M,
    STRUCTURE_CONTEXT_DISTANCE_SCALE_M,
    STRUCTURE_CONTEXT_WAKE_LENGTH_MULT,
    STRUCTURE_CONTEXT_WAKE_WIDTH_GROWTH,
    STRUCTURE_ENCODER_INPUT_MODE,
    STRUCTURE_HEIGHT_SCALE,
    resolve_structure_channel_mode,
    TRAIN_STRUCT_MODE,
    WEIGHTED_CHARB_HIGH_GAIN,
    WEIGHTED_CHARB_HIGH_START,
    WEIGHTED_CHARB_LOW_GAIN,
    WEIGHTED_CHARB_LOW_START,
    WEIGHTED_CHARB_POWER,
)
from utils import aspect_to_sin_cos, case_id, ensure_dir, is_canonical_case, normalize_abl_features, normalize_domain_size, read_json


@dataclass
class GridBundle:
    name: str
    category: str
    kind: str
    case_dir: Path
    roi_name: Optional[str]
    parent_name: Optional[str]
    bounds: tuple[float, float, float, float, float, float]
    x_coords: np.ndarray
    y_coords: np.ndarray
    z_levels: np.ndarray
    vertical_coordinate_mode: str
    terrain_raw: dict[str, np.ndarray]
    terrain_model: np.ndarray
    flow: np.ndarray
    is_fluid: np.ndarray
    nut: Optional[np.ndarray]
    phi_wall: Optional[np.ndarray]
    abl: dict[str, float]
    size_norm: dict[str, float]
    uref: float
    lref: float
    div_scale: float
    mom_scale: float
    meta: dict
    valid_i_range: tuple[int, int]   # (inclusive start, exclusive end) after lateral trim
    valid_j_range: tuple[int, int]
    valid_k_max: int                 # exclusive upper bound on k after z-cap
    near_ground_zrel_cap: float
    structure_model: Optional[np.ndarray] = None
    targeted_roi_sample_cache: Optional[dict] = None


@dataclass
class PointBatch:
    x_scaled: torch.Tensor
    y_scaled: torch.Tensor
    xy_local: torch.Tensor
    xy_global: Optional[torch.Tensor]
    sample_stats: Optional[dict[str, float]] = None


@dataclass
class BoundaryBatch:
    x_scaled: torch.Tensor
    xy_local: torch.Tensor
    normals: Optional[torch.Tensor]
    u_target: Optional[torch.Tensor]
    p_target: Optional[torch.Tensor]
    u_scale: float
    p_scale: float


@dataclass
class PatchBatch:
    x_scaled: torch.Tensor
    y_scaled: torch.Tensor
    x_volume_scaled: torch.Tensor
    y_volume_scaled: torch.Tensor
    xy_local: torch.Tensor
    xy_global: Optional[torch.Tensor]
    mask: torch.Tensor
    nut: Optional[torch.Tensor]
    x_coords: torch.Tensor
    y_coords: torch.Tensor
    z_levels: torch.Tensor
    shape: tuple[int, int, int]
    origin: tuple[int, int, int]
    div_scale: float
    mom_scale: float
    source_name: str


@dataclass
class ScalerBundle:
    x_scaler_global: Optional[MinMaxScaler]
    x_scaler_roi: Optional[MinMaxScaler]
    y_scaler: StandardScaler


_WARNED_NUT_SOURCES: set[str] = set()


def _vertical_coordinate_mode_from_meta(meta: dict) -> str:
    mode = str(
        meta.get('vertical_coordinate_mode')
        or (meta.get('preprocessing') or {}).get('vertical_coordinate_mode')
        or 'absolute'
    ).lower()
    if mode not in {'absolute', 'terrain_following'}:
        raise ValueError(f"Unsupported vertical_coordinate_mode={mode!r}")
    return mode


def bundle_uses_terrain_following_z(bundle: GridBundle) -> bool:
    return str(bundle.vertical_coordinate_mode).lower() == 'terrain_following'


def bundle_z_rel_at(bundle: GridBundle, ii: np.ndarray, jj: np.ndarray, kk: np.ndarray) -> np.ndarray:
    z = np.asarray(bundle.z_levels, dtype=np.float32)[kk]
    if bundle_uses_terrain_following_z(bundle):
        return np.maximum(z, 0.0).astype(np.float32, copy=False)
    elev = bundle.terrain_raw['elevation'][jj, ii]
    return np.maximum(z - elev, 0.0).astype(np.float32, copy=False)


def bundle_z_abs_at(bundle: GridBundle, ii: np.ndarray, jj: np.ndarray, kk: np.ndarray) -> np.ndarray:
    z = np.asarray(bundle.z_levels, dtype=np.float32)[kk]
    if bundle_uses_terrain_following_z(bundle):
        elev = bundle.terrain_raw['elevation'][jj, ii]
        return (elev + z).astype(np.float32, copy=False)
    return z.astype(np.float32, copy=False)


def _bundle_z_rel_volume(bundle: GridBundle) -> np.ndarray:
    z = np.asarray(bundle.z_levels, dtype=np.float32)[None, None, :]
    if bundle_uses_terrain_following_z(bundle):
        return np.broadcast_to(z, bundle.flow.shape[:3]).astype(np.float32, copy=False)
    elev = np.asarray(bundle.terrain_raw['elevation'], dtype=np.float32).T[:, :, None]
    return (z - elev).astype(np.float32, copy=False)


def _bundle_z_abs_volume(bundle: GridBundle) -> np.ndarray:
    z = np.asarray(bundle.z_levels, dtype=np.float32)[None, None, :]
    if bundle_uses_terrain_following_z(bundle):
        elev = np.asarray(bundle.terrain_raw['elevation'], dtype=np.float32).T[:, :, None]
        return (elev + z).astype(np.float32, copy=False)
    return np.broadcast_to(z, bundle.flow.shape[:3]).astype(np.float32, copy=False)


def _sanitize_nut_array(nut: np.ndarray, *, source: Path) -> np.ndarray:
    neg = np.isfinite(nut) & (nut < 0.0)
    if not np.any(neg):
        return nut
    out = np.array(nut, copy=True, dtype=np.float32)
    neg_count = int(np.count_nonzero(neg))
    min_neg = float(np.nanmin(out[neg]))
    out[neg] = 0.0
    key = str(source)
    if key not in _WARNED_NUT_SOURCES:
        print(
            f"[WARN] clamped {neg_count} negative nut values to 0 for {source} "
            f"(min={min_neg:.6g})",
            flush=True,
        )
        _WARNED_NUT_SOURCES.add(key)
    return out


class CaseRepository:
    def __init__(self, data_root: Path = DATA_CFD_ROOT, *, global_cache_limit: int = 8, roi_cache_limit: int = 6):
        self.data_root = Path(data_root)
        self.case_dirs: dict[str, Path] = {}
        for category in ['complexterrain_only', 'singlestructures', 'multistructures']:
            croot = self.data_root / category
            if not croot.exists():
                continue
            for case_dir in croot.iterdir():
                if case_dir.is_dir() and _is_grid_case_dir(case_dir):
                    self.case_dirs[case_dir.name] = case_dir
        self.global_cache_limit = int(global_cache_limit)
        self.roi_cache_limit = int(roi_cache_limit)
        self._global_cache: OrderedDict[str, GridBundle] = OrderedDict()
        self._roi_cache: OrderedDict[tuple[str, str], GridBundle] = OrderedDict()

    def canonical_case_names(self) -> list[str]:
        return sorted(self.case_dirs.keys(), key=lambda s: (case_id(s) or 10**9, s))

    def roi_names(self, case_name: str) -> list[str]:
        roi_root = self.case_dirs[case_name] / 'roi'
        if not roi_root.exists():
            return []
        return sorted([p.name for p in roi_root.iterdir() if p.is_dir() and p.name.startswith('roi_')])

    def load_global(self, case_name: str) -> GridBundle:
        if case_name in self._global_cache:
            self._global_cache.move_to_end(case_name)
            return self._global_cache[case_name]
        bundle = _load_grid_bundle(self.case_dirs[case_name], kind='global', roi_name=None, parent_name=None)
        self._global_cache[case_name] = bundle
        while len(self._global_cache) > self.global_cache_limit:
            self._global_cache.popitem(last=False)
        return bundle

    def load_roi(self, case_name: str, roi_name: str) -> GridBundle:
        key = (case_name, roi_name)
        if key in self._roi_cache:
            self._roi_cache.move_to_end(key)
            return self._roi_cache[key]
        bundle = _load_grid_bundle(self.case_dirs[case_name] / 'roi' / roi_name, kind='roi', roi_name=roi_name, parent_name=case_name)
        self._roi_cache[key] = bundle
        while len(self._roi_cache) > self.roi_cache_limit:
            self._roi_cache.popitem(last=False)
        return bundle


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as obj:
        return {k: np.asarray(obj[k]) for k in obj.files}


def _is_grid_case_dir(case_dir: Path) -> bool:
    return (
        (case_dir / 'meta.json').exists()
        and (case_dir / 'terrain.npz').exists()
        and (case_dir / 'flow.npz').exists()
    )


def _terrain_model_channels(terrain_raw: dict[str, np.ndarray]) -> np.ndarray:
    elev = np.asarray(terrain_raw['elevation'], dtype=np.float32)
    slope = np.asarray(terrain_raw['slope'], dtype=np.float32)
    aspect = np.asarray(terrain_raw['aspect'], dtype=np.float32)
    asin, acos = aspect_to_sin_cos(aspect)
    elev_centered = (elev - np.mean(elev, dtype=np.float64)).astype(np.float32) / 1000.0
    slope_norm = np.clip(slope / 90.0, 0.0, 1.0).astype(np.float32)
    return np.stack([elev_centered, slope_norm, asin.astype(np.float32), acos.astype(np.float32)], axis=0)


def _structure_bounds_records(bundle: GridBundle) -> list[dict[str, float]]:
    bounds_list = bundle.meta.get('structure_bounds') or []
    if not isinstance(bounds_list, list) or not bounds_list:
        return []
    records: list[dict[str, float]] = []
    for sb in bounds_list:
        if not isinstance(sb, dict):
            continue
        sb_min = sb.get('min')
        sb_max = sb.get('max')
        if not isinstance(sb_min, (list, tuple)) or not isinstance(sb_max, (list, tuple)) or len(sb_min) < 3 or len(sb_max) < 3:
            continue
        x_min, x_max = sorted((float(sb_min[0]), float(sb_max[0])))
        y_min, y_max = sorted((float(sb_min[1]), float(sb_max[1])))
        z_min, z_max = sorted((float(sb_min[2]), float(sb_max[2])))
        if not np.isfinite([x_min, x_max, y_min, y_max, z_min, z_max]).all():
            continue
        records.append({
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'z_min': z_min,
            'z_max': z_max,
            'cx': 0.5 * (x_min + x_max),
            'cy': 0.5 * (y_min + y_max),
            'half_x': 0.5 * max(x_max - x_min, 1e-6),
            'half_y': 0.5 * max(y_max - y_min, 1e-6),
            'height': max(z_max - z_min, 1e-6),
        })
    return records


def _structure_basic_channels(bundle: GridBundle, records: list[dict[str, float]]) -> np.ndarray:
    ny = int(bundle.terrain_raw['elevation'].shape[0])
    nx = int(bundle.terrain_raw['elevation'].shape[1])
    footprint = np.zeros((ny, nx), dtype=np.float32)
    height = np.zeros((ny, nx), dtype=np.float32)
    if not records:
        return np.stack([footprint, height], axis=0)
    elev = np.asarray(bundle.terrain_raw['elevation'], dtype=np.float32)
    h_scale = max(float(STRUCTURE_HEIGHT_SCALE), 1e-6)
    for rec in records:
        x_min, x_max = float(rec['x_min']), float(rec['x_max'])
        y_min, y_max = float(rec['y_min']), float(rec['y_max'])
        z_top = float(rec['z_max'])
        x_mask = (bundle.x_coords >= x_min) & (bundle.x_coords <= x_max)
        y_mask = (bundle.y_coords >= y_min) & (bundle.y_coords <= y_max)
        if not bool(np.any(x_mask)) or not bool(np.any(y_mask)):
            continue
        iy, ix = np.ix_(y_mask, x_mask)
        footprint[iy, ix] = 1.0
        local_height = np.maximum(z_top - elev[iy, ix], 0.0).astype(np.float32) / h_scale
        height[iy, ix] = np.maximum(height[iy, ix], local_height)
    return np.stack([footprint, height], axis=0)


def _structure_context_v2_channels(bundle: GridBundle, records: list[dict[str, float]]) -> np.ndarray:
    basic = _structure_basic_channels(bundle, records)
    footprint = basic[0]
    height = basic[1]
    ny = int(footprint.shape[0])
    nx = int(footprint.shape[1])
    if not records:
        zeros = np.zeros((6, ny, nx), dtype=np.float32)
        return np.concatenate([basic, zeros], axis=0)

    dist_scale = max(float(STRUCTURE_CONTEXT_DISTANCE_SCALE_M), 1e-6)
    wake_length_mult = max(float(STRUCTURE_CONTEXT_WAKE_LENGTH_MULT), 1.0)
    wake_width_growth = max(float(STRUCTURE_CONTEXT_WAKE_WIDTH_GROWTH), 0.0)
    density_sigma_m = max(float(STRUCTURE_CONTEXT_DENSITY_SIGMA_M), 1.0)

    wx = float(bundle.abl.get('flowDir_x', 1.0))
    wy = float(bundle.abl.get('flowDir_y', 0.0))
    wnorm = max(float(np.hypot(wx, wy)), 1e-6)
    wx /= wnorm
    wy /= wnorm
    cx_dir = -wy
    cy_dir = wx

    xg, yg = np.meshgrid(bundle.x_coords.astype(np.float32), bundle.y_coords.astype(np.float32), indexing='xy')
    best_abs_signed_dist = np.full((ny, nx), np.inf, dtype=np.float32)
    best_signed_dist = np.full((ny, nx), np.inf, dtype=np.float32)
    nearest_streamwise = np.zeros((ny, nx), dtype=np.float32)
    nearest_crosswise = np.zeros((ny, nx), dtype=np.float32)
    wake_max = np.zeros((ny, nx), dtype=np.float32)
    wake_sum = np.zeros((ny, nx), dtype=np.float32)
    density_sum = np.zeros((ny, nx), dtype=np.float32)

    for rec in records:
        x_min, x_max = float(rec['x_min']), float(rec['x_max'])
        y_min, y_max = float(rec['y_min']), float(rec['y_max'])
        cx = float(rec['cx'])
        cy = float(rec['cy'])
        half_x = float(rec['half_x'])
        half_y = float(rec['half_y'])
        struct_h = max(float(rec['height']), 1e-3)

        dx = xg - cx
        dy = yg - cy
        s_center = dx * wx + dy * wy
        c_center = dx * cx_dir + dy * cy_dir
        half_along = max(abs(wx) * half_x + abs(wy) * half_y, 1e-6)
        half_cross = max(abs(cx_dir) * half_x + abs(cy_dir) * half_y, 1e-6)

        dx_out = np.maximum(np.maximum(x_min - xg, 0.0), xg - x_max)
        dy_out = np.maximum(np.maximum(y_min - yg, 0.0), yg - y_max)
        outside_dist = np.hypot(dx_out, dy_out).astype(np.float32)
        inside_margin = np.minimum.reduce([xg - x_min, x_max - xg, yg - y_min, y_max - yg]).astype(np.float32)
        signed_dist = np.where(inside_margin >= 0.0, -inside_margin, outside_dist).astype(np.float32)
        abs_signed_dist = np.abs(signed_dist)

        s_edge = np.sign(s_center) * np.maximum(np.abs(s_center) - half_along, 0.0)
        c_edge = np.sign(c_center) * np.maximum(np.abs(c_center) - half_cross, 0.0)
        s_scale = max(2.0 * half_along, dist_scale)
        c_scale = max(2.0 * half_cross, 0.5 * dist_scale)
        s_edge_norm = np.tanh(s_edge / s_scale).astype(np.float32)
        c_edge_norm = np.tanh(c_edge / c_scale).astype(np.float32)

        update = abs_signed_dist < best_abs_signed_dist
        if np.any(update):
            best_abs_signed_dist[update] = abs_signed_dist[update]
            best_signed_dist[update] = signed_dist[update]
            nearest_streamwise[update] = s_edge_norm[update]
            nearest_crosswise[update] = c_edge_norm[update]

        trailing_down = np.maximum(s_center - half_along, 0.0)
        struct_scale = max(struct_h, 2.0 * half_along, 2.0 * half_cross, 1.0)
        wake_len = max(dist_scale, wake_length_mult * struct_scale)
        wake_half_width = np.maximum(max(half_cross, 0.5 * struct_h) + wake_width_growth * trailing_down, 1e-3)
        wake = np.exp(-trailing_down / wake_len) * np.exp(-0.5 * (c_center / wake_half_width) ** 2)
        wake = np.where(trailing_down > 0.0, wake, 0.0).astype(np.float32)
        wake_max = np.maximum(wake_max, wake)
        wake_sum += wake

        sigma = max(density_sigma_m, 2.0 * half_x, 2.0 * half_y, struct_h)
        density_sum += np.exp(-0.5 * ((dx * dx + dy * dy) / (sigma * sigma))).astype(np.float32)

    signed_dist_norm = np.tanh(best_signed_dist / dist_scale).astype(np.float32)
    wake_accum = (1.0 - np.exp(-wake_sum)).astype(np.float32)
    density = (1.0 - np.exp(-density_sum)).astype(np.float32)
    return np.stack(
        [
            footprint,
            height,
            signed_dist_norm,
            nearest_streamwise,
            nearest_crosswise,
            np.clip(wake_max, 0.0, 1.0).astype(np.float32),
            np.clip(wake_accum, 0.0, 1.0).astype(np.float32),
            np.clip(density, 0.0, 1.0).astype(np.float32),
        ],
        axis=0,
    )


def _structure_wall_orientation_channels(bundle: GridBundle) -> np.ndarray:
    ny = int(bundle.terrain_raw['elevation'].shape[0])
    nx = int(bundle.terrain_raw['elevation'].shape[1])
    zeros = np.zeros((4, ny, nx), dtype=np.float32)
    if bundle.phi_wall is None:
        return zeros

    phi = np.asarray(bundle.phi_wall, dtype=np.float32)
    if phi.ndim != 3 or phi.shape[0] != nx or phi.shape[1] != ny:
        return zeros
    if len(bundle.x_coords) < 2 or len(bundle.y_coords) < 2 or len(bundle.z_levels) < 2:
        return zeros

    finite = np.isfinite(phi)
    if not bool(np.any(finite)):
        return zeros

    # ROI-only path. Terrain-following ROIs are not supported yet: this gradient
    # treats z_levels as rectilinear physical coordinates.
    # Signed-distance gradients give deterministic wall normals without new data.
    # The minus sign follows the convention used for windward-facing features.
    phi_safe = np.where(finite, phi, 0.0).astype(np.float32, copy=False)
    try:
        dphi_dx, dphi_dy, dphi_dz = np.gradient(
            phi_safe,
            bundle.x_coords.astype(np.float32),
            bundle.y_coords.astype(np.float32),
            bundle.z_levels.astype(np.float32),
            edge_order=1,
        )
    except Exception:
        return zeros

    norm = np.sqrt(dphi_dx * dphi_dx + dphi_dy * dphi_dy + dphi_dz * dphi_dz).astype(np.float32)
    norm = np.where(norm > 1e-6, norm, 1.0).astype(np.float32, copy=False)
    n_x = (-dphi_dx / norm).astype(np.float32)
    n_y = (-dphi_dy / norm).astype(np.float32)
    n_z = (-dphi_dz / norm).astype(np.float32)

    abs_phi = np.where(finite, np.abs(phi), np.inf).astype(np.float32, copy=False)
    k_nearest = np.argmin(abs_phi, axis=2)
    has_nearest = np.isfinite(np.min(abs_phi, axis=2))
    ii = np.arange(nx)[:, None]
    jj = np.arange(ny)[None, :]

    n_x2 = np.where(has_nearest, n_x[ii, jj, k_nearest], 0.0).astype(np.float32)
    n_y2 = np.where(has_nearest, n_y[ii, jj, k_nearest], 0.0).astype(np.float32)
    n_z2 = np.where(has_nearest, n_z[ii, jj, k_nearest], 0.0).astype(np.float32)

    wx = float(bundle.abl.get('flowDir_x', 1.0))
    wy = float(bundle.abl.get('flowDir_y', 0.0))
    wnorm = max(float(np.hypot(wx, wy)), 1e-6)
    wx /= wnorm
    wy /= wnorm
    windward = np.clip(n_x2 * wx + n_y2 * wy, -1.0, 1.0).astype(np.float32)

    return np.stack(
        [
            n_x2.T.astype(np.float32, copy=False),
            n_y2.T.astype(np.float32, copy=False),
            n_z2.T.astype(np.float32, copy=False),
            windward.T.astype(np.float32, copy=False),
        ],
        axis=0,
    )


def _structure_context_v3_channels(bundle: GridBundle, records: list[dict[str, float]]) -> np.ndarray:
    context_v2 = _structure_context_v2_channels(bundle, records)
    wall_orientation = _structure_wall_orientation_channels(bundle)
    return np.concatenate([context_v2, wall_orientation], axis=0).astype(np.float32, copy=False)


def _structure_model_channels_for_mode(bundle: GridBundle, mode: str) -> np.ndarray:
    mode = resolve_structure_channel_mode(mode)
    if mode == 'none':
        ny = int(bundle.terrain_raw['elevation'].shape[0])
        nx = int(bundle.terrain_raw['elevation'].shape[1])
        return np.zeros((0, ny, nx), dtype=np.float32)
    records = _structure_bounds_records(bundle)
    if mode == 'basic':
        return _structure_basic_channels(bundle, records)
    if mode == 'context_v2':
        return _structure_context_v2_channels(bundle, records)
    if mode == 'context_v3':
        return _structure_context_v3_channels(bundle, records)
    raise ValueError(f"Unsupported structure channel mode: {mode!r}")


def _structure_model_channels(bundle: GridBundle) -> np.ndarray:
    return _structure_model_channels_for_mode(bundle, STRUCTURE_ENCODER_INPUT_MODE)


def _compute_valid_ranges(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_levels: np.ndarray,
    bounds: tuple,
    terrain_raw: dict[str, np.ndarray],
    vertical_coordinate_mode: str = "absolute",
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """Compute valid index ranges after lateral trim and z-cap masks (TO_READ.md)."""
    x0, x1, y0, y1, _, _ = bounds

    # Lateral trim: if bounds start near 0 the export had no edge buffer
    if x0 < LATERAL_TRIM_THRESHOLD:
        i_min = int(np.searchsorted(x_coords, x0 + LATERAL_TRIM_M, side='left'))
        i_max = int(np.searchsorted(x_coords, x1 - LATERAL_TRIM_M, side='right'))
    else:
        i_min, i_max = 0, len(x_coords)
    if y0 < LATERAL_TRIM_THRESHOLD:
        j_min = int(np.searchsorted(y_coords, y0 + LATERAL_TRIM_M, side='left'))
        j_max = int(np.searchsorted(y_coords, y1 - LATERAL_TRIM_M, side='right'))
    else:
        j_min, j_max = 0, len(y_coords)

    i_max = max(i_max, i_min + 1)
    j_max = max(j_max, j_min + 1)

    if str(vertical_coordinate_mode).lower() == "terrain_following":
        # Terrain-following exports already encode the z-cap in relative coordinates.
        k_max = len(z_levels)
    else:
        # Z-cap: ignore upper freestream based on domain extent
        domain_xy = max(x1 - x0, y1 - y0)
        terrain_z_max = float(np.nanmax(terrain_raw['elevation']))
        if domain_xy <= 600:
            z_cap = terrain_z_max + 200.0
        elif domain_xy <= 1200:
            z_cap = terrain_z_max + 300.0
        else:
            z_cap = terrain_z_max + 500.0

        k_max = int(np.searchsorted(z_levels, z_cap, side='right'))
        k_max = max(k_max, min(2, len(z_levels)))
        k_max = min(k_max, len(z_levels))

    return (i_min, i_max), (j_min, j_max), k_max


def _compute_near_ground_zrel_cap(
    terrain_raw: dict[str, np.ndarray],
    flow: np.ndarray,
    is_fluid: np.ndarray,
    z_levels: np.ndarray,
    *,
    vertical_coordinate_mode: str = "absolute",
    quantile: float = GLOBAL_SUPERVISED_GROUND_K_FRAC,
) -> float:
    valid = (is_fluid > FD_FLUID_MASK_THRESHOLD) & np.isfinite(flow).all(axis=-1)
    if not np.any(valid):
        return 0.0
    z = np.asarray(z_levels, dtype=np.float32)[None, None, :]
    if str(vertical_coordinate_mode).lower() == "terrain_following":
        z_rel = np.broadcast_to(z, flow.shape[:3])
    else:
        terrain_ij = np.asarray(terrain_raw['elevation'], dtype=np.float32).T[:, :, None]
        z_rel = z - terrain_ij
    vals = z_rel[valid]
    vals = vals[np.isfinite(vals) & (vals >= 0.0)]
    if vals.size == 0:
        return 0.0
    q = float(np.clip(quantile, 0.05, 1.0))
    return max(float(np.quantile(vals, q)), 1e-3)


def _load_grid_bundle(case_dir: Path, *, kind: str, roi_name: Optional[str], parent_name: Optional[str]) -> GridBundle:
    meta = read_json(case_dir / 'meta.json')
    terrain_raw = _load_npz_dict(case_dir / 'terrain.npz')
    flow_raw = _load_npz_dict(case_dir / 'flow.npz')
    phi_wall = None
    phi_path = case_dir / 'phi_wall.npy'
    if phi_path.exists():
        phi_wall = np.load(phi_path, allow_pickle=False).astype(np.float32)
    nut = None
    nut_path = case_dir / 'nut.npy'
    if nut_path.exists():
        nut = np.load(nut_path, allow_pickle=False).astype(np.float32)
        nut = _sanitize_nut_array(nut, source=nut_path)
    bounds = tuple(float(v) for v in meta['bounds'])
    nx, ny, nz = [int(v) for v in meta['grid_shape']]
    x0, x1, y0, y1, _, _ = bounds
    x_coords = np.linspace(x0, x1, nx, dtype=np.float32)
    y_coords = np.linspace(y0, y1, ny, dtype=np.float32)
    z_levels = np.asarray(meta['z_levels'], dtype=np.float32)
    vertical_coordinate_mode = _vertical_coordinate_mode_from_meta(meta)
    flow = np.stack([
        np.asarray(flow_raw['Ux'], dtype=np.float32),
        np.asarray(flow_raw['Uy'], dtype=np.float32),
        np.asarray(flow_raw['Uz'], dtype=np.float32),
        np.asarray(flow_raw['p'], dtype=np.float32),
    ], axis=-1)
    is_fluid = np.asarray(flow_raw['is_fluid'], dtype=np.float32)
    abl = meta.get('ABL', {}) if isinstance(meta.get('ABL', {}), dict) else {}
    abl_norm = normalize_abl_features(abl)
    size_norm = normalize_domain_size(bounds)
    uref = float(abl.get('Uref', 1.0)) if float(abl.get('Uref', 1.0)) > 0 else 1.0
    lref = float(abl.get('Zref', 50.0)) if float(abl.get('Zref', 50.0)) > 0 else 50.0
    terrain_model = _terrain_model_channels(terrain_raw)
    vi_range, vj_range, vk_max = _compute_valid_ranges(
        x_coords,
        y_coords,
        z_levels,
        bounds,
        terrain_raw,
        vertical_coordinate_mode=vertical_coordinate_mode,
    )
    near_ground_zrel_cap = _compute_near_ground_zrel_cap(
        terrain_raw,
        flow,
        is_fluid,
        z_levels,
        vertical_coordinate_mode=vertical_coordinate_mode,
    )
    return GridBundle(
        name=case_dir.name if kind == 'global' else parent_name,
        category=str(meta.get('category', 'unknown')),
        kind=str(kind),
        case_dir=case_dir,
        roi_name=roi_name,
        parent_name=parent_name,
        bounds=bounds,
        x_coords=x_coords,
        y_coords=y_coords,
        z_levels=z_levels,
        vertical_coordinate_mode=vertical_coordinate_mode,
        terrain_raw={k: np.asarray(v, dtype=np.float32) for k, v in terrain_raw.items()},
        terrain_model=terrain_model.astype(np.float32),
        flow=flow,
        is_fluid=is_fluid,
        nut=nut,
        phi_wall=phi_wall,
        abl={**abl_norm, 'Uref': uref, 'Zref': lref, 'z0': float(abl.get('z0', 0.1))},
        size_norm=size_norm,
        uref=uref,
        lref=lref,
        div_scale=max(float(uref / lref), 1e-12),
        mom_scale=max(float(uref * uref / lref), 1e-12),
        meta=meta,
        valid_i_range=vi_range,
        valid_j_range=vj_range,
        valid_k_max=vk_max,
        near_ground_zrel_cap=float(near_ground_zrel_cap),
    )


def load_split(split_json: Path) -> dict:
    payload = read_json(Path(split_json))
    return {
        'train': [str(v) for v in payload.get('train', []) if str(v).strip()],
        'val': [str(v) for v in payload.get('val', []) if str(v).strip()],
        'test': [str(v) for v in payload.get('test', []) if str(v).strip()],
        'raw': payload,
    }


def _valid_candidates(bundle: GridBundle, ii: np.ndarray, jj: np.ndarray, kk: np.ndarray) -> np.ndarray:
    fluid = bundle.is_fluid[ii, jj, kk] > FD_FLUID_MASK_THRESHOLD
    finite = np.isfinite(bundle.flow[ii, jj, kk, :]).all(axis=1)
    return fluid & finite


def _sample_valid_indices_core(
    bundle: GridBundle,
    n: int,
    rng: np.random.Generator,
    *,
    prefer_near_wall: bool = False,
    k_start: int = 0,
    k_stop: Optional[int] = None,
    max_z_rel: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i_lo, i_hi = bundle.valid_i_range
    j_lo, j_hi = bundle.valid_j_range
    k_hi = bundle.valid_k_max
    ks = max(int(k_start), 0)
    ke = k_hi if k_stop is None else min(int(k_stop), k_hi)
    if ke <= ks:
        ke = min(k_hi, ks + 1)
    ni, nj, nk = i_hi - i_lo, j_hi - j_lo, max(ke - ks, 1)
    total = int(ni * nj * nk)
    need = int(max(1, n))
    keep_i: list[int] = []
    keep_j: list[int] = []
    keep_k: list[int] = []
    tries = 0
    while len(keep_i) < need and tries < 24:
        tries += 1
        c = max(need * (8 if prefer_near_wall else 4), 1024)
        flat = rng.integers(0, total, size=c, endpoint=False)
        ii_loc, rem = np.divmod(flat, nj * nk)
        jj_loc, kk_loc = np.divmod(rem, nk)
        ii = ii_loc + i_lo
        jj = jj_loc + j_lo
        kk = kk_loc + ks
        valid = _valid_candidates(bundle, ii, jj, kk)
        if max_z_rel is not None:
            z_rel = bundle_z_rel_at(bundle, ii, jj, kk)
            valid = valid & np.isfinite(z_rel) & (z_rel >= 0.0) & (z_rel <= float(max_z_rel))
        if prefer_near_wall and bundle.phi_wall is not None:
            dist = np.abs(bundle.phi_wall[ii, jj, kk])
            valid = valid & np.isfinite(dist)
            if np.any(valid):
                w = 1.0 + 3.0 * np.exp(-np.clip(dist[valid], 0.0, None) / 5.0)
                pick = min(int(np.sum(valid)), need - len(keep_i))
                sel_local = rng.choice(np.flatnonzero(valid), size=pick, replace=False, p=w / w.sum())
                keep_i.extend(ii[sel_local].tolist())
                keep_j.extend(jj[sel_local].tolist())
                keep_k.extend(kk[sel_local].tolist())
                continue
        ii = ii[valid]
        jj = jj[valid]
        kk = kk[valid]
        pick = min(len(ii), need - len(keep_i))
        if pick > 0:
            keep_i.extend(ii[:pick].tolist())
            keep_j.extend(jj[:pick].tolist())
            keep_k.extend(kk[:pick].tolist())
    if len(keep_i) == 0:
        raise RuntimeError(f'Unable to sample valid points from {bundle.case_dir}')
    return np.asarray(keep_i[:need], dtype=np.int64), np.asarray(keep_j[:need], dtype=np.int64), np.asarray(keep_k[:need], dtype=np.int64)


def _random_valid_indices(
    bundle: GridBundle,
    n: int,
    rng: np.random.Generator,
    *,
    prefer_near_wall: bool = False,
    near_ground_frac: float = 0.0,
    ground_k_frac: float = GLOBAL_SUPERVISED_GROUND_K_FRAC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    need = int(max(1, n))
    frac = float(np.clip(near_ground_frac, 0.0, 1.0))
    if frac <= 0.0 or bundle.valid_k_max <= 1:
        return _sample_valid_indices_core(bundle, need, rng, prefer_near_wall=prefer_near_wall)

    n_ground = min(need, max(1, int(round(need * frac))))
    n_rest = max(0, need - n_ground)

    try:
        gi, gj, gk = _sample_valid_indices_core(
            bundle,
            n_ground,
            rng,
            prefer_near_wall=False,
            max_z_rel=float(bundle.near_ground_zrel_cap),
        )
    except RuntimeError:
        gi, gj, gk = _sample_valid_indices_core(bundle, n_ground, rng, prefer_near_wall=False)
    if n_rest > 0:
        ri, rj, rk = _sample_valid_indices_core(bundle, n_rest, rng, prefer_near_wall=prefer_near_wall)
        ii = np.concatenate([gi, ri], axis=0)
        jj = np.concatenate([gj, rj], axis=0)
        kk = np.concatenate([gk, rk], axis=0)
    else:
        ii, jj, kk = gi, gj, gk
    return ii[:need], jj[:need], kk[:need]


def _roi_zrel_volume(bundle: GridBundle) -> np.ndarray:
    return _bundle_z_rel_volume(bundle)


def _roi_structure_height_cap(bundle: GridBundle) -> Optional[float]:
    # ROI-only path. If terrain-following ROIs are added later, use physical
    # z_abs helpers here instead of comparing structure z against raw z_levels.
    records = _structure_bounds_records(bundle)
    if not records:
        return None
    caps = []
    for rec in records:
        z_min = float(rec.get('z_min', 0.0))
        z_max = float(rec.get('z_max', z_min))
        height = max(float(rec.get('height', z_max - z_min)), 1e-3)
        caps.append(z_max + float(ROI_TARGET_MAX_ABOVE_STRUCTURE_H) * height)
    if not caps:
        return None
    return float(max(caps))


def _flat_valid_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))


def _sample_disjoint_pool(
    pool: np.ndarray,
    *,
    quota: int,
    rng: np.random.Generator,
    used: set[int],
) -> tuple[list[int], int]:
    if quota <= 0:
        return [], 0
    pool = np.asarray(pool, dtype=np.int64)
    if pool.size == 0:
        return [], int(quota)
    out: list[int] = []

    # Small/medium pools are scanned exactly after shuffling. Large random pools are
    # sampled by repeated draws to avoid materialising millions of candidate ints.
    if pool.size <= max(int(quota) * 4, 200_000):
        order = rng.permutation(pool.size)
        for pos in order.tolist():
            idx = int(pool[int(pos)])
            if idx in used:
                continue
            used.add(idx)
            out.append(idx)
            if len(out) >= int(quota):
                break
    else:
        attempts = 0
        while len(out) < int(quota) and attempts < 16:
            attempts += 1
            need = int(quota) - len(out)
            draw = min(int(pool.size), max(need * 4, 4096))
            pos = rng.choice(pool.size, size=draw, replace=False)
            for raw in pool[pos].tolist():
                idx = int(raw)
                if idx in used:
                    continue
                used.add(idx)
                out.append(idx)
                if len(out) >= int(quota):
                    break
    return out, int(max(0, quota - len(out)))


def _bounded_repeat_topup(
    selected_unique: list[int],
    *,
    quota_shortfall: int,
    rng: np.random.Generator,
    max_repeat: int,
) -> list[int]:
    if quota_shortfall <= 0 or max_repeat <= 1 or not selected_unique:
        return []
    repeats = int(max_repeat) - 1
    candidates = np.repeat(np.asarray(selected_unique, dtype=np.int64), repeats)
    take = min(int(quota_shortfall), int(candidates.size))
    if take <= 0:
        return []
    if candidates.size > take:
        pos = rng.choice(candidates.size, size=take, replace=False)
        return [int(v) for v in candidates[pos].tolist()]
    return [int(v) for v in candidates.tolist()]


def _quota_counts(n: int, fracs: list[float]) -> list[int]:
    need = int(max(1, n))
    vals = np.asarray([max(float(v), 0.0) for v in fracs], dtype=np.float64)
    total = float(vals.sum())
    if total <= 0.0:
        out = [0 for _ in fracs]
        out[-1] = need
        return out
    vals = vals / total
    raw = vals * need
    counts = np.floor(raw).astype(np.int64)
    rem = int(need - int(counts.sum()))
    if rem > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:rem]:
            counts[int(idx)] += 1
    return [int(v) for v in counts.tolist()]


def _random_pool_excluding(valid_flat: np.ndarray, *, used: set[int], rng: np.random.Generator) -> list[int]:
    candidates = [int(v) for v in valid_flat.tolist() if int(v) not in used]
    if not candidates:
        return []
    order = rng.permutation(len(candidates))
    return [candidates[int(i)] for i in order.tolist()]


def sample_targeted_roi_supervised_batch(
    bundle: GridBundle,
    *,
    x_scaler: MinMaxScaler,
    y_scaler: StandardScaler,
    n_points: int,
    rng: np.random.Generator,
    parent_global: Optional[GridBundle] = None,
) -> PointBatch:
    if bundle.kind != 'roi' or bundle.phi_wall is None:
        return sample_supervised_batch(
            bundle,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            n_points=n_points,
            rng=rng,
            parent_global=parent_global,
        )

    cache_key = (
        float(ROI_TARGET_VERY_NEAR_WALL_DMAX),
        float(ROI_TARGET_NEAR_WALL_DMAX),
        float(ROI_TARGET_NEAR_WALL_BACKFILL_DMAX),
        float(ROI_TARGET_WAKE_MIN_CONTEXT),
        float(ROI_TARGET_WAKE_ZREL_MAX),
        float(ROI_TARGET_LOW_SPEED_RATIO_MAX),
        float(ROI_TARGET_LOW_SPEED_ZREL_MAX),
        float(ROI_TARGET_HIGH_SPEED_RATIO_MIN),
        float(ROI_TARGET_HIGH_SPEED_ZREL_MAX),
        float(ROI_TARGET_MAX_ABOVE_STRUCTURE_H),
    )
    cache = bundle.targeted_roi_sample_cache
    if not isinstance(cache, dict) or cache.get('key') != cache_key:
        valid = (bundle.is_fluid > FD_FLUID_MASK_THRESHOLD) & np.isfinite(bundle.flow).all(axis=-1)
        i_lo, i_hi = bundle.valid_i_range
        j_lo, j_hi = bundle.valid_j_range
        range_mask = np.zeros_like(valid, dtype=bool)
        range_mask[i_lo:i_hi, j_lo:j_hi, :bundle.valid_k_max] = True
        valid = valid & range_mask
        if not bool(np.any(valid)):
            return sample_supervised_batch(
                bundle,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                n_points=n_points,
                rng=rng,
                parent_global=parent_global,
            )

        z_rel = _roi_zrel_volume(bundle)
        z_abs = _bundle_z_abs_volume(bundle)
        speed = np.linalg.norm(bundle.flow[..., :3], axis=-1)
        speed_ratio = speed / max(float(bundle.uref), 1e-6)
        phi_abs = np.abs(bundle.phi_wall)
        finite_common = valid & np.isfinite(phi_abs) & np.isfinite(z_rel) & np.isfinite(speed_ratio)

        z_cap = _roi_structure_height_cap(bundle)
        below_struct_cap = np.ones_like(valid, dtype=bool)
        if z_cap is not None and float(ROI_TARGET_MAX_ABOVE_STRUCTURE_H) > 0.0:
            below_struct_cap = z_abs <= float(z_cap)

        very_near_mask = finite_common & (phi_abs <= float(ROI_TARGET_VERY_NEAR_WALL_DMAX)) & below_struct_cap
        near_mask = finite_common & (phi_abs <= float(ROI_TARGET_NEAR_WALL_DMAX)) & below_struct_cap
        near_backfill_mask = finite_common & (phi_abs <= float(ROI_TARGET_NEAR_WALL_BACKFILL_DMAX)) & below_struct_cap

        structure_model = _structure_model_channels_for_mode(bundle, 'context_v2')
        if structure_model.shape[0] >= 7:
            wake2d = np.maximum(structure_model[5], structure_model[6]).astype(np.float32)
            wake3d = wake2d.T[:, :, None]
        else:
            wake2d = np.zeros(bundle.terrain_raw['elevation'].shape, dtype=np.float32)
            wake3d = np.zeros(valid.shape, dtype=np.float32)
        geom_wake_mask = (
            finite_common
            & below_struct_cap
            & (wake3d >= float(ROI_TARGET_WAKE_MIN_CONTEXT))
            & (z_rel >= 0.0)
            & (z_rel <= float(ROI_TARGET_WAKE_ZREL_MAX))
        )
        low_speed_mask = (
            finite_common
            & below_struct_cap
            & (speed_ratio <= float(ROI_TARGET_LOW_SPEED_RATIO_MAX))
            & (z_rel >= 0.0)
            & (z_rel <= float(ROI_TARGET_LOW_SPEED_ZREL_MAX))
        )
        high_speed_mask = (
            finite_common
            & below_struct_cap
            & (speed_ratio >= float(ROI_TARGET_HIGH_SPEED_RATIO_MIN))
            & (z_rel >= 0.0)
            & (z_rel <= float(ROI_TARGET_HIGH_SPEED_ZREL_MAX))
        )

        cache = {
            'key': cache_key,
            'shape': valid.shape,
            'wake2d_xy': wake2d.T.astype(np.float32, copy=False),
            'pools': {
                'very_near_wall': _flat_valid_indices(very_near_mask),
                'near_wall': _flat_valid_indices(near_mask),
                'near_wall_backfill': _flat_valid_indices(near_backfill_mask),
                'geom_wake': _flat_valid_indices(geom_wake_mask),
                'low_speed': _flat_valid_indices(low_speed_mask),
                'high_speed': _flat_valid_indices(high_speed_mask),
                'random': _flat_valid_indices(valid),
            },
        }
        bundle.targeted_roi_sample_cache = cache

    pools = cache['pools']
    valid_shape = tuple(cache['shape'])

    labels = ['very_near_wall', 'near_wall', 'geom_wake', 'low_speed', 'high_speed', 'random']
    quotas = _quota_counts(
        int(n_points),
        [
            float(ROI_TARGET_VERY_NEAR_WALL_FRAC),
            float(ROI_TARGET_NEAR_WALL_FRAC),
            float(ROI_TARGET_GEOM_WAKE_FRAC),
            float(ROI_TARGET_LOW_SPEED_FRAC),
            float(ROI_TARGET_HIGH_SPEED_FRAC),
            float(ROI_TARGET_RANDOM_FRAC),
        ],
    )
    used: set[int] = set()
    picked_by_label: dict[str, list[int]] = {label: [] for label in labels}
    underfill: dict[str, int] = {label: 0 for label in labels}

    picked, short = _sample_disjoint_pool(pools['very_near_wall'], quota=quotas[0], rng=rng, used=used)
    picked_by_label['very_near_wall'].extend(picked)
    if short > 0:
        repeated = _bounded_repeat_topup(
            picked,
            quota_shortfall=short,
            rng=rng,
            max_repeat=int(ROI_TARGET_VERY_NEAR_WALL_MAX_REPEAT),
        )
        picked_by_label['very_near_wall'].extend(repeated)
        short = max(0, short - len(repeated))
    underfill['very_near_wall'] += short

    near_quota = quotas[1] + short
    picked, short = _sample_disjoint_pool(pools['near_wall'], quota=near_quota, rng=rng, used=used)
    picked_by_label['near_wall'].extend(picked)
    if short > 0:
        extra, short = _sample_disjoint_pool(pools['near_wall_backfill'], quota=short, rng=rng, used=used)
        picked_by_label['near_wall'].extend(extra)
    underfill['near_wall'] += short

    wake_quota = quotas[2] + short
    picked, short = _sample_disjoint_pool(pools['geom_wake'], quota=wake_quota, rng=rng, used=used)
    picked_by_label['geom_wake'].extend(picked)
    underfill['geom_wake'] += short

    low_quota = quotas[3] + short
    picked, short = _sample_disjoint_pool(pools['low_speed'], quota=low_quota, rng=rng, used=used)
    picked_by_label['low_speed'].extend(picked)
    underfill['low_speed'] += short

    high_quota = quotas[4] + short
    picked, short = _sample_disjoint_pool(pools['high_speed'], quota=high_quota, rng=rng, used=used)
    picked_by_label['high_speed'].extend(picked)
    underfill['high_speed'] += short

    random_quota = quotas[5] + short
    picked, short = _sample_disjoint_pool(pools['random'], quota=random_quota, rng=rng, used=used)
    picked_by_label['random'].extend(picked)
    underfill['random'] += short

    if short > 0 and used:
        # Last-resort replacement only after every unique valid cell has been exhausted.
        all_used = np.asarray(list(used), dtype=np.int64)
        extra = rng.choice(all_used, size=int(short), replace=True)
        picked_by_label['random'].extend([int(v) for v in extra.tolist()])

    flat_selected: list[int] = []
    for label in labels:
        flat_selected.extend(picked_by_label[label])
    flat_selected = flat_selected[: int(max(1, n_points))]
    if len(flat_selected) == 0:
        return sample_supervised_batch(
            bundle,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            n_points=n_points,
            rng=rng,
            parent_global=parent_global,
        )

    ii, jj, kk = np.unravel_index(np.asarray(flat_selected, dtype=np.int64), valid_shape)
    x_rows = _feature_rows(bundle, ii, jj, kk, include_phi_wall=True)
    y_rows = _target_rows(bundle, ii, jj, kk)
    xy_local = _xy_norm_from_bounds(bundle.x_coords[ii], bundle.y_coords[jj], bundle.bounds)
    xy_global = None
    if parent_global is not None:
        xy_global = _xy_norm_from_bounds(bundle.x_coords[ii], bundle.y_coords[jj], parent_global.bounds)

    denom = max(float(len(flat_selected)), 1.0)
    wake2d_xy = cache['wake2d_xy']
    selected_phi = np.abs(bundle.phi_wall[ii, jj, kk])
    selected_zrel = bundle_z_rel_at(bundle, ii, jj, kk)
    selected_speed_ratio = np.linalg.norm(bundle.flow[ii, jj, kk, :3], axis=-1) / max(float(bundle.uref), 1e-6)
    selected_wake = wake2d_xy[ii, jj]
    stats: dict[str, float] = {
        'targeted_total': float(len(flat_selected)),
        'targeted_very_near_wall_frac': float(np.mean(selected_phi <= float(ROI_TARGET_VERY_NEAR_WALL_DMAX))),
        'targeted_near_wall_frac': float(np.mean(selected_phi <= float(ROI_TARGET_NEAR_WALL_DMAX))),
        'targeted_near_wall_backfill_frac': float(np.mean(selected_phi <= float(ROI_TARGET_NEAR_WALL_BACKFILL_DMAX))),
        'targeted_geom_wake_frac': float(np.mean((selected_wake >= float(ROI_TARGET_WAKE_MIN_CONTEXT)) & (selected_zrel <= float(ROI_TARGET_WAKE_ZREL_MAX)))),
        'targeted_low_speed_frac': float(np.mean((selected_speed_ratio <= float(ROI_TARGET_LOW_SPEED_RATIO_MAX)) & (selected_zrel <= float(ROI_TARGET_LOW_SPEED_ZREL_MAX)))),
        'targeted_high_speed_frac': float(np.mean((selected_speed_ratio >= float(ROI_TARGET_HIGH_SPEED_RATIO_MIN)) & (selected_zrel <= float(ROI_TARGET_HIGH_SPEED_ZREL_MAX)))),
    }
    for label in labels:
        stats[f'targeted_delivered_{label}_frac'] = float(len(picked_by_label[label]) / denom)
        stats[f'targeted_underfill_{label}'] = float(underfill[label])
        stats[f'targeted_pool_{label}'] = float(len(pools[label]))
    stats['targeted_pool_near_wall_backfill'] = float(len(pools['near_wall_backfill']))

    return PointBatch(
        x_scaled=_scale_inputs(x_rows, scaler=x_scaler),
        y_scaled=_scale_outputs(y_rows, scaler=y_scaler),
        xy_local=torch.as_tensor(xy_local, dtype=torch.float32),
        xy_global=None if xy_global is None else torch.as_tensor(xy_global, dtype=torch.float32),
        sample_stats=stats,
    )


def _xy_norm_from_bounds(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float, float, float, float, float]) -> np.ndarray:
    x0, x1, y0, y1, _, _ = bounds
    xn = 2.0 * ((x - x0) / max(x1 - x0, 1e-6)) - 1.0
    yn = 2.0 * ((y - y0) / max(y1 - y0, 1e-6)) - 1.0
    return np.stack([np.clip(xn, -1.0, 1.0), np.clip(yn, -1.0, 1.0)], axis=1).astype(np.float32)


def _feature_rows(bundle: GridBundle, ii: np.ndarray, jj: np.ndarray, kk: np.ndarray, *, include_phi_wall: bool) -> np.ndarray:
    x = bundle.x_coords[ii]
    y = bundle.y_coords[jj]
    z = bundle_z_abs_at(bundle, ii, jj, kk)
    elev = bundle.terrain_raw['elevation'][jj, ii]
    slope = bundle.terrain_raw['slope'][jj, ii]
    aspect = bundle.terrain_raw['aspect'][jj, ii]
    asin, acos = aspect_to_sin_cos(aspect)
    z_rel = bundle_z_rel_at(bundle, ii, jj, kk)
    phi_ground = np.tanh(z_rel / max(PHI_GROUND_H, 1e-6)).astype(np.float32)
    terrain_elev = ((elev - np.mean(bundle.terrain_raw['elevation'], dtype=np.float64)) / 1000.0).astype(np.float32)
    terrain_slope = np.clip(slope / 90.0, 0.0, 1.0).astype(np.float32)

    rows = [x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), z_rel.astype(np.float32), phi_ground]
    if include_phi_wall:
        if bundle.phi_wall is None:
            pw = np.ones_like(phi_ground, dtype=np.float32)
        else:
            pw = np.tanh(bundle.phi_wall[ii, jj, kk] / max(PHI_WALL_H, 1e-6)).astype(np.float32)
        rows.append(pw)
    rows.extend([
        terrain_elev,
        terrain_slope,
        asin.astype(np.float32),
        acos.astype(np.float32),
        np.full_like(phi_ground, float(bundle.abl['Uref_norm']), dtype=np.float32),
        np.full_like(phi_ground, float(bundle.abl['Zref_norm']), dtype=np.float32),
        np.full_like(phi_ground, float(bundle.abl['log10_z0_norm']), dtype=np.float32),
        np.full_like(phi_ground, float(bundle.abl['flowDir_x']), dtype=np.float32),
        np.full_like(phi_ground, float(bundle.abl['flowDir_y']), dtype=np.float32),
        np.full_like(phi_ground, float(bundle.abl['flowDir_z']), dtype=np.float32),
        np.full_like(phi_ground, float(bundle.size_norm['Lx_norm']), dtype=np.float32),
        np.full_like(phi_ground, float(bundle.size_norm['Ly_norm']), dtype=np.float32),
        np.full_like(phi_ground, float(bundle.size_norm['Lz_norm']), dtype=np.float32),
    ])
    out = np.stack(rows, axis=1).astype(np.float32)
    expected = len(ROI_INPUT_COLS) if include_phi_wall else len(GLOBAL_INPUT_COLS)
    assert out.shape[1] == expected, f'Feature shape mismatch for {bundle.case_dir}: got {out.shape[1]}, expected {expected}'
    return out


def _target_rows(bundle: GridBundle, ii: np.ndarray, jj: np.ndarray, kk: np.ndarray) -> np.ndarray:
    return bundle.flow[ii, jj, kk, :].astype(np.float32)


def _scale_inputs(x_rows: np.ndarray, *, scaler: MinMaxScaler) -> torch.Tensor:
    return torch.as_tensor(scaler.transform(x_rows), dtype=torch.float32)


def _scale_outputs(y_rows: np.ndarray, *, scaler: StandardScaler) -> torch.Tensor:
    return torch.as_tensor(scaler.transform(y_rows), dtype=torch.float32)


def sample_supervised_batch(
    bundle: GridBundle,
    *,
    x_scaler: MinMaxScaler,
    y_scaler: StandardScaler,
    n_points: int,
    rng: np.random.Generator,
    parent_global: Optional[GridBundle] = None,
) -> PointBatch:
    include_phi_wall = bundle.kind == 'roi'
    near_ground_frac = float(GLOBAL_SUPERVISED_NEAR_GROUND_FRAC) if bundle.kind == 'global' else 0.0
    ii, jj, kk = _random_valid_indices(bundle, n_points, rng, prefer_near_wall=include_phi_wall, near_ground_frac=near_ground_frac)
    x_rows = _feature_rows(bundle, ii, jj, kk, include_phi_wall=include_phi_wall)
    y_rows = _target_rows(bundle, ii, jj, kk)
    xy_local = _xy_norm_from_bounds(bundle.x_coords[ii], bundle.y_coords[jj], bundle.bounds)
    xy_global = None
    if bundle.kind == 'roi' and parent_global is not None:
        xy_global = _xy_norm_from_bounds(bundle.x_coords[ii], bundle.y_coords[jj], parent_global.bounds)
    return PointBatch(
        x_scaled=_scale_inputs(x_rows, scaler=x_scaler),
        y_scaled=_scale_outputs(y_rows, scaler=y_scaler),
        xy_local=torch.as_tensor(xy_local, dtype=torch.float32),
        xy_global=None if xy_global is None else torch.as_tensor(xy_global, dtype=torch.float32),
    )


def _choose_patch_origin(
    bundle: GridBundle,
    patch_shape: tuple[int, int, int],
    rng: np.random.Generator,
    *,
    near_ground_prob: float,
    near_wall_prob: float = 0.0,
    high_speed_prob: float = 0.0,
) -> tuple[int, int, int, tuple[int, int, int]]:
    i_lo, i_hi = bundle.valid_i_range
    j_lo, j_hi = bundle.valid_j_range
    k_hi = bundle.valid_k_max
    ni, nj, nk = i_hi - i_lo, j_hi - j_lo, k_hi
    px = min(int(patch_shape[0]), ni)
    py = min(int(patch_shape[1]), nj)
    pz = min(int(patch_shape[2]), nk)
    prefer_high_speed = bool(bundle.kind == 'roi' and float(high_speed_prob) > 0.0 and rng.random() < float(high_speed_prob))
    best = (0, 0, 0)
    best_score = -1.0
    for _ in range(24):
        i0 = i_lo + int(rng.integers(0, max(ni - px + 1, 1)))
        j0 = j_lo + int(rng.integers(0, max(nj - py + 1, 1)))
        if rng.random() < float(near_ground_prob):
            kmax = max(nk - pz + 1, 1)
            kcap = max(1, int(0.35 * kmax))
            k0 = int(rng.integers(0, kcap))
        else:
            k0 = int(rng.integers(0, max(nk - pz + 1, 1)))
        mask = bundle.is_fluid[i0:i0 + px, j0:j0 + py, k0:k0 + pz] > FD_FLUID_MASK_THRESHOLD
        score = float(mask.mean())
        if prefer_high_speed:
            flow = bundle.flow[i0:i0 + px, j0:j0 + py, k0:k0 + pz, :3]
            speed_ratio = np.linalg.norm(flow, axis=-1) / max(float(bundle.uref), 1e-6)
            z_rel = _bundle_z_rel_volume(bundle)[i0:i0 + px, j0:j0 + py, k0:k0 + pz]
            high = (
                mask
                & np.isfinite(speed_ratio)
                & np.isfinite(z_rel)
                & (speed_ratio >= float(ROI_PATCH_HIGH_SPEED_RATIO_MIN))
                & (z_rel >= 0.0)
                & (z_rel <= float(ROI_PATCH_HIGH_SPEED_ZREL_MAX))
            )
            score += 1.5 * float(high.mean())
        if near_wall_prob > 0.0 and bundle.phi_wall is not None and rng.random() < near_wall_prob:
            dw = np.abs(bundle.phi_wall[i0:i0 + px, j0:j0 + py, k0:k0 + pz])
            near = np.isfinite(dw) & (dw <= ROI_NEAR_WALL_DMAX)
            score += 0.5 * float(near.mean())
        if score > best_score:
            best_score = score
            best = (i0, j0, k0)
    return best[0], best[1], best[2], (px, py, pz)


def sample_patch_batch(
    bundle: GridBundle,
    *,
    x_scaler: MinMaxScaler,
    y_scaler: StandardScaler,
    patch_shape: tuple[int, int, int],
    rng: np.random.Generator,
    near_ground_prob: float,
    parent_global: Optional[GridBundle] = None,
    include_grid_unet_context: bool = False,
) -> PatchBatch:
    include_phi_wall = bundle.kind == 'roi'
    near_wall_prob = 0.6 if include_phi_wall else 0.0
    high_speed_prob = float(ROI_PATCH_HIGH_SPEED_PROB) if include_phi_wall else 0.0
    i0, j0, k0, (px, py, pz) = _choose_patch_origin(
        bundle,
        patch_shape,
        rng,
        near_ground_prob=near_ground_prob,
        near_wall_prob=near_wall_prob,
        high_speed_prob=high_speed_prob,
    )
    return extract_patch_batch(
        bundle,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        i0=i0,
        j0=j0,
        k0=k0,
        patch_shape=(px, py, pz),
        parent_global=parent_global,
        include_grid_unet_context=include_grid_unet_context,
    )


def extract_patch_batch(
    bundle: GridBundle,
    *,
    x_scaler: MinMaxScaler,
    y_scaler: StandardScaler,
    i0: int,
    j0: int,
    k0: int,
    patch_shape: tuple[int, int, int],
    parent_global: Optional[GridBundle] = None,
    include_grid_unet_context: bool = False,
) -> PatchBatch:
    include_phi_wall = bundle.kind == 'roi'
    px = int(max(1, patch_shape[0]))
    py = int(max(1, patch_shape[1]))
    pz = int(max(1, patch_shape[2]))
    ii, jj, kk = np.meshgrid(
        np.arange(i0, i0 + px, dtype=np.int64),
        np.arange(j0, j0 + py, dtype=np.int64),
        np.arange(k0, k0 + pz, dtype=np.int64),
        indexing='ij',
    )
    ii_f = ii.reshape(-1)
    jj_f = jj.reshape(-1)
    kk_f = kk.reshape(-1)
    x_rows = _feature_rows(bundle, ii_f, jj_f, kk_f, include_phi_wall=include_phi_wall)
    y_rows = _target_rows(bundle, ii_f, jj_f, kk_f)
    xy_local = _xy_norm_from_bounds(bundle.x_coords[ii_f], bundle.y_coords[jj_f], bundle.bounds)
    xy_global = None
    if bundle.kind == 'roi' and parent_global is not None:
        xy_global = _xy_norm_from_bounds(bundle.x_coords[ii_f], bundle.y_coords[jj_f], parent_global.bounds)
    try:
        source_name = str(bundle.case_dir.relative_to(DATA_CFD_ROOT))
    except ValueError:
        source_name = str(bundle.case_dir)
    x_scaled = _scale_inputs(x_rows, scaler=x_scaler)
    y_scaled = _scale_outputs(y_rows, scaler=y_scaler)
    x_volume_scaled = x_scaled.view(px, py, pz, -1).permute(3, 0, 1, 2).contiguous()
    if include_grid_unet_context and bundle.kind == 'roi':
        structure_mode = resolve_structure_channel_mode(GRID_UNET_ROI_STRUCTURE_MODE)
        if structure_mode != 'none':
            structure_model = _structure_model_channels_for_mode(bundle, structure_mode)
            if structure_model.shape[0] > 0:
                struct_patch = np.transpose(structure_model[:, j0:j0 + py, i0:i0 + px], (0, 2, 1))
                struct_volume = np.repeat(struct_patch[:, :, :, None], pz, axis=3)
                x_volume_scaled = torch.cat(
                    [x_volume_scaled, torch.as_tensor(struct_volume, dtype=torch.float32)],
                    dim=0,
                )
    return PatchBatch(
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        x_volume_scaled=x_volume_scaled,
        y_volume_scaled=y_scaled.view(px, py, pz, -1).permute(3, 0, 1, 2).contiguous(),
        xy_local=torch.as_tensor(xy_local, dtype=torch.float32),
        xy_global=None if xy_global is None else torch.as_tensor(xy_global, dtype=torch.float32),
        mask=torch.as_tensor(bundle.is_fluid[i0:i0 + px, j0:j0 + py, k0:k0 + pz], dtype=torch.float32),
        nut=None if bundle.nut is None else torch.as_tensor(bundle.nut[i0:i0 + px, j0:j0 + py, k0:k0 + pz], dtype=torch.float32),
        x_coords=torch.as_tensor(bundle.x_coords[i0:i0 + px], dtype=torch.float32),
        y_coords=torch.as_tensor(bundle.y_coords[j0:j0 + py], dtype=torch.float32),
        z_levels=torch.as_tensor(bundle.z_levels[k0:k0 + pz], dtype=torch.float32),
        shape=(px, py, pz),
        origin=(int(i0), int(j0), int(k0)),
        div_scale=float(bundle.div_scale),
        mom_scale=float(bundle.mom_scale),
        source_name=source_name,
    )


def _sample_boundary_indices(bundle: GridBundle, face: str, rng: np.random.Generator, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = bundle.flow.shape[:3]
    if face == 'inlet':
        ii, jj, kk = np.meshgrid(np.array([0]), np.arange(ny), np.arange(nz), indexing='ij')
        normal = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    elif face == 'outlet':
        ii, jj, kk = np.meshgrid(np.array([nx - 1]), np.arange(ny), np.arange(nz), indexing='ij')
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif face == 'side':
        ii1, jj1, kk1 = np.meshgrid(np.arange(nx), np.array([0]), np.arange(nz), indexing='ij')
        ii2, jj2, kk2 = np.meshgrid(np.arange(nx), np.array([ny - 1]), np.arange(nz), indexing='ij')
        ii = np.concatenate([ii1.reshape(-1), ii2.reshape(-1)])
        jj = np.concatenate([jj1.reshape(-1), jj2.reshape(-1)])
        kk = np.concatenate([kk1.reshape(-1), kk2.reshape(-1)])
        normals = np.vstack([
            np.tile(np.array([0.0, -1.0, 0.0], dtype=np.float32), (ii1.size, 1)),
            np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (ii2.size, 1)),
        ])
        valid = _valid_candidates(bundle, ii, jj, kk)
        ii, jj, kk, normals = ii[valid], jj[valid], kk[valid], normals[valid]
        if len(ii) > max_points:
            idx = rng.choice(len(ii), size=max_points, replace=False)
            ii, jj, kk, normals = ii[idx], jj[idx], kk[idx], normals[idx]
        return ii, jj, kk, normals
    elif face == 'top':
        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.array([nz - 1]), indexing='ij')
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        raise ValueError(face)
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)
    kk = kk.reshape(-1)
    valid = _valid_candidates(bundle, ii, jj, kk)
    ii, jj, kk = ii[valid], jj[valid], kk[valid]
    if len(ii) > max_points:
        idx = rng.choice(len(ii), size=max_points, replace=False)
        ii, jj, kk = ii[idx], jj[idx], kk[idx]
    normals = np.tile(normal.reshape(1, 3), (len(ii), 1))
    return ii, jj, kk, normals


def _inlet_targets(bundle: GridBundle, ii: np.ndarray, jj: np.ndarray, kk: np.ndarray) -> np.ndarray:
    z_rel = bundle_z_rel_at(bundle, ii, jj, kk)
    uref = float(bundle.meta.get('ABL', {}).get('Uref', bundle.abl['Uref']))
    zref = float(bundle.meta.get('ABL', {}).get('Zref', bundle.abl['Zref']))
    z0 = float(bundle.meta.get('ABL', {}).get('z0', bundle.abl['z0']))
    den = np.log((zref + z0) / z0)
    den = den if np.isfinite(den) and abs(den) > 1e-12 else 1.0
    speed = uref * np.log((z_rel + z0) / z0) / den
    flow_dir = np.array([
        float(bundle.abl['flowDir_x']),
        float(bundle.abl['flowDir_y']),
        float(bundle.abl['flowDir_z']),
    ], dtype=np.float32)
    return (speed[:, None].astype(np.float32) * flow_dir[None, :]).astype(np.float32)


def prepare_global_boundary_batches(bundle: GridBundle, *, x_scaler: MinMaxScaler, rng: np.random.Generator) -> dict[str, Optional[BoundaryBatch]]:
    out: dict[str, Optional[BoundaryBatch]] = {}
    max_points_by_face = {
        'inlet': int(max(1, BC_POINTS_INLET)),
        'outlet': int(max(1, BC_POINTS_OUTLET)),
        'side': int(max(1, BC_POINTS_SIDE)),
        'top': int(max(1, BC_POINTS_TOP)),
    }
    for face in ['inlet', 'outlet', 'side', 'top']:
        ii, jj, kk, normals = _sample_boundary_indices(bundle, face, rng, max_points_by_face[face])
        if len(ii) == 0:
            out[face] = None
            continue
        x_rows = _feature_rows(bundle, ii, jj, kk, include_phi_wall=False)
        xy_local = _xy_norm_from_bounds(bundle.x_coords[ii], bundle.y_coords[jj], bundle.bounds)
        u_target = _inlet_targets(bundle, ii, jj, kk) if face == 'inlet' else None
        p_target = np.zeros((len(ii),), dtype=np.float32) if face == 'outlet' else None
        out[face] = BoundaryBatch(
            x_scaled=_scale_inputs(x_rows, scaler=x_scaler),
            xy_local=torch.as_tensor(xy_local, dtype=torch.float32),
            normals=torch.as_tensor(normals, dtype=torch.float32),
            u_target=None if u_target is None else torch.as_tensor(u_target, dtype=torch.float32),
            p_target=None if p_target is None else torch.as_tensor(p_target, dtype=torch.float32),
            u_scale=float(bundle.uref),
            p_scale=float(bundle.uref * bundle.uref),
        )
    return out


def fit_scalers(repo: CaseRepository, train_names: Iterable[str], *, sample_points_per_grid: int = SCALER_POINTS_PER_GRID, seed: int = 42) -> ScalerBundle:
    rng = np.random.default_rng(seed)
    global_rows = []
    roi_rows = []
    y_rows = []
    for name in train_names:
        g = repo.load_global(name)
        ii, jj, kk = _random_valid_indices(g, sample_points_per_grid, rng, prefer_near_wall=False)
        global_rows.append(_feature_rows(g, ii, jj, kk, include_phi_wall=False))
        y_rows.append(_target_rows(g, ii, jj, kk))
        for roi_name in repo.roi_names(name):
            r = repo.load_roi(name, roi_name)
            ii, jj, kk = _random_valid_indices(r, sample_points_per_grid, rng, prefer_near_wall=True)
            roi_rows.append(_feature_rows(r, ii, jj, kk, include_phi_wall=True))
            y_rows.append(_target_rows(r, ii, jj, kk))
    if not global_rows:
        raise RuntimeError('No global training rows available to fit scalers.')
    if not roi_rows:
        raise RuntimeError('No ROI training rows available to fit scalers.')
    xg = np.concatenate(global_rows, axis=0)
    xr = np.concatenate(roi_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)

    x_scaler_global = MinMaxScaler()
    x_scaler_roi = MinMaxScaler()
    y_scaler = StandardScaler()
    x_scaler_global.fit(xg)
    x_scaler_roi.fit(xr)
    y_scaler.fit(y)
    return ScalerBundle(x_scaler_global=x_scaler_global, x_scaler_roi=x_scaler_roi, y_scaler=y_scaler)


def fit_scalers_global_only(repo: CaseRepository, train_names: Iterable[str], *, sample_points_per_grid: int = SCALER_POINTS_PER_GRID, seed: int = 42) -> ScalerBundle:
    rng = np.random.default_rng(seed)
    global_rows = []
    y_rows = []
    for name in train_names:
        g = repo.load_global(name)
        ii, jj, kk = _random_valid_indices(g, sample_points_per_grid, rng, prefer_near_wall=False)
        global_rows.append(_feature_rows(g, ii, jj, kk, include_phi_wall=False))
        y_rows.append(_target_rows(g, ii, jj, kk))
    if not global_rows:
        raise RuntimeError('No global training rows available to fit global-only scalers.')
    xg = np.concatenate(global_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)
    x_scaler_global = MinMaxScaler()
    y_scaler = StandardScaler()
    x_scaler_global.fit(xg)
    y_scaler.fit(y)
    return ScalerBundle(x_scaler_global=x_scaler_global, x_scaler_roi=None, y_scaler=y_scaler)


def fit_scalers_roi_only(repo: CaseRepository, train_names: Iterable[str], *, sample_points_per_grid: int = SCALER_POINTS_PER_GRID, seed: int = 42) -> ScalerBundle:
    rng = np.random.default_rng(seed)
    roi_rows = []
    y_rows = []
    for name in train_names:
        for roi_name in repo.roi_names(name):
            r = repo.load_roi(name, roi_name)
            ii, jj, kk = _random_valid_indices(r, sample_points_per_grid, rng, prefer_near_wall=True)
            roi_rows.append(_feature_rows(r, ii, jj, kk, include_phi_wall=True))
            y_rows.append(_target_rows(r, ii, jj, kk))
    if not roi_rows:
        raise RuntimeError('No ROI training rows available to fit ROI-only scalers.')
    xr = np.concatenate(roi_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)
    x_scaler_roi = MinMaxScaler()
    y_scaler = StandardScaler()
    x_scaler_roi.fit(xr)
    y_scaler.fit(y)
    return ScalerBundle(x_scaler_global=None, x_scaler_roi=x_scaler_roi, y_scaler=y_scaler)


def fit_scalers_roi_refs(repo: CaseRepository, roi_refs: Iterable[tuple[str, str]], *, sample_points_per_grid: int = SCALER_POINTS_PER_GRID, seed: int = 42) -> ScalerBundle:
    rng = np.random.default_rng(seed)
    roi_rows = []
    y_rows = []
    for name, roi_name in roi_refs:
        r = repo.load_roi(str(name), str(roi_name))
        ii, jj, kk = _random_valid_indices(r, sample_points_per_grid, rng, prefer_near_wall=True)
        roi_rows.append(_feature_rows(r, ii, jj, kk, include_phi_wall=True))
        y_rows.append(_target_rows(r, ii, jj, kk))
    if not roi_rows:
        raise RuntimeError('No ROI training rows available to fit ROI-only scalers.')
    xr = np.concatenate(roi_rows, axis=0)
    y = np.concatenate(y_rows, axis=0)
    x_scaler_roi = MinMaxScaler()
    y_scaler = StandardScaler()
    x_scaler_roi.fit(xr)
    y_scaler.fit(y)
    return ScalerBundle(x_scaler_global=None, x_scaler_roi=x_scaler_roi, y_scaler=y_scaler)


def terrain_tensor(bundle: GridBundle) -> torch.Tensor:
    return torch.as_tensor(bundle.terrain_model[None, ...], dtype=torch.float32)


def structure_tensor(bundle: GridBundle) -> torch.Tensor:
    if bundle.structure_model is None:
        bundle.structure_model = _structure_model_channels(bundle)
    return torch.as_tensor(bundle.structure_model[None, ...], dtype=torch.float32)


def iter_fullgrid_predictions(
    bundle: GridBundle,
    *,
    x_scaler: MinMaxScaler,
    chunk_size: int,
    include_phi_wall: bool,
):
    valid = (bundle.is_fluid > FD_FLUID_MASK_THRESHOLD) & np.isfinite(bundle.flow).all(axis=-1)
    # Apply lateral-trim and z-cap masks
    range_mask = np.zeros_like(valid, dtype=bool)
    i_lo, i_hi = bundle.valid_i_range
    j_lo, j_hi = bundle.valid_j_range
    range_mask[i_lo:i_hi, j_lo:j_hi, :bundle.valid_k_max] = True
    valid = valid & range_mask
    flat = np.flatnonzero(valid.reshape(-1))
    nx, ny, nz = bundle.flow.shape[:3]
    for s in range(0, len(flat), int(max(1, chunk_size))):
        idx = flat[s:s + int(max(1, chunk_size))]
        ii, rem = np.divmod(idx, ny * nz)
        jj, kk = np.divmod(rem, nz)
        x_rows = _feature_rows(bundle, ii, jj, kk, include_phi_wall=include_phi_wall)
        y_rows = _target_rows(bundle, ii, jj, kk)
        xy_local = _xy_norm_from_bounds(bundle.x_coords[ii], bundle.y_coords[jj], bundle.bounds)
        # Raw geometric coordinates (in metres) for downstream subset masks
        # (near-ground = z_rel <= ..., near-wall = |phi_wall| <= ...).
        # x_rows columns: x, y, z, z_rel, phi_ground, [phi_wall_norm], ... — z_rel is column 3.
        z_rel_raw = x_rows[:, 3].astype(np.float32, copy=False)
        if include_phi_wall and bundle.phi_wall is not None:
            phi_wall_raw = bundle.phi_wall[ii, jj, kk].astype(np.float32, copy=False)
        else:
            phi_wall_raw = None
        yield (
            idx,
            _scale_inputs(x_rows, scaler=x_scaler),
            torch.as_tensor(y_rows, dtype=torch.float32),
            torch.as_tensor(xy_local, dtype=torch.float32),
            z_rel_raw,
            phi_wall_raw,
        )
