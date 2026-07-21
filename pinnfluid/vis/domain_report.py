#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LightSource, Normalize, TwoSlopeNorm
from matplotlib.transforms import Affine2D
import numpy as np
import rasterio
from rasterio.enums import Resampling

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    }
)

GLOBAL_TARGET_ZREL_M = 20.0
ROI_MIN_TARGET_ZREL_M = 2.0
ROI_MAX_TARGET_ZREL_M = 40.0
ROI_TARGET_HEIGHT_FRAC = 0.40
MAX_MAP_NX = 700
MAX_MAP_NY = 700
MAX_SEC_NX = 900
MAX_SEC_NZ = 220
MAX_HIRES_TERRAIN_PIX = 1200
GLOBAL_PROFILE_MAX_ZREL = 250.0
ROI_PROFILE_MAX_ZREL = 60.0


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_case_bundle(case_dir: Path) -> dict:
    meta = _load_json(case_dir / "meta.json")
    terr = np.load(case_dir / "terrain.npz")
    flow = np.load(case_dir / "flow.npz")
    phi_path = case_dir / "phi_wall.npy"
    phi_wall = np.load(phi_path) if phi_path.exists() else None
    return {
        "case_dir": case_dir,
        "meta": meta,
        "terrain": {k: terr[k] for k in terr.files},
        "flow": {k: flow[k] for k in flow.files},
        "phi_wall": phi_wall,
    }


def _case_axes(meta: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = [int(v) for v in meta["grid_shape"]]
    b = meta["bounds"]
    x = np.linspace(float(b[0]), float(b[1]), nx, dtype=np.float32)
    y = np.linspace(float(b[2]), float(b[3]), ny, dtype=np.float32)
    z_levels = meta.get("z_levels")
    if z_levels is not None and len(z_levels) == nz:
        z = np.asarray(z_levels, dtype=np.float32)
    else:
        z = np.linspace(float(b[4]), float(b[5]), nz, dtype=np.float32)
    return x, y, z


def _terrain_xy(bundle: dict) -> np.ndarray:
    return np.asarray(bundle["terrain"]["elevation"], dtype=np.float32).T


def _uref(meta: dict) -> float:
    abl = meta.get("ABL") or {}
    uref = float(abl.get("Uref", 1.0) or 1.0)
    return max(uref, 1e-6)


def _cp_field(p_field: np.ndarray, meta: dict) -> np.ndarray:
    # OpenFOAM simpleFoam p has dimensions [0 2 -2 ...], i.e. kinematic pressure p/rho.
    # Dividing by 0.5 * Uref^2 yields the standard pressure coefficient for constant density flow.
    denom = max(0.5 * _uref(meta) ** 2, 1e-6)
    return np.asarray(p_field, dtype=np.float32) / denom


def _speed_mag(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> np.ndarray:
    """Absolute wind speed magnitude |U| in m/s."""
    return np.sqrt(
        np.maximum(
            0.0,
            ux.astype(np.float32) ** 2 + uy.astype(np.float32) ** 2 + uz.astype(np.float32) ** 2,
        )
    )


def _structure_bounds(meta: dict) -> List[dict]:
    vals = meta.get("structure_bounds") or []
    return [v for v in vals if isinstance(v, dict) and "min" in v and "max" in v]


def _max_structure_height(meta: dict) -> float:
    hs = []
    for sb in _structure_bounds(meta):
        try:
            hs.append(float(sb["max"][2]) - float(sb["min"][2]))
        except Exception:
            continue
    return max(hs) if hs else 0.0


def _default_roi_target(meta: dict) -> float:
    h = _max_structure_height(meta)
    if h <= 0.0:
        return 5.0
    return float(min(max(ROI_MIN_TARGET_ZREL_M, ROI_TARGET_HEIGHT_FRAC * h), ROI_MAX_TARGET_ZREL_M))


def _roi_relative_paths(meta: dict, case_dir: Path) -> List[Path]:
    vals = meta.get("roi_paths") or meta.get("roi_relative_paths") or []
    out: List[Path] = []
    for rel in vals:
        p = case_dir / rel
        if (p / "meta.json").exists():
            out.append(p)
    if out:
        return out
    roi_root = case_dir / "roi"
    if roi_root.exists():
        return sorted([p for p in roi_root.iterdir() if (p / "meta.json").exists()])
    return []


def _roi_bounds_list(case_dir: Path, meta: dict) -> List[Tuple[Path, dict]]:
    out = []
    for rp in _roi_relative_paths(meta, case_dir):
        try:
            out.append((rp, _load_json(rp / "meta.json")))
        except Exception:
            continue
    return out


def _pick_section_j(meta: dict, y: np.ndarray) -> int:
    boxes = _structure_bounds(meta)
    if boxes:
        cy = np.mean([0.5 * (float(sb["min"][1]) + float(sb["max"][1])) for sb in boxes])
        return int(np.argmin(np.abs(y - cy)))
    return len(y) // 2


def _nearest_layer_by_zrel(
    field: np.ndarray,
    is_fluid: np.ndarray,
    terrain_xy: np.ndarray,
    z_levels: np.ndarray,
    target_zrel: float,
    *,
    search_radius: int = 2,
    require_finite: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = field.shape
    target_abs = terrain_xy + float(target_zrel)
    base = np.searchsorted(z_levels, target_abs, side="left")
    base = np.clip(base, 0, nz - 1)

    best_val = np.full((nx, ny), np.nan, dtype=np.float32)
    best_zrel = np.full((nx, ny), np.nan, dtype=np.float32)
    best_dist = np.full((nx, ny), np.inf, dtype=np.float32)

    for offset in range(-search_radius, search_radius + 1):
        kk = np.clip(base + offset, 0, nz - 1).astype(np.int64)
        vals = np.take_along_axis(field, kk[..., None], axis=2)[..., 0]
        fluid = np.take_along_axis(is_fluid, kk[..., None], axis=2)[..., 0]
        z_here = z_levels[kk]
        dist = np.abs(z_here - target_abs)
        valid = fluid > 0.5
        if require_finite:
            valid &= np.isfinite(vals)
        better = valid & (dist < best_dist)
        best_val[better] = vals[better]
        best_zrel[better] = (z_here - terrain_xy)[better]
        best_dist[better] = dist[better]

    unresolved = ~np.isfinite(best_val)
    if np.any(unresolved):
        valid_full = (np.asarray(is_fluid, dtype=np.float32) > 0.5) & np.isfinite(field)
        z_grid = np.asarray(z_levels, dtype=np.float32)[None, None, :]
        dist_full = np.abs(z_grid - target_abs[..., None])
        dist_full = np.where(valid_full, dist_full, np.inf)
        kk_best = np.argmin(dist_full, axis=2).astype(np.int64)
        has_valid = np.any(np.isfinite(dist_full), axis=2)
        ii, jj = np.indices(kk_best.shape, dtype=np.int64)
        fallback_val = np.asarray(field, dtype=np.float32)[ii, jj, kk_best]
        fallback_zrel = z_levels[kk_best] - terrain_xy
        use = unresolved & has_valid & np.isfinite(fallback_val)
        best_val[use] = fallback_val[use]
        best_zrel[use] = fallback_zrel[use]

    return best_val, best_zrel


def _downsample_map(
    x: np.ndarray,
    y: np.ndarray,
    data_xy: np.ndarray,
    *,
    max_nx: int = MAX_MAP_NX,
    max_ny: int = MAX_MAP_NY,
):
    sx = max(1, int(math.ceil(len(x) / max_nx)))
    sy = max(1, int(math.ceil(len(y) / max_ny)))
    return x[::sx], y[::sy], data_xy[::sx, ::sy]


def _downsample_section(
    x: np.ndarray,
    z: np.ndarray,
    data_xz: np.ndarray,
    terrain_line: np.ndarray,
    *,
    max_nx: int = MAX_SEC_NX,
    max_nz: int = MAX_SEC_NZ,
):
    sx = max(1, int(math.ceil(len(x) / max_nx)))
    sz = max(1, int(math.ceil(len(z) / max_nz)))
    return x[::sx], z[::sz], data_xz[::sx, ::sz], terrain_line[::sx]


def _finite_concat(*arrs: np.ndarray) -> np.ndarray:
    vals = []
    for arr in arrs:
        if arr is None:
            continue
        v = np.asarray(arr, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return np.zeros(0, dtype=float)
    return np.concatenate(vals)


def _map_quantiles(*arrs: np.ndarray) -> Tuple[float, float]:
    vals = _finite_concat(*arrs)
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _pressure_norm(*arrs: np.ndarray) -> Tuple[Optional[Normalize], Optional[float], Optional[float]]:
    lo, hi = _map_quantiles(*arrs)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, 0.0, 1.0
    if lo < 0.0 < hi:
        skew = min(abs(lo), abs(hi)) / max(abs(lo), abs(hi))
        if skew >= 0.15:
            return TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi), None, None
    return None, lo, hi


def _is_uniform_axis(a: np.ndarray, *, rel_tol: float = 1e-3) -> bool:
    """True if spacings in `a` are uniform within a small relative tolerance."""
    a = np.asarray(a, dtype=np.float64)
    if a.size < 3:
        return True
    d = np.diff(a)
    if d.size == 0:
        return True
    span = float(np.max(np.abs(d)))
    if span <= 0.0:
        return True
    return float(np.max(np.abs(d - d[0]))) <= rel_tol * span


def _upsample_field(
    x: np.ndarray,
    y: np.ndarray,
    data_xy: np.ndarray,
    *,
    target_min_cells: int = 120,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinearly upsample a small (nx, ny) field so blocky maps look smooth.

    NaN regions (solid cells, outside-fluid) are preserved. Only upsamples
    along axes that are uniform — for non-uniform axes (e.g. the vertical
    z-levels in section plots) we would otherwise remap data to wrong
    physical positions, which manifests as a visible gap between the
    terrain line and the fluid data.
    """
    data = np.asarray(data_xy, dtype=np.float32)
    if data.ndim != 2:
        return x, y, data
    nx, ny = data.shape

    x_uniform = _is_uniform_axis(np.asarray(x))
    y_uniform = _is_uniform_axis(np.asarray(y))

    fx = 1
    fy = 1
    if x_uniform and nx < target_min_cells:
        fx = max(1, int(np.ceil(float(target_min_cells) / max(nx, 1))))
    if y_uniform and ny < target_min_cells:
        fy = max(1, int(np.ceil(float(target_min_cells) / max(ny, 1))))
    if fx == 1 and fy == 1:
        return x, y, data

    try:
        from scipy.ndimage import zoom
    except Exception:
        return x, y, data
    mask = np.isfinite(data)
    if not mask.any():
        return x, y, data
    fill_value = float(np.nanmean(data[mask]))
    filled = np.where(mask, data, fill_value)
    fine = zoom(filled, (fx, fy), order=3, mode="nearest")
    mask_fine = zoom(mask.astype(np.float32), (fx, fy), order=1, mode="nearest") > 0.5
    fine = np.where(mask_fine, fine, np.nan).astype(np.float32)
    x_fine = (
        np.linspace(float(x[0]), float(x[-1]), fine.shape[0], dtype=np.float32)
        if fx > 1
        else np.asarray(x, dtype=np.float32)
    )
    y_fine = (
        np.linspace(float(y[0]), float(y[-1]), fine.shape[1], dtype=np.float32)
        if fy > 1
        else np.asarray(y, dtype=np.float32)
    )
    return x_fine, y_fine, fine


def _pmesh(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    data_xy: np.ndarray,
    *,
    cmap: str,
    vmin=None,
    vmax=None,
    norm=None,
    transform=None,
):
    xu, yu, du = _upsample_field(x, y, data_xy)
    m = np.ma.masked_invalid(np.asarray(du, dtype=float).T)
    return ax.pcolormesh(
        xu,
        yu,
        m,
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        transform=transform if transform is not None else ax.transData,
    )


def _display_rotation_deg(meta: dict) -> float:
    abl = meta.get("ABL") or {}
    w = abl.get("wind_from_deg")
    if w is None:
        return 0.0
    return float(270.0 - float(w))


def _map_transform(ax, meta: dict, x: np.ndarray, y: np.ndarray):
    angle = _display_rotation_deg(meta)
    cx = 0.5 * (float(np.nanmin(x)) + float(np.nanmax(x)))
    cy = 0.5 * (float(np.nanmin(y)) + float(np.nanmax(y)))
    corners = np.array(
        [
            [float(np.nanmin(x)), float(np.nanmin(y))],
            [float(np.nanmax(x)), float(np.nanmin(y))],
            [float(np.nanmax(x)), float(np.nanmax(y))],
            [float(np.nanmin(x)), float(np.nanmax(y))],
        ],
        dtype=float,
    )
    th = math.radians(angle)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=float)
    shifted = corners - np.array([cx, cy], dtype=float)
    turned = shifted @ rot.T + np.array([cx, cy], dtype=float)
    limits = (
        float(np.nanmin(turned[:, 0])),
        float(np.nanmax(turned[:, 0])),
        float(np.nanmin(turned[:, 1])),
        float(np.nanmax(turned[:, 1])),
    )
    trans = Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
    return trans, limits


def _apply_map_limits(ax, limits) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")


def _elevation_display_range(arr2d: np.ndarray) -> Tuple[float, float]:
    """Pick display-normalization limits for the terrain colormap.

    For essentially flat terrain (relief < 10m), widen the range ±200m around
    the mean so the image and colorbar sit in the middle of the 'terrain' cmap
    (green) instead of sweeping the whole blue→brown span.
    """
    z = np.asarray(arr2d, dtype=float)
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmax - vmin < 10.0:
        mid = 0.5 * (vmin + vmax)
        return mid - 200.0, mid + 200.0
    return vmin, vmax


def _terrain_rgb(arr2d: np.ndarray, *, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    z = np.asarray(arr2d, dtype=float)
    if not np.isfinite(z).any():
        z = np.zeros_like(z)
    if vmin is None or vmax is None:
        vmin, vmax = _elevation_display_range(z)
    ls = LightSource(azdeg=315, altdeg=45)
    norm = Normalize(vmin=float(vmin), vmax=float(vmax))
    try:
        return ls.shade(z, cmap=plt.get_cmap("terrain"), norm=norm, vert_exag=1.0, blend_mode="soft")
    except Exception:
        base = np.nan_to_num(z, nan=float(np.nanmean(z) if np.isfinite(z).any() else 0.0))
        return plt.get_cmap("terrain")((base - float(vmin)) / max(float(vmax) - float(vmin), 1e-6))


def _load_hires_terrain(meta: dict, fallback_xy: np.ndarray, *, realworld_elev: bool = False) -> Tuple[np.ndarray, Optional[Tuple[float, float, float, float]]]:
    pre = meta.get("preprocessing") or {}
    src_path = pre.get("terrain_source")
    zoff = float(pre.get("z_offset_applied", 0.0) or 0.0)
    if not zoff:
        zoff = float((meta.get("grid_info") or {}).get("z_offset_applied", 0.0) or 0.0)
    bounds = meta.get("bounds") or None
    if meta.get("grid_kind") == "roi":
        out = np.asarray(fallback_xy.T, dtype=np.float32)
        if realworld_elev and zoff:
            out = out - zoff
        return out, None
    if src_path:
        src = Path(src_path)
        if src.exists() and src.suffix.lower() in {".tif", ".tiff"}:
            try:
                with rasterio.open(src) as ds:
                    scale = max(ds.width / MAX_HIRES_TERRAIN_PIX, ds.height / MAX_HIRES_TERRAIN_PIX, 1.0)
                    out_w = max(2, int(math.ceil(ds.width / scale)))
                    out_h = max(2, int(math.ceil(ds.height / scale)))
                    arr = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
                    # rasterio returns row 0 = top of geotiff (max y for north-up).
                    # Callers feed this to imshow(origin="lower"), which expects
                    # row 0 = bottom (min y). Flip vertically here.
                    arr = np.ascontiguousarray(arr[::-1, :])
                    arr = np.asarray(arr, dtype=np.float32)
                    if not realworld_elev:
                        arr = arr + zoff
                    if bounds and len(bounds) >= 4:
                        extent = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
                    else:
                        extent = (0.0, float(arr.shape[1]), 0.0, float(arr.shape[0]))
                    return arr, extent
            except Exception:
                pass
    out = np.asarray(fallback_xy.T, dtype=np.float32)
    if realworld_elev and zoff:
        out = out - zoff
    return out, None


def _overlay_structure_boxes(
    ax,
    boxes: Sequence[dict],
    *,
    edgecolor: str = "k",
    linestyle: str = "-",
    linewidth: float = 1.0,
    alpha: float = 0.9,
    fill_alpha: float = 0.0,
    facecolor: str = "0.35",
    transform=None,
    number_labels: bool = False,
    number_color: str = "w",
    number_fontsize: float = 7.0,
):
    import matplotlib.patheffects as path_effects

    trans = transform if transform is not None else ax.transData
    # Number labels follow the same enumeration order as structure_bounds in
    # meta, which is the order used by the per-structure wind-load table, so
    # row N in the table is box N in the plot.
    for idx, sb in enumerate(boxes):
        try:
            xmin, ymin = float(sb["min"][0]), float(sb["min"][1])
            xmax, ymax = float(sb["max"][0]), float(sb["max"][1])
        except Exception:
            continue
        if fill_alpha > 0.0:
            fill_rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=True,
                facecolor=facecolor,
                edgecolor="none",
                alpha=float(fill_alpha),
                zorder=4,
                transform=trans,
            )
            ax.add_patch(fill_rect)
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=edgecolor,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=5,
            transform=trans,
        )
        ax.add_patch(rect)
        if number_labels:
            # Centre the structure number inside its box. A thin stroke keeps
            # the small digits legible over any background colour without a
            # filled badge that could spill onto neighbouring structures.
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            txt = ax.text(
                cx,
                cy,
                str(idx + 1),
                transform=trans,
                ha="center",
                va="center",
                fontsize=float(number_fontsize),
                fontweight="bold",
                color=number_color,
                zorder=6,
                clip_on=True,
            )
            txt.set_path_effects([
                path_effects.withStroke(linewidth=1.6, foreground="black"),
            ])


def _overlay_roi_boxes(ax, roi_bounds_list: Sequence[Tuple[Path, dict]], *, transform=None) -> None:
    trans = transform if transform is not None else ax.transData
    for roi_dir, meta in roi_bounds_list:
        b = meta.get("bounds")
        if not b or len(b) < 4:
            continue
        rect = patches.Rectangle(
            (float(b[0]), float(b[2])),
            float(b[1]) - float(b[0]),
            float(b[3]) - float(b[2]),
            fill=False,
            edgecolor="#d81b60",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            zorder=5,
            transform=trans,
        )
        ax.add_patch(rect)
        # Small label at the top-left corner so the reader knows which
        # ROI is which when more than one is shown.
        try:
            label = str(Path(roi_dir).name)
        except Exception:
            label = str(meta.get("roi_label", ""))
        if label:
            ax.text(
                float(b[0]) + 0.5,
                float(b[3]) - 0.5,
                label,
                fontsize=8,
                color="#d81b60",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#d81b60", alpha=0.85, linewidth=0.6),
                zorder=6,
                transform=trans,
            )


def _overlay_streamlines(ax, x: np.ndarray, y: np.ndarray, ux_layer: np.ndarray, uy_layer: np.ndarray, *, transform=None) -> None:
    u = np.asarray(ux_layer.T, dtype=float)
    v = np.asarray(uy_layer.T, dtype=float)
    if u.shape != (len(y), len(x)) or v.shape != (len(y), len(x)):
        return
    mask = np.isfinite(u) & np.isfinite(v)
    if np.count_nonzero(mask) < 20:
        return
    step_x = max(1, int(len(x) // 24))
    step_y = max(1, int(len(y) // 24))
    xs = x[::step_x]
    ys = y[::step_y]
    uu = u[::step_y, ::step_x]
    vv = v[::step_y, ::step_x]
    xx, yy = np.meshgrid(xs, ys)
    keep = np.isfinite(uu) & np.isfinite(vv)
    if np.count_nonzero(keep) < 8:
        return
    xx = xx[keep]
    yy = yy[keep]
    uu = uu[keep]
    vv = vv[keep]
    mag = np.hypot(uu, vv)
    scale = max(float(np.nanpercentile(mag, 90.0)), 1e-6) * 28.0
    ax.quiver(
        xx,
        yy,
        uu,
        vv,
        angles="xy",
        scale_units="xy",
        scale=scale,
        color="k",
        alpha=0.45,
        width=0.0017,
        zorder=4,
        transform=transform if transform is not None else ax.transData,
    )


def _overlay_section_line(ax, y_value: float, *, x_span: Tuple[float, float], transform=None) -> None:
    trans = transform if transform is not None else ax.transData
    xs = [float(x_span[0]), float(x_span[1])]
    ys = [float(y_value), float(y_value)]
    ax.plot(xs, ys, color="white", linewidth=1.0, linestyle=":", alpha=0.95, zorder=6, transform=trans)
    ax.plot(xs, ys, color="black", linewidth=0.35, linestyle=":", alpha=0.65, zorder=6, transform=trans)


def _overlay_wind_arrow(ax, x: np.ndarray, y: np.ndarray, *, transform=None, meta: Optional[dict] = None) -> None:
    """Draw a wind-direction arrow in axes-fraction coordinates so it is
    always visible regardless of the data-frame rotation used by the map
    transform. The arrow direction is computed from meta['ABL']['wind_from_deg']
    (meteorological: 0=N, 90=E, 180=S, 270=W). The domain is rotated at build
    time so wind always blows in +x in data coordinates; here we reconstruct
    the displayed direction directly from wind_from to keep the arrow
    consistent with where the wind actually comes from in the final plot."""
    wind_from = None
    if meta is not None:
        abl = meta.get("ABL") or {}
        wind_from = abl.get("wind_from_deg")
    if wind_from is None:
        # No wind info → no arrow
        return
    # Compass 'from' → compass 'to' (where the wind blows to)
    wind_to_compass = (float(wind_from) + 180.0) % 360.0
    # Compass to math-convention angle (CCW from east, where east=0, north=90)
    math_deg = (90.0 - wind_to_compass) % 360.0
    th = math.radians(math_deg)
    length = 0.14  # axes fraction
    cx, cy = 0.12, 0.88  # arrow center, top-left inset
    dx = 0.5 * length * math.cos(th)
    dy = 0.5 * length * math.sin(th)
    start = (cx - dx, cy - dy)
    end = (cx + dx, cy + dy)
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black", mutation_scale=14),
        zorder=8,
    )
    ax.text(
        cx,
        cy - 0.5 * length - 0.035,
        f"wind (from {float(wind_from):.0f}°)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.75, linewidth=0.0),
        zorder=8,
    )


def _show_terrain_context(ax, meta: dict, x: np.ndarray, y: np.ndarray, terrain_xy: np.ndarray, boxes: Sequence[dict], roi_bounds: Sequence[Tuple[Path, dict]], *, transform=None, realworld_elev: bool = False, number_labels: bool = False, number_fontsize: float = 6.0) -> None:
    hires, extent = _load_hires_terrain(meta, terrain_xy, realworld_elev=realworld_elev)
    vmin, vmax = _elevation_display_range(hires)
    rgb = _terrain_rgb(hires, vmin=vmin, vmax=vmax)
    if extent is None:
        b = meta["bounds"]
        extent = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    ax.imshow(
        rgb,
        origin="lower",
        extent=[extent[0], extent[1], extent[2], extent[3]],
        interpolation="nearest",
        aspect="equal",
        zorder=0,
        transform=transform if transform is not None else ax.transData,
    )
    _overlay_structure_boxes(ax, boxes, edgecolor="k", linewidth=1.0, transform=transform, number_labels=number_labels, number_fontsize=number_fontsize)
    _overlay_roi_boxes(ax, roi_bounds, transform=transform)
    _overlay_wind_arrow(ax, x, y, meta=meta)
    # Elevation colorbar matches the image normalization so a flat case shows
    # up at the middle of the 'terrain' cmap (green) rather than sweeping the
    # whole blue→brown range.
    finite = hires[np.isfinite(hires)] if hires.size else np.array([])
    if finite.size:
        sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=plt.get_cmap("terrain"))
        sm.set_array([])
        actual_min = float(np.nanmin(finite))
        actual_max = float(np.nanmax(finite))
        relief = actual_max - actual_min
        frame_label = "real" if realworld_elev else "domain"
        label = f"elevation [m]\n({frame_label}: {actual_min:.0f}–{actual_max:.0f} m, relief {relief:.0f} m)"
        ax.figure.colorbar(sm, ax=ax, label=label)


def _make_section(bundle: dict, *, j_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta = bundle["meta"]
    x, _, z = _case_axes(meta)
    terr_xy = _terrain_xy(bundle)
    terrain_line = terr_xy[:, j_idx]
    is_fluid = np.asarray(bundle["flow"]["is_fluid"], dtype=np.float32)
    ux = np.asarray(bundle["flow"]["Ux"], dtype=np.float32)[:, j_idx, :]
    uy = np.asarray(bundle["flow"]["Uy"], dtype=np.float32)[:, j_idx, :]
    uz = np.asarray(bundle["flow"]["Uz"], dtype=np.float32)[:, j_idx, :]
    p = np.asarray(bundle["flow"]["p"], dtype=np.float32)[:, j_idx, :]
    fluid = is_fluid[:, j_idx, :] > 0.5
    speed_mag = _speed_mag(ux, uy, uz)
    speed_mag = np.where(fluid, speed_mag, np.nan)
    p_mag = np.where(fluid, p, np.nan)
    return x, z, speed_mag, p_mag, terrain_line


def _map_layers(bundle: dict, target_zrel: float) -> Dict[str, np.ndarray]:
    meta = bundle["meta"]
    _, _, z = _case_axes(meta)
    terr_xy = _terrain_xy(bundle)
    is_fluid = np.asarray(bundle["flow"]["is_fluid"], dtype=np.float32)
    ux = np.asarray(bundle["flow"]["Ux"], dtype=np.float32)
    uy = np.asarray(bundle["flow"]["Uy"], dtype=np.float32)
    uz = np.asarray(bundle["flow"]["Uz"], dtype=np.float32)
    p = np.asarray(bundle["flow"]["p"], dtype=np.float32)

    ux_layer, actual_zrel = _nearest_layer_by_zrel(ux, is_fluid, terr_xy, z, target_zrel)
    uy_layer, _ = _nearest_layer_by_zrel(uy, is_fluid, terr_xy, z, target_zrel)
    uz_layer, _ = _nearest_layer_by_zrel(uz, is_fluid, terr_xy, z, target_zrel)
    p_layer, _ = _nearest_layer_by_zrel(p, is_fluid, terr_xy, z, target_zrel)
    speed_mag = _speed_mag(ux_layer, uy_layer, uz_layer)
    return {
        "ux": ux_layer,
        "uy": uy_layer,
        "speed_mag": speed_mag,
        "p": p_layer,
        "actual_zrel": actual_zrel,
    }


def _surface_field(field: np.ndarray, is_fluid: np.ndarray, *, require_finite: bool = True) -> np.ndarray:
    fluid = np.asarray(is_fluid, dtype=np.float32) > 0.5
    has_fluid = np.any(fluid, axis=2)
    kk = np.argmax(fluid, axis=2).astype(np.int64)
    ii, jj = np.indices(kk.shape, dtype=np.int64)
    vals = np.asarray(field, dtype=np.float32)[ii, jj, kk]
    out = np.full(kk.shape, np.nan, dtype=np.float32)
    valid = has_fluid
    if require_finite:
        valid &= np.isfinite(vals)
    out[valid] = vals[valid]
    return out


def _surface_pressure_map(bundle: dict) -> np.ndarray:
    """Raw kinematic pressure (m²/s²) at the first fluid cell above terrain,
    per (x,y) column. Returns a 2D map shaped (nx, ny)."""
    is_fluid = np.asarray(bundle["flow"]["is_fluid"], dtype=np.float32)
    p = np.asarray(bundle["flow"]["p"], dtype=np.float32)
    return _surface_field(p, is_fluid)


def _near_structure_pressure_map(bundle: dict, *, max_dist: float = 2.0) -> Optional[np.ndarray]:
    """For each (x,y) column, pick the fluid cell with the smallest |phi_wall|
    within max_dist and return its kinematic pressure (m²/s²). Columns with no
    near-wall fluid cell fall back to the first-fluid-above-terrain surface
    pressure. Returns None if the bundle has no phi_wall (terrain-only or
    global grid without a wall field).

    This captures pressure on the sides of structures (stagnation upstream,
    suction downstream) in addition to the top, which the "first fluid above"
    surface pressure map cannot see.
    """
    phi = bundle.get("phi_wall")
    if phi is None:
        return None
    is_fluid = np.asarray(bundle["flow"]["is_fluid"], dtype=np.float32) > 0.5
    p = np.asarray(bundle["flow"]["p"], dtype=np.float32)

    abs_phi = np.abs(np.asarray(phi, dtype=np.float32))
    near_wall = is_fluid & np.isfinite(abs_phi) & (abs_phi <= float(max_dist))

    masked = np.where(near_wall, abs_phi, np.inf)
    kk_near = np.argmin(masked, axis=2)
    ii, jj = np.indices(kk_near.shape, dtype=np.int64)
    p_near = p[ii, jj, kk_near]
    has_near = np.any(near_wall, axis=2) & np.isfinite(p_near)

    fallback = _surface_field(p, is_fluid.astype(np.float32))
    out = np.where(has_near, p_near, fallback).astype(np.float32)
    return out


def _surface_line_panel(ax, x: np.ndarray, values: np.ndarray, *, boxes: Sequence[dict], title: str, ylabel: str) -> None:
    vals = np.asarray(values, dtype=np.float32)
    keep = np.isfinite(vals)
    if np.any(keep):
        ax.plot(x[keep], vals[keep], color="#b2182b", linewidth=1.8)
    for sb in boxes:
        try:
            ax.axvspan(float(sb["min"][0]), float(sb["max"][0]), color="k", alpha=0.08, linewidth=0.0)
        except Exception:
            continue
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _extract_profile(bundle: dict, i_idx: int, j_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    meta = bundle["meta"]
    _, _, z = _case_axes(meta)
    terr_xy = _terrain_xy(bundle)
    is_fluid = np.asarray(bundle["flow"]["is_fluid"], dtype=np.float32)
    ux = np.asarray(bundle["flow"]["Ux"], dtype=np.float32)
    uy = np.asarray(bundle["flow"]["Uy"], dtype=np.float32)
    uz = np.asarray(bundle["flow"]["Uz"], dtype=np.float32)
    speed = _speed_mag(ux, uy, uz)
    zrel = z - float(terr_xy[i_idx, j_idx])
    vals = speed[i_idx, j_idx, :]
    fluid = is_fluid[i_idx, j_idx, :] > 0.5
    keep = np.isfinite(zrel) & np.isfinite(vals) & fluid
    zr = np.asarray(zrel[keep], dtype=np.float32)
    vv = np.asarray(vals[keep], dtype=np.float32)
    if zr.size == 0:
        return zr, vv
    order = np.argsort(zr)
    return zr[order], vv[order]


def _profile_points(meta: dict, x: np.ndarray, y: np.ndarray, j_idx: int) -> List[Tuple[str, int, int]]:
    boxes = _structure_bounds(meta)
    bounds = meta.get("bounds") or [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    x0, x1 = float(bounds[0]), float(bounds[1])
    if boxes:
        xmin = min(float(sb["min"][0]) for sb in boxes)
        xmax = max(float(sb["max"][0]) for sb in boxes)
        xc = 0.5 * (xmin + xmax)
        h = max(_max_structure_height(meta), 1.0)
        x_up = max(x0, xmin - max(2.0 * h, 0.12 * (x1 - x0)))
        x_st = xc
        x_dn = min(x1, xmax + max(4.0 * h, 0.15 * (x1 - x0)))
        picks = [("upstream", x_up), ("at structure", x_st), ("downstream", x_dn)]
    else:
        picks = [
            ("upstream", x0 + 0.25 * (x1 - x0)),
            ("mid-domain", x0 + 0.50 * (x1 - x0)),
            ("downstream", x0 + 0.75 * (x1 - x0)),
        ]
    out = []
    for label, xv in picks:
        ii = int(np.argmin(np.abs(x - xv)))
        out.append((label, ii, j_idx))
    return out


def _plot_profiles(ax, bundle: dict, *, j_idx: int, max_zrel: float, y_section: Optional[float] = None) -> None:
    meta = bundle["meta"]
    x, y, _ = _case_axes(meta)
    picks = _profile_points(meta, x, y, j_idx)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (label, ii, jj), color in zip(picks, colors):
        zr, vv = _extract_profile(bundle, ii, jj)
        if zr.size == 0:
            continue
        keep = zr <= float(max_zrel)
        ax.plot(vv[keep], zr[keep], linewidth=1.8, color=color, label=f"{label} (x={float(x[ii]):.0f}m)")
    ax.set_xlabel("|U| [m/s]")
    ax.set_ylabel("z_rel [m]")
    title = "Vertical wind speed profiles"
    if y_section is not None:
        title += f" (y={float(y_section):.1f}m)"
    ax.set_title(title)
    ax.set_ylim(0.0, float(max_zrel))
    ax.grid(True, alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")


def _section_panel(
    ax,
    x: np.ndarray,
    z: np.ndarray,
    data_xz: np.ndarray,
    terrain_line: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm: Optional[Normalize] = None,
    structure_boxes: Optional[Sequence[dict]] = None,
    section_y: Optional[float] = None,
):
    x_ds, z_ds, data_ds, terr_ds = _downsample_section(x, z, data_xz, terrain_line)
    im = _pmesh(ax, x_ds, z_ds, data_ds, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.plot(x_ds, terr_ds, color="k", linewidth=1.0)
    ax.fill_between(x_ds, terr_ds, float(np.nanmin(z_ds)), color="k", alpha=0.12)
    # Overlay structure silhouettes that intersect the section y-slice.
    if structure_boxes and section_y is not None:
        for sb in structure_boxes:
            try:
                y_min = float(sb["min"][1])
                y_max = float(sb["max"][1])
                if float(section_y) < y_min or float(section_y) > y_max:
                    continue
                x_min = float(sb["min"][0])
                x_max = float(sb["max"][0])
                z_min = float(sb["min"][2])
                z_max = float(sb["max"][2])
            except Exception:
                continue
            fill_rect = patches.Rectangle(
                (x_min, z_min),
                x_max - x_min,
                z_max - z_min,
                fill=True,
                facecolor="0.35",
                edgecolor="none",
                alpha=0.35,
                zorder=4,
            )
            ax.add_patch(fill_rect)
            edge_rect = patches.Rectangle(
                (x_min, z_min),
                x_max - x_min,
                z_max - z_min,
                fill=False,
                edgecolor="k",
                linewidth=1.0,
                zorder=5,
            )
            ax.add_patch(edge_rect)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    return im


def _phi_planform(roi_dir: Path) -> Optional[np.ndarray]:
    path = roi_dir / "phi_wall.npy"
    if not path.exists():
        return None
    phi = np.load(path)
    phi = np.asarray(phi, dtype=np.float32)
    plan = np.nanmin(np.abs(phi), axis=2)
    return plan


def _title_suffix(meta: dict, *, roi_boxes: int = 0) -> str:
    abl = meta.get("ABL") or {}
    uref = abl.get("Uref")
    wind_from = abl.get("wind_from_deg")
    parts = [meta.get("category", "?")]
    if uref is not None:
        parts.append(f"Uref={float(uref):.1f} m/s")
    if wind_from is not None:
        parts.append(f"wind_from={float(wind_from):.0f}°")
    parts.append(f"grid={meta.get('grid_shape')}")
    if meta.get("n_structures") is not None:
        parts.append(f"n_struct={meta.get('n_structures')}")
    if roi_boxes:
        parts.append(f"roi={roi_boxes}")
    return " | ".join(parts)


def plot_global_overview(case_dir: Path, out_path: Path) -> Path:
    bundle = _load_case_bundle(case_dir)
    meta = bundle["meta"]
    x, y, z = _case_axes(meta)
    terr_xy = _terrain_xy(bundle)
    target_zrel = GLOBAL_TARGET_ZREL_M
    layers = _map_layers(bundle, target_zrel)
    p_surface = _surface_pressure_map(bundle)
    j_idx = _pick_section_j(meta, y)
    y_section = float(y[j_idx])
    _, _, speed_xz, _, terr_line = _make_section(bundle, j_idx=j_idx)
    roi_bounds = _roi_bounds_list(case_dir, meta)
    boxes = _structure_bounds(meta)

    x_map, y_map, speed_map = _downsample_map(x, y, layers["speed_mag"])
    _, _, p_map = _downsample_map(x, y, p_surface)
    _, _, ux_map = _downsample_map(x, y, layers["ux"])
    _, _, uy_map = _downsample_map(x, y, layers["uy"])

    speed_vmin, speed_vmax = _map_quantiles(speed_map, speed_xz)
    p_line = np.asarray(p_surface[:, j_idx], dtype=np.float32)
    p_norm, p_vmin, p_vmax = _pressure_norm(p_map, p_line)

    fig, axes = plt.subplots(2, 3, figsize=(19.0, 11.5), constrained_layout=True)

    ax = axes[0, 0]
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    _show_terrain_context(ax, meta, x, y, terr_xy, boxes, roi_bounds, transform=map_trans)
    _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
    ax.set_title("Terrain context")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _apply_map_limits(ax, map_limits)

    ax = axes[0, 1]
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    im = _pmesh(ax, x_map, y_map, speed_map, cmap="viridis", vmin=speed_vmin, vmax=speed_vmax, transform=map_trans)
    _overlay_streamlines(ax, x_map, y_map, ux_map, uy_map, transform=map_trans)
    _overlay_structure_boxes(ax, boxes, edgecolor="w", linewidth=0.9, fill_alpha=0.25, facecolor="0.35", transform=map_trans)
    _overlay_roi_boxes(ax, roi_bounds, transform=map_trans)
    _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
    _overlay_wind_arrow(ax, x, y, meta=meta)
    fig.colorbar(im, ax=ax, label="|U| [m/s]")
    med_z = float(np.nanmedian(layers["actual_zrel"])) if np.isfinite(layers["actual_zrel"]).any() else float("nan")
    ax.set_title(f"Global wind speed\n(target z_rel={target_zrel:.0f}m, actual med={med_z:.1f}m)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _apply_map_limits(ax, map_limits)

    ax = axes[0, 2]
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    im = _pmesh(ax, x_map, y_map, p_map, cmap="coolwarm", norm=p_norm, vmin=p_vmin, vmax=p_vmax, transform=map_trans)
    _overlay_structure_boxes(ax, boxes, edgecolor="k", linewidth=0.9, fill_alpha=0.25, facecolor="0.35", transform=map_trans)
    _overlay_roi_boxes(ax, roi_bounds, transform=map_trans)
    _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
    fig.colorbar(im, ax=ax, label="p  [m²/s²]")
    ax.set_title("Surface kinematic pressure\n(first fluid cell above terrain/structure)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _apply_map_limits(ax, map_limits)

    im = _section_panel(
        axes[1, 0],
        x,
        z,
        speed_xz,
        terr_line,
        title=f"Streamwise section |U| (y={y_section:.1f}m)",
        cmap="viridis",
        vmin=speed_vmin,
        vmax=speed_vmax,
        structure_boxes=boxes,
        section_y=y_section,
    )
    fig.colorbar(im, ax=axes[1, 0], label="|U| [m/s]")

    _plot_profiles(axes[1, 1], bundle, j_idx=j_idx, max_zrel=GLOBAL_PROFILE_MAX_ZREL, y_section=y_section)
    _surface_line_panel(
        axes[1, 2],
        x,
        p_line,
        boxes=boxes,
        title=f"Surface pressure along section line (y={y_section:.1f}m)",
        ylabel="p  [m²/s²]",
    )

    fig.suptitle(f"{case_dir.name} - global overview\n{_title_suffix(meta, roi_boxes=len(roi_bounds))}", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roi_overview(case_dir: Path, out_path: Path) -> Optional[Path]:
    case_meta = _load_json(case_dir / "meta.json")
    roi_dirs = _roi_relative_paths(case_meta, case_dir)
    if not roi_dirs:
        return None

    rows = []
    for roi_dir in roi_dirs:
        bundle = _load_case_bundle(roi_dir)
        meta = bundle["meta"]
        x, y, z = _case_axes(meta)
        terr_xy = _terrain_xy(bundle)
        target_zrel = _default_roi_target(meta)
        layers = _map_layers(bundle, target_zrel)
        # ROIs have phi_wall → use near-structure pressure (captures sides of
        # structures, falls back to first-fluid-above-terrain for far cells).
        _nsp = _near_structure_pressure_map(bundle)
        p_surface = _nsp if _nsp is not None else _surface_pressure_map(bundle)
        j_idx = _pick_section_j(meta, y)
        _, _, speed_xz, _, terr_line = _make_section(bundle, j_idx=j_idx)
        phi_plan = _phi_planform(roi_dir)
        rows.append((roi_dir.name, bundle, x, y, z, terr_xy, layers, p_surface, j_idx, speed_xz, terr_line, target_zrel, phi_plan))

    nrows = len(rows)
    fig, axes = plt.subplots(
        nrows,
        4,
        figsize=(21.0, max(5.0, 4.6 * nrows)),
        constrained_layout=True,
        squeeze=False,
    )

    for r, (roi_name, bundle, x, y, z, terr_xy, layers, p_surface, j_idx, speed_xz, terr_line, target_zrel, phi_plan) in enumerate(rows):
        meta = bundle["meta"]
        boxes = _structure_bounds(meta)
        y_section = float(y[j_idx])
        x_map, y_map, speed_map = _downsample_map(x, y, layers["speed_mag"])
        _, _, p_map = _downsample_map(x, y, p_surface)
        _, _, ux_map = _downsample_map(x, y, layers["ux"])
        _, _, uy_map = _downsample_map(x, y, layers["uy"])
        speed_vmin, speed_vmax = _map_quantiles(speed_map, speed_xz)
        p_norm, p_vmin, p_vmax = _pressure_norm(p_map)
        med_z = float(np.nanmedian(layers["actual_zrel"])) if np.isfinite(layers["actual_zrel"]).any() else float("nan")

        ax = axes[r, 0]
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        if phi_plan is not None:
            x_phi, y_phi, phi_ds = _downsample_map(x, y, np.asarray(phi_plan, dtype=np.float32))
            _, phi_vmax = _map_quantiles(phi_ds)
            im = _pmesh(ax, x_phi, y_phi, phi_ds, cmap="magma_r", vmin=0.0, vmax=phi_vmax, transform=map_trans)
            fig.colorbar(im, ax=ax, label="min |phi_wall| over z [m]")
            ax.set_title(f"{roi_name}: structure distance field")
        else:
            hires, extent = _load_hires_terrain(meta, terr_xy)
            rgb = _terrain_rgb(hires)
            if extent is None:
                b = meta["bounds"]
                extent = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            ax.imshow(
                rgb,
                origin="lower",
                extent=[extent[0], extent[1], extent[2], extent[3]],
                interpolation="nearest",
                aspect="equal",
                zorder=0,
                transform=map_trans,
            )
            ax.set_title(f"{roi_name}: terrain + structure")
        _overlay_structure_boxes(ax, boxes, edgecolor="white", linewidth=0.9, transform=map_trans)
        _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _apply_map_limits(ax, map_limits)

        ax = axes[r, 1]
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        im = _pmesh(ax, x_map, y_map, speed_map, cmap="viridis", vmin=speed_vmin, vmax=speed_vmax, transform=map_trans)
        _overlay_streamlines(ax, x_map, y_map, ux_map, uy_map, transform=map_trans)
        _overlay_structure_boxes(ax, boxes, edgecolor="w", linewidth=1.0, fill_alpha=0.25, facecolor="0.35", transform=map_trans)
        _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
        _overlay_wind_arrow(ax, x, y, meta=meta)
        ax.set_title(f"Wind speed\n(target z_rel={target_zrel:.1f}m, actual med={med_z:.1f}m)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _apply_map_limits(ax, map_limits)
        fig.colorbar(im, ax=ax, label="|U| [m/s]")

        ax = axes[r, 2]
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        im = _pmesh(ax, x_map, y_map, p_map, cmap="coolwarm", norm=p_norm, vmin=p_vmin, vmax=p_vmax, transform=map_trans)
        _overlay_structure_boxes(ax, boxes, edgecolor="k", linewidth=1.0, fill_alpha=0.25, facecolor="0.35", transform=map_trans)
        _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
        ax.set_title(f"Near-structure kinematic pressure (y={y_section:.1f}m)\n(nearest fluid cell with |phi_wall|≤2m; fallback: surface)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _apply_map_limits(ax, map_limits)
        fig.colorbar(im, ax=ax, label="p  [m²/s²]")

        ax = axes[r, 3]
        im = _section_panel(
            ax,
            x,
            z,
            speed_xz,
            terr_line,
            title=f"Wake section |U| (y={y_section:.1f}m)",
            cmap="viridis",
            vmin=speed_vmin,
            vmax=speed_vmax,
            structure_boxes=boxes,
            section_y=y_section,
        )
        fig.colorbar(im, ax=ax, label="|U| [m/s]")

    fig.suptitle(f"{case_dir.name} - ROI overview\n{_title_suffix(case_meta, roi_boxes=len(roi_dirs))}", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_training_case_report(case_dir: Path, out_dir: Path) -> Dict[str, str]:
    case_dir = Path(case_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: Dict[str, str] = {}
    files["global_overview"] = str(plot_global_overview(case_dir, out_dir / "global_overview.png"))
    roi_png = plot_roi_overview(case_dir, out_dir / "roi_overview.png")
    if roi_png is not None:
        files["roi_overview"] = str(roi_png)

    manifest = {
        "case": case_dir.name,
        "category": case_dir.parent.name,
        "source": str(case_dir),
        "files": files,
    }
    with open(out_dir / "plot_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return files
