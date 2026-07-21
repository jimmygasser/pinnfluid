"""Prediction-only plots for predict_web.

Each call returns ONE self-contained figure so the user can pick which ones
they want to download. Adapted from scripts/vis/eval_report.py.

Plot set depends on whether ROIs are present:
  Global-only (5): terrain_context, wind_speed, surface_pressure, profiles,
                   global_max_pressure
  + ROI (3 extra): roi_wind_speed, roi_surface_pressure, roi_max_cp
                   (each row = one ROI so grid cases don't explode the count)
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from vis.domain_report import (  # type: ignore  noqa: E402
    GLOBAL_PROFILE_MAX_ZREL,
    GLOBAL_TARGET_ZREL_M,
    _apply_map_limits,
    _case_axes,
    _default_roi_target,
    _downsample_map,
    _extract_profile,
    _load_case_bundle,
    _map_layers,
    _map_quantiles,
    _map_transform,
    _near_structure_pressure_map,
    _overlay_roi_boxes,
    _overlay_section_line,
    _overlay_streamlines,
    _overlay_structure_boxes,
    _overlay_wind_arrow,
    _phi_planform,
    _pick_section_j,
    _pmesh,
    _pressure_norm,
    _profile_points,
    _roi_bounds_list,
    _roi_relative_paths,
    _show_terrain_context,
    _structure_bounds,
    _surface_field,
    _surface_pressure_map,
    _terrain_xy,
    _title_suffix,
)


def _overlay_max_marker(ax, x: np.ndarray, y: np.ndarray, field_xy: np.ndarray, *,
                        label_fmt: str = "max |U|={val:.1f}",
                        transform=None) -> None:
    """Draw a small marker + label at the argmax of a 2D (nx, ny) field."""
    arr = np.asarray(field_xy)
    finite = np.isfinite(arr)
    if not finite.any():
        return
    flat = int(np.argmax(np.where(finite, arr, -np.inf)))
    i, j = np.unravel_index(flat, arr.shape)
    px = float(x[i])
    py = float(y[j])
    val = float(arr[i, j])
    trans = transform if transform is not None else ax.transData
    ax.scatter([px], [py], s=90, marker='X',
               c='#fff8b0', edgecolors='#222', linewidth=1.4,
               zorder=11, transform=trans)
    ax.annotate(label_fmt.format(val=val),
                xy=(px, py), xycoords=trans,
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color='#111',
                bbox=dict(boxstyle='round,pad=0.3', fc='#fff8b0',
                          ec='#888', alpha=0.92),
                zorder=12)


def _overlay_sampling_points(ax, sampling_points, x: np.ndarray, y: np.ndarray, *, transform=None) -> None:
    """Draw a small cross + label at each in-domain sampling point.

    `sampling_points` are dicts carrying local-domain coords `x`, `y` (metres)
    and a `label`. Points outside the flow grid extent are skipped silently.
    """
    if not sampling_points:
        return
    trans = transform if transform is not None else ax.transData
    x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
    y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
    for idx, sp in enumerate(sampling_points):
        xl, yl = sp.get("x"), sp.get("y")
        if xl is None or yl is None:
            continue
        if not (x0 <= float(xl) <= x1 and y0 <= float(yl) <= y1):
            continue
        label = sp.get("label") or f"SP{idx + 1}"
        ax.scatter([float(xl)], [float(yl)], s=95, marker="X",
                   c="#00e5ff", edgecolors="#01303a", linewidth=1.5,
                   zorder=13, transform=trans)
        ax.annotate(label, xy=(float(xl), float(yl)), xycoords=trans,
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, fontweight="bold", color="#01303a",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#b3f0ff",
                              ec="#01303a", alpha=0.85),
                    zorder=14)


def _map_layers_filled(pred_bundle: dict, target_zrel: float) -> dict:
    """`_map_layers` with surface fallback for high-terrain columns.

    Columns where `terrain + target_zrel` exceeds the available z range
    have no layer at the requested height. We backfill those cells with
    the value at the lowest fluid cell above terrain (≈ surface) so the
    2D map covers the entire ROI footprint where any fluid exists.
    """
    layers = _map_layers(pred_bundle, target_zrel)
    is_fluid = np.asarray(pred_bundle["flow"]["is_fluid"], dtype=np.float32)
    flow = pred_bundle["flow"]
    fill_ux = _surface_field(np.asarray(flow["Ux"], dtype=np.float32), is_fluid)
    fill_uy = _surface_field(np.asarray(flow["Uy"], dtype=np.float32), is_fluid)
    fill_uz = _surface_field(np.asarray(flow["Uz"], dtype=np.float32), is_fluid)
    fill_p  = _surface_field(np.asarray(flow["p" ], dtype=np.float32), is_fluid)
    for k, fill in (("ux", fill_ux), ("uy", fill_uy), ("p", fill_p)):
        mask = ~np.isfinite(layers[k])
        if mask.any():
            layers[k] = np.where(mask, fill, layers[k])
    mask = ~np.isfinite(layers["speed_mag"])
    if mask.any():
        fb_speed = np.sqrt(np.maximum(0.0, fill_ux ** 2 + fill_uy ** 2 + fill_uz ** 2))
        layers["speed_mag"] = np.where(mask, fb_speed, layers["speed_mag"])
    return layers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
WIND_MIN_SPAN_MS = 6.0  # widen narrow wind colour ranges (flat domains)


def _broaden(vmin: float, vmax: float, min_span: float) -> Tuple[float, float]:
    """Widen a colour range to at least `min_span`, kept centred, so a flat
    domain (e.g. 9-11 m/s) does not look falsely dramatic."""
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return vmin, vmax
    if (vmax - vmin) < min_span:
        mid = 0.5 * (vmin + vmax)
        return mid - 0.5 * min_span, mid + 0.5 * min_span
    return vmin, vmax


def _overlay_terrain_contours(ax, x, y, elev_xy, *, transform=None) -> None:
    """Faint terrain contour lines (fine 20 m, heavier 100 m) so a filled map
    carries a sense of the ground shape without hiding the field.

    `elev_xy` must be the [x, y]-indexed terrain (i.e. _terrain_xy(bundle)),
    matching the orientation the filled maps use.
    """
    import math
    z = np.asarray(elev_xy, dtype=float)
    if not np.isfinite(z).any():
        return
    zt = z.T  # (ny, nx) to match meshgrid(x, y)
    X, Y = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    if not (hi > lo):
        return
    kw = {"transform": transform} if transform is not None else {}
    fine = np.arange(math.floor(lo / 20.0) * 20.0, hi + 20.0, 20.0)
    bold = np.arange(math.floor(lo / 100.0) * 100.0, hi + 100.0, 100.0)
    if fine.size > 1:
        ax.contour(X, Y, zt, levels=fine, colors="k", linewidths=0.3, alpha=0.28, **kw)
    if bold.size > 1:
        ax.contour(X, Y, zt, levels=bold, colors="k", linewidths=0.7, alpha=0.42, **kw)


def _rebuild_pred_fluid_mask(truth_bundle: dict, shape: Tuple[int, int, int]) -> np.ndarray:
    """Rebuild is_fluid from terrain + explicit structure AABBs.

    The empty-export pipeline (predict_web) derives `is_fluid` from a signed
    wall distance whose sign convention can collapse most of the ROI to
    "inside structure", leaving viz plots with large white regions. The
    predicted flow is meaningful everywhere above terrain and outside the
    structure boxes, so we rebuild a viz-only mask here. The truth bundle's
    own is_fluid is left untouched for any downstream code that wants it.
    """
    meta = truth_bundle["meta"]
    nx, ny, nz = int(shape[0]), int(shape[1]), int(shape[2])
    bounds = meta.get("bounds")
    if not bounds or len(bounds) < 6:
        return np.ones(shape, dtype=np.float32)
    x = np.linspace(float(bounds[0]), float(bounds[1]), nx, dtype=np.float32)
    y = np.linspace(float(bounds[2]), float(bounds[3]), ny, dtype=np.float32)
    z_levels = meta.get("z_levels")
    if z_levels is not None and len(z_levels) == nz:
        z = np.asarray(z_levels, dtype=np.float32)
    else:
        z = np.linspace(float(bounds[4]), float(bounds[5]), nz, dtype=np.float32)
    elev = np.asarray(truth_bundle["terrain"]["elevation"], dtype=np.float32)
    # terrain_layout = [ny, nx] => transpose to (nx, ny) for broadcasting
    if elev.shape == (ny, nx):
        elev_ij = elev.T
    elif elev.shape == (nx, ny):
        elev_ij = elev
    else:
        return np.ones(shape, dtype=np.float32)
    above_ground = z[None, None, :] > (elev_ij[:, :, None] + 1e-6)
    solid = np.zeros(shape, dtype=bool)
    for sb in meta.get("structure_bounds") or []:
        try:
            xmin, ymin, zmin = (float(v) for v in sb["min"])
            xmax, ymax, zmax = (float(v) for v in sb["max"])
        except Exception:
            continue
        solid |= (
            (x[:, None, None] >= xmin) & (x[:, None, None] <= xmax)
            & (y[None, :, None] >= ymin) & (y[None, :, None] <= ymax)
            & (z[None, None, :] >= zmin) & (z[None, None, :] <= zmax)
        )
    return (above_ground & ~solid).astype(np.float32)


def _pred_bundle(truth_bundle: dict, pred_flow: np.ndarray) -> dict:
    """Overlay the predicted flow onto a viz-only rebuilt is_fluid mask.

    The rebuilt mask covers everywhere above terrain and outside the
    structure AABBs. For stale saved predictions that were generated before
    inference was widened (and contain literal zeros outside the old
    narrow mask), we intersect with `pred != 0` so the un-computed cells
    don't show as blue/zero pressure.
    """
    pred = np.asarray(pred_flow, dtype=np.float32)
    is_fluid_viz = _rebuild_pred_fluid_mask(truth_bundle, pred.shape[:3])
    # Safety net for stale predictions: hide cells where every channel is 0
    # AND the original is_fluid was 0 (legitimate predictions are virtually
    # never exactly 0 on all 4 channels at the same cell).
    truth_isf = np.asarray(truth_bundle["flow"]["is_fluid"], dtype=np.float32) > 0.5
    all_zero = (pred == 0.0).all(axis=-1)
    is_fluid_viz = is_fluid_viz * (truth_isf | ~all_zero).astype(np.float32)
    return {
        "case_dir": truth_bundle["case_dir"],
        "meta": truth_bundle["meta"],
        "terrain": truth_bundle["terrain"],
        "flow": {
            "Ux": pred[..., 0],
            "Uy": pred[..., 1],
            "Uz": pred[..., 2],
            "p": pred[..., 3],
            "is_fluid": is_fluid_viz,
        },
        "phi_wall": truth_bundle.get("phi_wall"),
    }


def _max_pressure_map(bundle: dict) -> np.ndarray:
    p = np.asarray(bundle["flow"]["p"], dtype=np.float32)
    is_fluid = np.asarray(bundle["flow"]["is_fluid"], dtype=bool)
    masked = np.where(is_fluid, p, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmax(masked, axis=2)


def _cp_extrema(bundle: dict) -> Tuple[np.ndarray, np.ndarray, float]:
    """Per-column max/min Cp over all fluid z-levels (full ROI coverage)."""
    meta = bundle["meta"]
    uref = float(meta.get("ABL", {}).get("Uref", 1.0)) or 1.0
    q = 0.5 * uref * uref
    p = np.asarray(bundle["flow"]["p"], dtype=np.float32)
    is_fluid = np.asarray(bundle["flow"]["is_fluid"], dtype=bool)
    cp = np.where(is_fluid, p / max(q, 1e-6), np.nan)
    has_any = np.any(np.isfinite(cp), axis=2)
    cp_max = np.full(cp.shape[:2], np.nan, dtype=np.float32)
    cp_min = np.full(cp.shape[:2], np.nan, dtype=np.float32)
    safe_max = np.max(np.where(np.isfinite(cp), cp, -np.inf), axis=2)
    safe_min = np.min(np.where(np.isfinite(cp), cp, np.inf), axis=2)
    cp_max[has_any] = safe_max[has_any]
    cp_min[has_any] = safe_min[has_any]
    return cp_max, cp_min, uref


def _save_fig(fig, out_path: Optional[Path]) -> Tuple[Optional[Path], bytes]:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.read()
    saved: Optional[Path] = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png_bytes)
        saved = out_path
    plt.close(fig)
    return saved, png_bytes


def _common_axes(case_dir: Path, pred_flow: np.ndarray):
    truth = _load_case_bundle(Path(case_dir))
    pred = _pred_bundle(truth, pred_flow)
    meta = pred["meta"]
    x, y, _ = _case_axes(meta)
    j_idx = _pick_section_j(meta, y)
    return {
        "truth": truth,
        "pred": pred,
        "meta": meta,
        "x": x,
        "y": y,
        "j_idx": j_idx,
        "y_section": float(y[j_idx]),
        "boxes": _structure_bounds(meta),
        "roi_bounds": _roi_bounds_list(Path(case_dir), meta),
    }


def _load_roi_rows(case_dir: Path, roi_pred_flows: Dict[str, np.ndarray]) -> list:
    """Match the pred array to its ROI dir via the parent meta.json."""
    import json as _json
    case_dir = Path(case_dir)
    case_meta = _json.loads((case_dir / "meta.json").read_text())
    roi_dirs = _roi_relative_paths(case_meta, case_dir)
    rows = []
    for roi_dir in roi_dirs:
        if roi_dir.name not in roi_pred_flows:
            continue
        truth = _load_case_bundle(roi_dir)
        pred = _pred_bundle(truth, roi_pred_flows[roi_dir.name])
        rows.append((roi_dir, truth, pred))
    return rows


# ---------------------------------------------------------------------------
# GLOBAL plots
# ---------------------------------------------------------------------------
def plot_terrain_context(case_dir: Path, pred_flow: np.ndarray, *, out_path: Optional[Path] = None,
                         sampling_points: Optional[list] = None) -> Tuple[Optional[Path], bytes]:
    ctx = _common_axes(case_dir, pred_flow)
    meta, x, y = ctx["meta"], ctx["x"], ctx["y"]
    terr_xy = _terrain_xy(ctx["pred"])

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 6.5), constrained_layout=True)
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    _show_terrain_context(ax, meta, x, y, terr_xy, ctx["boxes"], ctx["roi_bounds"], transform=map_trans, realworld_elev=True, number_labels=True)
    _overlay_section_line(ax, ctx["y_section"], x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
    _overlay_sampling_points(ax, sampling_points, x, y, transform=map_trans)
    ax.set_title("")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _apply_map_limits(ax, map_limits)
    fig.suptitle(f"{Path(case_dir).name} — terrain context", fontsize=11)
    return _save_fig(fig, out_path)


def plot_wind_speed(case_dir: Path, pred_flow: np.ndarray, *, out_path: Optional[Path] = None, target_zrel: float = GLOBAL_TARGET_ZREL_M) -> Tuple[Optional[Path], bytes]:
    ctx = _common_axes(case_dir, pred_flow)
    meta, x, y = ctx["meta"], ctx["x"], ctx["y"]
    layers = _map_layers_filled(ctx["pred"], target_zrel)
    x_map, y_map, speed_ds = _downsample_map(x, y, layers["speed_mag"])
    _, _, ux_ds = _downsample_map(x, y, layers["ux"])
    _, _, uy_ds = _downsample_map(x, y, layers["uy"])
    speed_vmin, speed_vmax = _broaden(*_map_quantiles(speed_ds), WIND_MIN_SPAN_MS)

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 6.5), constrained_layout=True)
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    im = _pmesh(ax, x_map, y_map, speed_ds, cmap="viridis", vmin=speed_vmin, vmax=speed_vmax, transform=map_trans)
    _overlay_terrain_contours(ax, x, y, _terrain_xy(ctx["pred"]), transform=map_trans)
    _overlay_streamlines(ax, x_map, y_map, ux_ds, uy_ds, transform=map_trans)
    _overlay_structure_boxes(ax, ctx["boxes"], edgecolor="w", linewidth=0.9, fill_alpha=0.25, facecolor="0.35", transform=map_trans)
    _overlay_roi_boxes(ax, ctx["roi_bounds"], transform=map_trans)
    _overlay_section_line(ax, ctx["y_section"], x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
    _overlay_wind_arrow(ax, x, y, meta=meta)
    _overlay_max_marker(ax, x_map, y_map, speed_ds, transform=map_trans)
    ax.set_title(f"Predicted wind speed (z_rel≈{target_zrel:.0f} m)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _apply_map_limits(ax, map_limits)
    fig.colorbar(im, ax=ax, label="|U| [m/s]")
    fig.suptitle(f"{Path(case_dir).name} — wind speed", fontsize=11)
    return _save_fig(fig, out_path)


def plot_surface_pressure(case_dir: Path, pred_flow: np.ndarray, *, out_path: Optional[Path] = None) -> Tuple[Optional[Path], bytes]:
    from units import RHO_AIR  # type: ignore  noqa: E402
    ctx = _common_axes(case_dir, pred_flow)
    meta, x, y = ctx["meta"], ctx["x"], ctx["y"]
    p_surface = RHO_AIR * _surface_pressure_map(ctx["pred"])  # m^2/s^2 -> Pa
    x_map, y_map, p_ds = _downsample_map(x, y, p_surface)
    p_norm, p_vmin, p_vmax = _pressure_norm(p_ds)

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 6.5), constrained_layout=True)
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    im = _pmesh(ax, x_map, y_map, p_ds, cmap="coolwarm", norm=p_norm, vmin=p_vmin, vmax=p_vmax, transform=map_trans)
    _overlay_terrain_contours(ax, x, y, _terrain_xy(ctx["pred"]), transform=map_trans)
    _overlay_structure_boxes(ax, ctx["boxes"], edgecolor="k", linewidth=0.9, fill_alpha=0.25, facecolor="0.35", transform=map_trans)
    _overlay_roi_boxes(ax, ctx["roi_bounds"], transform=map_trans)
    _overlay_section_line(ax, ctx["y_section"], x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
    ax.set_title("Predicted surface pressure (gauge, p_ref = outlet)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _apply_map_limits(ax, map_limits)
    fig.colorbar(im, ax=ax, label="p [Pa]")
    fig.suptitle(f"{Path(case_dir).name} — surface pressure", fontsize=11)
    return _save_fig(fig, out_path)


def plot_profiles(case_dir: Path, pred_flow: np.ndarray, *, out_path: Optional[Path] = None, max_zrel: float = GLOBAL_PROFILE_MAX_ZREL) -> Tuple[Optional[Path], bytes]:
    ctx = _common_axes(case_dir, pred_flow)
    meta, x, y = ctx["meta"], ctx["x"], ctx["y"]
    picks = _profile_points(meta, x, y, ctx["j_idx"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5), constrained_layout=True)
    for (label, ii, jj), color in zip(picks, colors):
        zr, vv = _extract_profile(ctx["pred"], ii, jj)
        if zr.size:
            keep = zr <= float(max_zrel)
            ax.plot(vv[keep], zr[keep], linewidth=1.8, color=color, label=label)
    ax.set_xlabel("|U| [m/s]")
    ax.set_ylabel("z_rel [m]")
    ax.set_title("Predicted vertical wind-speed profiles")
    ax.set_ylim(0.0, float(max_zrel))
    ax.grid(True, alpha=0.25)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", ncol=1)
    fig.suptitle(f"{Path(case_dir).name} — vertical profiles", fontsize=11)
    return _save_fig(fig, out_path)


def plot_sampling_profiles(
    case_dir: Path,
    pred_flow: np.ndarray,
    sampling_points: Optional[list],
    *,
    out_path: Optional[Path] = None,
    max_zrel: float = GLOBAL_PROFILE_MAX_ZREL,
) -> Optional[Tuple[Optional[Path], bytes]]:
    """Vertical wind-speed profile (|U| vs height above ground) at each
    user-placed sampling point. One line per in-domain point. Returns None if
    there are no sampling points inside the domain."""
    if not sampling_points:
        return None
    ctx = _common_axes(case_dir, pred_flow)
    x, y = ctx["x"], ctx["y"]
    x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
    y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))

    rows = []
    for idx, sp in enumerate(sampling_points):
        xl, yl = sp.get("x"), sp.get("y")
        if xl is None or yl is None:
            continue
        if not (x0 <= float(xl) <= x1 and y0 <= float(yl) <= y1):
            continue
        ii = int(np.argmin(np.abs(x - float(xl))))
        jj = int(np.argmin(np.abs(y - float(yl))))
        zr, vv = _extract_profile(ctx["pred"], ii, jj)
        if zr.size == 0:
            continue
        rows.append((sp.get("label") or f"SP{idx + 1}", zr, vv))
    if not rows:
        return None

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5), constrained_layout=True)
    for k, (label, zr, vv) in enumerate(rows):
        keep = zr <= float(max_zrel)
        ax.plot(vv[keep], zr[keep], linewidth=1.8, color=cmap(k % 10), label=label)
    ax.set_xlabel("|U| [m/s]")
    ax.set_ylabel("z_rel [m]")
    ax.set_title("Vertical wind-speed profiles at sampling point(s)")
    ax.set_ylim(0.0, float(max_zrel))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=1)
    fig.suptitle(f"{Path(case_dir).name} — sampling-point profiles", fontsize=11)
    return _save_fig(fig, out_path)


def plot_global_max_pressure(case_dir: Path, pred_flow: np.ndarray, *, out_path: Optional[Path] = None) -> Tuple[Optional[Path], bytes]:
    from units import RHO_AIR  # type: ignore  noqa: E402
    ctx = _common_axes(case_dir, pred_flow)
    meta, x, y = ctx["meta"], ctx["x"], ctx["y"]
    p_max = RHO_AIR * _max_pressure_map(ctx["pred"])  # m^2/s^2 -> Pa
    x_map, y_map, p_max_ds = _downsample_map(x, y, p_max)
    p_norm, p_vmin, p_vmax = _pressure_norm(p_max_ds)

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 6.0), constrained_layout=True)
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    im = _pmesh(ax, x_map, y_map, p_max_ds, cmap="coolwarm", norm=p_norm, vmin=p_vmin, vmax=p_vmax, transform=map_trans)
    _overlay_structure_boxes(ax, ctx["boxes"], edgecolor="k", linewidth=0.9, fill_alpha=0.25, facecolor="0.35", transform=map_trans)
    _overlay_roi_boxes(ax, ctx["roi_bounds"], transform=map_trans)
    ax.set_title("Predicted max pressure over z (gauge, p_ref = outlet)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _apply_map_limits(ax, map_limits)
    fig.colorbar(im, ax=ax, label="max p [Pa]")
    fig.suptitle(f"{Path(case_dir).name} — global max pressure", fontsize=11)
    return _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# ROI plots (each = one figure with one row per ROI)
# ---------------------------------------------------------------------------
def plot_roi_wind_speed(
    case_dir: Path,
    roi_pred_flows: Dict[str, np.ndarray],
    *,
    out_path: Optional[Path] = None,
) -> Optional[Tuple[Optional[Path], bytes]]:
    rows = _load_roi_rows(Path(case_dir), roi_pred_flows)
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.0, max(5.0, 5.0 * len(rows))), constrained_layout=True, squeeze=False)
    for ridx, (roi_dir, truth, pred) in enumerate(rows):
        meta = truth["meta"]
        x, y, _ = _case_axes(meta)
        boxes = _structure_bounds(meta)
        target_zrel = _default_roi_target(meta)
        layers = _map_layers_filled(pred, target_zrel)
        x_map, y_map, speed = _downsample_map(x, y, layers["speed_mag"])
        _, _, ux_ds = _downsample_map(x, y, layers["ux"])
        _, _, uy_ds = _downsample_map(x, y, layers["uy"])
        speed_vmin, speed_vmax = _broaden(*_map_quantiles(speed), WIND_MIN_SPAN_MS)

        ax = axes[ridx, 0]
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        im = _pmesh(ax, x_map, y_map, speed, cmap="viridis", vmin=speed_vmin, vmax=speed_vmax, transform=map_trans)
        _overlay_streamlines(ax, x_map, y_map, ux_ds, uy_ds, transform=map_trans)
        _overlay_structure_boxes(ax, boxes, edgecolor="w", linewidth=0.9, fill_alpha=0.25, facecolor="0.35", transform=map_trans, number_labels=True, number_fontsize=8.0)
        _overlay_wind_arrow(ax, x, y, meta=meta)
        ax.set_title(f"{roi_dir.name}: predicted wind speed (z_rel≈{target_zrel:.1f} m)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _apply_map_limits(ax, map_limits)
        fig.colorbar(im, ax=ax, label="|U| [m/s]")

    fig.suptitle(f"{Path(case_dir).name} — ROI wind speed", fontsize=12)
    return _save_fig(fig, out_path)


def plot_roi_surface_pressure(
    case_dir: Path,
    roi_pred_flows: Dict[str, np.ndarray],
    *,
    out_path: Optional[Path] = None,
) -> Optional[Tuple[Optional[Path], bytes]]:
    from units import RHO_AIR  # type: ignore  noqa: E402
    rows = _load_roi_rows(Path(case_dir), roi_pred_flows)
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.0, max(5.0, 5.0 * len(rows))), constrained_layout=True, squeeze=False)
    for ridx, (roi_dir, truth, pred) in enumerate(rows):
        meta = truth["meta"]
        x, y, _ = _case_axes(meta)
        boxes = _structure_bounds(meta)
        p = _near_structure_pressure_map(pred)
        if p is None:
            p = _surface_pressure_map(pred)
        p = RHO_AIR * p  # m^2/s^2 -> Pa
        x_map, y_map, p_ds = _downsample_map(x, y, p)
        p_norm, p_vmin, p_vmax = _pressure_norm(p_ds)

        ax = axes[ridx, 0]
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        im = _pmesh(ax, x_map, y_map, p_ds, cmap="coolwarm", norm=p_norm, vmin=p_vmin, vmax=p_vmax, transform=map_trans)
        _overlay_structure_boxes(ax, boxes, edgecolor="k", linewidth=0.9, fill_alpha=0.25, facecolor="0.35", transform=map_trans, number_labels=True, number_fontsize=8.0)
        ax.set_title(f"{roi_dir.name}: predicted near-structure pressure (gauge, p_ref = outlet)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _apply_map_limits(ax, map_limits)
        fig.colorbar(im, ax=ax, label="p [Pa]")
    fig.suptitle(f"{Path(case_dir).name} — ROI near-structure pressure", fontsize=12)
    return _save_fig(fig, out_path)


def plot_roi_max_cp(
    case_dir: Path,
    roi_pred_flows: Dict[str, np.ndarray],
    *,
    out_path: Optional[Path] = None,
) -> Optional[Tuple[Optional[Path], bytes]]:
    rows = _load_roi_rows(Path(case_dir), roi_pred_flows)
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 2, figsize=(13.0, max(5.0, 4.5 * len(rows))), constrained_layout=True, squeeze=False)
    for ridx, (roi_dir, truth, pred) in enumerate(rows):
        meta = truth["meta"]
        x, y, _ = _case_axes(meta)
        boxes = _structure_bounds(meta)
        cp_max, cp_min, uref = _cp_extrema(pred)
        x_map, y_map, cp_max_ds = _downsample_map(x, y, cp_max)
        _, _, cp_min_ds = _downsample_map(x, y, cp_min)
        max_norm, max_vmin, max_vmax = _pressure_norm(cp_max_ds)
        min_norm, min_vmin, min_vmax = _pressure_norm(cp_min_ds)

        for col, (field, title, norm, vmin, vmax) in enumerate([
            (cp_max_ds, f"{roi_dir.name}: pred max Cp (over z)", max_norm, max_vmin, max_vmax),
            (cp_min_ds, f"{roi_dir.name}: pred min Cp (suction)", min_norm, min_vmin, min_vmax),
        ]):
            ax = axes[ridx, col]
            map_trans, map_limits = _map_transform(ax, meta, x, y)
            im = _pmesh(ax, x_map, y_map, field, cmap="coolwarm", norm=norm, vmin=vmin, vmax=vmax, transform=map_trans)
            _overlay_structure_boxes(ax, boxes, edgecolor="k", linewidth=0.9, fill_alpha=0.25, facecolor="0.35", transform=map_trans)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            _apply_map_limits(ax, map_limits)
            fig.colorbar(im, ax=ax, label=f"Cp (Uref={uref:.2f})")
    fig.suptitle(f"{Path(case_dir).name} — ROI max/min Cp", fontsize=12)
    return _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Snow drift indicator (heuristic)
# ---------------------------------------------------------------------------
# Near-surface wind-speed threshold for snow transport. ~5 m/s at 1-2 m above
# ground is a common drift threshold for fresh/loose snow; this is a simple
# screening heuristic, NOT a snow transport simulation.
SNOW_TRANSPORT_THRESHOLD_MPS = 5.0
SNOW_TARGET_ZREL_M = 1.5


def _snow_classes(speed_map: np.ndarray, threshold: float) -> np.ndarray:
    """0 = deposition-prone (sheltered), 1 = neutral, 2 = erosion / scour."""
    cls = np.full(speed_map.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(speed_map)
    cls[finite & (speed_map < 0.5 * threshold)] = 0.0
    cls[finite & (speed_map >= 0.5 * threshold) & (speed_map <= threshold)] = 1.0
    cls[finite & (speed_map > threshold)] = 2.0
    return cls


def _snow_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap(["#8ab4e8", "#ececec", "#d9885a"])


def _snow_legend(ax, threshold: float) -> None:
    from matplotlib.patches import Patch
    handles = [
        Patch(fc="#8ab4e8", ec="#666", label=f"deposition-prone (|U| < {0.5 * threshold:.1f} m/s)"),
        Patch(fc="#ececec", ec="#666", label="neutral"),
        Patch(fc="#d9885a", ec="#666", label=f"erosion / scour (|U| > {threshold:.1f} m/s)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


def plot_snow_indicator(
    case_dir: Path,
    pred_flow: np.ndarray,
    *,
    out_path: Optional[Path] = None,
    threshold: float = SNOW_TRANSPORT_THRESHOLD_MPS,
) -> Tuple[Optional[Path], bytes]:
    """Heuristic snow drift indicator from the near-surface wind speed."""
    ctx = _common_axes(case_dir, pred_flow)
    meta, x, y = ctx["meta"], ctx["x"], ctx["y"]
    layers = _map_layers_filled(ctx["pred"], SNOW_TARGET_ZREL_M)
    x_map, y_map, speed_ds = _downsample_map(x, y, layers["speed_mag"])
    cls = _snow_classes(speed_ds, threshold)

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 6.5), constrained_layout=True)
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    _pmesh(ax, x_map, y_map, cls, cmap=_snow_cmap(), vmin=-0.5, vmax=2.5, transform=map_trans)
    _overlay_structure_boxes(ax, ctx["boxes"], edgecolor="k", linewidth=0.9, fill_alpha=0.3, facecolor="0.35", transform=map_trans)
    _overlay_roi_boxes(ax, ctx["roi_bounds"], transform=map_trans)
    _overlay_wind_arrow(ax, x, y, meta=meta)
    _snow_legend(ax, threshold)
    ax.set_title(f"Snow drift indicator (heuristic, |U| at z_rel≈{SNOW_TARGET_ZREL_M:.1f} m)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _apply_map_limits(ax, map_limits)
    fig.suptitle(
        f"{Path(case_dir).name} — snow drift indicator\n"
        "Threshold-based screening from predicted near-surface wind; not a snow transport simulation.",
        fontsize=11,
    )
    return _save_fig(fig, out_path)


def plot_roi_snow_indicator(
    case_dir: Path,
    roi_pred_flows: Dict[str, np.ndarray],
    *,
    out_path: Optional[Path] = None,
    threshold: float = SNOW_TRANSPORT_THRESHOLD_MPS,
) -> Optional[Tuple[Optional[Path], bytes]]:
    """Per-ROI snow drift indicator (one row per ROI), z_rel ≈ 1 m."""
    rows = _load_roi_rows(Path(case_dir), roi_pred_flows)
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.0, max(5.0, 5.0 * len(rows))), constrained_layout=True, squeeze=False)
    for ridx, (roi_dir, truth, pred) in enumerate(rows):
        meta = truth["meta"]
        x, y, _ = _case_axes(meta)
        boxes = _structure_bounds(meta)
        layers = _map_layers_filled(pred, 1.0)
        x_map, y_map, speed_ds = _downsample_map(x, y, layers["speed_mag"])
        cls = _snow_classes(speed_ds, threshold)
        ax = axes[ridx, 0]
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        _pmesh(ax, x_map, y_map, cls, cmap=_snow_cmap(), vmin=-0.5, vmax=2.5, transform=map_trans)
        _overlay_structure_boxes(ax, boxes, edgecolor="k", linewidth=0.9, fill_alpha=0.3, facecolor="0.35", transform=map_trans)
        _snow_legend(ax, threshold)
        ax.set_title(f"{roi_dir.name}: snow drift indicator (|U| at z_rel≈1 m)", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _apply_map_limits(ax, map_limits)
    fig.suptitle(
        f"{Path(case_dir).name} — ROI snow drift indicator (heuristic screening)",
        fontsize=12,
    )
    return _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Two-model disagreement (uncertainty proxy)
# ---------------------------------------------------------------------------
def _finite_pos(v, fallback: float = 1.0) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return fallback
    return v if (np.isfinite(v) and v > 0.0) else fallback



def plot_disagreement(
    case_dir: Path,
    pred_flow_a: np.ndarray,
    pred_flow_b: np.ndarray,
    *,
    label_a: str = "model A",
    label_b: str = "model B",
    out_path: Optional[Path] = None,
    target_zrel: float = GLOBAL_TARGET_ZREL_M,
) -> Tuple[Optional[Path], bytes]:
    """Map of |U_A - U_B| at z_rel≈target — where the two families disagree."""
    ctx = _common_axes(case_dir, pred_flow_a)
    meta, x, y = ctx["meta"], ctx["x"], ctx["y"]
    la = _map_layers_filled(ctx["pred"], target_zrel)
    lb = _map_layers_filled(_pred_bundle(ctx["truth"], pred_flow_b), target_zrel)
    dmap = np.abs(la["speed_mag"] - lb["speed_mag"])
    x_map, y_map, d_ds = _downsample_map(x, y, dmap)

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 6.5), constrained_layout=True)
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    im = _pmesh(ax, x_map, y_map, d_ds, cmap="magma", vmin=0.0, vmax=_finite_pos(np.nanquantile(d_ds, 0.99)), transform=map_trans)
    _overlay_structure_boxes(ax, ctx["boxes"], edgecolor="w", linewidth=0.9, fill_alpha=0.25, facecolor="0.35", transform=map_trans)
    _overlay_roi_boxes(ax, ctx["roi_bounds"], transform=map_trans)
    ax.set_title(f"Model disagreement |ΔU| (z_rel≈{target_zrel:.0f} m)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    _apply_map_limits(ax, map_limits)
    fig.colorbar(im, ax=ax, label="|ΔU| [m/s]")
    fig.suptitle(
        f"{Path(case_dir).name} — uncertainty proxy: |{label_a} − {label_b}|\n"
        "High disagreement = lower confidence; both models agree where this map is dark.",
        fontsize=11,
    )
    return _save_fig(fig, out_path)


def plot_roi_disagreement(
    case_dir: Path,
    roi_pred_a: Dict[str, np.ndarray],
    roi_pred_b: Dict[str, np.ndarray],
    *,
    label_a: str = "model A",
    label_b: str = "model B",
    out_path: Optional[Path] = None,
) -> Optional[Tuple[Optional[Path], bytes]]:
    rows = _load_roi_rows(Path(case_dir), roi_pred_a)
    rows = [(d, t, p) for (d, t, p) in rows if d.name in roi_pred_b]
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.0, max(5.0, 5.0 * len(rows))), constrained_layout=True, squeeze=False)
    for ridx, (roi_dir, truth, pred) in enumerate(rows):
        meta = truth["meta"]
        x, y, _ = _case_axes(meta)
        boxes = _structure_bounds(meta)
        la = _map_layers_filled(pred, 1.5)
        lb = _map_layers_filled(_pred_bundle(truth, roi_pred_b[roi_dir.name]), 1.5)
        dmap = np.abs(la["speed_mag"] - lb["speed_mag"])
        x_map, y_map, d_ds = _downsample_map(x, y, dmap)
        ax = axes[ridx, 0]
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        im = _pmesh(ax, x_map, y_map, d_ds, cmap="magma", vmin=0.0, vmax=_finite_pos(np.nanquantile(d_ds, 0.99)), transform=map_trans)
        _overlay_structure_boxes(ax, boxes, edgecolor="w", linewidth=0.9, fill_alpha=0.25, facecolor="0.35", transform=map_trans)
        ax.set_title(f"{roi_dir.name}: |ΔU| (z_rel≈1.5 m)", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _apply_map_limits(ax, map_limits)
        fig.colorbar(im, ax=ax, label="|ΔU| [m/s]")
    fig.suptitle(
        f"{Path(case_dir).name} — ROI uncertainty proxy: |{label_a} − {label_b}|",
        fontsize=12,
    )
    return _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# Report aggregator
# ---------------------------------------------------------------------------
# Fixed display order (web "Show plots" AND the PDF report both follow this):
#   terrain context -> wind speed -> surface pressure -> global max pressure
#   -> snow indicator (global, optional) -> vertical profiles.
# The profiles plot is intentionally LAST in the global group so it sits right
# before the ROI plots (and any sampling-point profiles that follow them).
_GLOBAL_PLOTS: list[Tuple[str, Callable, str, str]] = [
    ("terrain_context",    plot_terrain_context,    "terrain_context.png",    "Terrain context"),
    ("wind_speed",         plot_wind_speed,         "wind_speed.png",         "Wind speed"),
    ("surface_pressure",   plot_surface_pressure,   "surface_pressure.png",   "Surface pressure"),
    ("global_max_pressure", plot_global_max_pressure, "global_max_pressure.png", "Global max pressure"),
    ("snow_indicator",     plot_snow_indicator,     "snow_indicator.png",     "Snow drift indicator (heuristic)"),
    ("profiles",           plot_profiles,           "profiles.png",           "Vertical profiles"),
]

_ROI_PLOTS: list[Tuple[str, Callable, str, str]] = [
    ("roi_wind_speed",        plot_roi_wind_speed,       "roi_wind_speed.png",       "ROI wind speed"),
    ("roi_surface_pressure",  plot_roi_surface_pressure, "roi_surface_pressure.png", "ROI surface pressure"),
    ("roi_max_cp",            plot_roi_max_cp,           "roi_max_cp.png",           "ROI max/min Cp"),
    ("roi_snow_indicator",    plot_roi_snow_indicator,   "roi_snow_indicator.png",   "ROI snow drift indicator (heuristic)"),
]


def generate_prediction_report(
    case_dir: Path,
    pred_flow: np.ndarray,
    *,
    out_dir: Optional[Path] = None,
    target_zrel: float = GLOBAL_TARGET_ZREL_M,
    roi_pred_flows: Optional[Dict[str, np.ndarray]] = None,
    sampling_points: Optional[list] = None,
) -> Dict[str, dict]:
    """Generate all prediction plots.

    Global plots always run. ROI plots run iff roi_pred_flows is non-empty
    and the case's meta.json lists roi_paths that match the pred dict keys.
    Sampling-point profiles (one figure) run iff sampling_points has any point
    inside the domain; they are emitted AFTER the ROI plots so the display
    order stays terrain → global → ROI → sampling.
    """
    case_dir = Path(case_dir)
    out_dir_p = Path(out_dir) if out_dir else None
    if out_dir_p is not None:
        out_dir_p.mkdir(parents=True, exist_ok=True)

    out: Dict[str, dict] = {}
    for key, fn, fname, title in _GLOBAL_PLOTS:
        out_path = (out_dir_p / fname) if out_dir_p else None
        if key == "wind_speed":
            saved, png = fn(case_dir, pred_flow, out_path=out_path, target_zrel=target_zrel)
        elif key == "terrain_context":
            saved, png = fn(case_dir, pred_flow, out_path=out_path, sampling_points=sampling_points)
        else:
            saved, png = fn(case_dir, pred_flow, out_path=out_path)
        out[key] = {
            "path": str(saved) if saved else None,
            "filename": fname,
            "title": title,
            "png_base64": base64.b64encode(png).decode("ascii"),
        }

    if roi_pred_flows:
        for key, fn, fname, title in _ROI_PLOTS:
            out_path = (out_dir_p / fname) if out_dir_p else None
            result = fn(case_dir, roi_pred_flows, out_path=out_path)
            if result is None:
                continue
            saved, png = result
            out[key] = {
                "path": str(saved) if saved else None,
                "filename": fname,
                "title": title,
                "png_base64": base64.b64encode(png).decode("ascii"),
            }

    if sampling_points:
        out_path = (out_dir_p / "sampling_profiles.png") if out_dir_p else None
        result = plot_sampling_profiles(case_dir, pred_flow, sampling_points, out_path=out_path)
        if result is not None:
            saved, png = result
            out["sampling_profiles"] = {
                "path": str(saved) if saved else None,
                "filename": "sampling_profiles.png",
                "title": "Sampling-point profiles",
                "png_base64": base64.b64encode(png).decode("ascii"),
            }

    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate prediction-only plots from one predict_web run.")
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--pred-flow", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-zrel", type=float, default=GLOBAL_TARGET_ZREL_M)
    ap.add_argument("--roi-pred-flow", nargs="*", default=[], help="roi_label=path.npy pairs")
    args = ap.parse_args()

    pred_flow = np.load(args.pred_flow)
    roi_preds = None
    if args.roi_pred_flow:
        roi_preds = {}
        for entry in args.roi_pred_flow:
            k, _, v = entry.partition("=")
            roi_preds[k] = np.load(v)

    report = generate_prediction_report(
        Path(args.case_dir),
        pred_flow,
        out_dir=Path(args.out_dir),
        target_zrel=float(args.target_zrel),
        roi_pred_flows=roi_preds,
    )
    for key, meta in report.items():
        print(f"{key}: {meta['path']}")
