#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from .domain_report import (
    GLOBAL_PROFILE_MAX_ZREL,
    GLOBAL_TARGET_ZREL_M,
    _apply_map_limits,
    _case_axes,
    _default_roi_target,
    _downsample_map,
    _extract_profile,
    _load_case_bundle,
    _make_section,
    _map_layers,
    _map_quantiles,
    _map_transform,
    _overlay_roi_boxes,
    _overlay_section_line,
    _overlay_streamlines,
    _overlay_structure_boxes,
    _overlay_wind_arrow,
    _pmesh,
    _phi_planform,
    _pick_section_j,
    _pressure_norm,
    _profile_points,
    _roi_bounds_list,
    _roi_relative_paths,
    _section_panel,
    _show_terrain_context,
    _structure_bounds,
    _surface_pressure_map,
    _near_structure_pressure_map,
    _terrain_xy,
    _title_suffix,
)


def _flow_dict_from_array(pred_flow: np.ndarray, truth_bundle: dict) -> dict:
    return {
        'Ux': np.asarray(pred_flow[..., 0], dtype=np.float32),
        'Uy': np.asarray(pred_flow[..., 1], dtype=np.float32),
        'Uz': np.asarray(pred_flow[..., 2], dtype=np.float32),
        'p': np.asarray(pred_flow[..., 3], dtype=np.float32),
        'is_fluid': np.asarray(truth_bundle['flow']['is_fluid'], dtype=np.float32),
    }


def _bundle_with_pred_flow(truth_bundle: dict, pred_flow: np.ndarray) -> dict:
    return {
        'case_dir': truth_bundle['case_dir'],
        'meta': truth_bundle['meta'],
        'terrain': truth_bundle['terrain'],
        'flow': _flow_dict_from_array(pred_flow, truth_bundle),
        'phi_wall': truth_bundle.get('phi_wall'),
    }


def _diff_norm(*arrs: np.ndarray):
    vals = []
    for arr in arrs:
        if arr is None:
            continue
        flat = np.asarray(arr, dtype=np.float32).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size:
            vals.append(flat)
    if not vals:
        return TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0), 1.0
    vals = np.concatenate(vals)
    vmax = float(np.percentile(np.abs(vals), 98.0))
    if not np.isfinite(vmax) or vmax <= 1e-9:
        vmax = 1.0
    return TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax), vmax


def _plot_profile_overlay(ax, truth_bundle: dict, pred_bundle: dict, *, j_idx: int, max_zrel: float) -> None:
    meta = truth_bundle['meta']
    x, y, _ = _case_axes(meta)
    picks = _profile_points(meta, x, y, j_idx)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for (label, ii, jj), color in zip(picks, colors):
        zr_t, vv_t = _extract_profile(truth_bundle, ii, jj)
        zr_p, vv_p = _extract_profile(pred_bundle, ii, jj)
        if zr_t.size:
            keep = zr_t <= float(max_zrel)
            ax.plot(vv_t[keep], zr_t[keep], linewidth=1.8, color=color, label=f'{label} truth')
        if zr_p.size:
            keep = zr_p <= float(max_zrel)
            ax.plot(vv_p[keep], zr_p[keep], linewidth=1.5, color=color, linestyle='--', label=f'{label} pred')
    ax.set_xlabel('|U| [m/s]')
    ax.set_ylabel('z_rel [m]')
    ax.set_title('Vertical wind speed profiles: truth vs pred')
    ax.set_ylim(0.0, float(max_zrel))
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', ncol=1)


def plot_global_eval_comparison(case_dir: Path, out_path: Path, *, global_pred_flow: np.ndarray) -> Path:
    truth = _load_case_bundle(case_dir)
    pred = _bundle_with_pred_flow(truth, global_pred_flow)
    meta = truth['meta']
    x, y, _ = _case_axes(meta)
    terr_xy = _terrain_xy(truth)
    target_zrel = GLOBAL_TARGET_ZREL_M
    truth_layers = _map_layers(truth, target_zrel)
    pred_layers = _map_layers(pred, target_zrel)
    truth_p_surface = _surface_pressure_map(truth)
    pred_p_surface = _surface_pressure_map(pred)
    speed_diff = pred_layers['speed_mag'] - truth_layers['speed_mag']
    p_diff = pred_p_surface - truth_p_surface
    j_idx = _pick_section_j(meta, y)
    y_section = float(y[j_idx])
    roi_bounds = _roi_bounds_list(case_dir, meta)
    boxes = _structure_bounds(meta)

    x_map, y_map, speed_true = _downsample_map(x, y, truth_layers['speed_mag'])
    _, _, speed_pred = _downsample_map(x, y, pred_layers['speed_mag'])
    _, _, speed_delta = _downsample_map(x, y, speed_diff)
    _, _, p_true = _downsample_map(x, y, truth_p_surface)
    _, _, p_pred = _downsample_map(x, y, pred_p_surface)
    _, _, p_delta = _downsample_map(x, y, p_diff)
    _, _, ux_true = _downsample_map(x, y, truth_layers['ux'])
    _, _, uy_true = _downsample_map(x, y, truth_layers['uy'])
    _, _, ux_pred = _downsample_map(x, y, pred_layers['ux'])
    _, _, uy_pred = _downsample_map(x, y, pred_layers['uy'])

    speed_vmin, speed_vmax = _map_quantiles(speed_true, speed_pred)
    speed_diff_norm, _ = _diff_norm(speed_delta)
    p_norm, p_vmin, p_vmax = _pressure_norm(p_true, p_pred)
    p_diff_norm, _ = _diff_norm(p_delta)

    fig, axes = plt.subplots(2, 4, figsize=(22.0, 10.0), constrained_layout=True)

    ax = axes[0, 0]
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    _show_terrain_context(ax, meta, x, y, terr_xy, boxes, roi_bounds, transform=map_trans)
    _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
    ax.set_title('Terrain context')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    _apply_map_limits(ax, map_limits)

    for ax, field, ux, uy, title in [
        (axes[0, 1], speed_true, ux_true, uy_true, 'Truth wind speed'),
        (axes[0, 2], speed_pred, ux_pred, uy_pred, 'Predicted wind speed'),
    ]:
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        im = _pmesh(ax, x_map, y_map, field, cmap='viridis', vmin=speed_vmin, vmax=speed_vmax, transform=map_trans)
        _overlay_streamlines(ax, x_map, y_map, ux, uy, transform=map_trans)
        _overlay_structure_boxes(ax, boxes, edgecolor='w', linewidth=0.9, fill_alpha=0.25, facecolor='0.35', transform=map_trans)
        _overlay_roi_boxes(ax, roi_bounds, transform=map_trans)
        _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
        _overlay_wind_arrow(ax, x, y, meta=meta)
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        _apply_map_limits(ax, map_limits)
        fig.colorbar(im, ax=ax, label='|U| [m/s]')

    ax = axes[0, 3]
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    im = _pmesh(ax, x_map, y_map, speed_delta, cmap='coolwarm', norm=speed_diff_norm, transform=map_trans)
    _overlay_structure_boxes(ax, boxes, edgecolor='k', linewidth=0.9, fill_alpha=0.25, facecolor='0.35', transform=map_trans)
    _overlay_roi_boxes(ax, roi_bounds, transform=map_trans)
    _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
    ax.set_title('Wind speed difference (pred - truth)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    _apply_map_limits(ax, map_limits)
    fig.colorbar(im, ax=ax, label='Δ |U| [m/s]')

    for ax, field, title in [
        (axes[1, 0], p_true, 'Truth surface kinematic pressure'),
        (axes[1, 1], p_pred, 'Predicted surface kinematic pressure'),
    ]:
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        im = _pmesh(ax, x_map, y_map, field, cmap='coolwarm', norm=p_norm, vmin=p_vmin, vmax=p_vmax, transform=map_trans)
        _overlay_structure_boxes(ax, boxes, edgecolor='k', linewidth=0.9, fill_alpha=0.25, facecolor='0.35', transform=map_trans)
        _overlay_roi_boxes(ax, roi_bounds, transform=map_trans)
        _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        _apply_map_limits(ax, map_limits)
        fig.colorbar(im, ax=ax, label='p  [m²/s²]')

    ax = axes[1, 2]
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    im = _pmesh(ax, x_map, y_map, p_delta, cmap='coolwarm', norm=p_diff_norm, transform=map_trans)
    _overlay_structure_boxes(ax, boxes, edgecolor='k', linewidth=0.9, fill_alpha=0.25, facecolor='0.35', transform=map_trans)
    _overlay_roi_boxes(ax, roi_bounds, transform=map_trans)
    _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
    ax.set_title('Surface pressure difference (pred - truth)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    _apply_map_limits(ax, map_limits)
    fig.colorbar(im, ax=ax, label='Δ p  [m²/s²]')

    _plot_profile_overlay(axes[1, 3], truth, pred, j_idx=j_idx, max_zrel=GLOBAL_PROFILE_MAX_ZREL)

    fig.suptitle(f"{case_dir.name} - global evaluation\n{_title_suffix(meta, roi_boxes=len(roi_bounds))}", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_roi_eval_comparison(case_dir: Path, out_path: Path, *, roi_pred_flows: Dict[str, np.ndarray]) -> Optional[Path]:
    case_meta = json.load(open(case_dir / 'meta.json', 'r'))
    roi_dirs = _roi_relative_paths(case_meta, case_dir)
    rows = []
    for roi_dir in roi_dirs:
        if roi_dir.name not in roi_pred_flows:
            continue
        truth = _load_case_bundle(roi_dir)
        pred = _bundle_with_pred_flow(truth, roi_pred_flows[roi_dir.name])
        meta = truth['meta']
        x, y, z = _case_axes(meta)
        target_zrel = _default_roi_target(meta)
        truth_layers = _map_layers(truth, target_zrel)
        pred_layers = _map_layers(pred, target_zrel)
        _t = _near_structure_pressure_map(truth)
        _p = _near_structure_pressure_map(pred)
        truth_p_surface = _t if _t is not None else _surface_pressure_map(truth)
        pred_p_surface = _p if _p is not None else _surface_pressure_map(pred)
        speed_diff = pred_layers['speed_mag'] - truth_layers['speed_mag']
        p_diff = pred_p_surface - truth_p_surface
        j_idx = _pick_section_j(meta, y)
        y_section = float(y[j_idx])
        _, _, speed_xz_truth, _, terr_line = _make_section(truth, j_idx=j_idx)
        _, _, speed_xz_pred, _, _ = _make_section(pred, j_idx=j_idx)
        speed_xz_diff = speed_xz_pred - speed_xz_truth
        phi_plan = _phi_planform(roi_dir)
        rows.append((roi_dir.name, truth, pred, x, y, z, truth_layers, pred_layers, truth_p_surface, pred_p_surface, speed_diff, p_diff, y_section, terr_line, speed_xz_diff, phi_plan))

    if not rows:
        return None

    nrows = len(rows) * 2
    fig, axes = plt.subplots(nrows, 4, figsize=(21.0, max(7.0, 3.8 * nrows)), constrained_layout=True, squeeze=False)

    for ridx, (roi_name, truth, pred, x, y, z, truth_layers, pred_layers, truth_p_surface, pred_p_surface, speed_diff, p_diff, y_section, terr_line, speed_xz_diff, phi_plan) in enumerate(rows):
        meta = truth['meta']
        boxes = _structure_bounds(meta)
        row0 = 2 * ridx
        row1 = row0 + 1

        x_map, y_map, speed_true = _downsample_map(x, y, truth_layers['speed_mag'])
        _, _, speed_pred = _downsample_map(x, y, pred_layers['speed_mag'])
        _, _, speed_delta = _downsample_map(x, y, speed_diff)
        _, _, p_true = _downsample_map(x, y, truth_p_surface)
        _, _, p_pred = _downsample_map(x, y, pred_p_surface)
        _, _, p_delta = _downsample_map(x, y, p_diff)

        speed_vmin, speed_vmax = _map_quantiles(speed_true, speed_pred)
        speed_diff_norm, _ = _diff_norm(speed_delta, speed_xz_diff)
        p_norm, p_vmin, p_vmax = _pressure_norm(p_true, p_pred)
        p_diff_norm, _ = _diff_norm(p_delta)

        ax = axes[row0, 0]
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        if phi_plan is not None:
            x_phi, y_phi, phi_ds = _downsample_map(x, y, np.asarray(phi_plan, dtype=np.float32))
            vmax = float(np.percentile(phi_ds[np.isfinite(phi_ds)], 98.0)) if np.isfinite(phi_ds).any() else 1.0
            im = _pmesh(ax, x_phi, y_phi, phi_ds, cmap='magma_r', vmin=0.0, vmax=max(vmax, 1e-6), transform=map_trans)
            fig.colorbar(im, ax=ax, label='min |phi_wall| [m]')
        _overlay_structure_boxes(ax, boxes, edgecolor='white', linewidth=0.9, transform=map_trans)
        _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
        ax.set_title(f'{roi_name}: structure distance field')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        _apply_map_limits(ax, map_limits)

        for ax, field, title in [
            (axes[row0, 1], speed_true, 'Truth wind speed'),
            (axes[row0, 2], speed_pred, 'Predicted wind speed'),
        ]:
            map_trans, map_limits = _map_transform(ax, meta, x, y)
            im = _pmesh(ax, x_map, y_map, field, cmap='viridis', vmin=speed_vmin, vmax=speed_vmax, transform=map_trans)
            _overlay_structure_boxes(ax, boxes, edgecolor='w', linewidth=0.9, fill_alpha=0.25, facecolor='0.35', transform=map_trans)
            _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
            _overlay_wind_arrow(ax, x, y, meta=meta)
            ax.set_title(title)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            _apply_map_limits(ax, map_limits)
            fig.colorbar(im, ax=ax, label='|U| [m/s]')

        ax = axes[row0, 3]
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        im = _pmesh(ax, x_map, y_map, speed_delta, cmap='coolwarm', norm=speed_diff_norm, transform=map_trans)
        _overlay_structure_boxes(ax, boxes, edgecolor='k', linewidth=0.9, fill_alpha=0.25, facecolor='0.35', transform=map_trans)
        _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
        ax.set_title('Wind speed difference')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        _apply_map_limits(ax, map_limits)
        fig.colorbar(im, ax=ax, label='Δ |U| [m/s]')

        for ax, field, title in [
            (axes[row1, 0], p_true, 'Truth near-structure pressure'),
            (axes[row1, 1], p_pred, 'Predicted near-structure pressure'),
        ]:
            map_trans, map_limits = _map_transform(ax, meta, x, y)
            im = _pmesh(ax, x_map, y_map, field, cmap='coolwarm', norm=p_norm, vmin=p_vmin, vmax=p_vmax, transform=map_trans)
            _overlay_structure_boxes(ax, boxes, edgecolor='k', linewidth=0.9, fill_alpha=0.25, facecolor='0.35', transform=map_trans)
            _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
            ax.set_title(title)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            _apply_map_limits(ax, map_limits)
            fig.colorbar(im, ax=ax, label='p  [m²/s²]')

        ax = axes[row1, 2]
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        im = _pmesh(ax, x_map, y_map, p_delta, cmap='coolwarm', norm=p_diff_norm, transform=map_trans)
        _overlay_structure_boxes(ax, boxes, edgecolor='k', linewidth=0.9, fill_alpha=0.25, facecolor='0.35', transform=map_trans)
        _overlay_section_line(ax, y_section, x_span=(float(np.nanmin(x)), float(np.nanmax(x))), transform=map_trans)
        ax.set_title('Near-structure pressure difference')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        _apply_map_limits(ax, map_limits)
        fig.colorbar(im, ax=ax, label='Δ p  [m²/s²]')

        im = _section_panel(
            axes[row1, 3],
            x,
            z,
            speed_xz_diff,
            terr_line,
            title=f'Streamwise speed difference (y={y_section:.1f}m)',
            cmap='coolwarm',
            norm=speed_diff_norm,
            structure_boxes=boxes,
            section_y=y_section,
        )
        fig.colorbar(im, ax=axes[row1, 3], label='Δ |U| [m/s]')

    fig.suptitle(f"{case_dir.name} - ROI evaluation\n{_title_suffix(case_meta, roi_boxes=len(rows))}", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _max_pressure_map(bundle: dict) -> np.ndarray:
    p = np.asarray(bundle['flow']['p'], dtype=np.float32)
    is_fluid = np.asarray(bundle['flow']['is_fluid'], dtype=bool)
    masked = np.where(is_fluid, p, np.nan)
    with np.errstate(invalid='ignore'):
        return np.nanmax(masked, axis=2)


def _cp_extrema_maps(bundle: dict, *, near_wall_only: bool = False, dmax: float = 2.0) -> tuple[np.ndarray, np.ndarray, float]:
    meta = bundle['meta']
    uref = float(meta.get('ABL', {}).get('Uref', 1.0)) or 1.0
    q = 0.5 * uref * uref
    p = np.asarray(bundle['flow']['p'], dtype=np.float32)
    is_fluid = np.asarray(bundle['flow']['is_fluid'], dtype=bool)
    mask = is_fluid
    if near_wall_only:
        phi = bundle.get('phi_wall')
        if phi is not None:
            mask = mask & (np.abs(np.asarray(phi, dtype=np.float32)) <= float(dmax))
    cp = np.where(mask, p / max(q, 1e-6), np.nan)
    has_any = np.any(np.isfinite(cp), axis=2)
    cp_max = np.full(cp.shape[:2], np.nan, dtype=np.float32)
    cp_min = np.full(cp.shape[:2], np.nan, dtype=np.float32)
    safe_max = np.max(np.where(np.isfinite(cp), cp, -np.inf), axis=2)
    safe_min = np.min(np.where(np.isfinite(cp), cp, np.inf), axis=2)
    cp_max[has_any] = safe_max[has_any]
    cp_min[has_any] = safe_min[has_any]
    return cp_max, cp_min, uref


def plot_global_max_pressure(case_dir: Path, out_path: Path, *, global_pred_flow: np.ndarray) -> Path:
    truth = _load_case_bundle(case_dir)
    pred = _bundle_with_pred_flow(truth, global_pred_flow)
    meta = truth['meta']
    x, y, _ = _case_axes(meta)
    boxes = _structure_bounds(meta)
    roi_bounds = _roi_bounds_list(case_dir, meta)
    p_true = _max_pressure_map(truth)
    p_pred = _max_pressure_map(pred)
    p_diff = p_pred - p_true
    x_map, y_map, p_true_ds = _downsample_map(x, y, p_true)
    _, _, p_pred_ds = _downsample_map(x, y, p_pred)
    _, _, p_diff_ds = _downsample_map(x, y, p_diff)
    p_norm, p_vmin, p_vmax = _pressure_norm(p_true_ds, p_pred_ds)
    p_diff_norm, _ = _diff_norm(p_diff_ds)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), constrained_layout=True)
    for ax, field, title in [
        (axes[0], p_true_ds, 'Truth max kinematic pressure (over z)'),
        (axes[1], p_pred_ds, 'Predicted max kinematic pressure (over z)'),
    ]:
        map_trans, map_limits = _map_transform(ax, meta, x, y)
        im = _pmesh(ax, x_map, y_map, field, cmap='coolwarm', norm=p_norm, vmin=p_vmin, vmax=p_vmax, transform=map_trans)
        _overlay_structure_boxes(ax, boxes, edgecolor='k', linewidth=0.9, fill_alpha=0.25, facecolor='0.35', transform=map_trans)
        _overlay_roi_boxes(ax, roi_bounds, transform=map_trans)
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        _apply_map_limits(ax, map_limits)
        fig.colorbar(im, ax=ax, label='max p [m²/s²]')
    ax = axes[2]
    map_trans, map_limits = _map_transform(ax, meta, x, y)
    im = _pmesh(ax, x_map, y_map, p_diff_ds, cmap='coolwarm', norm=p_diff_norm, transform=map_trans)
    _overlay_structure_boxes(ax, boxes, edgecolor='k', linewidth=0.9, fill_alpha=0.25, facecolor='0.35', transform=map_trans)
    _overlay_roi_boxes(ax, roi_bounds, transform=map_trans)
    ax.set_title('Max-pressure difference (pred - truth)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    _apply_map_limits(ax, map_limits)
    fig.colorbar(im, ax=ax, label='Δ max p [m²/s²]')

    fig.suptitle(f"{case_dir.name} - global max-pressure\n{_title_suffix(meta, roi_boxes=len(roi_bounds))}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_roi_max_cp(case_dir: Path, out_path: Path, *, roi_pred_flows: Dict[str, np.ndarray], dmax: float = 2.0) -> Optional[Path]:
    case_meta = json.load(open(case_dir / 'meta.json', 'r'))
    roi_dirs = _roi_relative_paths(case_meta, case_dir)
    rows = []
    for roi_dir in roi_dirs:
        if roi_dir.name not in roi_pred_flows:
            continue
        truth = _load_case_bundle(roi_dir)
        pred = _bundle_with_pred_flow(truth, roi_pred_flows[roi_dir.name])
        cp_max_t, cp_min_t, uref = _cp_extrema_maps(truth, near_wall_only=True, dmax=dmax)
        cp_max_p, cp_min_p, _ = _cp_extrema_maps(pred, near_wall_only=True, dmax=dmax)
        meta = truth['meta']
        x, y, _ = _case_axes(meta)
        rows.append((roi_dir.name, meta, x, y, cp_max_t, cp_max_p, cp_min_t, cp_min_p, uref))
    if not rows:
        return None

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 4, figsize=(20.0, max(4.5, 4.0 * nrows)), constrained_layout=True, squeeze=False)
    for ridx, (roi_name, meta, x, y, cp_max_t, cp_max_p, cp_min_t, cp_min_p, uref) in enumerate(rows):
        boxes = _structure_bounds(meta)
        x_map, y_map, a = _downsample_map(x, y, cp_max_t)
        _, _, b = _downsample_map(x, y, cp_max_p)
        _, _, c = _downsample_map(x, y, cp_min_t)
        _, _, d = _downsample_map(x, y, cp_min_p)
        max_norm, max_vmin, max_vmax = _pressure_norm(a, b)
        min_norm, min_vmin, min_vmax = _pressure_norm(c, d)
        for col, (field, title, norm, vmin, vmax, label) in enumerate([
            (a, f'{roi_name}: truth max Cp (near-wall ≤{dmax}m)', max_norm, max_vmin, max_vmax, 'Cp'),
            (b, f'{roi_name}: pred max Cp', max_norm, max_vmin, max_vmax, 'Cp'),
            (c, f'{roi_name}: truth min Cp (suction)', min_norm, min_vmin, min_vmax, 'Cp'),
            (d, f'{roi_name}: pred min Cp (suction)', min_norm, min_vmin, min_vmax, 'Cp'),
        ]):
            ax = axes[ridx, col]
            map_trans, map_limits = _map_transform(ax, meta, x, y)
            im = _pmesh(ax, x_map, y_map, field, cmap='coolwarm', norm=norm, vmin=vmin, vmax=vmax, transform=map_trans)
            _overlay_structure_boxes(ax, boxes, edgecolor='k', linewidth=0.9, fill_alpha=0.25, facecolor='0.35', transform=map_trans)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            _apply_map_limits(ax, map_limits)
            fig.colorbar(im, ax=ax, label=f'{label} (Uref={uref:.2f})')

    fig.suptitle(f"{case_dir.name} - ROI max/min Cp near-wall\n{_title_suffix(case_meta, roi_boxes=len(rows))}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def generate_eval_case_report(case_dir: Path, out_dir: Path, *, global_pred_flow: np.ndarray, roi_pred_flows: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, str]:
    case_dir = Path(case_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: Dict[str, str] = {}
    files['global_eval'] = str(plot_global_eval_comparison(case_dir, out_dir / 'global_eval.png', global_pred_flow=global_pred_flow))
    files['global_max_p'] = str(plot_global_max_pressure(case_dir, out_dir / 'global_max_pressure.png', global_pred_flow=global_pred_flow))
    if roi_pred_flows:
        roi_png = plot_roi_eval_comparison(case_dir, out_dir / 'roi_eval.png', roi_pred_flows=roi_pred_flows)
        if roi_png is not None:
            files['roi_eval'] = str(roi_png)
        roi_cp_png = plot_roi_max_cp(case_dir, out_dir / 'roi_max_cp.png', roi_pred_flows=roi_pred_flows)
        if roi_cp_png is not None:
            files['roi_max_cp'] = str(roi_cp_png)

    manifest = {
        'case': case_dir.name,
        'category': case_dir.parent.name,
        'source': str(case_dir),
        'files': files,
    }
    with open(out_dir / 'plot_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    return files
