"""Shared utilities for the unified terrain-structure model."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from config import (
    ABL_UREF_MAX,
    ABL_Z0_MAX,
    ABL_Z0_MIN,
    ABL_ZREF_MAX,
    CANONICAL_CASE_MAX,
    CANONICAL_CASE_MIN,
    SEED,
    SIZE_NORM_MAX,
)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Dict:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=False)


_CASE_ID_RE = re.compile(r'^(\d+)_')


def case_id(name: str) -> Optional[int]:
    m = _CASE_ID_RE.match(str(name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_canonical_case(name: str) -> bool:
    cid = case_id(name)
    return cid is not None and CANONICAL_CASE_MIN <= cid <= CANONICAL_CASE_MAX


def sorted_canonical(names: Iterable[str]) -> list[str]:
    out = [str(n) for n in names if is_canonical_case(str(n))]
    return sorted(out, key=lambda s: (case_id(s) or 10**9, s))


def normalize_abl_features(abl_dict: Dict) -> Dict[str, float]:
    uref = float(abl_dict.get('Uref', 1.0))
    zref = float(abl_dict.get('Zref', 50.0))
    z0 = float(abl_dict.get('z0', 0.1))
    flow = np.asarray(abl_dict.get('flowDir', [1.0, 0.0, 0.0]), dtype=float)
    flow = flow / (np.linalg.norm(flow) + 1e-12)
    z0_log = np.log10(np.clip(z0, ABL_Z0_MIN, ABL_Z0_MAX))
    return {
        'Uref_norm': float(np.clip(uref / ABL_UREF_MAX, 0.0, 1.0)),
        'Zref_norm': float(np.clip(zref / ABL_ZREF_MAX, 0.0, 1.0)),
        'log10_z0_norm': float((z0_log - np.log10(ABL_Z0_MIN)) / (np.log10(ABL_Z0_MAX) - np.log10(ABL_Z0_MIN))),
        'flowDir_x': float(flow[0]),
        'flowDir_y': float(flow[1]),
        'flowDir_z': float(flow[2]),
    }


def normalize_domain_size(bounds: Iterable[float]) -> Dict[str, float]:
    x0, x1, y0, y1, z0, z1 = [float(v) for v in bounds]
    lx = max(x1 - x0, 1e-6)
    ly = max(y1 - y0, 1e-6)
    lz = max(z1 - z0, 1e-6)
    return {
        'Lx_norm': float(np.clip(lx / SIZE_NORM_MAX, 0.0, 1.0)),
        'Ly_norm': float(np.clip(ly / SIZE_NORM_MAX, 0.0, 1.0)),
        'Lz_norm': float(np.clip(lz / SIZE_NORM_MAX, 0.0, 1.0)),
    }


def velocity_direction_error_deg(y_true: np.ndarray, y_pred: np.ndarray, min_speed: float = 0.5) -> float:
    ut = np.asarray(y_true[:, :2], dtype=float)
    up = np.asarray(y_pred[:, :2], dtype=float)
    nt = np.linalg.norm(ut, axis=1)
    npd = np.linalg.norm(up, axis=1)
    mask = np.isfinite(nt) & np.isfinite(npd) & (nt > float(min_speed)) & (npd > float(min_speed))
    if not np.any(mask):
        return float('nan')
    ut = ut[mask] / (nt[mask, None] + 1e-12)
    up = up[mask] / (npd[mask, None] + 1e-12)
    dot = np.clip(np.sum(ut * up, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    return float(np.mean(ang))


def init_regression_metrics_accumulator() -> Dict[str, float]:
    return {
        'n_points': 0.0,
        'sum_sq_uvec': 0.0,
        'sum_sq_umag': 0.0,
        'sum_abs_umag': 0.0,
        'sum_umag_bias': 0.0,
        'sum_sq_p': 0.0,
        'sum_abs_p': 0.0,
        'sum_p_bias': 0.0,
        'sum_dir_err_deg': 0.0,
        'n_dir': 0.0,
    }


def update_regression_metrics_accumulator(acc: Dict[str, float], y_true: np.ndarray, y_pred: np.ndarray, *, min_speed_dir: float = 0.5) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch for metrics update: {y_true.shape} vs {y_pred.shape}")

    err = y_pred - y_true
    um_true = np.linalg.norm(y_true[:, :3], axis=1)
    um_pred = np.linalg.norm(y_pred[:, :3], axis=1)
    du_mag = um_pred - um_true
    dp = err[:, 3]

    acc['n_points'] += float(y_true.shape[0])
    acc['sum_sq_uvec'] += float(np.sum(np.sum(err[:, :3] ** 2, axis=1)))
    acc['sum_sq_umag'] += float(np.sum(du_mag ** 2))
    acc['sum_abs_umag'] += float(np.sum(np.abs(du_mag)))
    acc['sum_umag_bias'] += float(np.sum(du_mag))
    acc['sum_sq_p'] += float(np.sum(dp ** 2))
    acc['sum_abs_p'] += float(np.sum(np.abs(dp)))
    acc['sum_p_bias'] += float(np.sum(dp))

    ut = np.asarray(y_true[:, :2], dtype=float)
    up = np.asarray(y_pred[:, :2], dtype=float)
    nt = np.linalg.norm(ut, axis=1)
    npd = np.linalg.norm(up, axis=1)
    mask = np.isfinite(nt) & np.isfinite(npd) & (nt > float(min_speed_dir)) & (npd > float(min_speed_dir))
    if np.any(mask):
        ut = ut[mask] / (nt[mask, None] + 1e-12)
        up = up[mask] / (npd[mask, None] + 1e-12)
        dot = np.clip(np.sum(ut * up, axis=1), -1.0, 1.0)
        ang = np.degrees(np.arccos(dot))
        acc['sum_dir_err_deg'] += float(np.sum(ang))
        acc['n_dir'] += float(ang.size)


def finalize_regression_metrics(acc: Dict[str, float], *, uref: float) -> Dict[str, float]:
    n = int(acc.get('n_points', 0.0))
    if n <= 0:
        return {
            'n_points': 0,
            'rmse_u': float('nan'),
            'rmse_umag': float('nan'),
            'nrmse_umag': float('nan'),
            'mae_umag': float('nan'),
            'bias_umag': float('nan'),
            'rmse_p': float('nan'),
            'nrmse_p': float('nan'),
            'rmse_p_gauge': float('nan'),
            'nrmse_p_gauge': float('nan'),
            'mae_p': float('nan'),
            'bias_p': float('nan'),
            'dir_err_deg': float('nan'),
        }
    uref = max(float(uref), 1e-6)
    qref = max(0.5 * uref * uref, 1e-6)
    n_f = float(n)
    n_dir = float(acc.get('n_dir', 0.0))
    gauge_sum_sq_p = max(float(acc['sum_sq_p']) - (float(acc['sum_p_bias']) ** 2) / max(n_f, 1.0), 0.0)
    return {
        'n_points': n,
        'rmse_u': float(np.sqrt(acc['sum_sq_uvec'] / n_f)),
        'rmse_umag': float(np.sqrt(acc['sum_sq_umag'] / n_f)),
        'nrmse_umag': float(np.sqrt(acc['sum_sq_umag'] / n_f) / uref),
        'mae_umag': float(acc['sum_abs_umag'] / n_f),
        'bias_umag': float(acc['sum_umag_bias'] / n_f),
        'rmse_p': float(np.sqrt(acc['sum_sq_p'] / n_f)),
        'nrmse_p': float(np.sqrt(acc['sum_sq_p'] / n_f) / qref),
        'rmse_p_gauge': float(np.sqrt(gauge_sum_sq_p / n_f)),
        'nrmse_p_gauge': float(np.sqrt(gauge_sum_sq_p / n_f) / qref),
        'mae_p': float(acc['sum_abs_p'] / n_f),
        'bias_p': float(acc['sum_p_bias'] / n_f),
        'dir_err_deg': float(acc['sum_dir_err_deg'] / n_dir) if n_dir > 0 else float('nan'),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, uref: float) -> Dict[str, float]:
    acc = init_regression_metrics_accumulator()
    update_regression_metrics_accumulator(acc, y_true, y_pred, min_speed_dir=0.5)
    return finalize_regression_metrics(acc, uref=uref)


# ---------------------------------------------------------------------------
# Engineering-relevant subset metrics
# ---------------------------------------------------------------------------
# Thresholds are physical (metres). Tuned to two engineering use cases:
#   - structural loading on panels/turbines    -> near_wall (|phi_wall| <= 2 m)
#   - snow drift / deposition near structures  -> near_ground (z_rel <= 5 m)
#                                                  and the combined pocket
NEAR_WALL_DIST_M = 2.0
NEAR_GROUND_ZREL_M = 5.0
NEAR_GROUND_NEAR_WALL_DIST_M = 5.0  # widen wall band; deposition pocket extends past surface


def init_subset_accumulators(subset_names) -> Dict[str, Dict[str, float]]:
    return {name: init_regression_metrics_accumulator() for name in subset_names}


def build_eval_subset_masks(z_rel: np.ndarray, phi_wall) -> Dict[str, np.ndarray]:
    """Return boolean masks for each engineering subset present given inputs.

    z_rel is required (in metres). phi_wall (in metres, signed distance to nearest
    structure surface) is optional — when None, only `near_ground` is produced.
    """
    z_rel = np.asarray(z_rel, dtype=np.float32)
    masks: Dict[str, np.ndarray] = {
        'near_ground': np.isfinite(z_rel) & (z_rel <= float(NEAR_GROUND_ZREL_M)),
    }
    if phi_wall is not None:
        phi = np.abs(np.asarray(phi_wall, dtype=np.float32))
        masks['near_wall'] = np.isfinite(phi) & (phi <= float(NEAR_WALL_DIST_M))
        masks['near_ground_near_wall'] = (
            np.isfinite(phi)
            & np.isfinite(z_rel)
            & (phi <= float(NEAR_GROUND_NEAR_WALL_DIST_M))
            & (z_rel <= float(NEAR_GROUND_ZREL_M))
        )
    return masks


def update_subset_accumulators(
    accs: Dict[str, Dict[str, float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> None:
    for name, mask in masks.items():
        if name not in accs or not np.any(mask):
            continue
        update_regression_metrics_accumulator(accs[name], y_true[mask], y_pred[mask])


def finalize_subset_metrics(
    accs: Dict[str, Dict[str, float]],
    *,
    uref: float,
) -> Dict[str, Dict[str, float]]:
    return {name: finalize_regression_metrics(acc, uref=uref) for name, acc in accs.items()}


def aspect_to_sin_cos(aspect_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ang = np.deg2rad(np.asarray(aspect_deg, dtype=np.float32))
    return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_wandb_tags(tags: str) -> list[str]:
    if tags is None:
        return []
    parts = [t.strip() for t in str(tags).split(',')]
    return [t for t in parts if t]




def _stable_wandb_run_id(project: str, name: str) -> str:
    raw = f"{project}::{name}".encode('utf-8')
    return f"pts_{hashlib.sha1(raw).hexdigest()[:24]}"


def maybe_wandb_init(
    *,
    enabled: bool,
    project: Optional[str],
    name: str,
    config: Optional[dict] = None,
    entity: Optional[str] = None,
    tags: Optional[list[str]] = None,
    wandb_dir: Optional[Path] = None,
    resume: bool = True,
):
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"[WARN] wandb import failed, disabling wandb logging: {type(e).__name__}: {e}", flush=True)
        return None

    project_name = str(project or 'pinn_terr_struc')
    init_kwargs = {
        'project': project_name,
        'name': str(name),
        'config': config or {},
    }
    if resume:
        init_kwargs['id'] = _stable_wandb_run_id(project_name, str(name))
        init_kwargs['resume'] = 'allow'
    if entity:
        init_kwargs['entity'] = str(entity)
    if tags:
        init_kwargs['tags'] = list(tags)
    if wandb_dir is not None:
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ['WANDB_DIR'] = str(wandb_dir)
        init_kwargs['dir'] = str(wandb_dir)
    try:
        return wandb.init(**init_kwargs)
    except Exception as e:
        print(f"[WARN] wandb.init failed, disabling wandb logging: {type(e).__name__}: {e}", flush=True)
        return None


def wandb_log(run, data: dict, step: Optional[int] = None) -> None:
    if run is None:
        return
    try:
        if step is None:
            run.log(data)
        else:
            run.log(data, step=int(step))
    except Exception as e:
        print(f"[WARN] wandb log failed: {type(e).__name__}: {e}", flush=True)


def wandb_finish(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as e:
        print(f"[WARN] wandb finish failed: {type(e).__name__}: {e}", flush=True)
