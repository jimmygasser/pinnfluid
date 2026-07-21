"""Training loop for the first unified terrain-structure hybrid PINN."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import shutil
import zipfile
import zlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from config import (
    CHARB_EPS,
    DATA_P_WEIGHT,
    AMP_DTYPE,
    CASCADE_EDGE_BAND_XY_M,
    CASCADE_EDGE_BAND_Z_M,
    CASCADE_EDGE_WEIGHT,
    CASCADE_MIN_STRUCTURE_CASES,
    CASCADE_STAGE2_GRID_MAX_ROI_CELLS,
    CASCADE_STAGE2_REFINER_KIND,
    CASCADE_STAGE2_MS_REPEAT_ENABLED,
    CASCADE_STAGE2_MS_REPEAT_MAX,
    CASCADE_STAGE2_MS_REPEAT_N2,
    CASCADE_STAGE2_MS_REPEAT_N3,
    EVAL_GLOBAL_PATCHES_PER_CASE,
    EVAL_ROI_PATCHES_PER_CASE,
    FD_FLUID_MASK_THRESHOLD,
    GLOBAL_DATA_P_WEIGHT,
    GLOBAL_TERRAIN_TENSOR_CACHE_LIMIT,
    GLOBAL_INPUT_COLS,
    GLOBAL_PATCHES_PER_DOMAIN,
    GLOBAL_PATCH_SHAPE,
    GLOBAL_POINTS_PER_DOMAIN,
    GRAD_CLIP_MAX_NORM,
    HARD_GROUND_BC,
    EARLY_STOPPING_PATIENCE,
    LATEST_CKPT_EVERY,
    LR,
    MAX_PLOT_FLOW_POINTS,
    MIN_EPOCH_FOR_BEST,
    MOMENTUM_LOSS_MODE,
    ONECYCLE_DIV_FACTOR,
    ONECYCLE_FINAL_DIV_FACTOR,
    ONECYCLE_PCT_START,
    PATCH_NEAR_GROUND_PROB,
    PHYS_RAMP_APPLY_BC,
    PHYS_RAMP_APPLY_GLOBAL,
    PHYS_RAMP_APPLY_ROI,
    PHYS_RAMP_ENABLED,
    PHYS_RAMP_END_EPOCH,
    PHYS_RAMP_START_EPOCH,
    PLOT_EVAL,
    ROI_TERRAIN_TENSOR_CACHE_LIMIT,
    ROI_INPUT_COLS,
    ROI_PATCHES_PER_DOMAIN,
    ROI_PATCH_SHAPE,
    ROI_POINTS_PER_DOMAIN,
    ROI_DATA_P_WEIGHT,
    ROI_SUPERVISED_SAMPLER_MODE,
    ROI_WALL_BC_DMAX,
    SCHEDULER_MODE,
    SEED,
    STRUCTURE_TENSOR_CACHE_LIMIT,
    TRAIN_LOSS,
    TRAIN_MODE,
    TRAIN_STRUCT_MODE,
    TRAIN_STRUCT_WEIGHT,
    USE_AMP,
    VAL_SELECTOR_P_WEIGHT,
    VAL_SELECTOR_USE_GAUGE_P,
    VAL_SELECTOR_MS_ROI_UMAG_WEIGHT,
    W_BC_INLET,
    W_BC_OUTLET,
    W_BC_WALL_ROI,
    W_BC_SIDE,
    W_BC_TOP,
    W_DATA_GLOBAL,
    W_DATA_ROI,
    W_DIV_GLOBAL,
    W_DIV_ROI,
    W_MOM_GLOBAL,
    W_MOM_ROI,
    W_PHYS_GLOBAL,
    W_PHYS_ROI,
)
from data_loader import (
    BoundaryBatch,
    CaseRepository,
    bundle_uses_terrain_following_z,
    bundle_z_rel_at,
    extract_patch_batch,
    fit_scalers,
    fit_scalers_global_only,
    fit_scalers_roi_refs,
    fit_scalers_roi_only,
    iter_fullgrid_predictions,
    prepare_global_boundary_batches,
    sample_patch_batch,
    sample_supervised_batch,
    sample_targeted_roi_supervised_batch,
    structure_tensor,
    terrain_tensor,
)
from losses import (
    abl_velocity_baseline_from_scaled_inputs,
    apply_output_constraints_from_scaled_inputs,
    compose_prediction_with_velocity_baseline,
    compute_patch_physics_losses_from_pred,
    inverse_minmax_column_from_scaled_inputs,
    inlet_bc_loss_from_phys,
    normal_velocity_bc_loss_from_phys,
    outlet_bc_loss_from_phys,
    roi_wall_velocity_bc_loss_from_phys,
    scale_outputs,
    structured_patch_loss_from_pred,
    supervised_data_loss_from_pred,
)
from utils import (
    build_eval_subset_masks,
    ensure_dir,
    finalize_regression_metrics,
    finalize_subset_metrics,
    init_regression_metrics_accumulator,
    init_subset_accumulators,
    update_regression_metrics_accumulator,
    update_subset_accumulators,
    wandb_log,
    write_json,
)


def _to_device(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(device, non_blocking=True)


def _use_cuda_amp(device: str, enabled: bool) -> bool:
    return bool(enabled) and str(device).startswith('cuda')


def _amp_torch_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name or AMP_DTYPE).strip().lower()
    if name in {'bf16', 'bfloat16'}:
        return torch.bfloat16
    if name in {'fp16', 'float16', 'half'}:
        return torch.float16
    raise ValueError(f'Unsupported AMP dtype: {dtype_name}')


def _training_rng(start_epoch: int) -> np.random.Generator:
    return np.random.default_rng(int(SEED) + int(start_epoch))


def _autocast_context(*, device: str, enabled: bool, amp_dtype: str):
    if not _use_cuda_amp(device, enabled):
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=_amp_torch_dtype(amp_dtype))


def _stable_patch_seed(*parts: str) -> int:
    token = '::'.join(str(p) for p in parts).encode('utf-8')
    return int((zlib.crc32(token) + int(SEED)) & 0xFFFFFFFF)


def _bundle_cache_key(bundle) -> str:
    return str(bundle.case_dir)


def _model_kind(model) -> str:
    return str(getattr(model, 'model_kind', 'hybrid')).strip().lower()


@dataclass
class CascadeConditioner:
    model: torch.nn.Module
    scalers: object
    checkpoint_path: str
    config_snapshot: Optional[dict] = None
    grid_global_cache: Optional[OrderedDict[str, np.ndarray]] = None


class _DeviceTensorCache:
    def __init__(self, *, device: str):
        self.device = str(device)
        self._terrain_global: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._terrain_roi: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._structure_roi: OrderedDict[str, torch.Tensor] = OrderedDict()

    @staticmethod
    def _get_or_put(cache: OrderedDict[str, torch.Tensor], key: str, value_factory, *, limit: int) -> torch.Tensor:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        value = value_factory()
        cache[key] = value
        while len(cache) > max(int(limit), 0):
            cache.popitem(last=False)
        return value

    def terrain(self, bundle) -> torch.Tensor:
        key = _bundle_cache_key(bundle)
        if str(bundle.kind) == 'global':
            return self._get_or_put(
                self._terrain_global,
                key,
                lambda: terrain_tensor(bundle).to(self.device, non_blocking=True),
                limit=int(GLOBAL_TERRAIN_TENSOR_CACHE_LIMIT),
            )
        return self._get_or_put(
            self._terrain_roi,
            key,
            lambda: terrain_tensor(bundle).to(self.device, non_blocking=True),
            limit=int(ROI_TERRAIN_TENSOR_CACHE_LIMIT),
        )

    def structure(self, bundle) -> torch.Tensor:
        key = _bundle_cache_key(bundle)
        return self._get_or_put(
            self._structure_roi,
            key,
            lambda: structure_tensor(bundle).to(self.device, non_blocking=True),
            limit=int(STRUCTURE_TENSOR_CACHE_LIMIT),
        )


def _physics_ramp_multiplier(epoch: int) -> float:
    if not bool(PHYS_RAMP_ENABLED):
        return 1.0
    start = int(PHYS_RAMP_START_EPOCH)
    end = int(PHYS_RAMP_END_EPOCH)
    epoch = int(epoch)
    if epoch <= start:
        return 0.0
    if end <= start:
        return 1.0
    if epoch >= end:
        return 1.0
    return float((epoch - start) / max(end - start, 1))


def _effective_weights(train_mode: str, *, epoch: int | None = None) -> dict[str, float]:
    mode = str(train_mode or TRAIN_MODE).strip().lower()
    if mode == 'dl':
        return {
            'w_phys_global': 0.0,
            'w_phys_roi': 0.0,
            'w_div_global': 0.0,
            'w_div_roi': 0.0,
            'w_mom_global': 0.0,
            'w_mom_roi': 0.0,
            'w_bc_inlet': 0.0,
            'w_bc_outlet': 0.0,
            'w_bc_side': 0.0,
            'w_bc_top': 0.0,
            'w_bc_wall_roi': 0.0,
        }
    weights = {
        'w_phys_global': float(W_PHYS_GLOBAL),
        'w_phys_roi': float(W_PHYS_ROI),
        'w_div_global': float(W_DIV_GLOBAL),
        'w_div_roi': float(W_DIV_ROI),
        'w_mom_global': float(W_MOM_GLOBAL),
        'w_mom_roi': float(W_MOM_ROI),
        'w_bc_inlet': float(W_BC_INLET),
        'w_bc_outlet': float(W_BC_OUTLET),
        'w_bc_side': float(W_BC_SIDE),
        'w_bc_top': float(W_BC_TOP),
        'w_bc_wall_roi': float(W_BC_WALL_ROI),
    }
    if epoch is not None and bool(PHYS_RAMP_ENABLED):
        ramp = _physics_ramp_multiplier(int(epoch))
        if bool(PHYS_RAMP_APPLY_GLOBAL):
            for key in ('w_phys_global',):
                weights[key] *= ramp
        if bool(PHYS_RAMP_APPLY_ROI):
            for key in ('w_phys_roi',):
                weights[key] *= ramp
        if bool(PHYS_RAMP_APPLY_BC):
            for key in ('w_bc_inlet', 'w_bc_outlet', 'w_bc_side', 'w_bc_top', 'w_bc_wall_roi'):
                weights[key] *= ramp
    return weights


def _tf_global_pde_enabled(weights: dict[str, float]) -> bool:
    return (
        float(weights.get('w_phys_global', 0.0)) > 0.0
        and (
            float(weights.get('w_div_global', 0.0)) > 0.0
            or float(weights.get('w_mom_global', 0.0)) > 0.0
        )
    )


def _tf_roi_pde_enabled(weights: dict[str, float]) -> bool:
    return (
        float(weights.get('w_phys_roi', 0.0)) > 0.0
        and (
            float(weights.get('w_div_roi', 0.0)) > 0.0
            or float(weights.get('w_mom_roi', 0.0)) > 0.0
        )
    )


def _tf_global_bc_enabled(weights: dict[str, float]) -> bool:
    return any(float(weights.get(k, 0.0)) > 0.0 for k in ('w_bc_inlet', 'w_bc_outlet', 'w_bc_side', 'w_bc_top'))


def _tf_roi_bc_enabled(weights: dict[str, float]) -> bool:
    return float(weights.get('w_bc_wall_roi', 0.0)) > 0.0


def _tf_skipped_physics_metrics() -> dict[str, float | bool]:
    return {
        'physics_patch_count': 0,
        'physics_div_rms': float('nan'),
        'physics_mom_rms_constant': float('nan'),
        'physics_mom_rms_nut': float('nan'),
        'physics_skipped_terrain_following': True,
    }


def _raise_if_tf_nondata_loss(
    bundle,
    weights: dict[str, float],
    *,
    scope: str,
    structured_loss_enabled: bool = False,
) -> None:
    if not bundle_uses_terrain_following_z(bundle):
        return
    scope_l = str(scope).lower()
    if scope_l == 'global':
        enabled = _tf_global_pde_enabled(weights) or _tf_global_bc_enabled(weights)
    elif scope_l == 'roi':
        enabled = _tf_roi_pde_enabled(weights) or _tf_roi_bc_enabled(weights)
    else:
        enabled = (
            _tf_global_pde_enabled(weights)
            or _tf_roi_pde_enabled(weights)
            or _tf_global_bc_enabled(weights)
            or _tf_roi_bc_enabled(weights)
        )
    if not (enabled or bool(structured_loss_enabled)):
        return
    raise RuntimeError(
        "Physics/BC/structured losses are not supported on terrain-following grids yet. "
        f"Case={getattr(bundle, 'name', 'unknown')} kind={getattr(bundle, 'kind', 'unknown')} "
        f"path={getattr(bundle, 'case_dir', '')}. Use mode=dl or an absolute-grid export."
    )


def _conditioner_snapshot_value(conditioner: CascadeConditioner, key: str, default):
    snapshot = conditioner.config_snapshot if isinstance(conditioner.config_snapshot, dict) else {}
    return snapshot.get(key, default)


def _conditioner_global_patch_shape(conditioner: CascadeConditioner) -> tuple[int, int, int]:
    raw = _conditioner_snapshot_value(conditioner, 'GLOBAL_PATCH_SHAPE', GLOBAL_PATCH_SHAPE)
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().tolist()
    vals = list(raw) if isinstance(raw, (list, tuple)) else list(GLOBAL_PATCH_SHAPE)
    if len(vals) != 3:
        vals = list(GLOBAL_PATCH_SHAPE)
    return tuple(max(1, int(v)) for v in vals)


def _is_grid_unet_conditioner(conditioner: CascadeConditioner) -> bool:
    return _model_kind(conditioner.model) == 'grid_unet'


def _is_cascade_grid_refiner(model) -> bool:
    return bool(getattr(model, 'uses_cascade_grid_refiner', False))


_CATEGORY_SHORT = {
    'complexterrain_only': 'ct',
    'singlestructures': 'ss',
    'multistructures': 'ms',
}


def _category_metrics(repo: CaseRepository, metrics_by_key: dict[str, dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = {
        short: {'nrmse_umag': [], 'nrmse_p': [], 'nrmse_p_gauge': []} for short in _CATEGORY_SHORT.values()
    }
    for key, metrics in metrics_by_key.items():
        case_name = str(key).split('/', 1)[0]
        category = repo.load_global(case_name).category
        short = _CATEGORY_SHORT.get(category)
        if short is None:
            continue
        for metric_name in ['nrmse_umag', 'nrmse_p', 'nrmse_p_gauge']:
            val = float(metrics.get(metric_name, float('nan')))
            if np.isfinite(val):
                buckets[short][metric_name].append(val)
    out: dict[str, dict[str, float]] = {}
    for short, vals in buckets.items():
        out[short] = {
            'nrmse_umag': float(np.mean(vals['nrmse_umag'])) if vals['nrmse_umag'] else float('nan'),
            'nrmse_p': float(np.mean(vals['nrmse_p'])) if vals['nrmse_p'] else float('nan'),
            'nrmse_p_gauge': float(np.mean(vals['nrmse_p_gauge'])) if vals['nrmse_p_gauge'] else float('nan'),
        }
    return out


def _mean_case_metric(metrics_by_key: dict[str, dict], metric_name: str) -> float:
    vals = [float(m.get(metric_name, float('nan'))) for m in metrics_by_key.values()]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float('nan')


def _selector_components(
    *,
    val_global_metrics: dict[str, dict],
    val_roi_metrics: dict[str, dict],
    val_roi_by_cat: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    p_metric = 'nrmse_p_gauge' if bool(VAL_SELECTOR_USE_GAUGE_P) else 'nrmse_p'
    global_score = _mean_case_metric(val_global_metrics, 'nrmse_umag')
    global_p_score = _mean_case_metric(val_global_metrics, p_metric)
    roi_score = _mean_case_metric(val_roi_metrics, 'nrmse_umag')
    roi_p_score = _mean_case_metric(val_roi_metrics, p_metric)
    selector_umag_terms = [v for v in [global_score, roi_score] if np.isfinite(v)]
    selector_p_terms = [v for v in [global_p_score, roi_p_score] if np.isfinite(v)]
    selector_umag = float(np.mean(selector_umag_terms)) if selector_umag_terms else float('inf')
    selector_p = float(np.mean(selector_p_terms)) if selector_p_terms else 0.0
    ms_roi_umag = float('nan')
    if val_roi_by_cat:
        ms_roi_umag = float((val_roi_by_cat.get('ms') or {}).get('nrmse_umag', float('nan')))
    selector = float(selector_umag + float(VAL_SELECTOR_P_WEIGHT) * selector_p)
    if np.isfinite(ms_roi_umag) and float(VAL_SELECTOR_MS_ROI_UMAG_WEIGHT) != 0.0:
        selector += float(VAL_SELECTOR_MS_ROI_UMAG_WEIGHT) * float(ms_roi_umag)
    return {
        'global_umag': global_score,
        'global_p': global_p_score,
        'roi_umag': roi_score,
        'roi_p': roi_p_score,
        'selector_umag': selector_umag,
        'selector_p': selector_p,
        'selector_ms_roi_umag': ms_roi_umag,
        'selector': selector,
    }


def _build_scheduler(optimizer: torch.optim.Optimizer, *, epochs: int, steps_per_epoch: int, scheduler_mode: str):
    mode = str(scheduler_mode or SCHEDULER_MODE).strip().lower()
    if mode == 'none' or epochs <= 0 or steps_per_epoch <= 0:
        return None
    if mode != 'onecycle':
        raise ValueError(f'Unsupported scheduler mode: {scheduler_mode}')
    total_steps = int(max(1, epochs) * max(1, steps_per_epoch))
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(LR),
        total_steps=total_steps,
        pct_start=float(ONECYCLE_PCT_START),
        anneal_strategy='cos',
        div_factor=float(ONECYCLE_DIV_FACTOR),
        final_div_factor=float(ONECYCLE_FINAL_DIV_FACTOR),
    )


def _predict_global(
    model,
    bundle,
    batch,
    y_scaler,
    x_scaler,
    *,
    device: str,
    hard_ground_bc: bool,
    terr: Optional[torch.Tensor] = None,
    gfeat: Optional[torch.Tensor] = None,
):
    terr = terrain_tensor(bundle).to(device) if terr is None else terr
    gfeat = model.encode_global(terr) if gfeat is None else gfeat
    raw = model.forward_global_from_encoded(gfeat, _to_device(batch.x_scaled, device), _to_device(batch.xy_local, device))
    pred_scaled, pred_phys = _compose_global_prediction_from_raw(
        model,
        raw,
        x_batch=_to_device(batch.x_scaled, device),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        hard_ground_bc=hard_ground_bc,
    )
    return pred_scaled, pred_phys


def _compose_global_prediction_from_raw(
    model,
    raw: torch.Tensor,
    *,
    x_batch: torch.Tensor,
    x_scaler,
    y_scaler,
    hard_ground_bc: bool,
):
    if bool(getattr(model, "uses_abl_velocity_baseline", False)):
        return compose_prediction_with_velocity_baseline(
            raw,
            x_batch=x_batch,
            x_scaler=x_scaler,
            input_cols=GLOBAL_INPUT_COLS,
            y_scaler=y_scaler,
            hard_ground_bc=hard_ground_bc,
        )
    return apply_output_constraints_from_scaled_inputs(
        raw,
        x_batch=x_batch,
        x_scaler=x_scaler,
        input_cols=GLOBAL_INPUT_COLS,
        y_scaler=y_scaler,
        hard_ground_bc=hard_ground_bc,
    )


def _predict_roi(
    model,
    global_bundle,
    roi_bundle,
    batch,
    y_scaler,
    x_scaler,
    *,
    device: str,
    hard_ground_bc: bool,
    gterr: Optional[torch.Tensor] = None,
    rterr: Optional[torch.Tensor] = None,
    sterr: Optional[torch.Tensor] = None,
    gfeat: Optional[torch.Tensor] = None,
    rfeat: Optional[torch.Tensor] = None,
    sfeat: Optional[torch.Tensor] = None,
):
    gterr = terrain_tensor(global_bundle).to(device) if gterr is None else gterr
    rterr = terrain_tensor(roi_bundle).to(device) if rterr is None else rterr
    gfeat = model.encode_global(gterr) if gfeat is None else gfeat
    rfeat = model.encode_roi(rterr) if rfeat is None else rfeat
    sfeat = model.encode_structure(sterr) if sfeat is None else sfeat
    raw = model.forward_roi_from_encoded(
        gfeat,
        rfeat,
        _to_device(batch.x_scaled, device),
        _to_device(batch.xy_global, device),
        _to_device(batch.xy_local, device),
        s_feat=sfeat,
    )
    pred_scaled, pred_phys = apply_output_constraints_from_scaled_inputs(
        raw,
        x_batch=_to_device(batch.x_scaled, device),
        x_scaler=x_scaler,
        input_cols=ROI_INPUT_COLS,
        y_scaler=y_scaler,
        hard_ground_bc=hard_ground_bc,
    )
    return pred_scaled, pred_phys


def _roi_xscaled_to_global_xscaled(
    x_scaled_roi: torch.Tensor,
    *,
    x_scaler_roi,
    x_scaler_global,
    device: str,
) -> torch.Tensor:
    cols_roi = list(ROI_INPUT_COLS)
    rows = []
    for col in GLOBAL_INPUT_COLS:
        roi_idx = int(cols_roi.index(col))
        phys = inverse_minmax_column_from_scaled_inputs(x_scaled_roi.to(device), x_scaler_roi, roi_idx)
        g_idx = int(GLOBAL_INPUT_COLS.index(col))
        xmin = torch.as_tensor(float(x_scaler_global.data_min_[g_idx]), dtype=phys.dtype, device=device)
        xmax = torch.as_tensor(float(x_scaler_global.data_max_[g_idx]), dtype=phys.dtype, device=device)
        denom = torch.clamp(xmax - xmin, min=1e-6)
        rows.append(((phys - xmin) / denom).unsqueeze(1))
    return torch.cat(rows, dim=1)


def _xy_global_from_roi_scaled(
    x_scaled_roi: torch.Tensor,
    *,
    x_scaler_roi,
    global_bundle,
    device: str,
) -> torch.Tensor:
    x_idx = int(ROI_INPUT_COLS.index("x"))
    y_idx = int(ROI_INPUT_COLS.index("y"))
    x_phys = inverse_minmax_column_from_scaled_inputs(x_scaled_roi.to(device), x_scaler_roi, x_idx)
    y_phys = inverse_minmax_column_from_scaled_inputs(x_scaled_roi.to(device), x_scaler_roi, y_idx)
    gx0, gx1, gy0, gy1, _, _ = global_bundle.bounds
    return torch.stack(
        [
            2.0 * ((x_phys - float(gx0)) / max(float(gx1 - gx0), 1e-6)) - 1.0,
            2.0 * ((y_phys - float(gy0)) / max(float(gy1 - gy0), 1e-6)) - 1.0,
        ],
        dim=1,
    )


def _cascade_conditioner_predict_on_roi_inputs(
    conditioner: CascadeConditioner,
    global_bundle,
    *,
    x_scaled_roi: torch.Tensor,
    x_scaler_roi,
    target_y_scaler,
    device: str,
    hard_ground_bc: bool,
    gterr: Optional[torch.Tensor] = None,
    gfeat: Optional[torch.Tensor] = None,
):
    if conditioner is None:
        raise RuntimeError("cascade stage2 requires a stage1 conditioner")
    bg_model = conditioner.model
    bg_scalers = conditioner.scalers
    if bg_scalers is None or getattr(bg_scalers, "x_scaler_global", None) is None:
        raise RuntimeError("cascade stage1 conditioner is missing global scalers")
    gterr = terrain_tensor(global_bundle).to(device) if gterr is None else gterr
    if _is_grid_unet_conditioner(conditioner):
        x_idx = int(ROI_INPUT_COLS.index("x"))
        y_idx = int(ROI_INPUT_COLS.index("y"))
        z_idx = int(ROI_INPUT_COLS.index("z"))
        x_dev = x_scaled_roi.to(device)
        query_xyz = torch.stack(
            [
                inverse_minmax_column_from_scaled_inputs(x_dev, x_scaler_roi, x_idx),
                inverse_minmax_column_from_scaled_inputs(x_dev, x_scaler_roi, y_idx),
                inverse_minmax_column_from_scaled_inputs(x_dev, x_scaler_roi, z_idx),
            ],
            dim=1,
        ).detach().cpu().numpy()
        pred_volume = _grid_conditioner_cached_global_volume(
            conditioner,
            global_bundle,
            device=device,
            hard_ground_bc=hard_ground_bc,
            terr=gterr,
        )
        bg_phys_np = _trilinear_sample_volume(
            pred_volume,
            global_bundle.x_coords,
            global_bundle.y_coords,
            global_bundle.z_levels,
            query_xyz,
        )
        bg_phys = torch.as_tensor(bg_phys_np, dtype=torch.float32, device=device)
        bg_scaled_target = scale_outputs(bg_phys, target_y_scaler, device=str(bg_phys.device))
        return bg_scaled_target, bg_phys, gterr, None

    gfeat = bg_model.encode_global(gterr) if gfeat is None else gfeat
    x_scaled_global = _roi_xscaled_to_global_xscaled(
        x_scaled_roi,
        x_scaler_roi=x_scaler_roi,
        x_scaler_global=bg_scalers.x_scaler_global,
        device=device,
    )
    xy_global = _xy_global_from_roi_scaled(
        x_scaled_roi,
        x_scaler_roi=x_scaler_roi,
        global_bundle=global_bundle,
        device=device,
    )
    raw = bg_model.forward_global_from_encoded(gfeat, x_scaled_global, xy_global)
    _, bg_phys = _compose_global_prediction_from_raw(
        bg_model,
        raw,
        x_batch=x_scaled_global,
        x_scaler=bg_scalers.x_scaler_global,
        y_scaler=bg_scalers.y_scaler,
        hard_ground_bc=hard_ground_bc,
    )
    bg_scaled_target = scale_outputs(bg_phys, target_y_scaler, device=str(bg_phys.device))
    return bg_scaled_target, bg_phys, gterr, gfeat


def _cascade_edge_residual_loss(
    residual_scaled: torch.Tensor,
    *,
    x_batch: torch.Tensor,
    x_scaler,
    input_cols: list[str],
    bounds: tuple[float, float, float, float, float, float],
    device: str,
) -> torch.Tensor:
    weight = float(CASCADE_EDGE_WEIGHT)
    if weight <= 0.0:
        return residual_scaled.sum() * 0.0
    cols = list(input_cols)
    x = inverse_minmax_column_from_scaled_inputs(x_batch.to(device), x_scaler, int(cols.index("x")))
    y = inverse_minmax_column_from_scaled_inputs(x_batch.to(device), x_scaler, int(cols.index("y")))
    z = inverse_minmax_column_from_scaled_inputs(x_batch.to(device), x_scaler, int(cols.index("z")))
    x0, x1, y0, y1, z0, z1 = [float(v) for v in bounds]
    dx = torch.minimum(x - x0, x1 - x)
    dy = torch.minimum(y - y0, y1 - y)
    dz = torch.minimum(z - z0, z1 - z)
    wx = 1.0 - torch.clamp(dx / max(float(CASCADE_EDGE_BAND_XY_M), 1e-6), min=0.0, max=1.0)
    wy = 1.0 - torch.clamp(dy / max(float(CASCADE_EDGE_BAND_XY_M), 1e-6), min=0.0, max=1.0)
    wz = 1.0 - torch.clamp(dz / max(float(CASCADE_EDGE_BAND_Z_M), 1e-6), min=0.0, max=1.0)
    edge_w = torch.maximum(torch.maximum(wx, wy), wz)
    edge_mask = edge_w > 0.0
    if not bool(edge_mask.any().item()):
        return residual_scaled.sum() * 0.0
    return float(weight) * torch.mean((residual_scaled[edge_mask] ** 2) * edge_w[edge_mask].unsqueeze(1))


def _predict_global_boundary(
    model,
    bundle,
    batch,
    scalers,
    *,
    device: str,
    hard_ground_bc: bool,
    terr: Optional[torch.Tensor] = None,
    gfeat: Optional[torch.Tensor] = None,
):
    terr = terrain_tensor(bundle).to(device) if terr is None else terr
    gfeat = model.encode_global(terr) if gfeat is None else gfeat
    raw = model.forward_global_from_encoded(gfeat, _to_device(batch.x_scaled, device), _to_device(batch.xy_local, device))
    _, pred_phys = _compose_global_prediction_from_raw(
        model,
        raw,
        x_batch=_to_device(batch.x_scaled, device),
        x_scaler=scalers.x_scaler_global,
        y_scaler=scalers.y_scaler,
        hard_ground_bc=hard_ground_bc,
    )
    return pred_phys


def _grid_patch_pred_scaled(
    model,
    patch,
    *,
    device: str,
    x_scaler,
    input_cols: list[str],
    y_scaler,
    hard_ground_bc: bool,
    roi: bool,
    terrain_context_2d: Optional[torch.Tensor] = None,
):
    x_volume = patch.x_volume_scaled.unsqueeze(0).to(device)
    if roi:
        raw_volume = model.forward_roi_grid(x_volume, terrain_context_2d=terrain_context_2d)
    else:
        raw_volume = model.forward_global_grid(x_volume, terrain_context_2d=terrain_context_2d)
    raw_flat = raw_volume.permute(0, 2, 3, 4, 1).reshape(-1, raw_volume.shape[1])
    pred_scaled, pred_phys = apply_output_constraints_from_scaled_inputs(
        raw_flat,
        x_batch=patch.x_scaled.to(device),
        x_scaler=x_scaler,
        input_cols=input_cols,
        y_scaler=y_scaler,
        hard_ground_bc=hard_ground_bc,
    )
    return pred_scaled, pred_phys


def _cascade_grid_patch_pred_scaled(
    model,
    conditioner: CascadeConditioner,
    global_bundle,
    patch,
    scalers,
    *,
    device: str,
    hard_ground_bc: bool,
    bgterr: Optional[torch.Tensor] = None,
    bgfeat: Optional[torch.Tensor] = None,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
):
    with torch.no_grad():
        with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
            bg_scaled_flat, _, bgterr, bgfeat = _cascade_conditioner_predict_on_roi_inputs(
                conditioner,
                global_bundle,
                x_scaled_roi=patch.x_scaled.to(device),
                x_scaler_roi=scalers.x_scaler_roi,
                target_y_scaler=scalers.y_scaler,
                device=device,
                hard_ground_bc=hard_ground_bc,
                gterr=bgterr,
                gfeat=bgfeat,
            )
    px, py, pz = patch.shape
    bg_volume = bg_scaled_flat.view(px, py, pz, -1).permute(3, 0, 1, 2).contiguous().unsqueeze(0)
    x_volume = patch.x_volume_scaled.unsqueeze(0).to(device)
    with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
        raw_resid_volume = model.forward_roi_grid(x_volume, bg_volume.to(device))
        raw_resid_flat = raw_resid_volume.permute(0, 2, 3, 4, 1).reshape(-1, raw_resid_volume.shape[1])
        pred_scaled, pred_phys = apply_output_constraints_from_scaled_inputs(
            raw_resid_flat + bg_scaled_flat,
            x_batch=patch.x_scaled.to(device),
            x_scaler=scalers.x_scaler_roi,
            input_cols=ROI_INPUT_COLS,
            y_scaler=scalers.y_scaler,
            hard_ground_bc=hard_ground_bc,
        )
    return pred_scaled, pred_phys, raw_resid_flat, bgterr, bgfeat


def _patch_valid_flat_mask(patch, pred_scaled: torch.Tensor, y_scaled: torch.Tensor, *, device: str) -> torch.Tensor:
    valid = patch.mask.reshape(-1).to(device) > float(FD_FLUID_MASK_THRESHOLD)
    valid = valid & torch.isfinite(y_scaled).all(dim=1) & torch.isfinite(pred_scaled).all(dim=1)
    return valid


def _grid_patch_terrain_context_2d(
    model,
    bundle,
    patch,
    *,
    device: str,
    terr: Optional[torch.Tensor] = None,
    feat2d: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not bool(getattr(model, 'uses_grid_terrain_context', False)):
        return None, terr, feat2d
    terr = terrain_tensor(bundle).to(device) if terr is None else terr
    feat2d = model.encode_grid_terrain_context(terr) if feat2d is None else feat2d
    if feat2d is None:
        return None, terr, feat2d
    i0, j0, _ = patch.origin
    px, py, _ = patch.shape
    patch2d = feat2d[:, :, j0:j0 + py, i0:i0 + px].permute(0, 1, 3, 2).contiguous()
    return patch2d, terr, feat2d


def _predict_global_grid_unet_volume(
    model,
    bundle,
    scalers,
    *,
    device: str,
    hard_ground_bc: bool,
    patch_shape: tuple[int, int, int],
    terr: Optional[torch.Tensor] = None,
    ctx2d: Optional[torch.Tensor] = None,
) -> np.ndarray:
    pred_acc = np.zeros(bundle.flow.shape, dtype=np.float32)
    weight_acc = np.zeros(bundle.flow.shape[:3], dtype=np.float32)
    if bool(getattr(model, 'uses_grid_terrain_context', False)):
        terr = terrain_tensor(bundle).to(device) if terr is None else terr
        ctx2d = model.encode_grid_terrain_context(terr) if ctx2d is None else ctx2d
    with torch.no_grad():
        for i0, j0, k0, shape in _iter_case_tiles(bundle, patch_shape):
            patch = extract_patch_batch(
                bundle,
                x_scaler=scalers.x_scaler_global,
                y_scaler=scalers.y_scaler,
                i0=i0,
                j0=j0,
                k0=k0,
                patch_shape=shape,
                include_grid_unet_context=True,
            )
            ctx_patch, terr, ctx2d = _grid_patch_terrain_context_2d(
                model,
                bundle,
                patch,
                device=device,
                terr=terr,
                feat2d=ctx2d,
            )
            pred_flat_phys = _grid_patch_pred_scaled(
                model,
                patch,
                device=device,
                x_scaler=scalers.x_scaler_global,
                input_cols=GLOBAL_INPUT_COLS,
                y_scaler=scalers.y_scaler,
                hard_ground_bc=hard_ground_bc,
                roi=False,
                terrain_context_2d=ctx_patch,
            )[1].float().cpu().numpy()
            sx, sy, sz = shape
            pred_patch = pred_flat_phys.reshape(sx, sy, sz, -1)
            pred_acc[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz, :] += pred_patch
            weight_acc[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz] += 1.0
    pred_flow = pred_acc / np.maximum(weight_acc[..., None], 1.0e-6)
    pred_flow[weight_acc <= 0.0] = np.nan
    return pred_flow.astype(np.float32, copy=False)


def _grid_conditioner_cached_global_volume(
    conditioner: CascadeConditioner,
    global_bundle,
    *,
    device: str,
    hard_ground_bc: bool,
    terr: Optional[torch.Tensor] = None,
) -> np.ndarray:
    if conditioner.grid_global_cache is None:
        conditioner.grid_global_cache = OrderedDict()
    key = str(global_bundle.name)
    cache = conditioner.grid_global_cache
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    pred_flow = _predict_global_grid_unet_volume(
        conditioner.model,
        global_bundle,
        conditioner.scalers,
        device=device,
        hard_ground_bc=hard_ground_bc,
        patch_shape=_conditioner_global_patch_shape(conditioner),
        terr=terr,
    )
    cache[key] = pred_flow
    # Structure-conditioner global grids are small, but keep a bounded LRU cache
    # so long evaluations cannot grow without limit.
    while len(cache) > 128:
        cache.popitem(last=False)
    return pred_flow


def _trilinear_sample_volume(
    volume: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_levels: np.ndarray,
    query_xyz: np.ndarray,
) -> np.ndarray:
    if query_xyz.size == 0:
        return np.empty((0, volume.shape[-1]), dtype=np.float32)

    def _axis_indices(coords: np.ndarray, q: np.ndarray):
        coords = np.asarray(coords, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if coords.size <= 1:
            zeros = np.zeros_like(q, dtype=np.int64)
            return zeros, zeros, np.zeros_like(q, dtype=np.float32)
        q_clip = np.clip(q, float(coords[0]), float(coords[-1]))
        lo = np.searchsorted(coords, q_clip, side='right') - 1
        lo = np.clip(lo, 0, coords.size - 2).astype(np.int64)
        hi = lo + 1
        den = np.maximum(coords[hi] - coords[lo], 1.0e-12)
        t = ((q_clip - coords[lo]) / den).astype(np.float32)
        return lo, hi, t

    q = np.asarray(query_xyz, dtype=np.float64)
    i0, i1, tx = _axis_indices(x_coords, q[:, 0])
    j0, j1, ty = _axis_indices(y_coords, q[:, 1])
    k0, k1, tz = _axis_indices(z_levels, q[:, 2])
    tx = tx[:, None]
    ty = ty[:, None]
    tz = tz[:, None]

    c000 = volume[i0, j0, k0]
    c100 = volume[i1, j0, k0]
    c010 = volume[i0, j1, k0]
    c110 = volume[i1, j1, k0]
    c001 = volume[i0, j0, k1]
    c101 = volume[i1, j0, k1]
    c011 = volume[i0, j1, k1]
    c111 = volume[i1, j1, k1]

    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    out = c0 * (1.0 - tz) + c1 * tz
    return out.astype(np.float32, copy=False)


def _supervised_patch_loss_from_pred(
    pred_scaled: torch.Tensor,
    patch,
    *,
    device: str,
    x_scaler,
    input_cols: list[str],
    y_scaler,
    loss_mode: str,
    charbonnier_eps: float,
    p_weight: float = DATA_P_WEIGHT,
):
    valid = (patch.mask.to(device) > 0.5).reshape(-1)
    y_true = patch.y_scaled.to(device)
    finite = torch.isfinite(y_true).all(dim=1)
    keep = valid & finite
    if not bool(keep.any().item()):
        return pred_scaled.sum() * 0.0
    return supervised_data_loss_from_pred(
        pred_scaled[keep],
        y_true[keep],
        x_batch=patch.x_scaled.to(device)[keep],
        x_scaler=x_scaler,
        input_cols=input_cols,
        y_scaler=y_scaler,
        p_weight=float(p_weight),
        loss_mode=loss_mode,
        charbonnier_eps=charbonnier_eps,
    )


def _grid_supervised_patch_count(points_per_domain: int, patch_shape: tuple[int, int, int]) -> int:
    patch_pts = int(np.prod(np.asarray(patch_shape, dtype=np.int64), dtype=np.int64))
    if patch_pts <= 0:
        return 1
    return max(1, int((int(points_per_domain) + patch_pts - 1) // patch_pts))


def _grid_inlet_targets(bundle, ii: np.ndarray, jj: np.ndarray, kk: np.ndarray) -> np.ndarray:
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


def _tile_starts(size: int, tile: int, *, overlap_fraction: float = 0.0) -> list[int]:
    size = int(size)
    tile = max(1, int(tile))
    if overlap_fraction <= 0.0:
        starts: list[int] = []
        pos = 0
        while pos < size:
            starts.append(pos)
            pos += tile
        return starts
    stride = max(1, int(round(tile * (1.0 - float(overlap_fraction)))))
    last = max(size - tile, 0)
    starts = list(range(0, last + 1, stride))
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(int(v) for v in starts))


def _iter_case_tiles(bundle, patch_shape: tuple[int, int, int], *, overlap_fraction: float = 0.0):
    nx, ny, nz = bundle.flow.shape[:3]
    px = max(1, min(int(patch_shape[0]), nx))
    py = max(1, min(int(patch_shape[1]), ny))
    pz = max(1, min(int(patch_shape[2]), nz))
    for i0 in _tile_starts(nx, px, overlap_fraction=overlap_fraction):
        for j0 in _tile_starts(ny, py, overlap_fraction=overlap_fraction):
            for k0 in _tile_starts(nz, pz, overlap_fraction=overlap_fraction):
                sx = min(px, nx - i0)
                sy = min(py, ny - j0)
                sz = min(pz, nz - k0)
                yield int(i0), int(j0), int(k0), (int(sx), int(sy), int(sz))


def _blend_window_1d(n: int) -> np.ndarray:
    n = int(max(1, n))
    if n == 1:
        return np.ones((1,), dtype=np.float32)
    w = np.hanning(n).astype(np.float32)
    return np.clip(w, 1.0e-3, None)


def _blend_window_3d(shape: tuple[int, int, int]) -> np.ndarray:
    wx = _blend_window_1d(shape[0])[:, None, None]
    wy = _blend_window_1d(shape[1])[None, :, None]
    wz = _blend_window_1d(shape[2])[None, None, :]
    return (wx * wy * wz).astype(np.float32)


def _sample_global_boundary_patch(bundle, *, x_scaler, y_scaler, face: str, rng: np.random.Generator):
    nx, ny, nz = bundle.flow.shape[:3]
    px = max(1, min(int(GLOBAL_PATCH_SHAPE[0]), nx))
    py = max(1, min(int(GLOBAL_PATCH_SHAPE[1]), ny))
    pz = max(1, min(int(GLOBAL_PATCH_SHAPE[2]), nz))

    def _rand_start(size: int, patch_size: int) -> int:
        return 0 if size <= patch_size else int(rng.integers(0, size - patch_size + 1))

    if face == 'inlet':
        i0, j0, k0 = 0, _rand_start(ny, py), _rand_start(nz, pz)
        face_local = np.meshgrid(
            np.array([0], dtype=np.int64),
            np.arange(py, dtype=np.int64),
            np.arange(pz, dtype=np.int64),
            indexing='ij',
        )
        normals = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        p_target = None
        build_u_target = True
    elif face == 'outlet':
        i0, j0, k0 = nx - px, _rand_start(ny, py), _rand_start(nz, pz)
        face_local = np.meshgrid(
            np.array([px - 1], dtype=np.int64),
            np.arange(py, dtype=np.int64),
            np.arange(pz, dtype=np.int64),
            indexing='ij',
        )
        normals = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        p_target = 0.0
        build_u_target = False
    elif face == 'side_lo':
        i0, j0, k0 = _rand_start(nx, px), 0, _rand_start(nz, pz)
        face_local = np.meshgrid(
            np.arange(px, dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.arange(pz, dtype=np.int64),
            indexing='ij',
        )
        normals = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        p_target = None
        build_u_target = False
    elif face == 'side_hi':
        i0, j0, k0 = _rand_start(nx, px), ny - py, _rand_start(nz, pz)
        face_local = np.meshgrid(
            np.arange(px, dtype=np.int64),
            np.array([py - 1], dtype=np.int64),
            np.arange(pz, dtype=np.int64),
            indexing='ij',
        )
        normals = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        p_target = None
        build_u_target = False
    elif face == 'top':
        i0, j0, k0 = _rand_start(nx, px), _rand_start(ny, py), nz - pz
        face_local = np.meshgrid(
            np.arange(px, dtype=np.int64),
            np.arange(py, dtype=np.int64),
            np.array([pz - 1], dtype=np.int64),
            indexing='ij',
        )
        normals = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        p_target = None
        build_u_target = False
    else:
        raise ValueError(face)

    patch = extract_patch_batch(
        bundle,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        i0=i0,
        j0=j0,
        k0=k0,
        patch_shape=(px, py, pz),
        include_grid_unet_context=True,
    )
    ii_local, jj_local, kk_local = [v.reshape(-1) for v in face_local]
    ii = ii_local + int(i0)
    jj = jj_local + int(j0)
    kk = kk_local + int(k0)
    valid = (bundle.is_fluid[ii, jj, kk] > 0.5) & np.isfinite(bundle.flow[ii, jj, kk, :]).all(axis=1)
    if not np.any(valid):
        return patch, None, None
    ii_local = ii_local[valid]
    jj_local = jj_local[valid]
    kk_local = kk_local[valid]
    ii = ii[valid]
    jj = jj[valid]
    kk = kk[valid]
    flat_idx = np.ravel_multi_index((ii_local, jj_local, kk_local), dims=patch.shape)
    batch = BoundaryBatch(
        x_scaled=patch.x_scaled[flat_idx],
        xy_local=torch.empty((len(flat_idx), 2), dtype=torch.float32),
        normals=torch.as_tensor(np.tile(normals.reshape(1, 3), (len(flat_idx), 1)), dtype=torch.float32),
        u_target=None if not build_u_target else torch.as_tensor(_grid_inlet_targets(bundle, ii, jj, kk), dtype=torch.float32),
        p_target=None if p_target is None else torch.full((len(flat_idx),), float(p_target), dtype=torch.float32),
        u_scale=float(bundle.uref),
        p_scale=float(bundle.uref * bundle.uref),
    )
    return patch, batch, torch.as_tensor(flat_idx, dtype=torch.long)


def _evaluate_case_global_grid_unet(
    model,
    bundle,
    scalers,
    *,
    device: str,
    hard_ground_bc: bool,
    return_pred_flow: bool = False,
    patch_shape: Optional[tuple[int, int, int]] = None,
):
    tile_shape = tuple(int(v) for v in (patch_shape or GLOBAL_PATCH_SHAPE))
    gterr = None
    gctx2d = None
    if bool(getattr(model, 'uses_grid_terrain_context', False)):
        with torch.no_grad():
            gterr = terrain_tensor(bundle).to(device)
            gctx2d = model.encode_grid_terrain_context(gterr)
    if return_pred_flow:
        pred_acc = np.zeros(bundle.flow.shape, dtype=np.float32)
        weight_acc = np.zeros(bundle.flow.shape[:3], dtype=np.float32)
        for i0, j0, k0, shape in _iter_case_tiles(bundle, tile_shape, overlap_fraction=0.5):
            patch = extract_patch_batch(
                bundle,
                x_scaler=scalers.x_scaler_global,
                y_scaler=scalers.y_scaler,
                i0=i0,
                j0=j0,
                k0=k0,
                patch_shape=shape,
                include_grid_unet_context=True,
            )
            with torch.no_grad():
                ctx_patch, gterr, gctx2d = _grid_patch_terrain_context_2d(
                    model,
                    bundle,
                    patch,
                    device=device,
                    terr=gterr,
                    feat2d=gctx2d,
                )
                pred_flat_phys = _grid_patch_pred_scaled(
                    model,
                    patch,
                    device=device,
                    x_scaler=scalers.x_scaler_global,
                    input_cols=GLOBAL_INPUT_COLS,
                    y_scaler=scalers.y_scaler,
                    hard_ground_bc=hard_ground_bc,
                    roi=False,
                    terrain_context_2d=ctx_patch,
                )[1].cpu().numpy()
            pred_patch = pred_flat_phys.reshape(shape[0], shape[1], shape[2], -1)
            blend = _blend_window_3d(shape)
            pred_acc[i0:i0 + shape[0], j0:j0 + shape[1], k0:k0 + shape[2], :] += pred_patch * blend[..., None]
            weight_acc[i0:i0 + shape[0], j0:j0 + shape[1], k0:k0 + shape[2]] += blend
        pred_flow = pred_acc / np.maximum(weight_acc[..., None], 1.0e-6)
        pred_flow[weight_acc <= 0.0] = np.nan
        ii, jj, kk = np.meshgrid(
            np.arange(bundle.flow.shape[0], dtype=np.int64),
            np.arange(bundle.flow.shape[1], dtype=np.int64),
            np.arange(bundle.flow.shape[2], dtype=np.int64),
            indexing='ij',
        )
        valid = (bundle.is_fluid > 0.5) & np.isfinite(bundle.flow).all(axis=-1) & np.isfinite(pred_flow).all(axis=-1)
        i_lo, i_hi = bundle.valid_i_range
        j_lo, j_hi = bundle.valid_j_range
        valid &= (ii >= i_lo) & (ii < i_hi) & (jj >= j_lo) & (jj < j_hi) & (kk < int(bundle.valid_k_max))
        metrics_acc = init_regression_metrics_accumulator()
        subset_accs = init_subset_accumulators(['near_ground'])
        if np.any(valid):
            y_np = bundle.flow[valid]
            pred_np = pred_flow[valid]
            update_regression_metrics_accumulator(metrics_acc, y_np, pred_np)
            z_rel_raw = bundle_z_rel_at(bundle, ii[valid], jj[valid], kk[valid]).astype(np.float32)
            masks = build_eval_subset_masks(z_rel_raw, None)
            update_subset_accumulators(subset_accs, y_np, pred_np, masks)
        case_metrics = finalize_regression_metrics(metrics_acc, uref=float(bundle.uref))
        case_metrics['subsets'] = finalize_subset_metrics(subset_accs, uref=float(bundle.uref))
        return case_metrics, pred_flow

    metrics_acc = init_regression_metrics_accumulator()
    subset_accs = init_subset_accumulators(['near_ground'])
    for i0, j0, k0, shape in _iter_case_tiles(bundle, tile_shape):
        patch = extract_patch_batch(
            bundle,
            x_scaler=scalers.x_scaler_global,
            y_scaler=scalers.y_scaler,
            i0=i0,
            j0=j0,
            k0=k0,
            patch_shape=shape,
            include_grid_unet_context=True,
        )
        with torch.no_grad():
            ctx_patch, gterr, gctx2d = _grid_patch_terrain_context_2d(
                model,
                bundle,
                patch,
                device=device,
                terr=gterr,
                feat2d=gctx2d,
            )
            pred_flat_phys = _grid_patch_pred_scaled(
                model,
                patch,
                device=device,
                x_scaler=scalers.x_scaler_global,
                input_cols=GLOBAL_INPUT_COLS,
                y_scaler=scalers.y_scaler,
                hard_ground_bc=hard_ground_bc,
                roi=False,
                terrain_context_2d=ctx_patch,
            )[1].cpu().numpy()
        sx, sy, sz = shape
        pred_patch = pred_flat_phys.reshape(sx, sy, sz, -1)
        ii, jj, kk = np.meshgrid(
            np.arange(i0, i0 + sx, dtype=np.int64),
            np.arange(j0, j0 + sy, dtype=np.int64),
            np.arange(k0, k0 + sz, dtype=np.int64),
            indexing='ij',
        )
        valid = (bundle.is_fluid[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz] > 0.5) & np.isfinite(bundle.flow[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz, :]).all(axis=-1)
        i_lo, i_hi = bundle.valid_i_range
        j_lo, j_hi = bundle.valid_j_range
        valid &= (ii >= i_lo) & (ii < i_hi) & (jj >= j_lo) & (jj < j_hi) & (kk < int(bundle.valid_k_max))
        if np.any(valid):
            y_np = bundle.flow[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz, :][valid]
            pred_np = pred_patch[valid]
            update_regression_metrics_accumulator(metrics_acc, y_np, pred_np)
            z_rel_raw = bundle_z_rel_at(bundle, ii[valid], jj[valid], kk[valid]).astype(np.float32)
            masks = build_eval_subset_masks(z_rel_raw, None)
            update_subset_accumulators(subset_accs, y_np, pred_np, masks)
    case_metrics = finalize_regression_metrics(metrics_acc, uref=float(bundle.uref))
    case_metrics['subsets'] = finalize_subset_metrics(subset_accs, uref=float(bundle.uref))
    return case_metrics, None


def _evaluate_case_roi_grid_unet(
    model,
    global_bundle,
    roi_bundle,
    scalers,
    *,
    device: str,
    hard_ground_bc: bool,
    return_pred_flow: bool = False,
):
    rterr = None
    rctx2d = None
    if bool(getattr(model, 'uses_grid_terrain_context', False)):
        with torch.no_grad():
            rterr = terrain_tensor(roi_bundle).to(device)
            rctx2d = model.encode_grid_terrain_context(rterr)
    if return_pred_flow:
        pred_acc = np.zeros(roi_bundle.flow.shape, dtype=np.float32)
        weight_acc = np.zeros(roi_bundle.flow.shape[:3], dtype=np.float32)
        for i0, j0, k0, shape in _iter_case_tiles(roi_bundle, ROI_PATCH_SHAPE, overlap_fraction=0.5):
            patch = extract_patch_batch(
                roi_bundle,
                x_scaler=scalers.x_scaler_roi,
                y_scaler=scalers.y_scaler,
                i0=i0,
                j0=j0,
                k0=k0,
                patch_shape=shape,
                parent_global=global_bundle,
                include_grid_unet_context=True,
            )
            with torch.no_grad():
                ctx_patch, rterr, rctx2d = _grid_patch_terrain_context_2d(
                    model,
                    roi_bundle,
                    patch,
                    device=device,
                    terr=rterr,
                    feat2d=rctx2d,
                )
                pred_flat_phys = _grid_patch_pred_scaled(
                    model,
                    patch,
                    device=device,
                    x_scaler=scalers.x_scaler_roi,
                    input_cols=ROI_INPUT_COLS,
                    y_scaler=scalers.y_scaler,
                    hard_ground_bc=hard_ground_bc,
                    roi=True,
                    terrain_context_2d=ctx_patch,
                )[1].cpu().numpy()
            pred_patch = pred_flat_phys.reshape(shape[0], shape[1], shape[2], -1)
            blend = _blend_window_3d(shape)
            pred_acc[i0:i0 + shape[0], j0:j0 + shape[1], k0:k0 + shape[2], :] += pred_patch * blend[..., None]
            weight_acc[i0:i0 + shape[0], j0:j0 + shape[1], k0:k0 + shape[2]] += blend
        pred_flow = pred_acc / np.maximum(weight_acc[..., None], 1.0e-6)
        pred_flow[weight_acc <= 0.0] = np.nan
        ii, jj, kk = np.meshgrid(
            np.arange(roi_bundle.flow.shape[0], dtype=np.int64),
            np.arange(roi_bundle.flow.shape[1], dtype=np.int64),
            np.arange(roi_bundle.flow.shape[2], dtype=np.int64),
            indexing='ij',
        )
        valid = (roi_bundle.is_fluid > 0.5) & np.isfinite(roi_bundle.flow).all(axis=-1) & np.isfinite(pred_flow).all(axis=-1)
        i_lo, i_hi = roi_bundle.valid_i_range
        j_lo, j_hi = roi_bundle.valid_j_range
        valid &= (ii >= i_lo) & (ii < i_hi) & (jj >= j_lo) & (jj < j_hi) & (kk < int(roi_bundle.valid_k_max))
        metrics_acc = init_regression_metrics_accumulator()
        subset_accs = init_subset_accumulators(['near_wall', 'near_ground', 'near_ground_near_wall'])
        if np.any(valid):
            y_np = roi_bundle.flow[valid]
            pred_np = pred_flow[valid]
            update_regression_metrics_accumulator(metrics_acc, y_np, pred_np)
            z_rel_raw = bundle_z_rel_at(roi_bundle, ii[valid], jj[valid], kk[valid]).astype(np.float32)
            phi_wall_raw = roi_bundle.phi_wall[valid] if roi_bundle.phi_wall is not None else None
            masks = build_eval_subset_masks(z_rel_raw, phi_wall_raw)
            update_subset_accumulators(subset_accs, y_np, pred_np, masks)
        case_metrics = finalize_regression_metrics(metrics_acc, uref=float(roi_bundle.uref))
        case_metrics['subsets'] = finalize_subset_metrics(subset_accs, uref=float(roi_bundle.uref))
        return case_metrics, pred_flow

    metrics_acc = init_regression_metrics_accumulator()
    subset_accs = init_subset_accumulators(['near_wall', 'near_ground', 'near_ground_near_wall'])
    for i0, j0, k0, shape in _iter_case_tiles(roi_bundle, ROI_PATCH_SHAPE):
        patch = extract_patch_batch(
            roi_bundle,
            x_scaler=scalers.x_scaler_roi,
            y_scaler=scalers.y_scaler,
            i0=i0,
            j0=j0,
            k0=k0,
            patch_shape=shape,
            parent_global=global_bundle,
            include_grid_unet_context=True,
        )
        with torch.no_grad():
            ctx_patch, rterr, rctx2d = _grid_patch_terrain_context_2d(
                model,
                roi_bundle,
                patch,
                device=device,
                terr=rterr,
                feat2d=rctx2d,
            )
            pred_flat_phys = _grid_patch_pred_scaled(
                model,
                patch,
                device=device,
                x_scaler=scalers.x_scaler_roi,
                input_cols=ROI_INPUT_COLS,
                y_scaler=scalers.y_scaler,
                hard_ground_bc=hard_ground_bc,
                roi=True,
                terrain_context_2d=ctx_patch,
            )[1].cpu().numpy()
        sx, sy, sz = shape
        pred_patch = pred_flat_phys.reshape(sx, sy, sz, -1)
        ii, jj, kk = np.meshgrid(
            np.arange(i0, i0 + sx, dtype=np.int64),
            np.arange(j0, j0 + sy, dtype=np.int64),
            np.arange(k0, k0 + sz, dtype=np.int64),
            indexing='ij',
        )
        valid = (roi_bundle.is_fluid[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz] > 0.5) & np.isfinite(roi_bundle.flow[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz, :]).all(axis=-1)
        i_lo, i_hi = roi_bundle.valid_i_range
        j_lo, j_hi = roi_bundle.valid_j_range
        valid &= (ii >= i_lo) & (ii < i_hi) & (jj >= j_lo) & (jj < j_hi) & (kk < int(roi_bundle.valid_k_max))
        if np.any(valid):
            y_np = roi_bundle.flow[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz, :][valid]
            pred_np = pred_patch[valid]
            update_regression_metrics_accumulator(metrics_acc, y_np, pred_np)
            z_rel_raw = bundle_z_rel_at(roi_bundle, ii[valid], jj[valid], kk[valid]).astype(np.float32)
            phi_wall_raw = roi_bundle.phi_wall[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz][valid] if roi_bundle.phi_wall is not None else None
            masks = build_eval_subset_masks(z_rel_raw, phi_wall_raw)
            update_subset_accumulators(subset_accs, y_np, pred_np, masks)
    case_metrics = finalize_regression_metrics(metrics_acc, uref=float(roi_bundle.uref))
    case_metrics['subsets'] = finalize_subset_metrics(subset_accs, uref=float(roi_bundle.uref))
    return case_metrics, None


def _evaluate_case_physics_grid_unet(
    model,
    bundle,
    scalers,
    *,
    device: str,
    hard_ground_bc: bool,
    n_patches: int,
    parent_global=None,
) -> dict[str, float]:
    if bundle_uses_terrain_following_z(bundle):
        return _tf_skipped_physics_metrics()
    if int(n_patches) <= 0:
        return {
            'physics_patch_count': 0,
            'physics_div_rms': float('nan'),
            'physics_mom_rms_constant': float('nan'),
            'physics_mom_rms_nut': float('nan'),
        }
    rng = np.random.default_rng(_stable_patch_seed(bundle.name, bundle.roi_name or bundle.kind, 'eval_phys_grid'))
    terr = None
    feat2d = None
    if bool(getattr(model, 'uses_grid_terrain_context', False)):
        with torch.no_grad():
            terr = terrain_tensor(bundle).to(device)
            feat2d = model.encode_grid_terrain_context(terr)
    div_vals: list[float] = []
    mom_const_vals: list[float] = []
    mom_nut_vals: list[float] = []
    for _ in range(int(n_patches)):
        if str(bundle.kind) == 'global':
            patch = sample_patch_batch(
                bundle,
                x_scaler=scalers.x_scaler_global,
                y_scaler=scalers.y_scaler,
                patch_shape=GLOBAL_PATCH_SHAPE,
                rng=rng,
                near_ground_prob=PATCH_NEAR_GROUND_PROB,
                include_grid_unet_context=True,
            )
            with torch.no_grad():
                ctx_patch, terr, feat2d = _grid_patch_terrain_context_2d(
                    model,
                    bundle,
                    patch,
                    device=device,
                    terr=terr,
                    feat2d=feat2d,
                )
                patch_pred = _grid_patch_pred_scaled(
                    model,
                    patch,
                    device=device,
                    x_scaler=scalers.x_scaler_global,
                    input_cols=GLOBAL_INPUT_COLS,
                    y_scaler=scalers.y_scaler,
                    hard_ground_bc=hard_ground_bc,
                    roi=False,
                    terrain_context_2d=ctx_patch,
                )[0]
        else:
            patch = sample_patch_batch(
                bundle,
                x_scaler=scalers.x_scaler_roi,
                y_scaler=scalers.y_scaler,
                patch_shape=ROI_PATCH_SHAPE,
                rng=rng,
                near_ground_prob=PATCH_NEAR_GROUND_PROB,
                parent_global=parent_global,
                include_grid_unet_context=True,
            )
            with torch.no_grad():
                ctx_patch, terr, feat2d = _grid_patch_terrain_context_2d(
                    model,
                    bundle,
                    patch,
                    device=device,
                    terr=terr,
                    feat2d=feat2d,
                )
                patch_pred = _grid_patch_pred_scaled(
                    model,
                    patch,
                    device=device,
                    x_scaler=scalers.x_scaler_roi,
                    input_cols=ROI_INPUT_COLS,
                    y_scaler=scalers.y_scaler,
                    hard_ground_bc=hard_ground_bc,
                    roi=True,
                    terrain_context_2d=ctx_patch,
                )[0]
        phys_const = compute_patch_physics_losses_from_pred(
            patch_pred,
            patch,
            scalers.y_scaler,
            device=device,
            momentum_loss_mode='constant',
        )
        div_vals.append(float(phys_const['div_rms'].detach().cpu().item()))
        mom_const_vals.append(float(phys_const['mom_rms'].detach().cpu().item()))
        if patch.nut is not None:
            phys_nut = compute_patch_physics_losses_from_pred(
                patch_pred,
                patch,
                scalers.y_scaler,
                device=device,
                momentum_loss_mode='nut',
            )
            mom_nut_vals.append(float(phys_nut['mom_rms'].detach().cpu().item()))
    return {
        'physics_patch_count': int(n_patches),
        'physics_div_rms': float(np.mean(div_vals)) if div_vals else float('nan'),
        'physics_mom_rms_constant': float(np.mean(mom_const_vals)) if mom_const_vals else float('nan'),
        'physics_mom_rms_nut': float(np.mean(mom_nut_vals)) if mom_nut_vals else float('nan'),
    }


def _mean_metric(case_metrics: dict[str, dict], key: str) -> float:
    vals = []
    for metrics in case_metrics.values():
        v = float(metrics.get(key, float('nan')))
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float('nan')


def _evaluate_case_physics(
    model,
    bundle,
    scalers,
    *,
    device: str,
    hard_ground_bc: bool,
    n_patches: int,
    parent_global=None,
    gterr: Optional[torch.Tensor] = None,
    rterr: Optional[torch.Tensor] = None,
    sterr: Optional[torch.Tensor] = None,
    gfeat: Optional[torch.Tensor] = None,
    rfeat: Optional[torch.Tensor] = None,
    sfeat: Optional[torch.Tensor] = None,
    use_amp: bool,
    amp_dtype: str,
) -> dict[str, float]:
    if bundle_uses_terrain_following_z(bundle):
        return _tf_skipped_physics_metrics()
    if int(n_patches) <= 0:
        return {
            'physics_patch_count': 0,
            'physics_div_rms': float('nan'),
            'physics_mom_rms_constant': float('nan'),
            'physics_mom_rms_nut': float('nan'),
        }
    if str(bundle.kind) == 'roi' and parent_global is None:
        raise ValueError('ROI physics evaluation requires parent_global.')
    rng = np.random.default_rng(_stable_patch_seed(bundle.name, bundle.roi_name or bundle.kind, 'eval_phys'))
    div_vals: list[float] = []
    mom_const_vals: list[float] = []
    mom_nut_vals: list[float] = []
    with torch.no_grad():
        for _ in range(int(n_patches)):
            if str(bundle.kind) == 'global':
                patch = sample_patch_batch(
                    bundle,
                    x_scaler=scalers.x_scaler_global,
                    y_scaler=scalers.y_scaler,
                    patch_shape=GLOBAL_PATCH_SHAPE,
                    rng=rng,
                    near_ground_prob=PATCH_NEAR_GROUND_PROB,
                )
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    patch_pred_raw = model.forward_global_from_encoded(
                        gfeat if gfeat is not None else model.encode_global(gterr if gterr is not None else terrain_tensor(bundle).to(device)),
                        patch.x_scaled.to(device),
                        patch.xy_local.to(device),
                    )
                    patch_pred, _ = _compose_global_prediction_from_raw(
                        model,
                        patch_pred_raw,
                        x_batch=patch.x_scaled.to(device),
                        x_scaler=scalers.x_scaler_global,
                        y_scaler=scalers.y_scaler,
                        hard_ground_bc=hard_ground_bc,
                    )
                    phys_const = compute_patch_physics_losses_from_pred(
                        patch_pred,
                        patch,
                        scalers.y_scaler,
                        device=device,
                        momentum_loss_mode='constant',
                    )
            else:
                patch = sample_patch_batch(
                    bundle,
                    x_scaler=scalers.x_scaler_roi,
                    y_scaler=scalers.y_scaler,
                    patch_shape=ROI_PATCH_SHAPE,
                    rng=rng,
                    near_ground_prob=PATCH_NEAR_GROUND_PROB,
                    parent_global=parent_global,
                )
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    patch_pred_raw = model.forward_roi_from_encoded(
                        gfeat if gfeat is not None else model.encode_global(gterr if gterr is not None else terrain_tensor(parent_global).to(device)),
                        rfeat if rfeat is not None else model.encode_roi(rterr if rterr is not None else terrain_tensor(bundle).to(device)),
                        patch.x_scaled.to(device),
                        patch.xy_global.to(device),
                        patch.xy_local.to(device),
                        s_feat=sfeat if sfeat is not None else model.encode_structure(sterr),
                    )
                    patch_pred, _ = apply_output_constraints_from_scaled_inputs(
                        patch_pred_raw,
                        x_batch=patch.x_scaled.to(device),
                        x_scaler=scalers.x_scaler_roi,
                        input_cols=ROI_INPUT_COLS,
                        y_scaler=scalers.y_scaler,
                        hard_ground_bc=hard_ground_bc,
                    )
                    phys_const = compute_patch_physics_losses_from_pred(
                        patch_pred,
                        patch,
                        scalers.y_scaler,
                        device=device,
                        momentum_loss_mode='constant',
                    )
            div_vals.append(float(phys_const['div_rms'].detach().cpu().item()))
            mom_const_vals.append(float(phys_const['mom_rms'].detach().cpu().item()))
            if patch.nut is not None:
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    phys_nut = compute_patch_physics_losses_from_pred(
                        patch_pred,
                        patch,
                        scalers.y_scaler,
                        device=device,
                        momentum_loss_mode='nut',
                    )
                mom_nut_vals.append(float(phys_nut['mom_rms'].detach().cpu().item()))
    return {
        'physics_patch_count': int(n_patches),
        'physics_div_rms': float(np.mean(div_vals)) if div_vals else float('nan'),
        'physics_mom_rms_constant': float(np.mean(mom_const_vals)) if mom_const_vals else float('nan'),
        'physics_mom_rms_nut': float(np.mean(mom_nut_vals)) if mom_nut_vals else float('nan'),
    }


def _evaluate_case_global(
    model,
    bundle,
    scalers,
    *,
    device: str,
    pred_batch_size: int,
    hard_ground_bc: bool,
    return_pred_flow: bool = False,
    terr: Optional[torch.Tensor] = None,
    gfeat: Optional[torch.Tensor] = None,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
):
    metrics_acc = init_regression_metrics_accumulator()
    subset_accs = init_subset_accumulators(['near_ground'])
    pred_flow = None
    if return_pred_flow:
        pred_flow = np.full(bundle.flow.shape, np.nan, dtype=np.float32)
    terr = terrain_tensor(bundle).to(device) if terr is None else terr
    gfeat = model.encode_global(terr) if gfeat is None else gfeat
    for idx, x_scaled, y_true, xy_local, z_rel_raw, _ in iter_fullgrid_predictions(
        bundle, x_scaler=scalers.x_scaler_global, chunk_size=pred_batch_size, include_phi_wall=False,
    ):
        x_dev = x_scaled.to(device)
        with torch.no_grad():
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                raw = model.forward_global_from_encoded(gfeat, x_dev, xy_local.to(device))
                _, pred_phys = _compose_global_prediction_from_raw(
                    model,
                    raw,
                    x_batch=x_dev,
                    x_scaler=scalers.x_scaler_global,
                    y_scaler=scalers.y_scaler,
                    hard_ground_bc=hard_ground_bc,
                )
                pred_np = pred_phys.cpu().numpy()
        y_np = y_true.numpy()
        update_regression_metrics_accumulator(metrics_acc, y_np, pred_np)
        masks = build_eval_subset_masks(z_rel_raw, None)
        update_subset_accumulators(subset_accs, y_np, pred_np, masks)
        if pred_flow is not None:
            pred_flow.reshape(-1, pred_flow.shape[-1])[idx] = pred_np
    case_metrics = finalize_regression_metrics(metrics_acc, uref=float(bundle.uref))
    case_metrics['subsets'] = finalize_subset_metrics(subset_accs, uref=float(bundle.uref))
    case_metrics.update(
        _evaluate_case_physics(
            model,
            bundle,
            scalers,
            device=device,
            hard_ground_bc=hard_ground_bc,
            n_patches=int(EVAL_GLOBAL_PATCHES_PER_CASE),
            gterr=terr,
            gfeat=gfeat,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    )
    return case_metrics, pred_flow


def _evaluate_case_roi(
    model,
    global_bundle,
    roi_bundle,
    scalers,
    *,
    device: str,
    pred_batch_size: int,
    hard_ground_bc: bool,
    return_pred_flow: bool = False,
    gterr: Optional[torch.Tensor] = None,
    rterr: Optional[torch.Tensor] = None,
    sterr: Optional[torch.Tensor] = None,
    gfeat: Optional[torch.Tensor] = None,
    rfeat: Optional[torch.Tensor] = None,
    sfeat: Optional[torch.Tensor] = None,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
):
    metrics_acc = init_regression_metrics_accumulator()
    subset_accs = init_subset_accumulators(['near_wall', 'near_ground', 'near_ground_near_wall'])
    pred_flow = None
    if return_pred_flow:
        pred_flow = np.full(roi_bundle.flow.shape, np.nan, dtype=np.float32)
    gterr = terrain_tensor(global_bundle).to(device) if gterr is None else gterr
    rterr = terrain_tensor(roi_bundle).to(device) if rterr is None else rterr
    gfeat = model.encode_global(gterr) if gfeat is None else gfeat
    rfeat = model.encode_roi(rterr) if rfeat is None else rfeat
    sfeat = model.encode_structure(sterr) if sfeat is None else sfeat
    x_idx = ROI_INPUT_COLS.index('x')
    y_idx = ROI_INPUT_COLS.index('y')
    x_min = float(scalers.x_scaler_roi.data_min_[x_idx])
    x_max = float(scalers.x_scaler_roi.data_max_[x_idx])
    y_min = float(scalers.x_scaler_roi.data_min_[y_idx])
    y_max = float(scalers.x_scaler_roi.data_max_[y_idx])
    gx0, gx1, gy0, gy1, _, _ = global_bundle.bounds
    for idx, x_scaled, y_true, xy_local, z_rel_raw, phi_wall_raw in iter_fullgrid_predictions(
        roi_bundle, x_scaler=scalers.x_scaler_roi, chunk_size=pred_batch_size, include_phi_wall=True,
    ):
        x_dev = x_scaled.to(device)
        x_phys = x_scaled[:, x_idx].numpy() * (x_max - x_min) + x_min
        y_phys = x_scaled[:, y_idx].numpy() * (y_max - y_min) + y_min
        xy_global = torch.as_tensor(np.stack([
            2.0 * ((x_phys - gx0) / max(gx1 - gx0, 1e-6)) - 1.0,
            2.0 * ((y_phys - gy0) / max(gy1 - gy0, 1e-6)) - 1.0,
        ], axis=1), dtype=torch.float32)
        with torch.no_grad():
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                raw = model.forward_roi_from_encoded(
                    gfeat,
                    rfeat,
                    x_dev,
                    xy_global.to(device),
                    xy_local.to(device),
                    s_feat=sfeat,
                )
                _, pred_phys = apply_output_constraints_from_scaled_inputs(
                    raw,
                    x_batch=x_dev,
                    x_scaler=scalers.x_scaler_roi,
                    input_cols=ROI_INPUT_COLS,
                    y_scaler=scalers.y_scaler,
                    hard_ground_bc=hard_ground_bc,
                )
                pred_np = pred_phys.cpu().numpy()
        y_np = y_true.numpy()
        update_regression_metrics_accumulator(metrics_acc, y_np, pred_np)
        masks = build_eval_subset_masks(z_rel_raw, phi_wall_raw)
        update_subset_accumulators(subset_accs, y_np, pred_np, masks)
        if pred_flow is not None:
            pred_flow.reshape(-1, pred_flow.shape[-1])[idx] = pred_np
    case_metrics = finalize_regression_metrics(metrics_acc, uref=float(roi_bundle.uref))
    case_metrics['subsets'] = finalize_subset_metrics(subset_accs, uref=float(roi_bundle.uref))
    case_metrics.update(
        _evaluate_case_physics(
            model,
            roi_bundle,
            scalers,
            device=device,
            hard_ground_bc=hard_ground_bc,
            n_patches=int(EVAL_ROI_PATCHES_PER_CASE),
            parent_global=global_bundle,
            gterr=gterr,
            rterr=rterr,
            sterr=sterr,
            gfeat=gfeat,
            rfeat=rfeat,
            sfeat=sfeat,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    )
    return case_metrics, pred_flow


def _evaluate_case_roi_cascade(
    model,
    conditioner: CascadeConditioner,
    global_bundle,
    roi_bundle,
    scalers,
    *,
    device: str,
    pred_batch_size: int,
    hard_ground_bc: bool,
    return_pred_flow: bool = False,
    rterr: Optional[torch.Tensor] = None,
    sterr: Optional[torch.Tensor] = None,
    rfeat: Optional[torch.Tensor] = None,
    sfeat: Optional[torch.Tensor] = None,
    bgterr: Optional[torch.Tensor] = None,
    bgfeat: Optional[torch.Tensor] = None,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
):
    if _is_cascade_grid_refiner(model):
        return _evaluate_case_roi_cascade_grid(
            model,
            conditioner,
            global_bundle,
            roi_bundle,
            scalers,
            device=device,
            hard_ground_bc=hard_ground_bc,
            return_pred_flow=return_pred_flow,
            bgterr=bgterr,
            bgfeat=bgfeat,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    metrics_acc = init_regression_metrics_accumulator()
    subset_accs = init_subset_accumulators(['near_wall', 'near_ground', 'near_ground_near_wall'])
    pred_flow = None
    if return_pred_flow:
        pred_flow = np.full(roi_bundle.flow.shape, np.nan, dtype=np.float32)
    rterr = terrain_tensor(roi_bundle).to(device) if rterr is None else rterr
    rfeat = model.encode_roi(rterr) if rfeat is None else rfeat
    sfeat = model.encode_structure(sterr) if sfeat is None else sfeat
    for idx, x_scaled, y_true, xy_local, z_rel_raw, phi_wall_raw in iter_fullgrid_predictions(
        roi_bundle, x_scaler=scalers.x_scaler_roi, chunk_size=pred_batch_size, include_phi_wall=True,
    ):
        x_dev = x_scaled.to(device)
        xy_local_dev = xy_local.to(device)
        with torch.no_grad():
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                bg_scaled, _, bgterr, bgfeat = _cascade_conditioner_predict_on_roi_inputs(
                    conditioner,
                    global_bundle,
                    x_scaled_roi=x_dev,
                    x_scaler_roi=scalers.x_scaler_roi,
                    target_y_scaler=scalers.y_scaler,
                    device=device,
                    hard_ground_bc=hard_ground_bc,
                    gterr=bgterr,
                    gfeat=bgfeat,
                )
                raw_resid_scaled = model.forward_roi_from_encoded(
                    rfeat,
                    x_dev,
                    xy_local_dev,
                    bg_scaled,
                    s_feat=sfeat,
                )
                pred_scaled, pred_phys = apply_output_constraints_from_scaled_inputs(
                    raw_resid_scaled + bg_scaled,
                    x_batch=x_dev,
                    x_scaler=scalers.x_scaler_roi,
                    input_cols=ROI_INPUT_COLS,
                    y_scaler=scalers.y_scaler,
                    hard_ground_bc=hard_ground_bc,
                )
                pred_np = pred_phys.cpu().numpy()
        y_np = y_true.numpy()
        update_regression_metrics_accumulator(metrics_acc, y_np, pred_np)
        masks = build_eval_subset_masks(z_rel_raw, phi_wall_raw)
        update_subset_accumulators(subset_accs, y_np, pred_np, masks)
        if pred_flow is not None:
            pred_flow.reshape(-1, pred_flow.shape[-1])[idx] = pred_np
    case_metrics = finalize_regression_metrics(metrics_acc, uref=float(roi_bundle.uref))
    case_metrics['subsets'] = finalize_subset_metrics(subset_accs, uref=float(roi_bundle.uref))
    case_metrics.update(
        _evaluate_case_physics_cascade_roi(
            model,
            conditioner,
            global_bundle,
            roi_bundle,
            scalers,
            device=device,
            hard_ground_bc=hard_ground_bc,
            n_patches=int(EVAL_ROI_PATCHES_PER_CASE),
            rterr=rterr,
            sterr=sterr,
            rfeat=rfeat,
            sfeat=sfeat,
            bgterr=bgterr,
            bgfeat=bgfeat,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    )
    return case_metrics, pred_flow


def _evaluate_case_roi_cascade_grid(
    model,
    conditioner: CascadeConditioner,
    global_bundle,
    roi_bundle,
    scalers,
    *,
    device: str,
    hard_ground_bc: bool,
    return_pred_flow: bool = False,
    bgterr: Optional[torch.Tensor] = None,
    bgfeat: Optional[torch.Tensor] = None,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
):
    metrics_acc = init_regression_metrics_accumulator()
    subset_accs = init_subset_accumulators(['near_wall', 'near_ground', 'near_ground_near_wall'])
    pred_flow = None
    if return_pred_flow:
        pred_acc = np.zeros(roi_bundle.flow.shape, dtype=np.float32)
        weight_acc = np.zeros(roi_bundle.flow.shape[:3], dtype=np.float32)
        tile_iter = _iter_case_tiles(roi_bundle, ROI_PATCH_SHAPE, overlap_fraction=0.5)
    else:
        tile_iter = _iter_case_tiles(roi_bundle, ROI_PATCH_SHAPE)

    for i0, j0, k0, shape in tile_iter:
        patch = extract_patch_batch(
            roi_bundle,
            x_scaler=scalers.x_scaler_roi,
            y_scaler=scalers.y_scaler,
            i0=i0,
            j0=j0,
            k0=k0,
            patch_shape=shape,
            parent_global=global_bundle,
            include_grid_unet_context=True,
        )
        with torch.no_grad():
            pred_scaled, pred_phys, _, bgterr, bgfeat = _cascade_grid_patch_pred_scaled(
                model,
                conditioner,
                global_bundle,
                patch,
                scalers,
                device=device,
                hard_ground_bc=hard_ground_bc,
                bgterr=bgterr,
                bgfeat=bgfeat,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        sx, sy, sz = shape
        pred_patch = pred_phys.detach().cpu().numpy().reshape(sx, sy, sz, -1)
        ii, jj, kk = np.meshgrid(
            np.arange(i0, i0 + sx, dtype=np.int64),
            np.arange(j0, j0 + sy, dtype=np.int64),
            np.arange(k0, k0 + sz, dtype=np.int64),
            indexing='ij',
        )
        valid = (roi_bundle.is_fluid[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz] > FD_FLUID_MASK_THRESHOLD) & np.isfinite(roi_bundle.flow[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz, :]).all(axis=-1)
        i_lo, i_hi = roi_bundle.valid_i_range
        j_lo, j_hi = roi_bundle.valid_j_range
        valid &= (ii >= i_lo) & (ii < i_hi) & (jj >= j_lo) & (jj < j_hi) & (kk < int(roi_bundle.valid_k_max))
        if np.any(valid):
            y_np = roi_bundle.flow[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz, :][valid]
            pred_np = pred_patch[valid]
            update_regression_metrics_accumulator(metrics_acc, y_np, pred_np)
            z_rel_raw = bundle_z_rel_at(roi_bundle, ii[valid], jj[valid], kk[valid]).astype(np.float32)
            phi_wall_raw = roi_bundle.phi_wall[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz][valid] if roi_bundle.phi_wall is not None else None
            masks = build_eval_subset_masks(z_rel_raw, phi_wall_raw)
            update_subset_accumulators(subset_accs, y_np, pred_np, masks)
        if return_pred_flow:
            blend = _blend_window_3d(shape)
            pred_acc[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz, :] += pred_patch * blend[..., None]
            weight_acc[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz] += blend

    if return_pred_flow:
        pred_flow = pred_acc / np.maximum(weight_acc[..., None], 1.0e-6)
        pred_flow[weight_acc <= 0.0] = np.nan
    case_metrics = finalize_regression_metrics(metrics_acc, uref=float(roi_bundle.uref))
    case_metrics['subsets'] = finalize_subset_metrics(subset_accs, uref=float(roi_bundle.uref))
    case_metrics.update(
        _evaluate_case_physics_cascade_roi(
            model,
            conditioner,
            global_bundle,
            roi_bundle,
            scalers,
            device=device,
            hard_ground_bc=hard_ground_bc,
            n_patches=int(EVAL_ROI_PATCHES_PER_CASE),
            rterr=None,
            sterr=None,
            rfeat=None,
            sfeat=None,
            bgterr=bgterr,
            bgfeat=bgfeat,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    )
    return case_metrics, pred_flow


def _evaluate_case_physics_cascade_roi(
    model,
    conditioner: CascadeConditioner,
    global_bundle,
    roi_bundle,
    scalers,
    *,
    device: str,
    hard_ground_bc: bool,
    n_patches: int,
    rterr: Optional[torch.Tensor] = None,
    sterr: Optional[torch.Tensor] = None,
    rfeat: Optional[torch.Tensor] = None,
    sfeat: Optional[torch.Tensor] = None,
    bgterr: Optional[torch.Tensor] = None,
    bgfeat: Optional[torch.Tensor] = None,
    use_amp: bool,
    amp_dtype: str,
):
    if bundle_uses_terrain_following_z(roi_bundle):
        return _tf_skipped_physics_metrics()
    if int(n_patches) <= 0:
        return {
            'physics_patch_count': 0,
            'physics_div_rms': float('nan'),
            'physics_mom_rms_constant': float('nan'),
            'physics_mom_rms_nut': float('nan'),
        }
    rng = np.random.default_rng(_stable_patch_seed(roi_bundle.name, roi_bundle.roi_name or 'cascade', 'eval_phys_cascade'))
    rterr = terrain_tensor(roi_bundle).to(device) if rterr is None else rterr
    rfeat = model.encode_roi(rterr) if rfeat is None else rfeat
    sfeat = model.encode_structure(sterr) if sfeat is None else sfeat
    div_vals: list[float] = []
    mom_const_vals: list[float] = []
    mom_nut_vals: list[float] = []
    with torch.no_grad():
        for _ in range(int(n_patches)):
            patch = sample_patch_batch(
                roi_bundle,
                x_scaler=scalers.x_scaler_roi,
                y_scaler=scalers.y_scaler,
                patch_shape=ROI_PATCH_SHAPE,
                rng=rng,
                near_ground_prob=PATCH_NEAR_GROUND_PROB,
                parent_global=global_bundle,
                include_grid_unet_context=_is_cascade_grid_refiner(model),
            )
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                if _is_cascade_grid_refiner(model):
                    patch_pred, _, _, bgterr, bgfeat = _cascade_grid_patch_pred_scaled(
                        model,
                        conditioner,
                        global_bundle,
                        patch,
                        scalers,
                        device=device,
                        hard_ground_bc=hard_ground_bc,
                        bgterr=bgterr,
                        bgfeat=bgfeat,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                    )
                else:
                    bg_scaled, _, bgterr, bgfeat = _cascade_conditioner_predict_on_roi_inputs(
                        conditioner,
                        global_bundle,
                        x_scaled_roi=patch.x_scaled.to(device),
                        x_scaler_roi=scalers.x_scaler_roi,
                        target_y_scaler=scalers.y_scaler,
                        device=device,
                        hard_ground_bc=hard_ground_bc,
                        gterr=bgterr,
                        gfeat=bgfeat,
                    )
                    patch_pred_raw = model.forward_roi_from_encoded(
                        rfeat,
                        patch.x_scaled.to(device),
                        patch.xy_local.to(device),
                        bg_scaled,
                        s_feat=sfeat,
                    )
                    patch_pred, _ = apply_output_constraints_from_scaled_inputs(
                        patch_pred_raw + bg_scaled,
                        x_batch=patch.x_scaled.to(device),
                        x_scaler=scalers.x_scaler_roi,
                        input_cols=ROI_INPUT_COLS,
                        y_scaler=scalers.y_scaler,
                        hard_ground_bc=hard_ground_bc,
                    )
                phys_const = compute_patch_physics_losses_from_pred(
                    patch_pred,
                    patch,
                    scalers.y_scaler,
                    device=device,
                    momentum_loss_mode='constant',
                )
            div_vals.append(float(phys_const['div_rms'].detach().cpu().item()))
            mom_const_vals.append(float(phys_const['mom_rms'].detach().cpu().item()))
            if patch.nut is not None:
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    phys_nut = compute_patch_physics_losses_from_pred(
                        patch_pred,
                        patch,
                        scalers.y_scaler,
                        device=device,
                        momentum_loss_mode='nut',
                    )
                mom_nut_vals.append(float(phys_nut['mom_rms'].detach().cpu().item()))
    return {
        'physics_patch_count': int(n_patches),
        'physics_div_rms': float(np.mean(div_vals)) if div_vals else float('nan'),
        'physics_mom_rms_constant': float(np.mean(mom_const_vals)) if mom_const_vals else float('nan'),
        'physics_mom_rms_nut': float(np.mean(mom_nut_vals)) if mom_nut_vals else float('nan'),
    }


def _evaluate_split_cascade_stage1(
    model,
    repo: CaseRepository,
    split_names: list[str],
    scalers,
    *,
    device: str,
    pred_batch_size: int,
    output_dir: Path,
    split_label: str,
    hard_ground_bc: bool = HARD_GROUND_BC,
    plot_eval: bool = PLOT_EVAL,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
) -> Dict:
    plotter = None
    if plot_eval:
        from vis.eval_report import generate_eval_case_report
        plotter = generate_eval_case_report
    global_case_metrics = {}
    roi_case_metrics = {}
    ensure_dir(output_dir)
    tensor_cache = _DeviceTensorCache(device=device)
    for idx_name, name in enumerate(split_names, start=1):
        print(f"[EVAL] {split_label} {idx_name}/{len(split_names)}: {name} | model=cascade_stage1", flush=True)
        case_dir = output_dir / name
        ensure_dir(case_dir)
        g = repo.load_global(name)
        gterr = tensor_cache.terrain(g)
        with torch.no_grad():
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                gfeat = model.encode_global(gterr)
        g_metrics, g_pred = _evaluate_case_global(
            model,
            g,
            scalers,
            device=device,
            pred_batch_size=pred_batch_size,
            hard_ground_bc=hard_ground_bc,
            return_pred_flow=plot_eval,
            terr=gterr,
            gfeat=gfeat,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        global_case_metrics[name] = g_metrics
        payload = {'case': name, 'split': split_label, 'global': g_metrics, 'rois': {}}
        if plotter is not None and g_pred is not None:
            payload['plots'] = plotter(repo.case_dirs[name], case_dir / 'plots', global_pred_flow=g_pred, roi_pred_flows={})
        write_json(case_dir / 'metrics.json', payload)
    summary = {
        'global_cases': global_case_metrics,
        'roi_cases': roi_case_metrics,
        'global_by_category': _category_metrics(repo, global_case_metrics),
        'roi_by_category': _category_metrics(repo, roi_case_metrics),
        'global_mean_nrmse_umag': float(np.nanmean([m['nrmse_umag'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'global_mean_nrmse_p': float(np.nanmean([m['nrmse_p'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'global_mean_nrmse_p_gauge': float(np.nanmean([m['nrmse_p_gauge'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'roi_mean_nrmse_umag': float('nan'),
        'roi_mean_nrmse_p': float('nan'),
        'roi_mean_nrmse_p_gauge': float('nan'),
        'global_near_ground_nrmse_umag': float(np.nanmean([(m.get('subsets') or {}).get('near_ground', {}).get('nrmse_umag', float('nan')) for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'global_near_ground_nrmse_p': float(np.nanmean([(m.get('subsets') or {}).get('near_ground', {}).get('nrmse_p', float('nan')) for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'global_near_ground_nrmse_p_gauge': float(np.nanmean([(m.get('subsets') or {}).get('near_ground', {}).get('nrmse_p_gauge', float('nan')) for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'roi_near_wall_nrmse_umag': float('nan'),
        'roi_near_wall_nrmse_p': float('nan'),
        'roi_near_wall_nrmse_p_gauge': float('nan'),
        'roi_near_ground_nrmse_umag': float('nan'),
        'roi_near_ground_nrmse_p': float('nan'),
        'roi_near_ground_nrmse_p_gauge': float('nan'),
        'roi_near_ground_near_wall_nrmse_umag': float('nan'),
        'roi_near_ground_near_wall_nrmse_p': float('nan'),
        'roi_near_ground_near_wall_nrmse_p_gauge': float('nan'),
        'global_mean_physics_div_rms': _mean_metric(global_case_metrics, 'physics_div_rms'),
        'global_mean_physics_mom_rms_constant': _mean_metric(global_case_metrics, 'physics_mom_rms_constant'),
        'global_mean_physics_mom_rms_nut': _mean_metric(global_case_metrics, 'physics_mom_rms_nut'),
        'roi_mean_physics_div_rms': float('nan'),
        'roi_mean_physics_mom_rms_constant': float('nan'),
        'roi_mean_physics_mom_rms_nut': float('nan'),
        'global_case_count': int(len(global_case_metrics)),
        'roi_case_count': 0,
        'split': split_label,
        'hard_ground_bc': bool(hard_ground_bc),
        'plot_eval': bool(plot_eval),
        'use_amp': bool(use_amp),
        'amp_dtype': str(amp_dtype),
    }
    write_json(output_dir / 'summary.json', summary)
    return summary


def _evaluate_split_cascade_stage2(
    model,
    repo: CaseRepository,
    split_names: list[str],
    scalers,
    *,
    conditioner: CascadeConditioner,
    device: str,
    pred_batch_size: int,
    output_dir: Path,
    split_label: str,
    hard_ground_bc: bool = HARD_GROUND_BC,
    plot_eval: bool = PLOT_EVAL,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
) -> Dict:
    plotter = None
    if plot_eval:
        from vis.eval_report import generate_eval_case_report
        plotter = generate_eval_case_report
    global_case_metrics = {}
    roi_case_metrics = {}
    ensure_dir(output_dir)
    tensor_cache = _DeviceTensorCache(device=device)
    grid_refiner = _is_cascade_grid_refiner(model)
    skipped_large_roi: list[dict] = []
    for idx_name, name in enumerate(split_names, start=1):
        print(f"[EVAL] {split_label} {idx_name}/{len(split_names)}: {name} | model=cascade_stage2", flush=True)
        case_dir = output_dir / name
        ensure_dir(case_dir)
        g = repo.load_global(name)
        bgterr = tensor_cache.terrain(g)
        if _is_grid_unet_conditioner(conditioner):
            bgfeat = None
            g_metrics, g_pred = _evaluate_case_global_grid_unet(
                conditioner.model,
                g,
                conditioner.scalers,
                device=device,
                hard_ground_bc=hard_ground_bc,
                return_pred_flow=plot_eval,
                patch_shape=_conditioner_global_patch_shape(conditioner),
            )
        else:
            with torch.no_grad():
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    bgfeat = conditioner.model.encode_global(bgterr)
            g_metrics, g_pred = _evaluate_case_global(
                conditioner.model,
                g,
                conditioner.scalers,
                device=device,
                pred_batch_size=pred_batch_size,
                hard_ground_bc=hard_ground_bc,
                return_pred_flow=plot_eval,
                terr=bgterr,
                gfeat=bgfeat,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        global_case_metrics[name] = g_metrics
        roi_metrics_local = {}
        roi_pred_local = {}
        roi_refs = [(name, roi_name) for roi_name in repo.roi_names(name)]
        if grid_refiner:
            roi_refs = _filter_cascade_grid_roi_refs(
                repo,
                roi_refs,
                split_label=split_label,
                skipped=skipped_large_roi,
            )
        for _, roi_name in roi_refs:
            r = repo.load_roi(name, roi_name)
            rterr = tensor_cache.terrain(r)
            sterr = tensor_cache.structure(r)
            with torch.no_grad():
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    if grid_refiner:
                        rfeat = None
                        sfeat = None
                    else:
                        rfeat = model.encode_roi(rterr)
                        sfeat = model.encode_structure(sterr)
            roi_points = int(np.prod(r.flow.shape[:3], dtype=np.int64))
            want_roi_plot = bool(plot_eval and roi_points <= int(MAX_PLOT_FLOW_POINTS))
            r_metrics, r_pred = _evaluate_case_roi_cascade(
                model,
                conditioner,
                g,
                r,
                scalers,
                device=device,
                pred_batch_size=pred_batch_size,
                hard_ground_bc=hard_ground_bc,
                return_pred_flow=want_roi_plot,
                rterr=rterr,
                sterr=sterr,
                rfeat=rfeat,
                sfeat=sfeat,
                bgterr=bgterr,
                bgfeat=bgfeat,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            roi_metrics_local[roi_name] = r_metrics
            roi_case_metrics[f'{name}/{roi_name}'] = r_metrics
            if want_roi_plot and r_pred is not None:
                roi_pred_local[roi_name] = r_pred
        payload = {'case': name, 'split': split_label, 'global': g_metrics, 'rois': roi_metrics_local}
        if plotter is not None and g_pred is not None:
            payload['plots'] = plotter(repo.case_dirs[name], case_dir / 'plots', global_pred_flow=g_pred, roi_pred_flows=roi_pred_local)
        write_json(case_dir / 'metrics.json', payload)
    def _subset_mean(case_metrics: dict, subset_name: str, key: str) -> float:
        vals = []
        for m in case_metrics.values():
            sub = (m.get('subsets') or {}).get(subset_name)
            if not sub:
                continue
            v = sub.get(key)
            if v is None or not np.isfinite(v):
                continue
            vals.append(float(v))
        return float(np.mean(vals)) if vals else float('nan')
    summary = {
        'global_cases': global_case_metrics,
        'roi_cases': roi_case_metrics,
        'global_by_category': _category_metrics(repo, global_case_metrics),
        'roi_by_category': _category_metrics(repo, roi_case_metrics),
        'global_mean_nrmse_umag': float(np.nanmean([m['nrmse_umag'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'global_mean_nrmse_p': float(np.nanmean([m['nrmse_p'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'global_mean_nrmse_p_gauge': float(np.nanmean([m['nrmse_p_gauge'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'roi_mean_nrmse_umag': float(np.nanmean([m['nrmse_umag'] for m in roi_case_metrics.values()])) if roi_case_metrics else float('nan'),
        'roi_mean_nrmse_p': float(np.nanmean([m['nrmse_p'] for m in roi_case_metrics.values()])) if roi_case_metrics else float('nan'),
        'roi_mean_nrmse_p_gauge': float(np.nanmean([m['nrmse_p_gauge'] for m in roi_case_metrics.values()])) if roi_case_metrics else float('nan'),
        'global_near_ground_nrmse_umag': _subset_mean(global_case_metrics, 'near_ground', 'nrmse_umag'),
        'global_near_ground_nrmse_p': _subset_mean(global_case_metrics, 'near_ground', 'nrmse_p'),
        'global_near_ground_nrmse_p_gauge': _subset_mean(global_case_metrics, 'near_ground', 'nrmse_p_gauge'),
        'roi_near_wall_nrmse_umag': _subset_mean(roi_case_metrics, 'near_wall', 'nrmse_umag'),
        'roi_near_wall_nrmse_p': _subset_mean(roi_case_metrics, 'near_wall', 'nrmse_p'),
        'roi_near_wall_nrmse_p_gauge': _subset_mean(roi_case_metrics, 'near_wall', 'nrmse_p_gauge'),
        'roi_near_ground_nrmse_umag': _subset_mean(roi_case_metrics, 'near_ground', 'nrmse_umag'),
        'roi_near_ground_nrmse_p': _subset_mean(roi_case_metrics, 'near_ground', 'nrmse_p'),
        'roi_near_ground_nrmse_p_gauge': _subset_mean(roi_case_metrics, 'near_ground', 'nrmse_p_gauge'),
        'roi_near_ground_near_wall_nrmse_umag': _subset_mean(roi_case_metrics, 'near_ground_near_wall', 'nrmse_umag'),
        'roi_near_ground_near_wall_nrmse_p': _subset_mean(roi_case_metrics, 'near_ground_near_wall', 'nrmse_p'),
        'roi_near_ground_near_wall_nrmse_p_gauge': _subset_mean(roi_case_metrics, 'near_ground_near_wall', 'nrmse_p_gauge'),
        'global_mean_physics_div_rms': _mean_metric(global_case_metrics, 'physics_div_rms'),
        'global_mean_physics_mom_rms_constant': _mean_metric(global_case_metrics, 'physics_mom_rms_constant'),
        'global_mean_physics_mom_rms_nut': _mean_metric(global_case_metrics, 'physics_mom_rms_nut'),
        'roi_mean_physics_div_rms': _mean_metric(roi_case_metrics, 'physics_div_rms'),
        'roi_mean_physics_mom_rms_constant': _mean_metric(roi_case_metrics, 'physics_mom_rms_constant'),
        'roi_mean_physics_mom_rms_nut': _mean_metric(roi_case_metrics, 'physics_mom_rms_nut'),
        'global_case_count': int(len(global_case_metrics)),
        'roi_case_count': int(len(roi_case_metrics)),
        'split': split_label,
        'hard_ground_bc': bool(hard_ground_bc),
        'plot_eval': bool(plot_eval),
        'use_amp': bool(use_amp),
        'amp_dtype': str(amp_dtype),
        'cascade_stage1_checkpoint': str(conditioner.checkpoint_path),
        'skipped_large_roi': {
            'enabled': bool(grid_refiner),
            'threshold_cells': int(CASCADE_STAGE2_GRID_MAX_ROI_CELLS),
            'count': int(len(skipped_large_roi)),
            'skipped': skipped_large_roi,
        },
    }
    if grid_refiner:
        write_json(output_dir / 'skipped_large_roi.json', summary['skipped_large_roi'])
    write_json(output_dir / 'summary.json', summary)
    return summary


def _evaluate_split_grid_unet(
    model,
    repo: CaseRepository,
    split_names: list[str],
    scalers,
    *,
    device: str,
    pred_batch_size: int,
    output_dir: Path,
    split_label: str,
    hard_ground_bc: bool = HARD_GROUND_BC,
    plot_eval: bool = PLOT_EVAL,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
) -> Dict:
    plotter = None
    if plot_eval:
        from vis.eval_report import generate_eval_case_report
        plotter = generate_eval_case_report

    global_case_metrics = {}
    roi_case_metrics = {}
    ensure_dir(output_dir)
    total_roi = int(sum(len(repo.roi_names(name)) for name in split_names))
    print(
        f"[EVAL] {split_label}: {len(split_names)} domains, {total_roi} ROI box(es), "
        f"plot_eval={'yes' if plot_eval else 'no'} | model=grid_unet",
        flush=True,
    )

    for idx_name, name in enumerate(split_names, start=1):
        print(f"[EVAL] {split_label} {idx_name}/{len(split_names)}: {name}", flush=True)
        case_dir = output_dir / name
        ensure_dir(case_dir)
        g = repo.load_global(name)
        g_metrics, g_pred = _evaluate_case_global_grid_unet(
            model,
            g,
            scalers,
            device=device,
            hard_ground_bc=hard_ground_bc,
            return_pred_flow=plot_eval,
        )
        g_metrics.update(
            _evaluate_case_physics_grid_unet(
                model,
                g,
                scalers,
                device=device,
                hard_ground_bc=hard_ground_bc,
                n_patches=int(EVAL_GLOBAL_PATCHES_PER_CASE),
            )
        )
        global_case_metrics[name] = g_metrics

        roi_metrics_local = {}
        roi_pred_local = {}
        for roi_name in repo.roi_names(name):
            r = repo.load_roi(name, roi_name)
            roi_points = int(np.prod(r.flow.shape[:3], dtype=np.int64))
            want_roi_plot = bool(plot_eval and roi_points <= int(MAX_PLOT_FLOW_POINTS))
            if plot_eval and not want_roi_plot:
                print(
                    f"[EVAL] skipping ROI plot for {name}/{roi_name} "
                    f"(points={roi_points:,} > MAX_PLOT_FLOW_POINTS={int(MAX_PLOT_FLOW_POINTS):,})",
                    flush=True,
                )
            r_metrics, r_pred = _evaluate_case_roi_grid_unet(
                model,
                g,
                r,
                scalers,
                device=device,
                hard_ground_bc=hard_ground_bc,
                return_pred_flow=want_roi_plot,
            )
            r_metrics.update(
                _evaluate_case_physics_grid_unet(
                    model,
                    r,
                    scalers,
                    device=device,
                    hard_ground_bc=hard_ground_bc,
                    n_patches=int(EVAL_ROI_PATCHES_PER_CASE),
                    parent_global=g,
                )
            )
            roi_metrics_local[roi_name] = r_metrics
            roi_case_metrics[f'{name}/{roi_name}'] = r_metrics
            if want_roi_plot and r_pred is not None:
                roi_pred_local[roi_name] = r_pred

        payload = {
            'case': name,
            'split': split_label,
            'global': g_metrics,
            'rois': roi_metrics_local,
        }
        if plotter is not None and g_pred is not None:
            files = plotter(repo.case_dirs[name], case_dir / 'plots', global_pred_flow=g_pred, roi_pred_flows=roi_pred_local)
            payload['plots'] = files
        write_json(case_dir / 'metrics.json', payload)

    def _subset_mean(case_metrics: dict, subset_name: str, key: str) -> float:
        vals = []
        for m in case_metrics.values():
            sub = (m.get('subsets') or {}).get(subset_name)
            if not sub:
                continue
            v = sub.get(key)
            if v is None or not np.isfinite(v):
                continue
            vals.append(float(v))
        return float(np.mean(vals)) if vals else float('nan')

    summary = {
        'global_cases': global_case_metrics,
        'roi_cases': roi_case_metrics,
        'global_by_category': _category_metrics(repo, global_case_metrics),
        'roi_by_category': _category_metrics(repo, roi_case_metrics),
        'global_mean_nrmse_umag': float(np.nanmean([m['nrmse_umag'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'global_mean_nrmse_p': float(np.nanmean([m['nrmse_p'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'global_mean_nrmse_p_gauge': float(np.nanmean([m['nrmse_p_gauge'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'roi_mean_nrmse_umag': float(np.nanmean([m['nrmse_umag'] for m in roi_case_metrics.values()])) if roi_case_metrics else float('nan'),
        'roi_mean_nrmse_p': float(np.nanmean([m['nrmse_p'] for m in roi_case_metrics.values()])) if roi_case_metrics else float('nan'),
        'roi_mean_nrmse_p_gauge': float(np.nanmean([m['nrmse_p_gauge'] for m in roi_case_metrics.values()])) if roi_case_metrics else float('nan'),
        'global_near_ground_nrmse_umag': _subset_mean(global_case_metrics, 'near_ground', 'nrmse_umag'),
        'global_near_ground_nrmse_p': _subset_mean(global_case_metrics, 'near_ground', 'nrmse_p'),
        'global_near_ground_nrmse_p_gauge': _subset_mean(global_case_metrics, 'near_ground', 'nrmse_p_gauge'),
        'roi_near_wall_nrmse_umag': _subset_mean(roi_case_metrics, 'near_wall', 'nrmse_umag'),
        'roi_near_wall_nrmse_p': _subset_mean(roi_case_metrics, 'near_wall', 'nrmse_p'),
        'roi_near_wall_nrmse_p_gauge': _subset_mean(roi_case_metrics, 'near_wall', 'nrmse_p_gauge'),
        'roi_near_ground_nrmse_umag': _subset_mean(roi_case_metrics, 'near_ground', 'nrmse_umag'),
        'roi_near_ground_nrmse_p': _subset_mean(roi_case_metrics, 'near_ground', 'nrmse_p'),
        'roi_near_ground_nrmse_p_gauge': _subset_mean(roi_case_metrics, 'near_ground', 'nrmse_p_gauge'),
        'roi_near_ground_near_wall_nrmse_umag': _subset_mean(roi_case_metrics, 'near_ground_near_wall', 'nrmse_umag'),
        'roi_near_ground_near_wall_nrmse_p': _subset_mean(roi_case_metrics, 'near_ground_near_wall', 'nrmse_p'),
        'roi_near_ground_near_wall_nrmse_p_gauge': _subset_mean(roi_case_metrics, 'near_ground_near_wall', 'nrmse_p_gauge'),
        'global_mean_physics_div_rms': _mean_metric(global_case_metrics, 'physics_div_rms'),
        'global_mean_physics_mom_rms_constant': _mean_metric(global_case_metrics, 'physics_mom_rms_constant'),
        'global_mean_physics_mom_rms_nut': _mean_metric(global_case_metrics, 'physics_mom_rms_nut'),
        'roi_mean_physics_div_rms': _mean_metric(roi_case_metrics, 'physics_div_rms'),
        'roi_mean_physics_mom_rms_constant': _mean_metric(roi_case_metrics, 'physics_mom_rms_constant'),
        'roi_mean_physics_mom_rms_nut': _mean_metric(roi_case_metrics, 'physics_mom_rms_nut'),
        'global_case_count': int(len(global_case_metrics)),
        'roi_case_count': int(len(roi_case_metrics)),
        'split': split_label,
        'hard_ground_bc': bool(hard_ground_bc),
        'plot_eval': bool(plot_eval),
        'use_amp': bool(use_amp),
        'amp_dtype': str(amp_dtype),
    }
    write_json(output_dir / 'summary.json', summary)
    print(
        f"[EVAL] {split_label} done | global nRMSE(Umag)={summary['global_mean_nrmse_umag']:.4f} | "
        f"roi nRMSE(Umag)={summary['roi_mean_nrmse_umag']:.4f}",
        flush=True,
    )
    return summary


def _train_grid_unet_model(
    model,
    repo: CaseRepository,
    split: dict,
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    save_dir: Path,
    train_mode: str = TRAIN_MODE,
    train_loss: str = TRAIN_LOSS,
    momentum_loss_mode: str = MOMENTUM_LOSS_MODE,
    train_struct_mode: str = TRAIN_STRUCT_MODE,
    train_struct_weight: float = TRAIN_STRUCT_WEIGHT,
    scheduler_mode: str = SCHEDULER_MODE,
    hard_ground_bc: bool = HARD_GROUND_BC,
    charb_eps: float = CHARB_EPS,
    pred_batch_size: int = 200000,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
    resume_checkpoint: Optional[dict] = None,
    wandb_run=None,
) -> Dict:
    if resume_checkpoint is not None:
        scalers = resume_checkpoint['scalers']
        history = list(resume_checkpoint.get('history', []))
        best_score = float(resume_checkpoint.get('best_val', float('inf')))
        best_epoch = int(resume_checkpoint.get('best_epoch', -1))
        start_epoch = int(resume_checkpoint.get('epoch', 0))
    else:
        scalers = fit_scalers(repo, split['train'])
        history = []
        best_score = float('inf')
        best_epoch = -1
        start_epoch = 0
    rng = _training_rng(int(start_epoch))
    stop_epoch = int(epochs)
    ensure_dir(save_dir)
    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=_use_cuda_amp(device, use_amp) and _amp_torch_dtype(amp_dtype) == torch.float16,
    )

    train_roi_refs = [(name, roi_name) for name in split['train'] for roi_name in repo.roi_names(name)]
    val_roi_refs = [(name, roi_name) for name in split['val'] for roi_name in repo.roi_names(name)]
    weights = _effective_weights(train_mode)
    steps_per_epoch = len(split['train']) + len(train_roi_refs)
    scheduler = _build_scheduler(optimizer, epochs=int(epochs), steps_per_epoch=steps_per_epoch, scheduler_mode=scheduler_mode)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        if scaler.is_enabled() and 'scaler_state_dict' in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
    print(
        f"[TRAIN] mode={train_mode} | model=grid_unet | epochs={int(epochs)} | train domains={len(split['train'])} | "
        f"train ROI={len(train_roi_refs)} | steps/epoch={steps_per_epoch} | mom={momentum_loss_mode} | "
        f"amp={'off' if not use_amp else amp_dtype}",
        flush=True,
    )
    if start_epoch > 0:
        print(
            f"[TRAIN] resuming from epoch={int(start_epoch)} | best epoch={int(best_epoch)} | "
            f"best selector={float(best_score):.4f}",
            flush=True,
        )
    if int(start_epoch) >= int(epochs):
        print(
            f"[TRAIN] target epochs already reached in latest checkpoint "
            f"(epoch={int(start_epoch)} >= {int(epochs)}), skipping training loop",
            flush=True,
        )
        stop_epoch = int(start_epoch)

    for epoch in range(int(start_epoch) + 1, int(epochs) + 1):
        weights = _effective_weights(train_mode, epoch=epoch)
        model.train()
        train_global_losses = []
        train_roi_losses = []
        train_div_global = []
        train_div_roi = []
        train_mom_global = []
        train_mom_roi = []
        train_wall_roi = []
        rng.shuffle(split['train'])
        rng.shuffle(train_roi_refs)
        use_struct_loss = float(train_struct_weight) > 0.0 and str(train_struct_mode).lower() != 'none'

        for name in split['train']:
            g = repo.load_global(name)
            _raise_if_tf_nondata_loss(g, weights, scope='global', structured_loss_enabled=use_struct_loss)
            gterr = None
            gctx2d = None
            if bool(getattr(model, 'uses_grid_terrain_context', False)):
                gterr = terrain_tensor(g).to(device)
                gctx2d = model.encode_grid_terrain_context(gterr)
            sup_count = _grid_supervised_patch_count(int(GLOBAL_POINTS_PER_DOMAIN), tuple(int(v) for v in GLOBAL_PATCH_SHAPE))
            extra_patch_count = max(0, int(GLOBAL_PATCHES_PER_DOMAIN) - sup_count) if (weights['w_phys_global'] > 0 or use_struct_loss) else 0
            loss_data = torch.tensor(0.0, device=device)
            loss_phys = torch.tensor(0.0, device=device)
            struct_loss = torch.tensor(0.0, device=device)
            for patch_idx in range(sup_count + extra_patch_count):
                patch = sample_patch_batch(
                    g,
                    x_scaler=scalers.x_scaler_global,
                    y_scaler=scalers.y_scaler,
                    patch_shape=GLOBAL_PATCH_SHAPE,
                    rng=rng,
                    near_ground_prob=PATCH_NEAR_GROUND_PROB,
                    include_grid_unet_context=True,
                )
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    ctx_patch, gterr, gctx2d = _grid_patch_terrain_context_2d(
                        model,
                        g,
                        patch,
                        device=device,
                        terr=gterr,
                        feat2d=gctx2d,
                    )
                    patch_pred, _ = _grid_patch_pred_scaled(
                        model,
                        patch,
                        device=device,
                        x_scaler=scalers.x_scaler_global,
                        input_cols=GLOBAL_INPUT_COLS,
                        y_scaler=scalers.y_scaler,
                        hard_ground_bc=hard_ground_bc,
                        roi=False,
                        terrain_context_2d=ctx_patch,
                    )
                    if patch_idx < sup_count:
                        loss_data = loss_data + _supervised_patch_loss_from_pred(
                            patch_pred,
                            patch,
                            device=device,
                            x_scaler=scalers.x_scaler_global,
                            input_cols=GLOBAL_INPUT_COLS,
                            y_scaler=scalers.y_scaler,
                            loss_mode=train_loss,
                            charbonnier_eps=charb_eps,
                            p_weight=float(GLOBAL_DATA_P_WEIGHT),
                        )
                    if weights['w_phys_global'] > 0:
                        phys = compute_patch_physics_losses_from_pred(
                            patch_pred,
                            patch,
                            scalers.y_scaler,
                            device=device,
                            momentum_loss_mode=momentum_loss_mode,
                        )
                        loss_phys = loss_phys + weights['w_div_global'] * phys['div_loss'] + weights['w_mom_global'] * phys['mom_loss']
                        train_div_global.append(float(phys['div_rms'].detach().cpu().item()))
                        train_mom_global.append(float(phys['mom_rms'].detach().cpu().item()))
                    if use_struct_loss:
                        struct_loss = struct_loss + structured_patch_loss_from_pred(
                            patch_pred,
                            patch,
                            mode=train_struct_mode,
                            p_weight=float(GLOBAL_DATA_P_WEIGHT),
                            charbonnier_eps=charb_eps,
                        )
            inlet_loss = torch.tensor(0.0, device=device)
            outlet_loss = torch.tensor(0.0, device=device)
            side_loss = torch.tensor(0.0, device=device)
            top_loss = torch.tensor(0.0, device=device)
            if weights['w_bc_inlet'] > 0:
                patch, bc_batch, flat_idx = _sample_global_boundary_patch(
                    g,
                    x_scaler=scalers.x_scaler_global,
                    y_scaler=scalers.y_scaler,
                    face='inlet',
                    rng=rng,
                )
                if bc_batch is not None and flat_idx is not None:
                    with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                        ctx_patch, gterr, gctx2d = _grid_patch_terrain_context_2d(
                            model,
                            g,
                            patch,
                            device=device,
                            terr=gterr,
                            feat2d=gctx2d,
                        )
                        pred_phys = _grid_patch_pred_scaled(
                            model,
                            patch,
                            device=device,
                            x_scaler=scalers.x_scaler_global,
                            input_cols=GLOBAL_INPUT_COLS,
                            y_scaler=scalers.y_scaler,
                            hard_ground_bc=hard_ground_bc,
                            roi=False,
                            terrain_context_2d=ctx_patch,
                        )[1]
                        inlet_loss = inlet_bc_loss_from_phys(pred_phys[flat_idx.to(pred_phys.device)], bc_batch, device=device)
            if weights['w_bc_outlet'] > 0:
                patch, bc_batch, flat_idx = _sample_global_boundary_patch(
                    g,
                    x_scaler=scalers.x_scaler_global,
                    y_scaler=scalers.y_scaler,
                    face='outlet',
                    rng=rng,
                )
                if bc_batch is not None and flat_idx is not None:
                    with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                        ctx_patch, gterr, gctx2d = _grid_patch_terrain_context_2d(
                            model,
                            g,
                            patch,
                            device=device,
                            terr=gterr,
                            feat2d=gctx2d,
                        )
                        pred_phys = _grid_patch_pred_scaled(
                            model,
                            patch,
                            device=device,
                            x_scaler=scalers.x_scaler_global,
                            input_cols=GLOBAL_INPUT_COLS,
                            y_scaler=scalers.y_scaler,
                            hard_ground_bc=hard_ground_bc,
                            roi=False,
                            terrain_context_2d=ctx_patch,
                        )[1]
                        outlet_loss = outlet_bc_loss_from_phys(pred_phys[flat_idx.to(pred_phys.device)], bc_batch, device=device)
            if weights['w_bc_side'] > 0:
                side_terms = []
                for face in ('side_lo', 'side_hi'):
                    patch, bc_batch, flat_idx = _sample_global_boundary_patch(
                        g,
                        x_scaler=scalers.x_scaler_global,
                        y_scaler=scalers.y_scaler,
                        face=face,
                        rng=rng,
                    )
                    if bc_batch is None or flat_idx is None:
                        continue
                    with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                        ctx_patch, gterr, gctx2d = _grid_patch_terrain_context_2d(
                            model,
                            g,
                            patch,
                            device=device,
                            terr=gterr,
                            feat2d=gctx2d,
                        )
                        pred_phys = _grid_patch_pred_scaled(
                            model,
                            patch,
                            device=device,
                            x_scaler=scalers.x_scaler_global,
                            input_cols=GLOBAL_INPUT_COLS,
                            y_scaler=scalers.y_scaler,
                            hard_ground_bc=hard_ground_bc,
                            roi=False,
                            terrain_context_2d=ctx_patch,
                        )[1]
                        side_terms.append(normal_velocity_bc_loss_from_phys(pred_phys[flat_idx.to(pred_phys.device)], bc_batch, device=device))
                if side_terms:
                    side_loss = torch.stack(side_terms).mean()
            if weights['w_bc_top'] > 0:
                patch, bc_batch, flat_idx = _sample_global_boundary_patch(
                    g,
                    x_scaler=scalers.x_scaler_global,
                    y_scaler=scalers.y_scaler,
                    face='top',
                    rng=rng,
                )
                if bc_batch is not None and flat_idx is not None:
                    with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                        ctx_patch, gterr, gctx2d = _grid_patch_terrain_context_2d(
                            model,
                            g,
                            patch,
                            device=device,
                            terr=gterr,
                            feat2d=gctx2d,
                        )
                        pred_phys = _grid_patch_pred_scaled(
                            model,
                            patch,
                            device=device,
                            x_scaler=scalers.x_scaler_global,
                            input_cols=GLOBAL_INPUT_COLS,
                            y_scaler=scalers.y_scaler,
                            hard_ground_bc=hard_ground_bc,
                            roi=False,
                            terrain_context_2d=ctx_patch,
                        )[1]
                        top_loss = normal_velocity_bc_loss_from_phys(pred_phys[flat_idx.to(pred_phys.device)], bc_batch, device=device)
            total = (
                float(W_DATA_GLOBAL) * loss_data
                + weights['w_phys_global'] * loss_phys
                + float(train_struct_weight) * struct_loss
                + weights['w_bc_inlet'] * inlet_loss
                + weights['w_bc_outlet'] * outlet_loss
                + weights['w_bc_side'] * side_loss
                + weights['w_bc_top'] * top_loss
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(total).backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_global_losses.append(float(total.detach().cpu().item()))

        for name, roi_name in train_roi_refs:
            g = repo.load_global(name)
            r = repo.load_roi(name, roi_name)
            _raise_if_tf_nondata_loss(g, weights, scope='global', structured_loss_enabled=False)
            _raise_if_tf_nondata_loss(r, weights, scope='roi', structured_loss_enabled=use_struct_loss)
            rterr = None
            rctx2d = None
            if bool(getattr(model, 'uses_grid_terrain_context', False)):
                rterr = terrain_tensor(r).to(device)
                rctx2d = model.encode_grid_terrain_context(rterr)
            sup_count = _grid_supervised_patch_count(int(ROI_POINTS_PER_DOMAIN), tuple(int(v) for v in ROI_PATCH_SHAPE))
            extra_patch_count = max(0, int(ROI_PATCHES_PER_DOMAIN) - sup_count) if (weights['w_phys_roi'] > 0 or weights['w_bc_wall_roi'] > 0 or use_struct_loss) else 0
            loss_data = torch.tensor(0.0, device=device)
            loss_phys = torch.tensor(0.0, device=device)
            struct_loss = torch.tensor(0.0, device=device)
            wall_loss = torch.tensor(0.0, device=device)
            for patch_idx in range(sup_count + extra_patch_count):
                patch = sample_patch_batch(
                    r,
                    x_scaler=scalers.x_scaler_roi,
                    y_scaler=scalers.y_scaler,
                    patch_shape=ROI_PATCH_SHAPE,
                    rng=rng,
                    near_ground_prob=PATCH_NEAR_GROUND_PROB,
                    parent_global=g,
                    include_grid_unet_context=True,
                )
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    ctx_patch, rterr, rctx2d = _grid_patch_terrain_context_2d(
                        model,
                        r,
                        patch,
                        device=device,
                        terr=rterr,
                        feat2d=rctx2d,
                    )
                    patch_pred, patch_pred_phys = _grid_patch_pred_scaled(
                        model,
                        patch,
                        device=device,
                        x_scaler=scalers.x_scaler_roi,
                        input_cols=ROI_INPUT_COLS,
                        y_scaler=scalers.y_scaler,
                        hard_ground_bc=hard_ground_bc,
                        roi=True,
                        terrain_context_2d=ctx_patch,
                    )
                    if patch_idx < sup_count:
                        loss_data = loss_data + _supervised_patch_loss_from_pred(
                            patch_pred,
                            patch,
                            device=device,
                            x_scaler=scalers.x_scaler_roi,
                            input_cols=ROI_INPUT_COLS,
                            y_scaler=scalers.y_scaler,
                            loss_mode=train_loss,
                            charbonnier_eps=charb_eps,
                            p_weight=float(ROI_DATA_P_WEIGHT),
                        )
                    if weights['w_phys_roi'] > 0:
                        phys = compute_patch_physics_losses_from_pred(
                            patch_pred,
                            patch,
                            scalers.y_scaler,
                            device=device,
                            momentum_loss_mode=momentum_loss_mode,
                        )
                        loss_phys = loss_phys + weights['w_div_roi'] * phys['div_loss'] + weights['w_mom_roi'] * phys['mom_loss']
                        train_div_roi.append(float(phys['div_rms'].detach().cpu().item()))
                        train_mom_roi.append(float(phys['mom_rms'].detach().cpu().item()))
                    if use_struct_loss:
                        struct_loss = struct_loss + structured_patch_loss_from_pred(
                            patch_pred,
                            patch,
                            mode=train_struct_mode,
                            p_weight=float(ROI_DATA_P_WEIGHT),
                            charbonnier_eps=charb_eps,
                        )
                    if weights['w_bc_wall_roi'] > 0:
                        wall_loss = wall_loss + roi_wall_velocity_bc_loss_from_phys(
                            patch_pred_phys,
                            x_batch=patch.x_scaled,
                            x_scaler=scalers.x_scaler_roi,
                            input_cols=ROI_INPUT_COLS,
                            device=device,
                            u_scale=float(r.uref),
                            dmax=float(ROI_WALL_BC_DMAX),
                        )
            total = float(W_DATA_ROI) * loss_data + weights['w_phys_roi'] * loss_phys + float(train_struct_weight) * struct_loss + weights['w_bc_wall_roi'] * wall_loss
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(total).backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_roi_losses.append(float(total.detach().cpu().item()))
            if weights['w_bc_wall_roi'] > 0:
                train_wall_roi.append(float(wall_loss.detach().cpu().item()))

        model.eval()
        val_global_metrics = {}
        for name in split['val']:
            g = repo.load_global(name)
            metrics, _ = _evaluate_case_global_grid_unet(
                model,
                g,
                scalers,
                device=device,
                hard_ground_bc=hard_ground_bc,
                return_pred_flow=False,
            )
            metrics.update(
                _evaluate_case_physics_grid_unet(
                    model,
                    g,
                    scalers,
                    device=device,
                    hard_ground_bc=hard_ground_bc,
                    n_patches=int(EVAL_GLOBAL_PATCHES_PER_CASE),
                )
            )
            val_global_metrics[name] = metrics
        val_roi_metrics = {}
        for name, roi_name in val_roi_refs:
            g = repo.load_global(name)
            r = repo.load_roi(name, roi_name)
            metrics, _ = _evaluate_case_roi_grid_unet(
                model,
                g,
                r,
                scalers,
                device=device,
                hard_ground_bc=hard_ground_bc,
                return_pred_flow=False,
            )
            metrics.update(
                _evaluate_case_physics_grid_unet(
                    model,
                    r,
                    scalers,
                    device=device,
                    hard_ground_bc=hard_ground_bc,
                    n_patches=int(EVAL_ROI_PATCHES_PER_CASE),
                    parent_global=g,
                )
            )
            val_roi_metrics[f'{name}/{roi_name}'] = metrics

        global_score = float(np.nanmean([m['nrmse_umag'] for m in val_global_metrics.values()])) if val_global_metrics else float('nan')
        global_p_score = float(np.nanmean([m['nrmse_p'] for m in val_global_metrics.values()])) if val_global_metrics else float('nan')
        roi_score = float(np.nanmean([m['nrmse_umag'] for m in val_roi_metrics.values()])) if val_roi_metrics else float('nan')
        roi_p_score = float(np.nanmean([m['nrmse_p'] for m in val_roi_metrics.values()])) if val_roi_metrics else float('nan')
        val_global_by_cat = _category_metrics(repo, val_global_metrics)
        val_roi_by_cat = _category_metrics(repo, val_roi_metrics)
        selector_scores = _selector_components(
            val_global_metrics=val_global_metrics,
            val_roi_metrics=val_roi_metrics,
            val_roi_by_cat=val_roi_by_cat,
        )
        selector_umag = float(selector_scores['selector_umag'])
        selector_p = float(selector_scores['selector_p'])
        selector_ms_roi_umag = float(selector_scores['selector_ms_roi_umag'])
        selector = float(selector_scores['selector'])
        row = {
            'epoch': int(epoch),
            'lr': float(optimizer.param_groups[0]['lr']),
            'train_loss_global': float(np.mean(train_global_losses)) if train_global_losses else float('nan'),
            'train_loss_roi': float(np.mean(train_roi_losses)) if train_roi_losses else float('nan'),
            'train_div_rms_global': float(np.mean(train_div_global)) if train_div_global else float('nan'),
            'train_div_rms_roi': float(np.mean(train_div_roi)) if train_div_roi else float('nan'),
            'train_mom_rms_global': float(np.mean(train_mom_global)) if train_mom_global else float('nan'),
            'train_mom_rms_roi': float(np.mean(train_mom_roi)) if train_mom_roi else float('nan'),
            'train_wall_bc_roi': float(np.mean(train_wall_roi)) if train_wall_roi else float('nan'),
            'val_global_nrmse_umag': global_score,
            'val_global_nrmse_p': global_p_score,
            'val_roi_nrmse_umag': roi_score,
            'val_roi_nrmse_p': roi_p_score,
            'val_selector_umag': selector_umag,
            'val_selector_p': selector_p,
            'val_selector_ms_roi_umag': selector_ms_roi_umag,
            'val_selector': selector,
        }
        for short in _CATEGORY_SHORT.values():
            row[f'val_global_{short}_nrmse_umag'] = float(val_global_by_cat[short]['nrmse_umag'])
            row[f'val_global_{short}_nrmse_p'] = float(val_global_by_cat[short]['nrmse_p'])
            row[f'val_roi_{short}_nrmse_umag'] = float(val_roi_by_cat[short]['nrmse_umag'])
            row[f'val_roi_{short}_nrmse_p'] = float(val_roi_by_cat[short]['nrmse_p'])
        history.append(row)
        write_json(save_dir / 'logs' / 'history.json', history)
        print(
            f"Epoch {epoch:4d}/{int(epochs)} | lr={row['lr']:.2e} | "
            f"train G={row['train_loss_global']:.4f} R={row['train_loss_roi']:.4f} | "
            f"val Umag G={row['val_global_nrmse_umag']:.4f} R={row['val_roi_nrmse_umag']:.4f} | "
            f"val p G={row['val_global_nrmse_p']:.4f} R={row['val_roi_nrmse_p']:.4f} | "
            f"sel={row['val_selector']:.4f}",
            flush=True,
        )
        wandb_payload = {
            'train/loss_global': row['train_loss_global'],
            'train/loss_roi': row['train_loss_roi'],
            'train/div_rms_global': row['train_div_rms_global'],
            'train/div_rms_roi': row['train_div_rms_roi'],
            'train/mom_rms_global': row['train_mom_rms_global'],
            'train/mom_rms_roi': row['train_mom_rms_roi'],
            'train/wall_bc_roi': row['train_wall_bc_roi'],
            'val/global_nrmse_umag': row['val_global_nrmse_umag'],
            'val/global_nrmse_p': row['val_global_nrmse_p'],
            'val/roi_nrmse_umag': row['val_roi_nrmse_umag'],
            'val/roi_nrmse_p': row['val_roi_nrmse_p'],
            'val/selector_umag': row['val_selector_umag'],
            'val/selector_p': row['val_selector_p'],
            'val/selector_ms_roi_umag': row['val_selector_ms_roi_umag'],
            'val/selector': row['val_selector'],
            'train/lr': row['lr'],
        }
        for short in _CATEGORY_SHORT.values():
            wandb_payload[f'val/global_{short}_nrmse_umag'] = row[f'val_global_{short}_nrmse_umag']
            wandb_payload[f'val/global_{short}_nrmse_p'] = row[f'val_global_{short}_nrmse_p']
            wandb_payload[f'val/roi_{short}_nrmse_umag'] = row[f'val_roi_{short}_nrmse_umag']
            wandb_payload[f'val/roi_{short}_nrmse_p'] = row[f'val_roi_{short}_nrmse_p']
        wandb_log(wandb_run, wandb_payload, step=epoch)

        ckpt = {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scalers': scalers,
            'split': split,
            'history': history,
            'best_val': float(best_score),
            'best_epoch': int(best_epoch),
            'train_config': {
                'train_mode': str(train_mode),
                'train_loss': str(train_loss),
                'momentum_loss_mode': str(momentum_loss_mode),
                'train_struct_mode': str(train_struct_mode),
                'train_struct_weight': float(train_struct_weight),
                'scheduler_mode': str(scheduler_mode),
                'hard_ground_bc': bool(hard_ground_bc),
                'use_amp': bool(use_amp),
                'amp_dtype': str(amp_dtype),
                'model_kind': 'grid_unet',
            },
        }
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        if scaler.is_enabled():
            ckpt['scaler_state_dict'] = scaler.state_dict()
        (save_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, save_dir / 'checkpoints' / 'latest.pth')
        if epoch >= int(MIN_EPOCH_FOR_BEST) and selector < best_score:
            best_score = float(selector)
            best_epoch = int(epoch)
            ckpt['best_val'] = float(best_score)
            ckpt['best_epoch'] = int(best_epoch)
            torch.save(ckpt, save_dir / 'checkpoints' / 'best.pth')
        patience = int(max(0, EARLY_STOPPING_PATIENCE))
        if patience > 0 and best_epoch >= int(MIN_EPOCH_FOR_BEST):
            stale_epochs = int(epoch) - int(best_epoch)
            if stale_epochs >= patience:
                stop_epoch = int(epoch)
                print(
                    f"[TRAIN] early stop at epoch={int(epoch)} | best epoch={int(best_epoch)} | "
                    f"stale={int(stale_epochs)}",
                    flush=True,
                )
                break

    if best_epoch < 0:
        shutil.copy2(save_dir / 'checkpoints' / 'latest.pth', save_dir / 'checkpoints' / 'best.pth')
        best_epoch = int(stop_epoch)
        best_score = float(history[-1]['val_selector']) if history else float('inf')

    write_json(save_dir / 'checkpoint_paths.json', {
        'latest': str(save_dir / 'checkpoints' / 'latest.pth'),
        'best': str(save_dir / 'checkpoints' / 'best.pth'),
    })
    wandb_log(
        wandb_run,
        {
            'best/epoch': int(best_epoch),
            'best/val_selector': float(best_score),
        },
        step=int(stop_epoch) if stop_epoch > 0 else None,
    )
    print(f"[TRAIN] done | best epoch={int(best_epoch)} | best selector={float(best_score):.4f}", flush=True)
    return {
        'save_dir': str(save_dir),
        'stop_epoch': int(stop_epoch),
        'best_epoch': int(best_epoch),
        'best_val': float(best_score),
        'scalers': scalers,
        'history': history,
    }


def _train_cascade_stage1_model(
    model,
    repo: CaseRepository,
    split: dict,
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    save_dir: Path,
    train_mode: str = TRAIN_MODE,
    train_loss: str = TRAIN_LOSS,
    momentum_loss_mode: str = MOMENTUM_LOSS_MODE,
    train_struct_mode: str = TRAIN_STRUCT_MODE,
    train_struct_weight: float = TRAIN_STRUCT_WEIGHT,
    scheduler_mode: str = SCHEDULER_MODE,
    hard_ground_bc: bool = HARD_GROUND_BC,
    charb_eps: float = CHARB_EPS,
    pred_batch_size: int = 200000,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
    resume_checkpoint: Optional[dict] = None,
    config_snapshot: Optional[dict] = None,
    wandb_run=None,
) -> Dict:
    if resume_checkpoint is not None:
        scalers = resume_checkpoint['scalers']
        history = list(resume_checkpoint.get('history', []))
        best_score = float(resume_checkpoint.get('best_val', float('inf')))
        best_epoch = int(resume_checkpoint.get('best_epoch', -1))
        start_epoch = int(resume_checkpoint.get('epoch', 0))
    else:
        scalers = fit_scalers_global_only(repo, split['train'])
        history = []
        best_score = float('inf')
        best_epoch = -1
        start_epoch = 0
    rng = _training_rng(int(start_epoch))
    stop_epoch = int(epochs)
    ensure_dir(save_dir)
    tensor_cache = _DeviceTensorCache(device=device)
    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=_use_cuda_amp(device, use_amp) and _amp_torch_dtype(amp_dtype) == torch.float16,
    )

    weights = _effective_weights(train_mode)
    steps_per_epoch = len(split['train'])
    scheduler = _build_scheduler(optimizer, epochs=int(epochs), steps_per_epoch=steps_per_epoch, scheduler_mode=scheduler_mode)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        if scaler.is_enabled() and 'scaler_state_dict' in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
    train_structure_cases = int(sum(1 for name in split['train'] if repo.roi_names(name)))
    print(
        f"[TRAIN] mode={train_mode} | model=cascade_stage1 | epochs={int(epochs)} | "
        f"train domains={len(split['train'])} | train structure domains={train_structure_cases} | "
        f"steps/epoch={steps_per_epoch} | mom={momentum_loss_mode} | "
        f"amp={'off' if not use_amp else amp_dtype}",
        flush=True,
    )
    if start_epoch > 0:
        print(
            f"[TRAIN] resuming from epoch={int(start_epoch)} | best epoch={int(best_epoch)} | "
            f"best selector={float(best_score):.4f}",
            flush=True,
        )
    if int(start_epoch) >= int(epochs):
        print(
            f"[TRAIN] target epochs already reached in latest checkpoint "
            f"(epoch={int(start_epoch)} >= {int(epochs)}), skipping training loop",
            flush=True,
        )
        stop_epoch = int(start_epoch)

    for epoch in range(int(start_epoch) + 1, int(epochs) + 1):
        weights = _effective_weights(train_mode, epoch=epoch)
        model.train()
        train_global_losses = []
        train_div_global = []
        train_mom_global = []
        rng.shuffle(split['train'])
        use_struct_loss = float(train_struct_weight) > 0.0 and str(train_struct_mode).lower() != 'none'

        for name in split['train']:
            g = repo.load_global(name)
            _raise_if_tf_nondata_loss(g, weights, scope='global', structured_loss_enabled=use_struct_loss)
            gterr = tensor_cache.terrain(g)
            batch = sample_supervised_batch(
                g,
                x_scaler=scalers.x_scaler_global,
                y_scaler=scalers.y_scaler,
                n_points=GLOBAL_POINTS_PER_DOMAIN,
                rng=rng,
            )
            y_true = batch.y_scaled.to(device)
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                gfeat = model.encode_global(gterr)
                pred_scaled, _ = _predict_global(
                    model,
                    g,
                    batch,
                    scalers.y_scaler,
                    scalers.x_scaler_global,
                    device=device,
                    hard_ground_bc=hard_ground_bc,
                    terr=gterr,
                    gfeat=gfeat,
                )
                loss_data = supervised_data_loss_from_pred(
                    pred_scaled,
                    y_true,
                    x_batch=batch.x_scaled.to(device),
                    x_scaler=scalers.x_scaler_global,
                    input_cols=GLOBAL_INPUT_COLS,
                    y_scaler=scalers.y_scaler,
                    p_weight=float(GLOBAL_DATA_P_WEIGHT),
                    loss_mode=train_loss,
                    charbonnier_eps=charb_eps,
                )
                loss_phys = torch.tensor(0.0, device=device)
                struct_loss = torch.tensor(0.0, device=device)
                if weights['w_phys_global'] > 0 or use_struct_loss:
                    for _ in range(int(GLOBAL_PATCHES_PER_DOMAIN)):
                        patch = sample_patch_batch(
                            g,
                            x_scaler=scalers.x_scaler_global,
                            y_scaler=scalers.y_scaler,
                            patch_shape=GLOBAL_PATCH_SHAPE,
                            rng=rng,
                            near_ground_prob=PATCH_NEAR_GROUND_PROB,
                        )
                        patch_pred_raw = model.forward_global_from_encoded(
                            gfeat,
                            patch.x_scaled.to(device),
                            patch.xy_local.to(device),
                        )
                        patch_pred, _ = _compose_global_prediction_from_raw(
                            model,
                            patch_pred_raw,
                            x_batch=patch.x_scaled.to(device),
                            x_scaler=scalers.x_scaler_global,
                            y_scaler=scalers.y_scaler,
                            hard_ground_bc=hard_ground_bc,
                        )
                        if weights['w_phys_global'] > 0:
                            phys = compute_patch_physics_losses_from_pred(
                                patch_pred,
                                patch,
                                scalers.y_scaler,
                                device=device,
                                momentum_loss_mode=momentum_loss_mode,
                            )
                            loss_phys = (
                                loss_phys
                                + weights['w_div_global'] * phys['div_loss']
                                + weights['w_mom_global'] * phys['mom_loss']
                            )
                            train_div_global.append(float(phys['div_rms'].detach().cpu().item()))
                            train_mom_global.append(float(phys['mom_rms'].detach().cpu().item()))
                        if use_struct_loss:
                            struct_loss = struct_loss + structured_patch_loss_from_pred(
                                patch_pred,
                                patch,
                                mode=train_struct_mode,
                                p_weight=float(GLOBAL_DATA_P_WEIGHT),
                                charbonnier_eps=charb_eps,
                            )
            bc_batches = prepare_global_boundary_batches(g, x_scaler=scalers.x_scaler_global, rng=rng)
            inlet_loss = torch.tensor(0.0, device=device)
            outlet_loss = torch.tensor(0.0, device=device)
            side_loss = torch.tensor(0.0, device=device)
            top_loss = torch.tensor(0.0, device=device)
            if bc_batches.get('inlet') is not None and weights['w_bc_inlet'] > 0:
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    inlet_loss = inlet_bc_loss_from_phys(
                        _predict_global_boundary(
                            model,
                            g,
                            bc_batches['inlet'],
                            scalers,
                            device=device,
                            hard_ground_bc=hard_ground_bc,
                            terr=gterr,
                            gfeat=gfeat,
                        ),
                        bc_batches['inlet'],
                        device=device,
                    )
            if bc_batches.get('outlet') is not None and weights['w_bc_outlet'] > 0:
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    outlet_loss = outlet_bc_loss_from_phys(
                        _predict_global_boundary(
                            model,
                            g,
                            bc_batches['outlet'],
                            scalers,
                            device=device,
                            hard_ground_bc=hard_ground_bc,
                            terr=gterr,
                            gfeat=gfeat,
                        ),
                        bc_batches['outlet'],
                        device=device,
                    )
            if bc_batches.get('side') is not None and weights['w_bc_side'] > 0:
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    side_loss = normal_velocity_bc_loss_from_phys(
                        _predict_global_boundary(
                            model,
                            g,
                            bc_batches['side'],
                            scalers,
                            device=device,
                            hard_ground_bc=hard_ground_bc,
                            terr=gterr,
                            gfeat=gfeat,
                        ),
                        bc_batches['side'],
                        device=device,
                    )
            if bc_batches.get('top') is not None and weights['w_bc_top'] > 0:
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    top_loss = normal_velocity_bc_loss_from_phys(
                        _predict_global_boundary(
                            model,
                            g,
                            bc_batches['top'],
                            scalers,
                            device=device,
                            hard_ground_bc=hard_ground_bc,
                            terr=gterr,
                            gfeat=gfeat,
                        ),
                        bc_batches['top'],
                        device=device,
                    )
            total = (
                float(W_DATA_GLOBAL) * loss_data
                + weights['w_phys_global'] * loss_phys
                + float(train_struct_weight) * struct_loss
                + weights['w_bc_inlet'] * inlet_loss
                + weights['w_bc_outlet'] * outlet_loss
                + weights['w_bc_side'] * side_loss
                + weights['w_bc_top'] * top_loss
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(total).backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_global_losses.append(float(total.detach().cpu().item()))

        model.eval()
        val_global_metrics = {}
        for name in split['val']:
            g = repo.load_global(name)
            gterr = tensor_cache.terrain(g)
            with torch.no_grad():
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    gfeat = model.encode_global(gterr)
            val_global_metrics[name], _ = _evaluate_case_global(
                model,
                g,
                scalers,
                device=device,
                pred_batch_size=pred_batch_size,
                hard_ground_bc=hard_ground_bc,
                terr=gterr,
                gfeat=gfeat,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

        global_score = float(np.nanmean([m['nrmse_umag'] for m in val_global_metrics.values()])) if val_global_metrics else float('nan')
        global_p_score = float(np.nanmean([m['nrmse_p'] for m in val_global_metrics.values()])) if val_global_metrics else float('nan')
        val_global_by_cat = _category_metrics(repo, val_global_metrics)
        selector_scores = _selector_components(
            val_global_metrics=val_global_metrics,
            val_roi_metrics={},
            val_roi_by_cat={},
        )
        selector_umag = float(selector_scores['selector_umag'])
        selector_p = float(selector_scores['selector_p'])
        selector_ms_roi_umag = float(selector_scores['selector_ms_roi_umag'])
        selector = float(selector_scores['selector'])
        row = {
            'epoch': int(epoch),
            'lr': float(optimizer.param_groups[0]['lr']),
            'train_loss_global': float(np.mean(train_global_losses)) if train_global_losses else float('nan'),
            'train_loss_roi': float('nan'),
            'train_div_rms_global': float(np.mean(train_div_global)) if train_div_global else float('nan'),
            'train_div_rms_roi': float('nan'),
            'train_mom_rms_global': float(np.mean(train_mom_global)) if train_mom_global else float('nan'),
            'train_mom_rms_roi': float('nan'),
            'train_wall_bc_roi': float('nan'),
            'val_global_nrmse_umag': global_score,
            'val_global_nrmse_p': global_p_score,
            'val_roi_nrmse_umag': float('nan'),
            'val_roi_nrmse_p': float('nan'),
            'val_selector_umag': selector_umag,
            'val_selector_p': selector_p,
            'val_selector_ms_roi_umag': selector_ms_roi_umag,
            'val_selector': selector,
        }
        for short in _CATEGORY_SHORT.values():
            row[f'val_global_{short}_nrmse_umag'] = float(val_global_by_cat[short]['nrmse_umag'])
            row[f'val_global_{short}_nrmse_p'] = float(val_global_by_cat[short]['nrmse_p'])
            row[f'val_roi_{short}_nrmse_umag'] = float('nan')
            row[f'val_roi_{short}_nrmse_p'] = float('nan')
        history.append(row)
        write_json(save_dir / 'logs' / 'history.json', history)
        print(
            f"Epoch {epoch:4d}/{int(epochs)} | lr={row['lr']:.2e} | "
            f"train G={row['train_loss_global']:.4f} | "
            f"val Umag G={row['val_global_nrmse_umag']:.4f} | "
            f"val p G={row['val_global_nrmse_p']:.4f} | sel={row['val_selector']:.4f}",
            flush=True,
        )
        wandb_payload = {
            'train/loss_global': row['train_loss_global'],
            'train/div_rms_global': row['train_div_rms_global'],
            'train/mom_rms_global': row['train_mom_rms_global'],
            'val/global_nrmse_umag': row['val_global_nrmse_umag'],
            'val/global_nrmse_p': row['val_global_nrmse_p'],
            'val/selector_umag': row['val_selector_umag'],
            'val/selector_p': row['val_selector_p'],
            'val/selector_ms_roi_umag': row['val_selector_ms_roi_umag'],
            'val/selector': row['val_selector'],
            'train/lr': row['lr'],
        }
        for short in _CATEGORY_SHORT.values():
            wandb_payload[f'val/global_{short}_nrmse_umag'] = row[f'val_global_{short}_nrmse_umag']
            wandb_payload[f'val/global_{short}_nrmse_p'] = row[f'val_global_{short}_nrmse_p']
        wandb_log(wandb_run, wandb_payload, step=epoch)

        ckpt = {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scalers': scalers,
            'split': split,
            'history': history,
            'best_val': float(best_score),
            'best_epoch': int(best_epoch),
            'config_snapshot': dict(config_snapshot or {}),
            'train_config': {
                'train_mode': str(train_mode),
                'train_loss': str(train_loss),
                'momentum_loss_mode': str(momentum_loss_mode),
                'train_struct_mode': str(train_struct_mode),
                'train_struct_weight': float(train_struct_weight),
                'scheduler_mode': str(scheduler_mode),
                'hard_ground_bc': bool(hard_ground_bc),
                'use_amp': bool(use_amp),
                'amp_dtype': str(amp_dtype),
                'model_kind': 'cascade_stage1',
                'config_snapshot': dict(config_snapshot or {}),
            },
        }
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        if scaler.is_enabled():
            ckpt['scaler_state_dict'] = scaler.state_dict()
        (save_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, save_dir / 'checkpoints' / 'latest.pth')
        if epoch >= int(MIN_EPOCH_FOR_BEST) and selector < best_score:
            best_score = float(selector)
            best_epoch = int(epoch)
            ckpt['best_val'] = float(best_score)
            ckpt['best_epoch'] = int(best_epoch)
            torch.save(ckpt, save_dir / 'checkpoints' / 'best.pth')
        patience = int(max(0, EARLY_STOPPING_PATIENCE))
        if patience > 0 and best_epoch >= int(MIN_EPOCH_FOR_BEST):
            stale_epochs = int(epoch) - int(best_epoch)
            if stale_epochs >= patience:
                stop_epoch = int(epoch)
                print(
                    f"[TRAIN] early stop at epoch={int(epoch)} | best epoch={int(best_epoch)} | "
                    f"stale={int(stale_epochs)}",
                    flush=True,
                )
                break

    if best_epoch < 0:
        shutil.copy2(save_dir / 'checkpoints' / 'latest.pth', save_dir / 'checkpoints' / 'best.pth')
        best_epoch = int(stop_epoch)
        best_score = float(history[-1]['val_selector']) if history else float('inf')

    write_json(save_dir / 'checkpoint_paths.json', {
        'latest': str(save_dir / 'checkpoints' / 'latest.pth'),
        'best': str(save_dir / 'checkpoints' / 'best.pth'),
    })
    wandb_log(
        wandb_run,
        {
            'best/epoch': int(best_epoch),
            'best/val_selector': float(best_score),
        },
        step=int(stop_epoch) if stop_epoch > 0 else None,
    )
    print(f"[TRAIN] done | best epoch={int(best_epoch)} | best selector={float(best_score):.4f}", flush=True)
    return {
        'save_dir': str(save_dir),
        'stop_epoch': int(stop_epoch),
        'best_epoch': int(best_epoch),
        'best_val': float(best_score),
        'scalers': scalers,
        'history': history,
    }


def _structure_count_for_repeat(bundle) -> int:
    bounds = bundle.meta.get('structure_bounds') if isinstance(bundle.meta, dict) else None
    if isinstance(bounds, list):
        return len(bounds)
    count = bundle.meta.get('n_structures') if isinstance(bundle.meta, dict) else None
    try:
        return int(count)
    except Exception:
        return 0


def _cascade_stage2_ms_repeat_count(repo: CaseRepository, case_name: str) -> int:
    if not bool(CASCADE_STAGE2_MS_REPEAT_ENABLED):
        return 1
    try:
        n_structures = _structure_count_for_repeat(repo.load_global(case_name))
    except Exception:
        n_structures = 0
    repeat = 1
    if n_structures >= int(CASCADE_STAGE2_MS_REPEAT_N2):
        repeat = 2
    if n_structures >= int(CASCADE_STAGE2_MS_REPEAT_N3):
        repeat = 3
    return max(1, min(int(CASCADE_STAGE2_MS_REPEAT_MAX), int(repeat)))


def _expand_cascade_stage2_train_roi_refs(repo: CaseRepository, refs: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], dict[str, int]]:
    if not bool(CASCADE_STAGE2_MS_REPEAT_ENABLED):
        return refs, {}
    repeat_by_case: dict[str, int] = {}
    expanded: list[tuple[str, str]] = []
    for name, roi_name in refs:
        repeat = repeat_by_case.get(name)
        if repeat is None:
            repeat = _cascade_stage2_ms_repeat_count(repo, name)
            repeat_by_case[name] = repeat
        expanded.extend([(name, roi_name)] * repeat)
    return expanded, repeat_by_case


def _roi_ref_cell_count(repo: CaseRepository, case_name: str, roi_name: str) -> int:
    flow_path = Path(repo.case_dirs[case_name]) / 'roi' / roi_name / 'flow.npz'
    try:
        with zipfile.ZipFile(flow_path, 'r') as zf:
            names = zf.namelist()
            member = 'Ux.npy' if 'Ux.npy' in names else next(name for name in names if name.endswith('.npy'))
            with zf.open(member, 'r') as fp:
                version = np.lib.format.read_magic(fp)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(fp)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(fp)
                else:
                    shape, _, _ = np.lib.format._read_array_header(fp, version)  # type: ignore[attr-defined]
                shape = tuple(int(v) for v in shape)
    except Exception:
        bundle = repo.load_roi(case_name, roi_name)
        shape = tuple(int(v) for v in bundle.flow.shape[:3])
    return int(np.prod(shape, dtype=np.int64))


def _filter_cascade_grid_roi_refs(
    repo: CaseRepository,
    refs: list[tuple[str, str]],
    *,
    split_label: str,
    skipped: list[dict],
) -> list[tuple[str, str]]:
    threshold = int(CASCADE_STAGE2_GRID_MAX_ROI_CELLS)
    if threshold <= 0:
        return refs
    kept: list[tuple[str, str]] = []
    local_skipped: list[dict] = []
    for case_name, roi_name in refs:
        cells = _roi_ref_cell_count(repo, case_name, roi_name)
        if cells > threshold:
            record = {
                'split': str(split_label),
                'case': str(case_name),
                'roi': str(roi_name),
                'ref': f'{case_name}/{roi_name}',
                'cells': int(cells),
                'threshold': int(threshold),
                'reason': 'roi_cells_above_CASCADE_STAGE2_GRID_MAX_ROI_CELLS',
            }
            local_skipped.append(record)
            skipped.append(record)
            continue
        kept.append((case_name, roi_name))
    if local_skipped:
        refs_text = ', '.join(f"{r['ref']}={r['cells']}" for r in local_skipped[:12])
        if len(local_skipped) > 12:
            refs_text += f", ... +{len(local_skipped) - 12} more"
        print(
            f"[TRAIN] cascade_stage2:grid_unet skipped {len(local_skipped)} {split_label} ROI refs "
            f"above {threshold} cells: [{refs_text}]",
            flush=True,
        )
    return kept


def _write_skipped_large_roi(save_dir: Path, skipped: list[dict]) -> None:
    payload = {
        'enabled': True,
        'threshold_cells': int(CASCADE_STAGE2_GRID_MAX_ROI_CELLS),
        'count': int(len(skipped)),
        'skipped': skipped,
    }
    write_json(save_dir / 'skipped_large_roi.json', payload)


def _train_cascade_stage2_model(
    model,
    repo: CaseRepository,
    split: dict,
    *,
    conditioner: CascadeConditioner,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    save_dir: Path,
    train_mode: str = TRAIN_MODE,
    train_loss: str = TRAIN_LOSS,
    momentum_loss_mode: str = MOMENTUM_LOSS_MODE,
    train_struct_mode: str = TRAIN_STRUCT_MODE,
    train_struct_weight: float = TRAIN_STRUCT_WEIGHT,
    scheduler_mode: str = SCHEDULER_MODE,
    hard_ground_bc: bool = HARD_GROUND_BC,
    charb_eps: float = CHARB_EPS,
    pred_batch_size: int = 200000,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
    resume_checkpoint: Optional[dict] = None,
    config_snapshot: Optional[dict] = None,
    cascade_stage1_checkpoint: str = '',
    wandb_run=None,
) -> Dict:
    if conditioner is None:
        raise RuntimeError('cascade_stage2 requires a stage1 conditioner')
    ensure_dir(save_dir)
    grid_refiner = _is_cascade_grid_refiner(model)
    skipped_large_roi: list[dict] = []
    base_train_roi_refs = [(name, roi_name) for name in split['train'] for roi_name in repo.roi_names(name)]
    val_roi_refs = [(name, roi_name) for name in split['val'] for roi_name in repo.roi_names(name)]
    if grid_refiner:
        base_train_roi_refs = _filter_cascade_grid_roi_refs(
            repo,
            base_train_roi_refs,
            split_label='train',
            skipped=skipped_large_roi,
        )
        val_roi_refs = _filter_cascade_grid_roi_refs(
            repo,
            val_roi_refs,
            split_label='val',
            skipped=skipped_large_roi,
        )
        _write_skipped_large_roi(save_dir, skipped_large_roi)
    if resume_checkpoint is not None:
        scalers = resume_checkpoint['scalers']
        history = list(resume_checkpoint.get('history', []))
        best_score = float(resume_checkpoint.get('best_val', float('inf')))
        best_epoch = int(resume_checkpoint.get('best_epoch', -1))
        start_epoch = int(resume_checkpoint.get('epoch', 0))
    else:
        scalers = fit_scalers_roi_refs(repo, base_train_roi_refs) if grid_refiner else fit_scalers_roi_only(repo, split['train'])
        history = []
        best_score = float('inf')
        best_epoch = -1
        start_epoch = 0
    rng = _training_rng(int(start_epoch))
    stop_epoch = int(epochs)
    tensor_cache = _DeviceTensorCache(device=device)
    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=_use_cuda_amp(device, use_amp) and _amp_torch_dtype(amp_dtype) == torch.float16,
    )

    train_roi_refs, ms_repeat_by_case = _expand_cascade_stage2_train_roi_refs(repo, base_train_roi_refs)
    structure_case_count = int(len({name for name, _ in base_train_roi_refs}))
    if structure_case_count < int(CASCADE_MIN_STRUCTURE_CASES):
        raise RuntimeError(
            f'cascade_stage2 requires at least {int(CASCADE_MIN_STRUCTURE_CASES)} '
            f'structure train cases, found {structure_case_count}'
        )
    weights = _effective_weights(train_mode)
    grid_patches_per_roi = max(1, int(ROI_PATCHES_PER_DOMAIN)) if grid_refiner else 1
    steps_per_epoch = len(train_roi_refs) * grid_patches_per_roi
    scheduler = _build_scheduler(optimizer, epochs=int(epochs), steps_per_epoch=steps_per_epoch, scheduler_mode=scheduler_mode)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        if scaler.is_enabled() and 'scaler_state_dict' in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
    for p in conditioner.model.parameters():
        p.requires_grad_(False)
    conditioner.model.eval()
    if _is_grid_unet_conditioner(conditioner):
        conditioner.grid_global_cache = OrderedDict()
    print(
        f"[TRAIN] mode={train_mode} | model=cascade_stage2:{str(CASCADE_STAGE2_REFINER_KIND)} | epochs={int(epochs)} | "
        f"train structure domains={structure_case_count} | train ROI={len(base_train_roi_refs)} | "
        f"steps/epoch={steps_per_epoch} | mom={momentum_loss_mode} | "
        f"amp={'off' if not use_amp else amp_dtype}",
        flush=True,
    )
    if ms_repeat_by_case:
        repeated = {name: repeat for name, repeat in sorted(ms_repeat_by_case.items()) if int(repeat) > 1}
        print(
            f"[TRAIN] cascade_stage2 MS repeat enabled | repeated_cases={len(repeated)} | "
            f"expanded ROI steps={len(base_train_roi_refs)}->{len(train_roi_refs)} | "
            f"max_repeat={int(CASCADE_STAGE2_MS_REPEAT_MAX)}",
            flush=True,
        )
    if start_epoch > 0:
        print(
            f"[TRAIN] resuming from epoch={int(start_epoch)} | best epoch={int(best_epoch)} | "
            f"best selector={float(best_score):.4f}",
            flush=True,
        )
    if int(start_epoch) >= int(epochs):
        print(
            f"[TRAIN] target epochs already reached in latest checkpoint "
            f"(epoch={int(start_epoch)} >= {int(epochs)}), skipping training loop",
            flush=True,
        )
        stop_epoch = int(start_epoch)

    # Global validation metrics are inherited from the frozen stage1 conditioner.
    val_global_metrics = {}
    for name in split['val']:
        g = repo.load_global(name)
        gterr = tensor_cache.terrain(g)
        if _is_grid_unet_conditioner(conditioner):
            val_global_metrics[name], _ = _evaluate_case_global_grid_unet(
                conditioner.model,
                g,
                conditioner.scalers,
                device=device,
                hard_ground_bc=hard_ground_bc,
                patch_shape=_conditioner_global_patch_shape(conditioner),
            )
        else:
            with torch.no_grad():
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    gfeat = conditioner.model.encode_global(gterr)
            val_global_metrics[name], _ = _evaluate_case_global(
                conditioner.model,
                g,
                conditioner.scalers,
                device=device,
                pred_batch_size=pred_batch_size,
                hard_ground_bc=hard_ground_bc,
                terr=gterr,
                gfeat=gfeat,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
    global_score = float(np.nanmean([m['nrmse_umag'] for m in val_global_metrics.values()])) if val_global_metrics else float('nan')
    global_p_score = float(np.nanmean([m['nrmse_p'] for m in val_global_metrics.values()])) if val_global_metrics else float('nan')
    val_global_by_cat = _category_metrics(repo, val_global_metrics)

    for epoch in range(int(start_epoch) + 1, int(epochs) + 1):
        weights = _effective_weights(train_mode, epoch=epoch)
        model.train()
        conditioner.model.eval()
        train_roi_losses = []
        train_div_roi = []
        train_mom_roi = []
        train_wall_roi = []
        train_edge_roi = []
        train_sample_stats: dict[str, list[float]] = {}
        rng.shuffle(train_roi_refs)
        use_struct_loss = float(train_struct_weight) > 0.0 and str(train_struct_mode).lower() != 'none'

        for name, roi_name in train_roi_refs:
            g = repo.load_global(name)
            r = repo.load_roi(name, roi_name)
            _raise_if_tf_nondata_loss(g, weights, scope='global', structured_loss_enabled=False)
            _raise_if_tf_nondata_loss(r, weights, scope='roi', structured_loss_enabled=use_struct_loss)
            bgterr = tensor_cache.terrain(g)
            rterr = tensor_cache.terrain(r)
            sterr = tensor_cache.structure(r)
            bgfeat = None
            if not _is_grid_unet_conditioner(conditioner):
                with torch.no_grad():
                    with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                        bgfeat = conditioner.model.encode_global(bgterr)
            if grid_refiner:
                for _patch_idx in range(grid_patches_per_roi):
                    patch = sample_patch_batch(
                        r,
                        x_scaler=scalers.x_scaler_roi,
                        y_scaler=scalers.y_scaler,
                        patch_shape=ROI_PATCH_SHAPE,
                        rng=rng,
                        near_ground_prob=PATCH_NEAR_GROUND_PROB,
                        parent_global=g,
                        include_grid_unet_context=True,
                    )
                    y_true = patch.y_scaled.to(device)
                    pred_scaled, pred_phys, raw_resid_flat, bgterr, bgfeat = _cascade_grid_patch_pred_scaled(
                        model,
                        conditioner,
                        g,
                        patch,
                        scalers,
                        device=device,
                        hard_ground_bc=hard_ground_bc,
                        bgterr=bgterr,
                        bgfeat=bgfeat,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                    )
                    valid = _patch_valid_flat_mask(patch, pred_scaled, y_true, device=device)
                    if not bool(valid.any().item()):
                        continue
                    x_patch = patch.x_scaled.to(device)
                    with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                        loss_data = supervised_data_loss_from_pred(
                            pred_scaled[valid],
                            y_true[valid],
                            x_batch=x_patch[valid],
                            x_scaler=scalers.x_scaler_roi,
                            input_cols=ROI_INPUT_COLS,
                            y_scaler=scalers.y_scaler,
                            p_weight=float(ROI_DATA_P_WEIGHT),
                            loss_mode=train_loss,
                            charbonnier_eps=charb_eps,
                        )
                        edge_loss = _cascade_edge_residual_loss(
                            raw_resid_flat[valid],
                            x_batch=x_patch[valid],
                            x_scaler=scalers.x_scaler_roi,
                            input_cols=ROI_INPUT_COLS,
                            bounds=r.bounds,
                            device=device,
                        )
                        loss_phys = torch.tensor(0.0, device=device)
                        struct_loss = torch.tensor(0.0, device=device)
                        wall_loss = torch.tensor(0.0, device=device)
                        if weights['w_phys_roi'] > 0:
                            phys = compute_patch_physics_losses_from_pred(
                                pred_scaled,
                                patch,
                                scalers.y_scaler,
                                device=device,
                                momentum_loss_mode=momentum_loss_mode,
                            )
                            loss_phys = weights['w_div_roi'] * phys['div_loss'] + weights['w_mom_roi'] * phys['mom_loss']
                            train_div_roi.append(float(phys['div_rms'].detach().cpu().item()))
                            train_mom_roi.append(float(phys['mom_rms'].detach().cpu().item()))
                        if use_struct_loss:
                            struct_loss = structured_patch_loss_from_pred(
                                pred_scaled,
                                patch,
                                mode=train_struct_mode,
                                p_weight=float(ROI_DATA_P_WEIGHT),
                                charbonnier_eps=charb_eps,
                            )
                        if weights['w_bc_wall_roi'] > 0:
                            wall_loss = roi_wall_velocity_bc_loss_from_phys(
                                pred_phys,
                                x_batch=patch.x_scaled,
                                x_scaler=scalers.x_scaler_roi,
                                input_cols=ROI_INPUT_COLS,
                                device=device,
                                u_scale=float(r.uref),
                                dmax=float(ROI_WALL_BC_DMAX),
                            )
                        total = (
                            float(W_DATA_ROI) * loss_data
                            + weights['w_phys_roi'] * loss_phys
                            + float(train_struct_weight) * struct_loss
                            + weights['w_bc_wall_roi'] * wall_loss
                            + edge_loss
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.scale(total).backward()
                        if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        total.backward()
                        if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    train_roi_losses.append(float(total.detach().cpu().item()))
                    train_edge_roi.append(float(edge_loss.detach().cpu().item()))
                    if weights['w_bc_wall_roi'] > 0:
                        train_wall_roi.append(float(wall_loss.detach().cpu().item()))
                continue
            if str(ROI_SUPERVISED_SAMPLER_MODE).lower() == 'targeted_v1':
                batch = sample_targeted_roi_supervised_batch(
                    r,
                    x_scaler=scalers.x_scaler_roi,
                    y_scaler=scalers.y_scaler,
                    n_points=ROI_POINTS_PER_DOMAIN,
                    rng=rng,
                    parent_global=g,
                )
            else:
                batch = sample_supervised_batch(
                    r,
                    x_scaler=scalers.x_scaler_roi,
                    y_scaler=scalers.y_scaler,
                    n_points=ROI_POINTS_PER_DOMAIN,
                    rng=rng,
                    parent_global=g,
                )
            if batch.sample_stats:
                for key, value in batch.sample_stats.items():
                    train_sample_stats.setdefault(str(key), []).append(float(value))
            y_true = batch.y_scaled.to(device)
            with torch.no_grad():
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    bg_scaled, _, bgterr, bgfeat = _cascade_conditioner_predict_on_roi_inputs(
                        conditioner,
                        g,
                        x_scaled_roi=batch.x_scaled.to(device),
                        x_scaler_roi=scalers.x_scaler_roi,
                        target_y_scaler=scalers.y_scaler,
                        device=device,
                        hard_ground_bc=hard_ground_bc,
                        gterr=bgterr,
                        gfeat=bgfeat,
                    )
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                rfeat = model.encode_roi(rterr)
                sfeat = model.encode_structure(sterr)
                raw_resid_scaled = model.forward_roi_from_encoded(
                    rfeat,
                    batch.x_scaled.to(device),
                    batch.xy_local.to(device),
                    bg_scaled,
                    s_feat=sfeat,
                )
                pred_scaled, _ = apply_output_constraints_from_scaled_inputs(
                    raw_resid_scaled + bg_scaled,
                    x_batch=batch.x_scaled.to(device),
                    x_scaler=scalers.x_scaler_roi,
                    input_cols=ROI_INPUT_COLS,
                    y_scaler=scalers.y_scaler,
                    hard_ground_bc=hard_ground_bc,
                )
                loss_data = supervised_data_loss_from_pred(
                    pred_scaled,
                    y_true,
                    x_batch=batch.x_scaled.to(device),
                    x_scaler=scalers.x_scaler_roi,
                    input_cols=ROI_INPUT_COLS,
                    y_scaler=scalers.y_scaler,
                    p_weight=float(ROI_DATA_P_WEIGHT),
                    loss_mode=train_loss,
                    charbonnier_eps=charb_eps,
                )
                edge_loss = _cascade_edge_residual_loss(
                    raw_resid_scaled,
                    x_batch=batch.x_scaled,
                    x_scaler=scalers.x_scaler_roi,
                    input_cols=ROI_INPUT_COLS,
                    bounds=r.bounds,
                    device=device,
                )
                loss_phys = torch.tensor(0.0, device=device)
                struct_loss = torch.tensor(0.0, device=device)
                wall_loss = torch.tensor(0.0, device=device)
                patch_loop_count = int(ROI_PATCHES_PER_DOMAIN) if (
                    weights['w_phys_roi'] > 0 or weights['w_bc_wall_roi'] > 0 or use_struct_loss or float(CASCADE_EDGE_WEIGHT) > 0.0
                ) else 0
                for _ in range(patch_loop_count):
                    patch = sample_patch_batch(
                        r,
                        x_scaler=scalers.x_scaler_roi,
                        y_scaler=scalers.y_scaler,
                        patch_shape=ROI_PATCH_SHAPE,
                        rng=rng,
                        near_ground_prob=PATCH_NEAR_GROUND_PROB,
                        parent_global=g,
                    )
                    with torch.no_grad():
                        with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                            bg_scaled_patch, _, bgterr, bgfeat = _cascade_conditioner_predict_on_roi_inputs(
                                conditioner,
                                g,
                                x_scaled_roi=patch.x_scaled.to(device),
                                x_scaler_roi=scalers.x_scaler_roi,
                                target_y_scaler=scalers.y_scaler,
                                device=device,
                                hard_ground_bc=hard_ground_bc,
                                gterr=bgterr,
                                gfeat=bgfeat,
                            )
                    raw_resid_patch = model.forward_roi_from_encoded(
                        rfeat,
                        patch.x_scaled.to(device),
                        patch.xy_local.to(device),
                        bg_scaled_patch,
                        s_feat=sfeat,
                    )
                    patch_pred, patch_pred_phys = apply_output_constraints_from_scaled_inputs(
                        raw_resid_patch + bg_scaled_patch,
                        x_batch=patch.x_scaled.to(device),
                        x_scaler=scalers.x_scaler_roi,
                        input_cols=ROI_INPUT_COLS,
                        y_scaler=scalers.y_scaler,
                        hard_ground_bc=hard_ground_bc,
                    )
                    edge_loss = edge_loss + _cascade_edge_residual_loss(
                        raw_resid_patch,
                        x_batch=patch.x_scaled,
                        x_scaler=scalers.x_scaler_roi,
                        input_cols=ROI_INPUT_COLS,
                        bounds=r.bounds,
                        device=device,
                    )
                    if weights['w_phys_roi'] > 0:
                        phys = compute_patch_physics_losses_from_pred(
                            patch_pred,
                            patch,
                            scalers.y_scaler,
                            device=device,
                            momentum_loss_mode=momentum_loss_mode,
                        )
                        loss_phys = (
                            loss_phys
                            + weights['w_div_roi'] * phys['div_loss']
                            + weights['w_mom_roi'] * phys['mom_loss']
                        )
                        train_div_roi.append(float(phys['div_rms'].detach().cpu().item()))
                        train_mom_roi.append(float(phys['mom_rms'].detach().cpu().item()))
                    if use_struct_loss:
                        struct_loss = struct_loss + structured_patch_loss_from_pred(
                            patch_pred,
                            patch,
                            mode=train_struct_mode,
                            p_weight=float(ROI_DATA_P_WEIGHT),
                            charbonnier_eps=charb_eps,
                        )
                    if weights['w_bc_wall_roi'] > 0:
                        wall_loss = wall_loss + roi_wall_velocity_bc_loss_from_phys(
                            patch_pred_phys,
                            x_batch=patch.x_scaled,
                            x_scaler=scalers.x_scaler_roi,
                            input_cols=ROI_INPUT_COLS,
                            device=device,
                            u_scale=float(r.uref),
                            dmax=float(ROI_WALL_BC_DMAX),
                        )
            total = (
                float(W_DATA_ROI) * loss_data
                + weights['w_phys_roi'] * loss_phys
                + float(train_struct_weight) * struct_loss
                + weights['w_bc_wall_roi'] * wall_loss
                + edge_loss
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(total).backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_roi_losses.append(float(total.detach().cpu().item()))
            train_edge_roi.append(float(edge_loss.detach().cpu().item()))
            if weights['w_bc_wall_roi'] > 0:
                train_wall_roi.append(float(wall_loss.detach().cpu().item()))

        model.eval()
        val_roi_metrics = {}
        for name, roi_name in val_roi_refs:
            g = repo.load_global(name)
            r = repo.load_roi(name, roi_name)
            bgterr = tensor_cache.terrain(g)
            rterr = tensor_cache.terrain(r)
            sterr = tensor_cache.structure(r)
            with torch.no_grad():
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    bgfeat = None if _is_grid_unet_conditioner(conditioner) else conditioner.model.encode_global(bgterr)
                    if grid_refiner:
                        rfeat = None
                        sfeat = None
                    else:
                        rfeat = model.encode_roi(rterr)
                        sfeat = model.encode_structure(sterr)
            val_roi_metrics[f'{name}/{roi_name}'], _ = _evaluate_case_roi_cascade(
                model,
                conditioner,
                g,
                r,
                scalers,
                device=device,
                pred_batch_size=pred_batch_size,
                hard_ground_bc=hard_ground_bc,
                rterr=rterr,
                sterr=sterr,
                rfeat=rfeat,
                sfeat=sfeat,
                bgterr=bgterr,
                bgfeat=bgfeat,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

        roi_score = float(np.nanmean([m['nrmse_umag'] for m in val_roi_metrics.values()])) if val_roi_metrics else float('nan')
        roi_p_score = float(np.nanmean([m['nrmse_p'] for m in val_roi_metrics.values()])) if val_roi_metrics else float('nan')
        val_roi_by_cat = _category_metrics(repo, val_roi_metrics)
        selector_scores = _selector_components(
            val_global_metrics=val_global_metrics,
            val_roi_metrics=val_roi_metrics,
            val_roi_by_cat=val_roi_by_cat,
        )
        selector_umag = float(selector_scores['selector_umag'])
        selector_p = float(selector_scores['selector_p'])
        selector_ms_roi_umag = float(selector_scores['selector_ms_roi_umag'])
        selector = float(selector_scores['selector'])
        row = {
            'epoch': int(epoch),
            'lr': float(optimizer.param_groups[0]['lr']),
            'train_loss_global': float('nan'),
            'train_loss_roi': float(np.mean(train_roi_losses)) if train_roi_losses else float('nan'),
            'train_div_rms_global': float('nan'),
            'train_div_rms_roi': float(np.mean(train_div_roi)) if train_div_roi else float('nan'),
            'train_mom_rms_global': float('nan'),
            'train_mom_rms_roi': float(np.mean(train_mom_roi)) if train_mom_roi else float('nan'),
            'train_wall_bc_roi': float(np.mean(train_wall_roi)) if train_wall_roi else float('nan'),
            'train_edge_roi': float(np.mean(train_edge_roi)) if train_edge_roi else float('nan'),
            'val_global_nrmse_umag': global_score,
            'val_global_nrmse_p': global_p_score,
            'val_roi_nrmse_umag': roi_score,
            'val_roi_nrmse_p': roi_p_score,
            'val_selector_umag': selector_umag,
            'val_selector_p': selector_p,
            'val_selector_ms_roi_umag': selector_ms_roi_umag,
            'val_selector': selector,
        }
        for key, values in sorted(train_sample_stats.items()):
            row[f'train_{key}'] = float(np.mean(values)) if values else float('nan')
        for short in _CATEGORY_SHORT.values():
            row[f'val_global_{short}_nrmse_umag'] = float(val_global_by_cat[short]['nrmse_umag'])
            row[f'val_global_{short}_nrmse_p'] = float(val_global_by_cat[short]['nrmse_p'])
            row[f'val_roi_{short}_nrmse_umag'] = float(val_roi_by_cat[short]['nrmse_umag'])
            row[f'val_roi_{short}_nrmse_p'] = float(val_roi_by_cat[short]['nrmse_p'])
        history.append(row)
        write_json(save_dir / 'logs' / 'history.json', history)
        print(
            f"Epoch {epoch:4d}/{int(epochs)} | lr={row['lr']:.2e} | "
            f"train R={row['train_loss_roi']:.4f} | "
            f"sample vnw={row.get('train_targeted_very_near_wall_frac', float('nan')):.2f} "
            f"nw={row.get('train_targeted_near_wall_frac', float('nan')):.2f} "
            f"wake={row.get('train_targeted_geom_wake_frac', float('nan')):.2f} | "
            f"val Umag G={row['val_global_nrmse_umag']:.4f} R={row['val_roi_nrmse_umag']:.4f} | "
            f"val p G={row['val_global_nrmse_p']:.4f} R={row['val_roi_nrmse_p']:.4f} | "
            f"sel={row['val_selector']:.4f}",
            flush=True,
        )
        wandb_payload = {
            'train/loss_roi': row['train_loss_roi'],
            'train/div_rms_roi': row['train_div_rms_roi'],
            'train/mom_rms_roi': row['train_mom_rms_roi'],
            'train/wall_bc_roi': row['train_wall_bc_roi'],
            'train/edge_roi': row['train_edge_roi'],
            'val/global_nrmse_umag': row['val_global_nrmse_umag'],
            'val/global_nrmse_p': row['val_global_nrmse_p'],
            'val/roi_nrmse_umag': row['val_roi_nrmse_umag'],
            'val/roi_nrmse_p': row['val_roi_nrmse_p'],
            'val/selector_umag': row['val_selector_umag'],
            'val/selector_p': row['val_selector_p'],
            'val/selector_ms_roi_umag': row['val_selector_ms_roi_umag'],
            'val/selector': row['val_selector'],
            'train/lr': row['lr'],
        }
        for key in sorted(train_sample_stats.keys()):
            row_key = f'train_{key}'
            wandb_payload[f'train/{key}'] = row.get(row_key, float('nan'))
        for short in _CATEGORY_SHORT.values():
            wandb_payload[f'val/global_{short}_nrmse_umag'] = row[f'val_global_{short}_nrmse_umag']
            wandb_payload[f'val/global_{short}_nrmse_p'] = row[f'val_global_{short}_nrmse_p']
            wandb_payload[f'val/roi_{short}_nrmse_umag'] = row[f'val_roi_{short}_nrmse_umag']
            wandb_payload[f'val/roi_{short}_nrmse_p'] = row[f'val_roi_{short}_nrmse_p']
        wandb_log(wandb_run, wandb_payload, step=epoch)

        ckpt = {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scalers': scalers,
            'split': split,
            'history': history,
            'best_val': float(best_score),
            'best_epoch': int(best_epoch),
            'config_snapshot': dict(config_snapshot or {}),
            'cascade_stage1_checkpoint': str(cascade_stage1_checkpoint or conditioner.checkpoint_path),
            'skipped_large_roi': skipped_large_roi,
            'train_config': {
                'train_mode': str(train_mode),
                'train_loss': str(train_loss),
                'momentum_loss_mode': str(momentum_loss_mode),
                'train_struct_mode': str(train_struct_mode),
                'train_struct_weight': float(train_struct_weight),
                'scheduler_mode': str(scheduler_mode),
                'hard_ground_bc': bool(hard_ground_bc),
                'use_amp': bool(use_amp),
                'amp_dtype': str(amp_dtype),
                'model_kind': 'cascade_stage2',
                'config_snapshot': dict(config_snapshot or {}),
                'cascade_stage1_checkpoint': str(cascade_stage1_checkpoint or conditioner.checkpoint_path),
                'skipped_large_roi': skipped_large_roi,
            },
        }
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        if scaler.is_enabled():
            ckpt['scaler_state_dict'] = scaler.state_dict()
        (save_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, save_dir / 'checkpoints' / 'latest.pth')
        if epoch >= int(MIN_EPOCH_FOR_BEST) and selector < best_score:
            best_score = float(selector)
            best_epoch = int(epoch)
            ckpt['best_val'] = float(best_score)
            ckpt['best_epoch'] = int(best_epoch)
            torch.save(ckpt, save_dir / 'checkpoints' / 'best.pth')
        patience = int(max(0, EARLY_STOPPING_PATIENCE))
        if patience > 0 and best_epoch >= int(MIN_EPOCH_FOR_BEST):
            stale_epochs = int(epoch) - int(best_epoch)
            if stale_epochs >= patience:
                stop_epoch = int(epoch)
                print(
                    f"[TRAIN] early stop at epoch={int(epoch)} | best epoch={int(best_epoch)} | "
                    f"stale={int(stale_epochs)}",
                    flush=True,
                )
                break

    if best_epoch < 0:
        shutil.copy2(save_dir / 'checkpoints' / 'latest.pth', save_dir / 'checkpoints' / 'best.pth')
        best_epoch = int(stop_epoch)
        best_score = float(history[-1]['val_selector']) if history else float('inf')

    write_json(save_dir / 'checkpoint_paths.json', {
        'latest': str(save_dir / 'checkpoints' / 'latest.pth'),
        'best': str(save_dir / 'checkpoints' / 'best.pth'),
    })
    wandb_log(
        wandb_run,
        {
            'best/epoch': int(best_epoch),
            'best/val_selector': float(best_score),
        },
        step=int(stop_epoch) if stop_epoch > 0 else None,
    )
    print(f"[TRAIN] done | best epoch={int(best_epoch)} | best selector={float(best_score):.4f}", flush=True)
    return {
        'save_dir': str(save_dir),
        'stop_epoch': int(stop_epoch),
        'best_epoch': int(best_epoch),
        'best_val': float(best_score),
        'scalers': scalers,
        'history': history,
    }


def evaluate_split(
    model,
    repo: CaseRepository,
    split_names: list[str],
    scalers,
    *,
    conditioner: Optional[CascadeConditioner] = None,
    device: str,
    pred_batch_size: int,
    output_dir: Path,
    split_label: str,
    hard_ground_bc: bool = HARD_GROUND_BC,
    plot_eval: bool = PLOT_EVAL,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
) -> Dict:
    kind = _model_kind(model)
    if kind == 'grid_unet':
        return _evaluate_split_grid_unet(
            model,
            repo,
            split_names,
            scalers,
            device=device,
            pred_batch_size=pred_batch_size,
            output_dir=output_dir,
            split_label=split_label,
            hard_ground_bc=hard_ground_bc,
            plot_eval=plot_eval,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    if kind == 'cascade_stage1':
        return _evaluate_split_cascade_stage1(
            model,
            repo,
            split_names,
            scalers,
            device=device,
            pred_batch_size=pred_batch_size,
            output_dir=output_dir,
            split_label=split_label,
            hard_ground_bc=hard_ground_bc,
            plot_eval=plot_eval,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    if kind == 'cascade_stage2':
        if conditioner is None:
            raise RuntimeError('cascade_stage2 evaluation requires a stage1 conditioner')
        return _evaluate_split_cascade_stage2(
            model,
            repo,
            split_names,
            scalers,
            conditioner=conditioner,
            device=device,
            pred_batch_size=pred_batch_size,
            output_dir=output_dir,
            split_label=split_label,
            hard_ground_bc=hard_ground_bc,
            plot_eval=plot_eval,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    plotter = None
    if plot_eval:
        from vis.eval_report import generate_eval_case_report
        plotter = generate_eval_case_report

    global_case_metrics = {}
    roi_case_metrics = {}
    ensure_dir(output_dir)
    tensor_cache = _DeviceTensorCache(device=device)
    total_roi = int(sum(len(repo.roi_names(name)) for name in split_names))
    print(
        f"[EVAL] {split_label}: {len(split_names)} domains, {total_roi} ROI box(es), "
        f"plot_eval={'yes' if plot_eval else 'no'}",
        flush=True,
    )

    for idx_name, name in enumerate(split_names, start=1):
        print(f"[EVAL] {split_label} {idx_name}/{len(split_names)}: {name}", flush=True)
        case_dir = output_dir / name
        ensure_dir(case_dir)

        g = repo.load_global(name)
        gterr = tensor_cache.terrain(g)
        with torch.no_grad():
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                gfeat = model.encode_global(gterr)
        g_metrics, g_pred = _evaluate_case_global(
            model,
            g,
            scalers,
            device=device,
            pred_batch_size=pred_batch_size,
            hard_ground_bc=hard_ground_bc,
            return_pred_flow=plot_eval,
            terr=gterr,
            gfeat=gfeat,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        global_case_metrics[name] = g_metrics

        roi_metrics_local = {}
        roi_pred_local = {}
        for roi_name in repo.roi_names(name):
            r = repo.load_roi(name, roi_name)
            rterr = tensor_cache.terrain(r)
            sterr = tensor_cache.structure(r) if getattr(model, 'use_structure_encoder', False) else None
            with torch.no_grad():
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    rfeat = model.encode_roi(rterr)
                    sfeat = model.encode_structure(sterr)
            roi_points = int(np.prod(r.flow.shape[:3], dtype=np.int64))
            want_roi_plot = bool(plot_eval and roi_points <= int(MAX_PLOT_FLOW_POINTS))
            if plot_eval and not want_roi_plot:
                print(
                    f"[EVAL] skipping ROI plot for {name}/{roi_name} "
                    f"(points={roi_points:,} > MAX_PLOT_FLOW_POINTS={int(MAX_PLOT_FLOW_POINTS):,})",
                    flush=True,
                )
            r_metrics, r_pred = _evaluate_case_roi(
                model,
                g,
                r,
                scalers,
                device=device,
                pred_batch_size=pred_batch_size,
                hard_ground_bc=hard_ground_bc,
                return_pred_flow=want_roi_plot,
                gterr=gterr,
                rterr=rterr,
                sterr=sterr,
                gfeat=gfeat,
                rfeat=rfeat,
                sfeat=sfeat,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            roi_metrics_local[roi_name] = r_metrics
            roi_case_metrics[f'{name}/{roi_name}'] = r_metrics
            if want_roi_plot and r_pred is not None:
                roi_pred_local[roi_name] = r_pred

        payload = {
            'case': name,
            'split': split_label,
            'global': g_metrics,
            'rois': roi_metrics_local,
        }
        if plotter is not None and g_pred is not None:
            files = plotter(repo.case_dirs[name], case_dir / 'plots', global_pred_flow=g_pred, roi_pred_flows=roi_pred_local)
            payload['plots'] = files
        write_json(case_dir / 'metrics.json', payload)

    def _subset_mean(case_metrics: dict, subset_name: str, key: str) -> float:
        vals = []
        for m in case_metrics.values():
            sub = (m.get('subsets') or {}).get(subset_name)
            if not sub:
                continue
            v = sub.get(key)
            if v is None or not np.isfinite(v):
                continue
            vals.append(float(v))
        return float(np.mean(vals)) if vals else float('nan')

    summary = {
        'global_cases': global_case_metrics,
        'roi_cases': roi_case_metrics,
        'global_by_category': _category_metrics(repo, global_case_metrics),
        'roi_by_category': _category_metrics(repo, roi_case_metrics),
        'global_mean_nrmse_umag': float(np.nanmean([m['nrmse_umag'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'global_mean_nrmse_p': float(np.nanmean([m['nrmse_p'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'global_mean_nrmse_p_gauge': float(np.nanmean([m['nrmse_p_gauge'] for m in global_case_metrics.values()])) if global_case_metrics else float('nan'),
        'roi_mean_nrmse_umag': float(np.nanmean([m['nrmse_umag'] for m in roi_case_metrics.values()])) if roi_case_metrics else float('nan'),
        'roi_mean_nrmse_p': float(np.nanmean([m['nrmse_p'] for m in roi_case_metrics.values()])) if roi_case_metrics else float('nan'),
        'roi_mean_nrmse_p_gauge': float(np.nanmean([m['nrmse_p_gauge'] for m in roi_case_metrics.values()])) if roi_case_metrics else float('nan'),
        # Engineering KPIs: pressure & velocity restricted to where it matters.
        # global near_ground -> snow drift / wind shear at terrain surface.
        # roi near_wall      -> structural loading on panels/turbines.
        # roi near_ground_near_wall -> snow drift pocket at panel feet.
        'global_near_ground_nrmse_umag': _subset_mean(global_case_metrics, 'near_ground', 'nrmse_umag'),
        'global_near_ground_nrmse_p':    _subset_mean(global_case_metrics, 'near_ground', 'nrmse_p'),
        'global_near_ground_nrmse_p_gauge': _subset_mean(global_case_metrics, 'near_ground', 'nrmse_p_gauge'),
        'roi_near_wall_nrmse_umag':      _subset_mean(roi_case_metrics, 'near_wall', 'nrmse_umag'),
        'roi_near_wall_nrmse_p':         _subset_mean(roi_case_metrics, 'near_wall', 'nrmse_p'),
        'roi_near_wall_nrmse_p_gauge':   _subset_mean(roi_case_metrics, 'near_wall', 'nrmse_p_gauge'),
        'roi_near_ground_nrmse_umag':    _subset_mean(roi_case_metrics, 'near_ground', 'nrmse_umag'),
        'roi_near_ground_nrmse_p':       _subset_mean(roi_case_metrics, 'near_ground', 'nrmse_p'),
        'roi_near_ground_nrmse_p_gauge': _subset_mean(roi_case_metrics, 'near_ground', 'nrmse_p_gauge'),
        'roi_near_ground_near_wall_nrmse_umag': _subset_mean(roi_case_metrics, 'near_ground_near_wall', 'nrmse_umag'),
        'roi_near_ground_near_wall_nrmse_p':    _subset_mean(roi_case_metrics, 'near_ground_near_wall', 'nrmse_p'),
        'roi_near_ground_near_wall_nrmse_p_gauge': _subset_mean(roi_case_metrics, 'near_ground_near_wall', 'nrmse_p_gauge'),
        'global_mean_physics_div_rms': _mean_metric(global_case_metrics, 'physics_div_rms'),
        'global_mean_physics_mom_rms_constant': _mean_metric(global_case_metrics, 'physics_mom_rms_constant'),
        'global_mean_physics_mom_rms_nut': _mean_metric(global_case_metrics, 'physics_mom_rms_nut'),
        'roi_mean_physics_div_rms': _mean_metric(roi_case_metrics, 'physics_div_rms'),
        'roi_mean_physics_mom_rms_constant': _mean_metric(roi_case_metrics, 'physics_mom_rms_constant'),
        'roi_mean_physics_mom_rms_nut': _mean_metric(roi_case_metrics, 'physics_mom_rms_nut'),
        'global_case_count': int(len(global_case_metrics)),
        'roi_case_count': int(len(roi_case_metrics)),
        'split': split_label,
        'hard_ground_bc': bool(hard_ground_bc),
        'plot_eval': bool(plot_eval),
        'use_amp': bool(use_amp),
        'amp_dtype': str(amp_dtype),
    }
    write_json(output_dir / 'summary.json', summary)
    print(
        f"[EVAL] {split_label} done | global nRMSE(Umag)={summary['global_mean_nrmse_umag']:.4f} | "
        f"roi nRMSE(Umag)={summary['roi_mean_nrmse_umag']:.4f}",
        flush=True,
    )
    return summary


def train_model(
    model,
    repo: CaseRepository,
    split: dict,
    *,
    conditioner: Optional[CascadeConditioner] = None,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    save_dir: Path,
    train_mode: str = TRAIN_MODE,
    train_loss: str = TRAIN_LOSS,
    momentum_loss_mode: str = MOMENTUM_LOSS_MODE,
    train_struct_mode: str = TRAIN_STRUCT_MODE,
    train_struct_weight: float = TRAIN_STRUCT_WEIGHT,
    scheduler_mode: str = SCHEDULER_MODE,
    hard_ground_bc: bool = HARD_GROUND_BC,
    charb_eps: float = CHARB_EPS,
    pred_batch_size: int = 200000,
    use_amp: bool = USE_AMP,
    amp_dtype: str = AMP_DTYPE,
    resume_checkpoint: Optional[dict] = None,
    config_snapshot: Optional[dict] = None,
    cascade_stage1_checkpoint: str = '',
    wandb_run=None,
) -> Dict:
    kind = _model_kind(model)
    if kind == 'grid_unet':
        return _train_grid_unet_model(
            model,
            repo,
            split,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            save_dir=save_dir,
            train_mode=train_mode,
            train_loss=train_loss,
            momentum_loss_mode=momentum_loss_mode,
            train_struct_mode=train_struct_mode,
            train_struct_weight=train_struct_weight,
            scheduler_mode=scheduler_mode,
            hard_ground_bc=hard_ground_bc,
            charb_eps=charb_eps,
            pred_batch_size=pred_batch_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            resume_checkpoint=resume_checkpoint,
            wandb_run=wandb_run,
        )
    if kind == 'cascade_stage1':
        return _train_cascade_stage1_model(
            model,
            repo,
            split,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            save_dir=save_dir,
            train_mode=train_mode,
            train_loss=train_loss,
            momentum_loss_mode=momentum_loss_mode,
            train_struct_mode=train_struct_mode,
            train_struct_weight=train_struct_weight,
            scheduler_mode=scheduler_mode,
            hard_ground_bc=hard_ground_bc,
            charb_eps=charb_eps,
            pred_batch_size=pred_batch_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            resume_checkpoint=resume_checkpoint,
            config_snapshot=config_snapshot,
            wandb_run=wandb_run,
        )
    if kind == 'cascade_stage2':
        return _train_cascade_stage2_model(
            model,
            repo,
            split,
            conditioner=conditioner,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            save_dir=save_dir,
            train_mode=train_mode,
            train_loss=train_loss,
            momentum_loss_mode=momentum_loss_mode,
            train_struct_mode=train_struct_mode,
            train_struct_weight=train_struct_weight,
            scheduler_mode=scheduler_mode,
            hard_ground_bc=hard_ground_bc,
            charb_eps=charb_eps,
            pred_batch_size=pred_batch_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            resume_checkpoint=resume_checkpoint,
            config_snapshot=config_snapshot,
            cascade_stage1_checkpoint=cascade_stage1_checkpoint,
            wandb_run=wandb_run,
        )
    if resume_checkpoint is not None:
        scalers = resume_checkpoint['scalers']
        history = list(resume_checkpoint.get('history', []))
        best_score = float(resume_checkpoint.get('best_val', float('inf')))
        best_epoch = int(resume_checkpoint.get('best_epoch', -1))
        start_epoch = int(resume_checkpoint.get('epoch', 0))
    else:
        scalers = fit_scalers(repo, split['train'])
        history = []
        best_score = float('inf')
        best_epoch = -1
        start_epoch = 0
    rng = _training_rng(int(start_epoch))
    stop_epoch = int(epochs)
    ensure_dir(save_dir)
    tensor_cache = _DeviceTensorCache(device=device)
    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=_use_cuda_amp(device, use_amp) and _amp_torch_dtype(amp_dtype) == torch.float16,
    )

    train_roi_refs = [(name, roi_name) for name in split['train'] for roi_name in repo.roi_names(name)]
    val_roi_refs = [(name, roi_name) for name in split['val'] for roi_name in repo.roi_names(name)]
    weights = _effective_weights(train_mode)
    steps_per_epoch = len(split['train']) + len(train_roi_refs)
    scheduler = _build_scheduler(optimizer, epochs=int(epochs), steps_per_epoch=steps_per_epoch, scheduler_mode=scheduler_mode)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        if scaler.is_enabled() and 'scaler_state_dict' in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
    print(
        f"[TRAIN] mode={train_mode} | epochs={int(epochs)} | train domains={len(split['train'])} | "
        f"train ROI={len(train_roi_refs)} | steps/epoch={steps_per_epoch} | mom={momentum_loss_mode} | "
        f"amp={'off' if not use_amp else amp_dtype}",
        flush=True,
    )
    if start_epoch > 0:
        print(
            f"[TRAIN] resuming from epoch={int(start_epoch)} | best epoch={int(best_epoch)} | "
            f"best selector={float(best_score):.4f}",
            flush=True,
        )
    if int(start_epoch) >= int(epochs):
        print(
            f"[TRAIN] target epochs already reached in latest checkpoint "
            f"(epoch={int(start_epoch)} >= {int(epochs)}), skipping training loop",
            flush=True,
        )
        stop_epoch = int(start_epoch)

    for epoch in range(int(start_epoch) + 1, int(epochs) + 1):
        weights = _effective_weights(train_mode, epoch=epoch)
        model.train()
        train_global_losses = []
        train_roi_losses = []
        train_div_global = []
        train_div_roi = []
        train_mom_global = []
        train_mom_roi = []
        train_wall_roi = []
        rng.shuffle(split['train'])
        rng.shuffle(train_roi_refs)
        use_struct_loss = float(train_struct_weight) > 0.0 and str(train_struct_mode).lower() != 'none'

        for name in split['train']:
            g = repo.load_global(name)
            _raise_if_tf_nondata_loss(g, weights, scope='global', structured_loss_enabled=use_struct_loss)
            gterr = tensor_cache.terrain(g)
            batch = sample_supervised_batch(g, x_scaler=scalers.x_scaler_global, y_scaler=scalers.y_scaler, n_points=GLOBAL_POINTS_PER_DOMAIN, rng=rng)
            y_true = batch.y_scaled.to(device)
            if epoch == 1 and not bool(torch.isfinite(y_true).all().item()):
                raise RuntimeError(f'Non-finite supervised targets detected in global batch for {name}')
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                gfeat = model.encode_global(gterr)
                pred_scaled, pred_phys = _predict_global(
                    model,
                    g,
                    batch,
                    scalers.y_scaler,
                    scalers.x_scaler_global,
                    device=device,
                    hard_ground_bc=hard_ground_bc,
                    terr=gterr,
                    gfeat=gfeat,
                )
                loss_data = supervised_data_loss_from_pred(
                    pred_scaled,
                    y_true,
                    x_batch=batch.x_scaled.to(device),
                    x_scaler=scalers.x_scaler_global,
                    input_cols=GLOBAL_INPUT_COLS,
                    y_scaler=scalers.y_scaler,
                    p_weight=float(GLOBAL_DATA_P_WEIGHT),
                    loss_mode=train_loss,
                    charbonnier_eps=charb_eps,
                )
                loss_phys = torch.tensor(0.0, device=device)
                struct_loss = torch.tensor(0.0, device=device)
                if weights['w_phys_global'] > 0 or use_struct_loss:
                    for _ in range(int(GLOBAL_PATCHES_PER_DOMAIN)):
                        patch = sample_patch_batch(g, x_scaler=scalers.x_scaler_global, y_scaler=scalers.y_scaler, patch_shape=GLOBAL_PATCH_SHAPE, rng=rng, near_ground_prob=PATCH_NEAR_GROUND_PROB)
                        patch_pred_raw = model.forward_global_from_encoded(gfeat, patch.x_scaled.to(device), patch.xy_local.to(device))
                        patch_pred, _ = apply_output_constraints_from_scaled_inputs(
                            patch_pred_raw,
                            x_batch=patch.x_scaled.to(device),
                            x_scaler=scalers.x_scaler_global,
                            input_cols=GLOBAL_INPUT_COLS,
                            y_scaler=scalers.y_scaler,
                            hard_ground_bc=hard_ground_bc,
                        )
                        phys = compute_patch_physics_losses_from_pred(
                            patch_pred,
                            patch,
                            scalers.y_scaler,
                            device=device,
                            momentum_loss_mode=momentum_loss_mode,
                        )
                        if weights['w_phys_global'] > 0:
                            loss_phys = (
                                loss_phys
                                + weights['w_div_global'] * phys['div_loss']
                                + weights['w_mom_global'] * phys['mom_loss']
                            )
                            train_div_global.append(float(phys['div_rms'].detach().cpu().item()))
                            train_mom_global.append(float(phys['mom_rms'].detach().cpu().item()))
                        if use_struct_loss:
                            struct_loss = struct_loss + structured_patch_loss_from_pred(
                                patch_pred,
                                patch,
                                mode=train_struct_mode,
                                p_weight=float(GLOBAL_DATA_P_WEIGHT),
                                charbonnier_eps=charb_eps,
                            )
            bc_batches = prepare_global_boundary_batches(g, x_scaler=scalers.x_scaler_global, rng=rng)
            inlet_loss = torch.tensor(0.0, device=device)
            outlet_loss = torch.tensor(0.0, device=device)
            side_loss = torch.tensor(0.0, device=device)
            top_loss = torch.tensor(0.0, device=device)
            if bc_batches.get('inlet') is not None and weights['w_bc_inlet'] > 0:
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    inlet_loss = inlet_bc_loss_from_phys(
                        _predict_global_boundary(
                            model,
                            g,
                            bc_batches['inlet'],
                            scalers,
                            device=device,
                            hard_ground_bc=hard_ground_bc,
                            terr=gterr,
                            gfeat=gfeat,
                        ),
                        bc_batches['inlet'],
                        device=device,
                    )
            if bc_batches.get('outlet') is not None and weights['w_bc_outlet'] > 0:
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    outlet_loss = outlet_bc_loss_from_phys(
                        _predict_global_boundary(
                            model,
                            g,
                            bc_batches['outlet'],
                            scalers,
                            device=device,
                            hard_ground_bc=hard_ground_bc,
                            terr=gterr,
                            gfeat=gfeat,
                        ),
                        bc_batches['outlet'],
                        device=device,
                    )
            if bc_batches.get('side') is not None and weights['w_bc_side'] > 0:
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    side_loss = normal_velocity_bc_loss_from_phys(
                        _predict_global_boundary(
                            model,
                            g,
                            bc_batches['side'],
                            scalers,
                            device=device,
                            hard_ground_bc=hard_ground_bc,
                            terr=gterr,
                            gfeat=gfeat,
                        ),
                        bc_batches['side'],
                        device=device,
                    )
            if bc_batches.get('top') is not None and weights['w_bc_top'] > 0:
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    top_loss = normal_velocity_bc_loss_from_phys(
                        _predict_global_boundary(
                            model,
                            g,
                            bc_batches['top'],
                            scalers,
                            device=device,
                            hard_ground_bc=hard_ground_bc,
                            terr=gterr,
                            gfeat=gfeat,
                        ),
                        bc_batches['top'],
                        device=device,
                    )
            total = (
                float(W_DATA_GLOBAL) * loss_data
                + weights['w_phys_global'] * loss_phys
                + float(train_struct_weight) * struct_loss
                + weights['w_bc_inlet'] * inlet_loss
                + weights['w_bc_outlet'] * outlet_loss
                + weights['w_bc_side'] * side_loss
                + weights['w_bc_top'] * top_loss
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(total).backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_global_losses.append(float(total.detach().cpu().item()))

        for name, roi_name in train_roi_refs:
            g = repo.load_global(name)
            r = repo.load_roi(name, roi_name)
            _raise_if_tf_nondata_loss(g, weights, scope='global', structured_loss_enabled=False)
            _raise_if_tf_nondata_loss(r, weights, scope='roi', structured_loss_enabled=use_struct_loss)
            gterr = tensor_cache.terrain(g)
            rterr = tensor_cache.terrain(r)
            sterr = tensor_cache.structure(r) if getattr(model, 'use_structure_encoder', False) else None
            batch = sample_supervised_batch(r, x_scaler=scalers.x_scaler_roi, y_scaler=scalers.y_scaler, n_points=ROI_POINTS_PER_DOMAIN, rng=rng, parent_global=g)
            y_true = batch.y_scaled.to(device)
            if epoch == 1 and not bool(torch.isfinite(y_true).all().item()):
                raise RuntimeError(f'Non-finite supervised targets detected in ROI batch for {name}/{roi_name}')
            with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                gfeat = model.encode_global(gterr)
                rfeat = model.encode_roi(rterr)
                sfeat = model.encode_structure(sterr)
                pred_scaled, _ = _predict_roi(
                    model,
                    g,
                    r,
                    batch,
                    scalers.y_scaler,
                    scalers.x_scaler_roi,
                    device=device,
                    hard_ground_bc=hard_ground_bc,
                    gterr=gterr,
                    rterr=rterr,
                    sterr=sterr,
                    gfeat=gfeat,
                    rfeat=rfeat,
                    sfeat=sfeat,
                )
                loss_data = supervised_data_loss_from_pred(
                    pred_scaled,
                    y_true,
                    x_batch=batch.x_scaled.to(device),
                    x_scaler=scalers.x_scaler_roi,
                    input_cols=ROI_INPUT_COLS,
                    y_scaler=scalers.y_scaler,
                    p_weight=float(ROI_DATA_P_WEIGHT),
                    loss_mode=train_loss,
                    charbonnier_eps=charb_eps,
                )
                loss_phys = torch.tensor(0.0, device=device)
                struct_loss = torch.tensor(0.0, device=device)
                wall_loss = torch.tensor(0.0, device=device)
                if weights['w_phys_roi'] > 0 or weights['w_bc_wall_roi'] > 0 or use_struct_loss:
                    for _ in range(int(ROI_PATCHES_PER_DOMAIN)):
                        patch = sample_patch_batch(r, x_scaler=scalers.x_scaler_roi, y_scaler=scalers.y_scaler, patch_shape=ROI_PATCH_SHAPE, rng=rng, near_ground_prob=PATCH_NEAR_GROUND_PROB, parent_global=g)
                        patch_pred_raw = model.forward_roi_from_encoded(
                            gfeat,
                            rfeat,
                            patch.x_scaled.to(device),
                            patch.xy_global.to(device),
                            patch.xy_local.to(device),
                            s_feat=sfeat,
                        )
                        patch_pred, _ = apply_output_constraints_from_scaled_inputs(
                            patch_pred_raw,
                            x_batch=patch.x_scaled.to(device),
                            x_scaler=scalers.x_scaler_roi,
                            input_cols=ROI_INPUT_COLS,
                            y_scaler=scalers.y_scaler,
                            hard_ground_bc=hard_ground_bc,
                        )
                        if weights['w_phys_roi'] > 0:
                            phys = compute_patch_physics_losses_from_pred(
                                patch_pred,
                                patch,
                                scalers.y_scaler,
                                device=device,
                                momentum_loss_mode=momentum_loss_mode,
                            )
                            loss_phys = (
                                loss_phys
                                + weights['w_div_roi'] * phys['div_loss']
                                + weights['w_mom_roi'] * phys['mom_loss']
                            )
                            train_div_roi.append(float(phys['div_rms'].detach().cpu().item()))
                            train_mom_roi.append(float(phys['mom_rms'].detach().cpu().item()))
                        if use_struct_loss:
                            struct_loss = struct_loss + structured_patch_loss_from_pred(
                                patch_pred,
                                patch,
                                mode=train_struct_mode,
                                p_weight=float(ROI_DATA_P_WEIGHT),
                                charbonnier_eps=charb_eps,
                            )
                        if weights['w_bc_wall_roi'] > 0:
                            patch_pred_phys = apply_output_constraints_from_scaled_inputs(
                                patch_pred,
                                x_batch=patch.x_scaled.to(device),
                                x_scaler=scalers.x_scaler_roi,
                                input_cols=ROI_INPUT_COLS,
                                y_scaler=scalers.y_scaler,
                                hard_ground_bc=False,
                            )[1]
                            wall_loss = wall_loss + roi_wall_velocity_bc_loss_from_phys(
                                patch_pred_phys,
                                x_batch=patch.x_scaled,
                                x_scaler=scalers.x_scaler_roi,
                                input_cols=ROI_INPUT_COLS,
                                device=device,
                                u_scale=float(r.uref),
                                dmax=float(ROI_WALL_BC_DMAX),
                            )
            total = (
                float(W_DATA_ROI) * loss_data
                + weights['w_phys_roi'] * loss_phys
                + float(train_struct_weight) * struct_loss
                + weights['w_bc_wall_roi'] * wall_loss
            )
            if not bool(total.requires_grad):
                print(
                    f"[WARN] grid_unet roi step produced no grad-bearing loss for {name}/{roi_name}; skipping optimizer step",
                    flush=True,
                )
                continue
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(total).backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                if GRAD_CLIP_MAX_NORM is not None and float(GRAD_CLIP_MAX_NORM) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_MAX_NORM))
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_roi_losses.append(float(total.detach().cpu().item()))
            if weights['w_bc_wall_roi'] > 0:
                train_wall_roi.append(float(wall_loss.detach().cpu().item()))

        model.eval()
        val_global_metrics = {}
        for name in split['val']:
            g = repo.load_global(name)
            gterr = tensor_cache.terrain(g)
            with torch.no_grad():
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    gfeat = model.encode_global(gterr)
            val_global_metrics[name], _ = _evaluate_case_global(
                model,
                g,
                scalers,
                device=device,
                pred_batch_size=pred_batch_size,
                hard_ground_bc=hard_ground_bc,
                terr=gterr,
                gfeat=gfeat,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        val_roi_metrics = {}
        for name, roi_name in val_roi_refs:
            g = repo.load_global(name)
            r = repo.load_roi(name, roi_name)
            gterr = tensor_cache.terrain(g)
            rterr = tensor_cache.terrain(r)
            sterr = tensor_cache.structure(r) if getattr(model, 'use_structure_encoder', False) else None
            with torch.no_grad():
                with _autocast_context(device=device, enabled=use_amp, amp_dtype=amp_dtype):
                    gfeat = model.encode_global(gterr)
                    rfeat = model.encode_roi(rterr)
                    sfeat = model.encode_structure(sterr)
            val_roi_metrics[f'{name}/{roi_name}'], _ = _evaluate_case_roi(
                model,
                g,
                r,
                scalers,
                device=device,
                pred_batch_size=pred_batch_size,
                hard_ground_bc=hard_ground_bc,
                gterr=gterr,
                rterr=rterr,
                sterr=sterr,
                gfeat=gfeat,
                rfeat=rfeat,
                sfeat=sfeat,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

        global_score = float(np.nanmean([m['nrmse_umag'] for m in val_global_metrics.values()])) if val_global_metrics else float('nan')
        global_p_score = float(np.nanmean([m['nrmse_p'] for m in val_global_metrics.values()])) if val_global_metrics else float('nan')
        roi_score = float(np.nanmean([m['nrmse_umag'] for m in val_roi_metrics.values()])) if val_roi_metrics else float('nan')
        roi_p_score = float(np.nanmean([m['nrmse_p'] for m in val_roi_metrics.values()])) if val_roi_metrics else float('nan')
        val_global_by_cat = _category_metrics(repo, val_global_metrics)
        val_roi_by_cat = _category_metrics(repo, val_roi_metrics)
        selector_scores = _selector_components(
            val_global_metrics=val_global_metrics,
            val_roi_metrics=val_roi_metrics,
            val_roi_by_cat=val_roi_by_cat,
        )
        selector_umag = float(selector_scores['selector_umag'])
        selector_p = float(selector_scores['selector_p'])
        selector_ms_roi_umag = float(selector_scores['selector_ms_roi_umag'])
        selector = float(selector_scores['selector'])
        row = {
            'epoch': int(epoch),
            'lr': float(optimizer.param_groups[0]['lr']),
            'train_loss_global': float(np.mean(train_global_losses)) if train_global_losses else float('nan'),
            'train_loss_roi': float(np.mean(train_roi_losses)) if train_roi_losses else float('nan'),
            'train_div_rms_global': float(np.mean(train_div_global)) if train_div_global else float('nan'),
            'train_div_rms_roi': float(np.mean(train_div_roi)) if train_div_roi else float('nan'),
            'train_mom_rms_global': float(np.mean(train_mom_global)) if train_mom_global else float('nan'),
            'train_mom_rms_roi': float(np.mean(train_mom_roi)) if train_mom_roi else float('nan'),
            'train_wall_bc_roi': float(np.mean(train_wall_roi)) if train_wall_roi else float('nan'),
            'val_global_nrmse_umag': global_score,
            'val_global_nrmse_p': global_p_score,
            'val_roi_nrmse_umag': roi_score,
            'val_roi_nrmse_p': roi_p_score,
            'val_selector_umag': selector_umag,
            'val_selector_p': selector_p,
            'val_selector_ms_roi_umag': selector_ms_roi_umag,
            'val_selector': selector,
        }
        for short in _CATEGORY_SHORT.values():
            row[f'val_global_{short}_nrmse_umag'] = float(val_global_by_cat[short]['nrmse_umag'])
            row[f'val_global_{short}_nrmse_p'] = float(val_global_by_cat[short]['nrmse_p'])
            row[f'val_roi_{short}_nrmse_umag'] = float(val_roi_by_cat[short]['nrmse_umag'])
            row[f'val_roi_{short}_nrmse_p'] = float(val_roi_by_cat[short]['nrmse_p'])
        history.append(row)
        write_json(save_dir / 'logs' / 'history.json', history)
        print(
            f"Epoch {epoch:4d}/{int(epochs)} | lr={row['lr']:.2e} | "
            f"train G={row['train_loss_global']:.4f} R={row['train_loss_roi']:.4f} | "
            f"val Umag G={row['val_global_nrmse_umag']:.4f} R={row['val_roi_nrmse_umag']:.4f} | "
            f"val p G={row['val_global_nrmse_p']:.4f} R={row['val_roi_nrmse_p']:.4f} | "
            f"sel={row['val_selector']:.4f}",
            flush=True,
        )
        wandb_payload = {
            'train/loss_global': row['train_loss_global'],
            'train/loss_roi': row['train_loss_roi'],
            'train/div_rms_global': row['train_div_rms_global'],
            'train/div_rms_roi': row['train_div_rms_roi'],
            'train/mom_rms_global': row['train_mom_rms_global'],
            'train/mom_rms_roi': row['train_mom_rms_roi'],
            'train/wall_bc_roi': row['train_wall_bc_roi'],
            'val/global_nrmse_umag': row['val_global_nrmse_umag'],
            'val/global_nrmse_p': row['val_global_nrmse_p'],
            'val/roi_nrmse_umag': row['val_roi_nrmse_umag'],
            'val/roi_nrmse_p': row['val_roi_nrmse_p'],
            'val/selector_umag': row['val_selector_umag'],
            'val/selector_p': row['val_selector_p'],
            'val/selector_ms_roi_umag': row['val_selector_ms_roi_umag'],
            'val/selector': row['val_selector'],
            'train/lr': row['lr'],
        }
        for short in _CATEGORY_SHORT.values():
            wandb_payload[f'val/global_{short}_nrmse_umag'] = row[f'val_global_{short}_nrmse_umag']
            wandb_payload[f'val/global_{short}_nrmse_p'] = row[f'val_global_{short}_nrmse_p']
            wandb_payload[f'val/roi_{short}_nrmse_umag'] = row[f'val_roi_{short}_nrmse_umag']
            wandb_payload[f'val/roi_{short}_nrmse_p'] = row[f'val_roi_{short}_nrmse_p']
        wandb_log(wandb_run, wandb_payload, step=epoch)

        ckpt = {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scalers': scalers,
            'split': split,
            'history': history,
            'best_val': float(best_score),
            'best_epoch': int(best_epoch),
            'train_config': {
                'train_mode': str(train_mode),
                'train_loss': str(train_loss),
                'momentum_loss_mode': str(momentum_loss_mode),
                'train_struct_mode': str(train_struct_mode),
                'train_struct_weight': float(train_struct_weight),
                'scheduler_mode': str(scheduler_mode),
                'hard_ground_bc': bool(hard_ground_bc),
                'use_amp': bool(use_amp),
                'amp_dtype': str(amp_dtype),
                'model_kind': 'hybrid',
            },
        }
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        if scaler.is_enabled():
            ckpt['scaler_state_dict'] = scaler.state_dict()
        (save_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, save_dir / 'checkpoints' / 'latest.pth')
        if epoch >= int(MIN_EPOCH_FOR_BEST) and selector < best_score:
            best_score = float(selector)
            best_epoch = int(epoch)
            ckpt['best_val'] = float(best_score)
            ckpt['best_epoch'] = int(best_epoch)
            torch.save(ckpt, save_dir / 'checkpoints' / 'best.pth')
        patience = int(max(0, EARLY_STOPPING_PATIENCE))
        if patience > 0 and best_epoch >= int(MIN_EPOCH_FOR_BEST):
            stale_epochs = int(epoch) - int(best_epoch)
            if stale_epochs >= patience:
                stop_epoch = int(epoch)
                print(
                    f"[TRAIN] early stop at epoch={int(epoch)} | best epoch={int(best_epoch)} | "
                    f"stale={int(stale_epochs)}",
                    flush=True,
                )
                break

    if best_epoch < 0:
        shutil.copy2(save_dir / 'checkpoints' / 'latest.pth', save_dir / 'checkpoints' / 'best.pth')
        best_epoch = int(stop_epoch)
        best_score = float(history[-1]['val_selector']) if history else float('inf')

    write_json(save_dir / 'checkpoint_paths.json', {
        'latest': str(save_dir / 'checkpoints' / 'latest.pth'),
        'best': str(save_dir / 'checkpoints' / 'best.pth'),
    })
    wandb_log(
        wandb_run,
        {
            'best/epoch': int(best_epoch),
            'best/val_selector': float(best_score),
        },
        step=int(stop_epoch) if stop_epoch > 0 else None,
    )
    print(f"[TRAIN] done | best epoch={int(best_epoch)} | best selector={float(best_score):.4f}", flush=True)
    return {
        'save_dir': str(save_dir),
        'stop_epoch': int(stop_epoch),
        'best_epoch': int(best_epoch),
        'best_val': float(best_score),
        'scalers': scalers,
        'history': history,
    }
