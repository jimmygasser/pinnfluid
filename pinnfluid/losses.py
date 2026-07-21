"""Losses for the unified terrain-structure hybrid PINN."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from config import (
    ABL_UREF_MAX,
    ABL_Z0_MAX,
    ABL_Z0_MIN,
    ABL_ZREF_MAX,
    CHARB_EPS,
    DATA_P_WEIGHT,
    FFT_MIN_FLUID_FRAC,
    FD_FLUID_MASK_THRESHOLD,
    MOMENTUM_LOSS_MODE,
    NU,
    PHI_WALL_H,
    ROI_WALL_BC_DMAX,
    SUP_WEIGHT_NEAR_STRUCTURE_DMAX,
    SUP_WEIGHT_NEAR_STRUCTURE_GAIN,
    SUP_WEIGHT_WAKE_GAIN,
    SUP_WEIGHT_WAKE_POWER,
    SUP_WEIGHT_WAKE_SPEED_RATIO_MAX,
    SUP_WEIGHT_WAKE_ZREL_MAX,
    WEIGHTED_CHARB_HIGH_GAIN,
    WEIGHTED_CHARB_HIGH_START,
    WEIGHTED_CHARB_LOW_GAIN,
    WEIGHTED_CHARB_LOW_START,
    WEIGHTED_CHARB_POWER,
)
from data_loader import PatchBatch
from physics_grid import NonUniformGridPhysics


def _channel_weights(n_channels: int, *, p_weight: float, device, dtype) -> torch.Tensor:
    w = torch.ones(n_channels, device=device, dtype=dtype)
    if p_weight != 1.0 and n_channels > 3:
        w[3] = float(p_weight)
    return w


def _charbonnier(err: torch.Tensor, eps: float) -> torch.Tensor:
    eps_t = torch.as_tensor(float(max(eps, 1e-12)), dtype=err.dtype, device=err.device)
    return torch.sqrt(err * err + eps_t * eps_t) - eps_t


def inverse_scale_outputs(y_scaled: torch.Tensor, y_scaler, *, device: str) -> torch.Tensor:
    scale = torch.as_tensor(y_scaler.scale_, dtype=torch.float32, device=device).view(1, -1)
    mean = torch.as_tensor(y_scaler.mean_, dtype=torch.float32, device=device).view(1, -1)
    return y_scaled * scale + mean


def scale_outputs(y_phys: torch.Tensor, y_scaler, *, device: str) -> torch.Tensor:
    scale = torch.as_tensor(y_scaler.scale_, dtype=torch.float32, device=device).view(1, -1)
    mean = torch.as_tensor(y_scaler.mean_, dtype=torch.float32, device=device).view(1, -1)
    return (y_phys - mean) / scale


def _inverse_minmax_column(x_batch: torch.Tensor, x_scaler, idx: int) -> torch.Tensor:
    xmin = torch.as_tensor(float(x_scaler.data_min_[idx]), dtype=x_batch.dtype, device=x_batch.device)
    xmax = torch.as_tensor(float(x_scaler.data_max_[idx]), dtype=x_batch.dtype, device=x_batch.device)
    return x_batch[:, idx] * (xmax - xmin) + xmin


def inverse_minmax_column_from_scaled_inputs(x_batch: torch.Tensor, x_scaler, idx: int) -> torch.Tensor:
    return _inverse_minmax_column(x_batch, x_scaler, idx)


def abl_velocity_baseline_from_scaled_inputs(
    x_batch: torch.Tensor,
    *,
    x_scaler,
    input_cols: list[str],
) -> torch.Tensor:
    cols = list(input_cols)
    z_rel = _inverse_minmax_column(x_batch, x_scaler, int(cols.index('z_rel')))
    uref_norm = _inverse_minmax_column(x_batch, x_scaler, int(cols.index('Uref_norm')))
    zref_norm = _inverse_minmax_column(x_batch, x_scaler, int(cols.index('Zref_norm')))
    log10_z0_norm = _inverse_minmax_column(x_batch, x_scaler, int(cols.index('log10_z0_norm')))
    flow_x = _inverse_minmax_column(x_batch, x_scaler, int(cols.index('flowDir_x')))
    flow_y = _inverse_minmax_column(x_batch, x_scaler, int(cols.index('flowDir_y')))
    flow_z = _inverse_minmax_column(x_batch, x_scaler, int(cols.index('flowDir_z')))

    uref = torch.clamp(uref_norm * float(ABL_UREF_MAX), min=1e-6)
    zref = torch.clamp(zref_norm * float(ABL_ZREF_MAX), min=1e-6)
    log10_z0 = (
        log10_z0_norm
        * (float(torch.log10(torch.tensor(float(ABL_Z0_MAX))).item()) - float(torch.log10(torch.tensor(float(ABL_Z0_MIN))).item()))
        + float(torch.log10(torch.tensor(float(ABL_Z0_MIN))).item())
    )
    z0 = torch.clamp(torch.pow(torch.full_like(log10_z0, 10.0), log10_z0), min=float(ABL_Z0_MIN), max=float(ABL_Z0_MAX))
    den = torch.log((zref + z0) / z0).clamp(min=1e-6)
    speed = uref * torch.log((torch.clamp(z_rel, min=0.0) + z0) / z0) / den
    baseline = torch.zeros((x_batch.shape[0], 4), dtype=x_batch.dtype, device=x_batch.device)
    baseline[:, 0] = speed * flow_x
    baseline[:, 1] = speed * flow_y
    baseline[:, 2] = speed * flow_z
    return baseline


def compose_prediction_with_velocity_baseline(
    pred_scaled_resid: torch.Tensor,
    *,
    x_batch: torch.Tensor,
    x_scaler,
    input_cols: list[str],
    y_scaler,
    hard_ground_bc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline_phys = abl_velocity_baseline_from_scaled_inputs(
        x_batch,
        x_scaler=x_scaler,
        input_cols=input_cols,
    )
    baseline_scaled = scale_outputs(baseline_phys, y_scaler, device=str(pred_scaled_resid.device))
    pred_scaled = pred_scaled_resid + baseline_scaled
    return apply_output_constraints_from_scaled_inputs(
        pred_scaled,
        x_batch=x_batch,
        x_scaler=x_scaler,
        input_cols=input_cols,
        y_scaler=y_scaler,
        hard_ground_bc=hard_ground_bc,
    )


def _supervised_sample_weights(
    *,
    x_batch: Optional[torch.Tensor],
    x_scaler,
    input_cols: Optional[list[str]],
    y_true: torch.Tensor,
    y_scaler,
) -> Optional[torch.Tensor]:
    if x_batch is None or x_scaler is None or input_cols is None or y_scaler is None:
        return None

    cols = list(input_cols)
    sample_w = torch.ones(y_true.shape[0], dtype=y_true.dtype, device=y_true.device)

    near_structure_gain = float(SUP_WEIGHT_NEAR_STRUCTURE_GAIN)
    if near_structure_gain > 0.0 and 'phi_wall' in cols:
        phi_idx = int(cols.index('phi_wall'))
        phi_wall = _inverse_minmax_column(x_batch, x_scaler, phi_idx)
        phi_cap = float(torch.tanh(torch.tensor(float(max(SUP_WEIGHT_NEAR_STRUCTURE_DMAX, 1e-6)) / max(float(PHI_WALL_H), 1e-6))).item())
        phi_cap = max(phi_cap, 1e-6)
        near_term = torch.clamp((phi_cap - phi_wall) / phi_cap, min=0.0, max=1.0)
        sample_w = sample_w + near_structure_gain * near_term

    wake_gain = float(SUP_WEIGHT_WAKE_GAIN)
    if wake_gain > 0.0 and 'z_rel' in cols and 'Uref_norm' in cols:
        z_rel_idx = int(cols.index('z_rel'))
        uref_idx = int(cols.index('Uref_norm'))
        z_rel = _inverse_minmax_column(x_batch, x_scaler, z_rel_idx)
        uref_norm = _inverse_minmax_column(x_batch, x_scaler, uref_idx)
        uref = torch.clamp(uref_norm * float(ABL_UREF_MAX), min=1e-6)
        y_true_phys = inverse_scale_outputs(y_true, y_scaler, device=str(y_true.device))
        speed_ratio = torch.linalg.norm(y_true_phys[:, :3], dim=1) / uref
        low_speed = torch.clamp(
            (float(SUP_WEIGHT_WAKE_SPEED_RATIO_MAX) - speed_ratio) / max(float(SUP_WEIGHT_WAKE_SPEED_RATIO_MAX), 1e-6),
            min=0.0,
            max=1.0,
        )
        near_ground = torch.clamp(
            (float(SUP_WEIGHT_WAKE_ZREL_MAX) - z_rel) / max(float(SUP_WEIGHT_WAKE_ZREL_MAX), 1e-6),
            min=0.0,
            max=1.0,
        )
        wake_term = near_ground * low_speed.pow(float(max(SUP_WEIGHT_WAKE_POWER, 1e-6)))
        sample_w = sample_w + wake_gain * wake_term

    sample_w = sample_w / torch.clamp(sample_w.mean(), min=1e-6)
    return sample_w


def apply_output_constraints_from_scaled_inputs(
    pred_scaled: torch.Tensor,
    *,
    x_batch: Optional[torch.Tensor],
    x_scaler,
    input_cols: Optional[list[str]],
    y_scaler,
    hard_ground_bc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_phys = inverse_scale_outputs(pred_scaled, y_scaler, device=str(pred_scaled.device))
    if not hard_ground_bc or x_batch is None or x_scaler is None or input_cols is None:
        return pred_scaled, pred_phys
    try:
        phi_idx = int(list(input_cols).index('phi_ground'))
    except ValueError:
        return pred_scaled, pred_phys
    phi_ground = _inverse_minmax_column(x_batch, x_scaler, phi_idx)
    phi_ground = torch.nan_to_num(phi_ground, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    pred_phys = pred_phys.clone()
    pred_phys[:, 2] = pred_phys[:, 2] * phi_ground
    return scale_outputs(pred_phys, y_scaler, device=str(pred_scaled.device)), pred_phys


def supervised_data_loss_from_pred(
    pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    x_batch: Optional[torch.Tensor] = None,
    x_scaler=None,
    input_cols: Optional[list[str]] = None,
    y_scaler=None,
    p_weight: float = DATA_P_WEIGHT,
    loss_mode: str = 'rmse',
    charbonnier_eps: float = CHARB_EPS,
) -> torch.Tensor:
    mode = str(loss_mode or 'rmse').strip().lower()
    if mode == 'rmse':
        mode = 'mse'
    if mode not in {'mse', 'charb', 'charb_weighted'}:
        mode = 'mse'
    err = pred - y_true
    w = _channel_weights(pred.shape[1], p_weight=p_weight, device=pred.device, dtype=pred.dtype).unsqueeze(0)
    sample_w = _supervised_sample_weights(
        x_batch=x_batch,
        x_scaler=x_scaler,
        input_cols=input_cols,
        y_true=y_true,
        y_scaler=y_scaler,
    )
    if mode in {'charb', 'charb_weighted'}:
        base = _charbonnier(err, float(charbonnier_eps)) * w
        if mode == 'charb_weighted' and x_batch is not None and x_scaler is not None and y_scaler is not None and input_cols is not None:
            try:
                uref_idx = int(list(input_cols).index('Uref_norm'))
            except ValueError:
                uref_idx = -1
            if uref_idx >= 0:
                uref_norm = _inverse_minmax_column(x_batch, x_scaler, uref_idx)
                uref = torch.clamp(uref_norm * float(ABL_UREF_MAX), min=1e-6)
                y_phys = inverse_scale_outputs(y_true, y_scaler, device=str(y_true.device))
                umag = torch.linalg.norm(y_phys[:, :3], dim=1)
                ratio = torch.nan_to_num(umag / uref, nan=0.0, posinf=4.0, neginf=0.0).clamp(0.0, 4.0)
                low_term = torch.clamp((float(WEIGHTED_CHARB_LOW_START) - ratio) / max(float(WEIGHTED_CHARB_LOW_START), 1e-6), min=0.0, max=1.0)
                high_term = torch.clamp((ratio - float(WEIGHTED_CHARB_HIGH_START)) / max(1.0 - float(WEIGHTED_CHARB_HIGH_START), 1e-6), min=0.0, max=1.0)
                charb_extrema_w = 1.0 + float(WEIGHTED_CHARB_LOW_GAIN) * low_term.pow(float(WEIGHTED_CHARB_POWER)) + float(WEIGHTED_CHARB_HIGH_GAIN) * high_term.pow(float(WEIGHTED_CHARB_POWER))
                base = base * charb_extrema_w.unsqueeze(1)
        if sample_w is not None:
            base = base * sample_w.unsqueeze(1)
        return torch.mean(base)
    base = (err * err) * w
    if sample_w is not None:
        base = base * sample_w.unsqueeze(1)
    return torch.mean(base)


def compute_patch_physics_losses_from_pred(
    pred_scaled: torch.Tensor,
    patch: PatchBatch,
    y_scaler,
    *,
    device: str,
    momentum_loss_mode: str = MOMENTUM_LOSS_MODE,
) -> Dict[str, torch.Tensor]:
    pred = inverse_scale_outputs(pred_scaled, y_scaler, device=device)
    px, py, pz = patch.shape
    ux = pred[:, 0].view(px, py, pz)
    uy = pred[:, 1].view(px, py, pz)
    uz = pred[:, 2].view(px, py, pz)
    p = pred[:, 3].view(px, py, pz)
    mask = patch.mask.to(device) > FD_FLUID_MASK_THRESHOLD
    ops = NonUniformGridPhysics(patch.x_coords.to(device), patch.y_coords.to(device), patch.z_levels.to(device))
    div = ops.divergence(ux, uy, uz)
    mode = str(momentum_loss_mode or MOMENTUM_LOSS_MODE).strip().lower()
    if mode == 'nut':
        if patch.nut is None:
            raise RuntimeError(
                f"Momentum loss mode 'nut' requested but nut.npy is missing for patch source {patch.source_name}."
            )
        # nut.npy is NaN outside fluid cells. Finite differences use neighbours
        # before the fluid mask is applied, so non-finite nut values must be
        # replaced here or NaNs leak into adjacent valid residuals.
        nut_field = torch.nan_to_num(patch.nut.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        nut_field = torch.clamp(nut_field, min=0.0)
        nu_eff = torch.clamp(torch.as_tensor(float(NU), dtype=ux.dtype, device=device) + nut_field, min=1e-9)
        finite_nu = torch.isfinite(nu_eff)
        mask = mask & finite_nu
        rx, ry, rz = ops.momentum_residual_with_nu_field(ux, uy, uz, p, nu_eff)
    else:
        rx, ry, rz = ops.momentum_residual(ux, uy, uz, p)
    mom_sq = rx * rx + ry * ry + rz * rz
    div_scale = max(float(patch.div_scale), 1e-12)
    mom_scale = max(float(patch.mom_scale), 1e-12)
    if bool(mask.any().item()):
        div_loss = torch.mean((div[mask] / div_scale) ** 2)
        mom_loss = torch.mean(mom_sq[mask] / (mom_scale * mom_scale))
    else:
        zero = pred.sum() * 0.0
        div_loss = zero
        mom_loss = zero
    return {
        'div_loss': div_loss,
        'mom_loss': mom_loss,
        'div_rms': torch.sqrt(div_loss + 1e-12),
        'mom_rms': torch.sqrt(mom_loss + 1e-12),
    }


def _fluid_valid_mask(mask: torch.Tensor, axis: int) -> torch.Tensor:
    valid = mask.clone()
    if axis == 0:
        valid[1:-1, :, :] = mask[1:-1, :, :] & mask[:-2, :, :] & mask[2:, :, :]
        if mask.shape[0] > 1:
            valid[0, :, :] = mask[0, :, :] & mask[1, :, :]
            valid[-1, :, :] = mask[-1, :, :] & mask[-2, :, :]
    elif axis == 1:
        valid[:, 1:-1, :] = mask[:, 1:-1, :] & mask[:, :-2, :] & mask[:, 2:, :]
        if mask.shape[1] > 1:
            valid[:, 0, :] = mask[:, 0, :] & mask[:, 1, :]
            valid[:, -1, :] = mask[:, -1, :] & mask[:, -2, :]
    else:
        valid[:, :, 1:-1] = mask[:, :, 1:-1] & mask[:, :, :-2] & mask[:, :, 2:]
        if mask.shape[2] > 1:
            valid[:, :, 0] = mask[:, :, 0] & mask[:, :, 1]
            valid[:, :, -1] = mask[:, :, -1] & mask[:, :, -2]
    return valid


def _gradient_structural_loss(pred_grid: torch.Tensor, true_grid: torch.Tensor, mask: torch.Tensor, ops: NonUniformGridPhysics, *, p_weight: float, charbonnier_eps: float) -> torch.Tensor:
    channel_losses = []
    channel_weights = _channel_weights(pred_grid.shape[-1], p_weight=p_weight, device=pred_grid.device, dtype=pred_grid.dtype)
    for c in range(pred_grid.shape[-1]):
        pred_c = pred_grid[..., c]
        true_c = true_grid[..., c]
        grads_pred = (ops.grad_x(pred_c), ops.grad_y(pred_c), ops.grad_z(pred_c))
        grads_true = (ops.grad_x(true_c), ops.grad_y(true_c), ops.grad_z(true_c))
        grad_losses = []
        for axis, (gp, gt) in enumerate(zip(grads_pred, grads_true)):
            valid = _fluid_valid_mask(mask, axis=axis)
            if bool(valid.any().item()):
                grad_losses.append(torch.mean(_charbonnier(gp[valid] - gt[valid], float(charbonnier_eps))))
        if grad_losses:
            channel_losses.append(channel_weights[c] * torch.mean(torch.stack(grad_losses)))
    if not channel_losses:
        return pred_grid.sum() * 0.0
    return torch.mean(torch.stack(channel_losses))


def _fft_structural_loss(pred_grid: torch.Tensor, true_grid: torch.Tensor, mask: torch.Tensor, *, p_weight: float, min_fluid_frac: float) -> torch.Tensor:
    losses = []
    weights = _channel_weights(pred_grid.shape[-1], p_weight=p_weight, device=pred_grid.device, dtype=pred_grid.dtype)
    for k in range(pred_grid.shape[2]):
        mask2 = mask[:, :, k]
        if float(mask2.float().mean().item()) < float(min_fluid_frac):
            continue
        mask3 = mask2.unsqueeze(-1)
        pred2 = torch.where(mask3, pred_grid[:, :, k, :], torch.zeros_like(pred_grid[:, :, k, :]))
        true2 = torch.where(mask3, true_grid[:, :, k, :], torch.zeros_like(true_grid[:, :, k, :]))
        pred2 = torch.nan_to_num(pred2, nan=0.0, posinf=0.0, neginf=0.0)
        true2 = torch.nan_to_num(true2, nan=0.0, posinf=0.0, neginf=0.0)
        pred_fft = torch.fft.rfft2(pred2.permute(2, 0, 1), norm='ortho')
        true_fft = torch.fft.rfft2(true2.permute(2, 0, 1), norm='ortho')
        chan_loss = torch.mean(torch.abs(pred_fft - true_fft), dim=(1, 2))
        if bool(torch.isfinite(chan_loss).all().item()):
            losses.append(torch.mean(chan_loss * weights))
    if not losses:
        return pred_grid.sum() * 0.0
    return torch.mean(torch.stack(losses))


def structured_patch_loss_from_pred(pred_scaled: torch.Tensor, patch: PatchBatch, *, mode: str, p_weight: float = 1.0, charbonnier_eps: float = CHARB_EPS, fft_min_fluid_frac: float = FFT_MIN_FLUID_FRAC) -> torch.Tensor:
    mode = str(mode or 'none').strip().lower()
    if mode == 'none':
        return pred_scaled.sum() * 0.0
    px, py, pz = patch.shape
    pred_grid = pred_scaled.view(px, py, pz, pred_scaled.shape[1])
    true_grid = patch.y_scaled.to(pred_scaled.device).view(px, py, pz, pred_scaled.shape[1])
    mask = patch.mask.to(pred_scaled.device) > FD_FLUID_MASK_THRESHOLD
    ops = NonUniformGridPhysics(patch.x_coords.to(pred_scaled.device), patch.y_coords.to(pred_scaled.device), patch.z_levels.to(pred_scaled.device))
    if mode == 'grad':
        return _gradient_structural_loss(pred_grid, true_grid, mask, ops, p_weight=p_weight, charbonnier_eps=charbonnier_eps)
    if mode == 'fft':
        return _fft_structural_loss(pred_grid, true_grid, mask, p_weight=p_weight, min_fluid_frac=fft_min_fluid_frac)
    raise ValueError(f'Unsupported structural loss mode: {mode}')


def inlet_bc_loss_from_phys(y_pred_phys: torch.Tensor, batch, *, device: str) -> torch.Tensor:
    if batch is None or batch.u_target is None:
        return y_pred_phys.sum() * 0.0
    u_pred = y_pred_phys[:, :3]
    u_true = batch.u_target.to(device)
    u_scale = max(float(batch.u_scale), 1e-6)
    return torch.mean(torch.sum(((u_pred - u_true) / u_scale) ** 2, dim=1))


def outlet_bc_loss_from_phys(y_pred_phys: torch.Tensor, batch, *, device: str) -> torch.Tensor:
    if batch is None:
        return y_pred_phys.sum() * 0.0
    p_pred = y_pred_phys[:, 3]
    p_true = batch.p_target.to(device) if batch.p_target is not None else torch.zeros_like(p_pred)
    p_scale = max(float(batch.p_scale), 1e-6)
    return torch.mean(((p_pred - p_true) / p_scale) ** 2)


def normal_velocity_bc_loss_from_phys(y_pred_phys: torch.Tensor, batch, *, device: str) -> torch.Tensor:
    if batch is None or batch.normals is None:
        return y_pred_phys.sum() * 0.0
    normals = batch.normals.to(device)
    u_pred = y_pred_phys[:, :3]
    u_scale = max(float(batch.u_scale), 1e-6)
    u_n = torch.sum(u_pred * normals, dim=1)
    return torch.mean((u_n / u_scale) ** 2)


def roi_wall_velocity_bc_loss_from_phys(
    y_pred_phys: torch.Tensor,
    *,
    x_batch: Optional[torch.Tensor],
    x_scaler,
    input_cols: Optional[list[str]],
    device: str,
    u_scale: float,
    dmax: float = ROI_WALL_BC_DMAX,
) -> torch.Tensor:
    if x_batch is None or x_scaler is None or input_cols is None:
        return y_pred_phys.sum() * 0.0
    cols = list(input_cols)
    if 'phi_wall' not in cols:
        return y_pred_phys.sum() * 0.0
    phi_idx = int(cols.index('phi_wall'))
    phi_wall = _inverse_minmax_column(x_batch.to(device), x_scaler, phi_idx)
    phi_cap = float(torch.tanh(torch.tensor(float(max(dmax, 1e-6)) / max(float(PHI_WALL_H), 1e-6))).item())
    phi_cap = max(phi_cap, 1e-6)
    near = torch.isfinite(phi_wall) & (phi_wall <= phi_cap)
    if not bool(near.any().item()):
        return y_pred_phys.sum() * 0.0
    closeness = torch.clamp((phi_cap - phi_wall[near]) / phi_cap, min=0.0, max=1.0)
    u_pred = y_pred_phys[near, :3]
    u_scale = max(float(u_scale), 1e-6)
    return torch.mean(torch.sum((u_pred / u_scale) ** 2, dim=1) * (1.0 + closeness))
