"""Entry point for the unified terrain-structure hybrid PINN / DL model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import config as config_mod
from config import (
    AMP_DTYPE,
    CHARB_EPS,
    DEVICE,
    HARD_GROUND_BC,
    LR,
    MOMENTUM_LOSS_MODE,
    N_EPOCHS,
    PLOT_EVAL,
    PRED_BATCH_SIZE,
    RECOMMENDED_SPLIT_JSON,
    RESULTS_ROOT,
    SCHEDULER_MODE,
    TRAIN_LOSS,
    TRAIN_MODE,
    TRAIN_STRUCT_MODE,
    TRAIN_STRUCT_WEIGHT,
    USE_AMP,
    WANDB_ENABLED,
    WANDB_PROJECT_DL,
    WANDB_PROJECT_PINN,
    WEIGHT_DECAY,
)
from data_loader import CaseRepository, load_split
from model_runtime import load_model_bundle_from_checkpoint
from models import create_model
from training import CascadeConditioner, evaluate_split, train_model
from utils import (
    ensure_dir,
    maybe_wandb_init,
    parse_wandb_tags,
    seed_everything,
    wandb_finish,
    wandb_log,
    write_json,
)


_SNAPSHOT_KEYS = [
    'DATA_CFD_ROOT',
    'SEED',
    'TRAIN_MODE',
    'N_EPOCHS',
    'MIN_EPOCH_FOR_BEST',
    'EARLY_STOPPING_PATIENCE',
    'LATEST_CKPT_EVERY',
    'HIDDEN_DIM',
    'DEPTH',
    'GLOBAL_ENCODER_WIDTH',
    'ROI_ENCODER_WIDTH',
    'STRUCTURE_ENCODER_WIDTH',
    'ENCODER_DEPTH',
    'GLOBAL_ENCODER_DEPTH',
    'ROI_ENCODER_DEPTH',
    'STRUCTURE_ENCODER_DEPTH',
    'GLOBAL_ENCODER_DILATIONS',
    'ROI_ENCODER_DILATIONS',
    'STRUCTURE_ENCODER_DILATIONS',
    'DROPOUT',
    'NUM_FOURIER_FEATURES',
    'FOURIER_SIGMA',
    'USE_STRUCTURE_ENCODER',
    'STRUCTURE_ENCODER_INPUT_MODE',
    'STRUCTURE_HEIGHT_SCALE',
    'STRUCTURE_CONTEXT_DISTANCE_SCALE_M',
    'STRUCTURE_CONTEXT_WAKE_LENGTH_MULT',
    'STRUCTURE_CONTEXT_WAKE_WIDTH_GROWTH',
    'STRUCTURE_CONTEXT_DENSITY_SIGMA_M',
    'GRID_UNET_BASE_WIDTH',
    'GRID_UNET_LEVELS',
    'GRID_UNET_DROPOUT',
    'GRID_UNET_ROI_STRUCTURE_MODE',
    'GRID_UNET_USE_TERRAIN_CONTEXT',
    'GRID_UNET_TERRAIN_CONTEXT_WIDTH',
    'GRID_UNET_TERRAIN_CONTEXT_DEPTH',
    'GRID_UNET_TERRAIN_CONTEXT_DILATIONS',
    'CASCADE_STAGE',
    'CASCADE_STAGE2_REFINER_KIND',
    'CASCADE_USE_ABL_VELOCITY_BASELINE',
    'CASCADE_ZERO_INIT_HEAD',
    'CASCADE_EDGE_WEIGHT',
    'CASCADE_EDGE_BAND_XY_M',
    'CASCADE_EDGE_BAND_Z_M',
    'CASCADE_FREEZE_MAX_CT_UMAG',
    'CASCADE_FREEZE_MAX_CT_P',
    'CASCADE_FREEZE_MAX_SELECTOR_DELTA',
    'CASCADE_MIN_STRUCTURE_CASES',
    'CASCADE_STAGE2_GRID_MAX_ROI_CELLS',
    'CASCADE_STAGE2_MS_REPEAT_ENABLED',
    'CASCADE_STAGE2_MS_REPEAT_N2',
    'CASCADE_STAGE2_MS_REPEAT_N3',
    'CASCADE_STAGE2_MS_REPEAT_MAX',
    'GLOBAL_POINTS_PER_DOMAIN',
    'ROI_POINTS_PER_DOMAIN',
    'ROI_SUPERVISED_SAMPLER_MODE',
    'ROI_TARGET_VERY_NEAR_WALL_FRAC',
    'ROI_TARGET_NEAR_WALL_FRAC',
    'ROI_TARGET_GEOM_WAKE_FRAC',
    'ROI_TARGET_LOW_SPEED_FRAC',
    'ROI_TARGET_HIGH_SPEED_FRAC',
    'ROI_TARGET_RANDOM_FRAC',
    'ROI_TARGET_VERY_NEAR_WALL_DMAX',
    'ROI_TARGET_VERY_NEAR_WALL_MAX_REPEAT',
    'ROI_TARGET_NEAR_WALL_DMAX',
    'ROI_TARGET_NEAR_WALL_BACKFILL_DMAX',
    'ROI_TARGET_WAKE_MIN_CONTEXT',
    'ROI_TARGET_WAKE_ZREL_MAX',
    'ROI_TARGET_LOW_SPEED_RATIO_MAX',
    'ROI_TARGET_LOW_SPEED_ZREL_MAX',
    'ROI_TARGET_HIGH_SPEED_RATIO_MIN',
    'ROI_TARGET_HIGH_SPEED_ZREL_MAX',
    'ROI_TARGET_MAX_ABOVE_STRUCTURE_H',
    'ROI_PATCH_HIGH_SPEED_PROB',
    'ROI_PATCH_HIGH_SPEED_RATIO_MIN',
    'ROI_PATCH_HIGH_SPEED_ZREL_MAX',
    'GLOBAL_PATCH_SHAPE',
    'ROI_PATCH_SHAPE',
    'GLOBAL_PATCHES_PER_DOMAIN',
    'ROI_PATCHES_PER_DOMAIN',
    'EVAL_GLOBAL_PATCHES_PER_CASE',
    'EVAL_ROI_PATCHES_PER_CASE',
    'GLOBAL_TERRAIN_TENSOR_CACHE_LIMIT',
    'ROI_TERRAIN_TENSOR_CACHE_LIMIT',
    'STRUCTURE_TENSOR_CACHE_LIMIT',
    'PATCH_NEAR_GROUND_PROB',
    'GLOBAL_SUPERVISED_NEAR_GROUND_FRAC',
    'GLOBAL_SUPERVISED_GROUND_K_FRAC',
    'BC_POINTS_INLET',
    'BC_POINTS_OUTLET',
    'BC_POINTS_SIDE',
    'BC_POINTS_TOP',
    'TRAIN_LOSS',
    'MOMENTUM_LOSS_MODE',
    'TRAIN_STRUCT_MODE',
    'TRAIN_STRUCT_WEIGHT',
    'CHARB_EPS',
    'DATA_P_WEIGHT',
    'GLOBAL_DATA_P_WEIGHT',
    'ROI_DATA_P_WEIGHT',
    'VAL_SELECTOR_P_WEIGHT',
    'VAL_SELECTOR_USE_GAUGE_P',
    'VAL_SELECTOR_MS_ROI_UMAG_WEIGHT',
    'SUP_WEIGHT_NEAR_STRUCTURE_GAIN',
    'SUP_WEIGHT_NEAR_STRUCTURE_DMAX',
    'SUP_WEIGHT_WAKE_GAIN',
    'SUP_WEIGHT_WAKE_ZREL_MAX',
    'SUP_WEIGHT_WAKE_SPEED_RATIO_MAX',
    'SUP_WEIGHT_WAKE_POWER',
    'W_DATA_GLOBAL',
    'W_DATA_ROI',
    'W_PHYS_GLOBAL',
    'W_PHYS_ROI',
    'W_DIV_GLOBAL',
    'W_DIV_ROI',
    'W_MOM_GLOBAL',
    'W_MOM_ROI',
    'W_BC_INLET',
    'W_BC_OUTLET',
    'W_BC_SIDE',
    'W_BC_TOP',
    'W_BC_WALL_ROI',
    'ROI_WALL_BC_DMAX',
    'PHYS_RAMP_ENABLED',
    'PHYS_RAMP_START_EPOCH',
    'PHYS_RAMP_END_EPOCH',
    'PHYS_RAMP_APPLY_GLOBAL',
    'PHYS_RAMP_APPLY_ROI',
    'PHYS_RAMP_APPLY_BC',
    'HARD_GROUND_BC',
    'USE_AMP',
    'AMP_DTYPE',
    'LR',
    'WEIGHT_DECAY',
    'SCHEDULER_MODE',
    'ONECYCLE_PCT_START',
    'ONECYCLE_DIV_FACTOR',
    'ONECYCLE_FINAL_DIV_FACTOR',
]


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _config_snapshot() -> dict:
    snap = {}
    for key in _SNAPSHOT_KEYS:
        if hasattr(config_mod, key):
            snap[key] = _jsonable(getattr(config_mod, key))
    return snap


def _wandb_project_for_mode(mode: str) -> str:
    return str(WANDB_PROJECT_DL if str(mode).lower() == 'dl' else WANDB_PROJECT_PINN)


def _summary_log_payload(prefix: str, summary: dict) -> dict:
    return {
        f'{prefix}/global_nrmse_umag': float(summary.get('global_mean_nrmse_umag', float('nan'))),
        f'{prefix}/global_nrmse_p': float(summary.get('global_mean_nrmse_p', float('nan'))),
        f'{prefix}/global_nrmse_p_gauge': float(summary.get('global_mean_nrmse_p_gauge', float('nan'))),
        f'{prefix}/roi_nrmse_umag': float(summary.get('roi_mean_nrmse_umag', float('nan'))),
        f'{prefix}/roi_nrmse_p': float(summary.get('roi_mean_nrmse_p', float('nan'))),
        f'{prefix}/roi_nrmse_p_gauge': float(summary.get('roi_mean_nrmse_p_gauge', float('nan'))),
        f'{prefix}/global_case_count': int(summary.get('global_case_count', 0)),
        f'{prefix}/roi_case_count': int(summary.get('roi_case_count', 0)),
    }


def _resolve_model_kind_from_checkpoint(ckpt: dict, fallback: str) -> str:
    train_cfg = ckpt.get('train_config') if isinstance(ckpt, dict) else None
    if isinstance(train_cfg, dict):
        kind = str(train_cfg.get('model_kind', '')).strip().lower()
        if kind in {'hybrid', 'grid_unet', 'cascade_stage1', 'cascade_stage2'}:
            return kind
    return str(fallback)


def _resolve_cascade_stage1_checkpoint_path(arg_value: str, ckpt: dict | None) -> str:
    if str(arg_value or '').strip():
        return str(arg_value)
    if isinstance(ckpt, dict):
        direct = str(ckpt.get('cascade_stage1_checkpoint', '') or '').strip()
        if direct:
            return direct
        train_cfg = ckpt.get('train_config')
        if isinstance(train_cfg, dict):
            nested = str(train_cfg.get('cascade_stage1_checkpoint', '') or '').strip()
            if nested:
                return nested
    return ''


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Unified terrain-structure hybrid PINN')
    ap.add_argument('--experiment-name', required=True)
    ap.add_argument('--split-json', default=str(RECOMMENDED_SPLIT_JSON))
    ap.add_argument('--device', default=DEVICE)
    ap.add_argument('--epochs', type=int, default=N_EPOCHS)
    ap.add_argument('--model', default='hybrid', choices=['hybrid', 'grid_unet', 'unet', 'cascade', 'cascade_stage1', 'cascade_stage2'])
    ap.add_argument('--mode', default=TRAIN_MODE, choices=['pinn', 'dl'])
    ap.add_argument('--train-loss', default=TRAIN_LOSS, choices=['rmse', 'mse', 'charb', 'charb_weighted'])
    ap.add_argument('--momentum-loss-mode', default=MOMENTUM_LOSS_MODE, choices=['constant', 'nut'])
    ap.add_argument('--train-struc-mode', default=TRAIN_STRUCT_MODE, choices=['none', 'grad', 'fft'])
    ap.add_argument('--train-struc-weight', type=float, default=TRAIN_STRUCT_WEIGHT)
    ap.add_argument('--charb-eps', type=float, default=CHARB_EPS)
    ap.add_argument('--scheduler', default=SCHEDULER_MODE, choices=['onecycle', 'none'])
    ap.add_argument('--amp', action=argparse.BooleanOptionalAction, default=USE_AMP)
    ap.add_argument('--amp-dtype', default=AMP_DTYPE, choices=['bf16', 'fp16'])
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--checkpoint', default='')
    ap.add_argument('--init-checkpoint', default='')
    ap.add_argument('--cascade-stage1-checkpoint', default='')
    ap.add_argument('--pred-batch-size', type=int, default=PRED_BATCH_SIZE)
    ap.add_argument('--hard-ground-bc', action=argparse.BooleanOptionalAction, default=HARD_GROUND_BC)
    ap.add_argument('--plot-eval', action=argparse.BooleanOptionalAction, default=PLOT_EVAL)
    ap.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--no-wandb', action='store_true')
    ap.add_argument('--wandb-project', default='')
    ap.add_argument('--wandb-entity', default='')
    ap.add_argument('--wandb-tags', default='')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(int(getattr(config_mod, 'SEED', 42)))
    repo = CaseRepository()
    split = load_split(Path(args.split_json))
    train_roi_count = int(sum(len(repo.roi_names(name)) for name in split['train']))
    val_roi_count = int(sum(len(repo.roi_names(name)) for name in split['val']))
    test_roi_count = int(sum(len(repo.roi_names(name)) for name in split['test']))
    save_dir = RESULTS_ROOT / args.experiment_name
    ensure_dir(save_dir)
    model_kind = str(args.model)
    latest_ckpt = save_dir / 'checkpoints' / 'latest.pth'
    preload_ckpt = None
    init_ckpt = None
    init_ckpt_path = None
    if args.init_checkpoint:
        init_ckpt_path = Path(args.init_checkpoint)
        if not init_ckpt_path.is_absolute():
            init_ckpt_path = (config_mod.ROOT / init_ckpt_path).resolve()
    if args.eval_only and args.checkpoint:
        preload_ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model_kind = _resolve_model_kind_from_checkpoint(preload_ckpt, model_kind)
    elif bool(args.resume) and latest_ckpt.exists():
        preload_ckpt = torch.load(latest_ckpt, map_location=args.device, weights_only=False)
        model_kind = _resolve_model_kind_from_checkpoint(preload_ckpt, model_kind)
    elif init_ckpt_path is not None:
        init_ckpt = torch.load(init_ckpt_path, map_location=args.device, weights_only=False)
        model_kind = _resolve_model_kind_from_checkpoint(init_ckpt, model_kind)
    model = create_model(device=args.device, model_kind=model_kind)
    model_kind = str(getattr(model, 'model_kind', model_kind))
    cascade_stage1_checkpoint = ''
    cascade_conditioner = None
    if model_kind == 'cascade_stage2':
        cascade_stage1_checkpoint = _resolve_cascade_stage1_checkpoint_path(
            args.cascade_stage1_checkpoint,
            preload_ckpt if preload_ckpt is not None else init_ckpt,
        )
        if not cascade_stage1_checkpoint and latest_ckpt.exists() and not args.eval_only:
            try:
                latest_meta = preload_ckpt if preload_ckpt is not None else torch.load(latest_ckpt, map_location=args.device, weights_only=False)
            except TypeError:
                latest_meta = preload_ckpt if preload_ckpt is not None else torch.load(latest_ckpt, map_location=args.device)
            cascade_stage1_checkpoint = _resolve_cascade_stage1_checkpoint_path('', latest_meta)
        if not cascade_stage1_checkpoint:
            raise SystemExit('cascade_stage2 requires --cascade-stage1-checkpoint (or checkpoint metadata with that path)')
        stage1_path = Path(cascade_stage1_checkpoint)
        if not stage1_path.is_absolute():
            stage1_path = (config_mod.ROOT / stage1_path).resolve()
        cond_model, cond_scalers, cond_ckpt = load_model_bundle_from_checkpoint(stage1_path, args.device)
        cond_train_cfg = cond_ckpt.get('train_config') if isinstance(cond_ckpt, dict) else {}
        cond_snapshot = cond_ckpt.get('config_snapshot') if isinstance(cond_ckpt, dict) else {}
        if not cond_snapshot and isinstance(cond_train_cfg, dict):
            cond_snapshot = cond_train_cfg.get('config_snapshot', {}) or {}
        for p in cond_model.parameters():
            p.requires_grad_(False)
        cascade_conditioner = CascadeConditioner(
            model=cond_model,
            scalers=cond_scalers,
            checkpoint_path=str(stage1_path),
            config_snapshot=cond_snapshot if isinstance(cond_snapshot, dict) else {},
        )
        cascade_stage1_checkpoint = str(stage1_path)
    print(
        f"[RUN] exp={args.experiment_name} | model={model_kind} | mode={args.mode} | device={args.device} | "
        f"epochs={int(args.epochs)} | loss={args.train_loss} | mom={args.momentum_loss_mode} | "
        f"struc={args.train_struc_mode}:{float(args.train_struc_weight):.3g}",
        flush=True,
    )
    print(
        f"[RUN] split loaded | train={len(split['train'])} ({train_roi_count} ROI) | "
        f"val={len(split['val'])} ({val_roi_count} ROI) | test={len(split['test'])} ({test_roi_count} ROI)",
        flush=True,
    )

    run_config = {
        'experiment_name': str(args.experiment_name),
        'split_json': str(args.split_json),
        'device': str(args.device),
        'epochs': int(args.epochs),
        'model': str(model_kind),
        'mode': str(args.mode),
        'train_loss': str(args.train_loss),
        'momentum_loss_mode': str(args.momentum_loss_mode),
        'train_struc_mode': str(args.train_struc_mode),
        'train_struc_weight': float(args.train_struc_weight),
        'charb_eps': float(args.charb_eps),
        'scheduler': str(args.scheduler),
        'use_amp': bool(args.amp),
        'amp_dtype': str(args.amp_dtype),
        'hard_ground_bc': bool(args.hard_ground_bc),
        'plot_eval': bool(args.plot_eval),
        'resume': bool(args.resume),
        'init_checkpoint': '' if init_ckpt_path is None else str(init_ckpt_path),
        'cascade_stage1_checkpoint': str(cascade_stage1_checkpoint),
        'wandb_enabled': bool(WANDB_ENABLED and not args.no_wandb),
        'wandb_project': str(args.wandb_project or _wandb_project_for_mode(args.mode)),
        'wandb_entity': str(args.wandb_entity or ''),
        'wandb_tags': parse_wandb_tags(args.wandb_tags),
        'model_hparams': _jsonable(getattr(model, 'model_hparams', {})),
        'config_snapshot': _config_snapshot(),
        'split_counts': {
            'train': int(len(split['train'])),
            'val': int(len(split['val'])),
            'test': int(len(split['test'])),
        },
    }
    write_json(save_dir / 'run_config.json', run_config)

    wandb_run = maybe_wandb_init(
        enabled=bool(WANDB_ENABLED and not args.no_wandb),
        project=str(args.wandb_project or _wandb_project_for_mode(args.mode)),
        entity=str(args.wandb_entity or '') or None,
        name=str(args.experiment_name),
        tags=parse_wandb_tags(args.wandb_tags),
        config=run_config,
        wandb_dir=save_dir / 'logs' / 'wandb',
        resume=bool(args.resume),
    )

    try:
        if args.eval_only:
            if not args.checkpoint:
                raise SystemExit('--eval-only requires --checkpoint')
            print(f"[RUN] eval-only | checkpoint={args.checkpoint}", flush=True)
            ckpt = preload_ckpt if preload_ckpt is not None else torch.load(args.checkpoint, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            scalers = ckpt['scalers']
            val_summary = evaluate_split(
                model,
                repo,
                split['val'],
                scalers,
                conditioner=cascade_conditioner,
                device=args.device,
                pred_batch_size=args.pred_batch_size,
                output_dir=save_dir / 'eval' / 'val_cases',
                split_label='val',
                hard_ground_bc=args.hard_ground_bc,
                plot_eval=args.plot_eval,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
            )
            test_summary = evaluate_split(
                model,
                repo,
                split['test'],
                scalers,
                conditioner=cascade_conditioner,
                device=args.device,
                pred_batch_size=args.pred_batch_size,
                output_dir=save_dir / 'eval' / 'test_cases',
                split_label='test',
                hard_ground_bc=args.hard_ground_bc,
                plot_eval=args.plot_eval,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
            )
            write_json(save_dir / 'eval' / 'summary.json', {'val': val_summary, 'test': test_summary, 'run_config': run_config})
            wandb_log(wandb_run, {**_summary_log_payload('eval_val', val_summary), **_summary_log_payload('eval_test', test_summary)})
            print("[RUN] eval-only complete", flush=True)
            return

        eval_summary_path = save_dir / 'eval' / 'summary.json'
        if bool(args.resume) and eval_summary_path.exists():
            print(f"[RUN] existing completed run found at {eval_summary_path}, skipping", flush=True)
            return

        resume_ckpt = None
        if bool(args.resume) and latest_ckpt.exists():
            print(f"[RUN] resume checkpoint detected | path={latest_ckpt}", flush=True)
            resume_ckpt = preload_ckpt if preload_ckpt is not None else torch.load(latest_ckpt, map_location=args.device, weights_only=False)
            model.load_state_dict(resume_ckpt['model_state_dict'])
            print(
                f"[RUN] resuming experiment from epoch={int(resume_ckpt.get('epoch', 0))} | "
                f"best epoch={int(resume_ckpt.get('best_epoch', -1))}",
                flush=True,
            )
        elif init_ckpt is not None:
            print(f"[RUN] warm-start checkpoint detected | path={init_ckpt_path}", flush=True)
            model.load_state_dict(init_ckpt['model_state_dict'])

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        print("[RUN] training...", flush=True)
        out = train_model(
            model,
            repo,
            split,
            conditioner=cascade_conditioner,
            optimizer=optimizer,
            epochs=args.epochs,
            device=args.device,
            save_dir=save_dir,
            train_mode=args.mode,
            train_loss=args.train_loss,
            momentum_loss_mode=args.momentum_loss_mode,
            train_struct_mode=args.train_struc_mode,
            train_struct_weight=args.train_struc_weight,
            scheduler_mode=args.scheduler,
            hard_ground_bc=args.hard_ground_bc,
            charb_eps=args.charb_eps,
            pred_batch_size=args.pred_batch_size,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            resume_checkpoint=resume_ckpt,
            config_snapshot=run_config['config_snapshot'],
            cascade_stage1_checkpoint=cascade_stage1_checkpoint,
            wandb_run=wandb_run,
        )
        best_ckpt = save_dir / 'checkpoints' / 'best.pth'
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            scalers = ckpt['scalers']
        else:
            scalers = out['scalers']
        print(
            f"[RUN] training complete | best epoch={int(out['best_epoch'])} | "
            f"selector={float(out['best_val']):.4f} | evaluating val/test...",
            flush=True,
        )
        val_summary = evaluate_split(
            model,
            repo,
            split['val'],
            scalers,
            conditioner=cascade_conditioner,
            device=args.device,
            pred_batch_size=args.pred_batch_size,
            output_dir=save_dir / 'eval' / 'val_cases',
            split_label='val',
            hard_ground_bc=args.hard_ground_bc,
            plot_eval=args.plot_eval,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        test_summary = evaluate_split(
            model,
            repo,
            split['test'],
            scalers,
            conditioner=cascade_conditioner,
            device=args.device,
            pred_batch_size=args.pred_batch_size,
            output_dir=save_dir / 'eval' / 'test_cases',
            split_label='test',
            hard_ground_bc=args.hard_ground_bc,
            plot_eval=args.plot_eval,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        final_summary = {
            'val': val_summary,
            'test': test_summary,
            'best_epoch': out['best_epoch'],
            'best_val': out['best_val'],
            'run_config': run_config,
        }
        write_json(save_dir / 'eval' / 'summary.json', final_summary)
        wandb_log(
            wandb_run,
            {
                'best/epoch': int(out['best_epoch']),
                'best/val_selector': float(out['best_val']),
                **_summary_log_payload('eval_val', val_summary),
                **_summary_log_payload('eval_test', test_summary),
            },
            step=int(out.get('stop_epoch', out['best_epoch'])) if int(out.get('stop_epoch', out['best_epoch'])) > 0 else None,
        )
        print("[RUN] finished", flush=True)
    finally:
        wandb_finish(wandb_run)


if __name__ == '__main__':
    main()
