"""Configuration for the first unified terrain-structure hybrid PINN."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
DATA_CFD_ROOT = ROOT / "data" / "cfd"
RESULTS_ROOT = ROOT / "results"
SPLITS_ROOT = SCRIPTS_DIR / "splits"

SEED = 42
DEVICE = os.environ.get("PINN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

GLOBAL_INPUT_COLS = [
    "x",
    "y",
    "z",
    "z_rel",
    "phi_ground",
    "terrain_elev",
    "terrain_slope",
    "terrain_aspect_sin",
    "terrain_aspect_cos",
    "Uref_norm",
    "Zref_norm",
    "log10_z0_norm",
    "flowDir_x",
    "flowDir_y",
    "flowDir_z",
    "Lx_norm",
    "Ly_norm",
    "Lz_norm",
]

ROI_INPUT_COLS = [
    "x",
    "y",
    "z",
    "z_rel",
    "phi_ground",
    "phi_wall",
    "terrain_elev",
    "terrain_slope",
    "terrain_aspect_sin",
    "terrain_aspect_cos",
    "Uref_norm",
    "Zref_norm",
    "log10_z0_norm",
    "flowDir_x",
    "flowDir_y",
    "flowDir_z",
    "Lx_norm",
    "Ly_norm",
    "Lz_norm",
]

OUTPUT_COLS = ["Ux", "Uy", "Uz", "p"]
TERRAIN_CHANNELS = ["elevation", "slope", "aspect_sin", "aspect_cos"]

ABL_UREF_MAX = 20.0
ABL_ZREF_MAX = 200.0
ABL_Z0_MIN = 1e-4
ABL_Z0_MAX = 2.0
SIZE_NORM_MAX = 3000.0   # max domain extent supported by domain_builder.py

# Training-time masks (see TO_READ.md)
LATERAL_TRIM_M = 5.0              # metres to trim from each lateral edge
LATERAL_TRIM_THRESHOLD = 5.0      # apply trim if bound origin < this value

PHI_GROUND_H = 20.0
PHI_WALL_H = 2.0

GLOBAL_ENCODER_WIDTH = 32
ROI_ENCODER_WIDTH = 32
STRUCTURE_ENCODER_WIDTH = 16
ENCODER_DEPTH = 3
GLOBAL_ENCODER_DEPTH = None
ROI_ENCODER_DEPTH = None
STRUCTURE_ENCODER_DEPTH = None
GLOBAL_ENCODER_DILATIONS = ()
ROI_ENCODER_DILATIONS = ()
STRUCTURE_ENCODER_DILATIONS = ()
HIDDEN_DIM = 256
DEPTH = 4
DROPOUT = 0.05
NUM_FOURIER_FEATURES = 4
FOURIER_SIGMA = 1.0
USE_STRUCTURE_ENCODER = False
STRUCTURE_ENCODER_INPUT_MODE = "basic"  # supported: basic, context_v2, context_v3
STRUCTURE_HEIGHT_SCALE = 10.0
STRUCTURE_CONTEXT_DISTANCE_SCALE_M = 20.0
STRUCTURE_CONTEXT_WAKE_LENGTH_MULT = 12.0
STRUCTURE_CONTEXT_WAKE_WIDTH_GROWTH = 0.25
STRUCTURE_CONTEXT_DENSITY_SIGMA_M = 25.0
GRID_UNET_BASE_WIDTH = 32
GRID_UNET_LEVELS = 4
GRID_UNET_DROPOUT = 0.0
GRID_UNET_ROI_STRUCTURE_MODE = "context_v2"  # supported: none, basic, context_v2, context_v3, inherit
GRID_UNET_USE_TERRAIN_CONTEXT = False
GRID_UNET_TERRAIN_CONTEXT_WIDTH = 32
GRID_UNET_TERRAIN_CONTEXT_DEPTH = 3
GRID_UNET_TERRAIN_CONTEXT_DILATIONS = ()
CASCADE_STAGE = "stage1"  # supported: stage1, stage2
CASCADE_STAGE2_REFINER_KIND = "point"  # supported: point, grid_unet
CASCADE_USE_ABL_VELOCITY_BASELINE = False
CASCADE_ZERO_INIT_HEAD = False
CASCADE_EDGE_WEIGHT = 0.0
CASCADE_EDGE_BAND_XY_M = 15.0
CASCADE_EDGE_BAND_Z_M = 10.0
CASCADE_FREEZE_MAX_CT_UMAG = 0.46
CASCADE_FREEZE_MAX_CT_P = 0.78
CASCADE_FREEZE_MAX_SELECTOR_DELTA = 0.03
CASCADE_MIN_STRUCTURE_CASES = 50
CASCADE_STAGE2_GRID_MAX_ROI_CELLS = 10_000_000
CASCADE_STAGE2_MS_REPEAT_ENABLED = False
CASCADE_STAGE2_MS_REPEAT_N2 = 8
CASCADE_STAGE2_MS_REPEAT_N3 = 24
CASCADE_STAGE2_MS_REPEAT_MAX = 3

TRAIN_MODE = "pinn"           # supported: pinn, dl
N_EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-4
MIN_EPOCH_FOR_BEST = 100
EARLY_STOPPING_PATIENCE = 80   # 0 disables early stopping
LATEST_CKPT_EVERY = 10
GRAD_CLIP_MAX_NORM = 1.0
SCHEDULER_MODE = "onecycle"   # supported: onecycle, none
ONECYCLE_PCT_START = 0.1
ONECYCLE_DIV_FACTOR = 25.0
ONECYCLE_FINAL_DIV_FACTOR = 1.0e3

GLOBAL_POINTS_PER_DOMAIN = 4096
ROI_POINTS_PER_DOMAIN = 4096
SCALER_POINTS_PER_GRID = 2048
PRED_BATCH_SIZE = 200000
MAX_PLOT_FLOW_POINTS = 20_000_000
GLOBAL_TERRAIN_TENSOR_CACHE_LIMIT = 8
ROI_TERRAIN_TENSOR_CACHE_LIMIT = 6
STRUCTURE_TENSOR_CACHE_LIMIT = 6

GLOBAL_PATCH_SHAPE = (16, 16, 16)
ROI_PATCH_SHAPE = (24, 24, 24)
GLOBAL_PATCHES_PER_DOMAIN = 1
ROI_PATCHES_PER_DOMAIN = 1
PATCH_NEAR_GROUND_PROB = 0.6
GLOBAL_SUPERVISED_NEAR_GROUND_FRAC = 0.5
GLOBAL_SUPERVISED_GROUND_K_FRAC = 0.3
ROI_PATCH_NEAR_WALL_PROB = 0.6
ROI_NEAR_WALL_DMAX = 5.0
FD_FLUID_MASK_THRESHOLD = 0.5
NU = 1.5e-5

ROI_SUPERVISED_SAMPLER_MODE = "random"  # random, targeted_v1
ROI_TARGET_VERY_NEAR_WALL_FRAC = 0.20
ROI_TARGET_NEAR_WALL_FRAC = 0.25
ROI_TARGET_GEOM_WAKE_FRAC = 0.25
ROI_TARGET_LOW_SPEED_FRAC = 0.20
ROI_TARGET_HIGH_SPEED_FRAC = 0.0
ROI_TARGET_RANDOM_FRAC = 0.10
ROI_TARGET_VERY_NEAR_WALL_DMAX = 1.0
ROI_TARGET_VERY_NEAR_WALL_MAX_REPEAT = 4
ROI_TARGET_NEAR_WALL_DMAX = 5.0
ROI_TARGET_NEAR_WALL_BACKFILL_DMAX = 10.0
ROI_TARGET_WAKE_MIN_CONTEXT = 0.15
ROI_TARGET_WAKE_ZREL_MAX = 30.0
ROI_TARGET_LOW_SPEED_RATIO_MAX = 0.75
ROI_TARGET_LOW_SPEED_ZREL_MAX = 30.0
ROI_TARGET_HIGH_SPEED_RATIO_MIN = 1.05
ROI_TARGET_HIGH_SPEED_ZREL_MAX = 30.0
ROI_TARGET_MAX_ABOVE_STRUCTURE_H = 2.0
ROI_PATCH_HIGH_SPEED_PROB = 0.0
ROI_PATCH_HIGH_SPEED_RATIO_MIN = 1.05
ROI_PATCH_HIGH_SPEED_ZREL_MAX = 30.0

BC_POINTS_INLET = 2048
BC_POINTS_OUTLET = 2048
BC_POINTS_SIDE = 512
BC_POINTS_TOP = 512

SUP_WEIGHT_NEAR_STRUCTURE_GAIN = 0.0
SUP_WEIGHT_NEAR_STRUCTURE_DMAX = 2.0
SUP_WEIGHT_WAKE_GAIN = 0.0
SUP_WEIGHT_WAKE_ZREL_MAX = 20.0
SUP_WEIGHT_WAKE_SPEED_RATIO_MAX = 0.7
SUP_WEIGHT_WAKE_POWER = 2.0

TRAIN_LOSS = "rmse"            # supported: rmse (legacy alias of mse), mse, charb, charb_weighted
TRAIN_STRUCT_MODE = "none"     # supported: none, grad, fft
TRAIN_STRUCT_WEIGHT = 0.0
MOMENTUM_LOSS_MODE = "constant"  # supported: constant, nut
CHARB_EPS = 1e-3
FFT_MIN_FLUID_FRAC = 0.5
DATA_P_WEIGHT = 1.0
GLOBAL_DATA_P_WEIGHT = None
ROI_DATA_P_WEIGHT = None
VAL_SELECTOR_P_WEIGHT = 0.3
VAL_SELECTOR_USE_GAUGE_P = False
VAL_SELECTOR_MS_ROI_UMAG_WEIGHT = 0.0
WEIGHTED_CHARB_LOW_START = 0.35
WEIGHTED_CHARB_HIGH_START = 0.95
WEIGHTED_CHARB_LOW_GAIN = 0.75
WEIGHTED_CHARB_HIGH_GAIN = 0.75
WEIGHTED_CHARB_POWER = 2.0

W_DATA_GLOBAL = 1.0
W_DATA_ROI = 1.0
W_PHYS_GLOBAL = 1.0
W_PHYS_ROI = 1.0
W_DIV_GLOBAL = 1.0
W_MOM_GLOBAL = 1.0
W_DIV_ROI = 1.0
W_MOM_ROI = 1.0
W_BC_INLET = 0.5
W_BC_OUTLET = 0.5
W_BC_SIDE = 0.2
W_BC_TOP = 0.1
W_BC_WALL_ROI = 0.1
ROI_WALL_BC_DMAX = 1.0
PHYS_RAMP_ENABLED = False
PHYS_RAMP_START_EPOCH = 25
PHYS_RAMP_END_EPOCH = 75
PHYS_RAMP_APPLY_GLOBAL = True
PHYS_RAMP_APPLY_ROI = True
PHYS_RAMP_APPLY_BC = False
HARD_GROUND_BC = False
PLOT_EVAL = True
EVAL_GLOBAL_PATCHES_PER_CASE = 4
EVAL_ROI_PATCHES_PER_CASE = 4
USE_AMP = False
AMP_DTYPE = "bf16"  # supported: bf16, fp16

# Off by default in the public release so a fresh clone trains without needing
# a Weights & Biases account. Enable it by setting this True and exporting
# WANDB_API_KEY, or pass --wandb-project on the command line.
WANDB_ENABLED = False
WANDB_PROJECT_PINN = "pinn_terr_struc"
WANDB_PROJECT_DL = "dl_terr_struc"

CANONICAL_CASE_MIN = 1
CANONICAL_CASE_MAX = 120

RECOMMENDED_SPLIT_JSON = SPLITS_ROOT / 'recommended_120domains.json'


def resolve_structure_channel_mode(mode: str) -> str:
    name = str(mode or "basic").strip().lower()
    if name == "inherit":
        name = str(STRUCTURE_ENCODER_INPUT_MODE or "basic").strip().lower()
    if name not in {"none", "basic", "context_v2", "context_v3"}:
        raise ValueError(f"Unsupported structure channel mode: {mode!r}")
    return name


def structure_channel_count(mode: str) -> int:
    name = resolve_structure_channel_mode(mode)
    if name == "none":
        return 0
    if name == "basic":
        return 2
    if name == "context_v2":
        return 8
    if name == "context_v3":
        return 12
    raise ValueError(f"Unsupported structure channel mode: {mode!r}")


def _finalize_pressure_weights() -> None:
    global GLOBAL_DATA_P_WEIGHT, ROI_DATA_P_WEIGHT
    legacy = float(DATA_P_WEIGHT)
    if GLOBAL_DATA_P_WEIGHT is None:
        GLOBAL_DATA_P_WEIGHT = legacy
    if ROI_DATA_P_WEIGHT is None:
        ROI_DATA_P_WEIGHT = legacy


def _finalize_encoder_depths() -> None:
    global GLOBAL_ENCODER_DEPTH, ROI_ENCODER_DEPTH, STRUCTURE_ENCODER_DEPTH
    legacy = int(ENCODER_DEPTH)
    if GLOBAL_ENCODER_DEPTH is None:
        GLOBAL_ENCODER_DEPTH = legacy
    if ROI_ENCODER_DEPTH is None:
        ROI_ENCODER_DEPTH = legacy
    if STRUCTURE_ENCODER_DEPTH is None:
        STRUCTURE_ENCODER_DEPTH = legacy

def _apply_override_namespace(ns: dict, *, label: str) -> None:
    applied = []
    for key, value in ns.items():
        if key.isupper():
            globals()[key] = value
            applied.append(key)
    if applied:
        print(f"Loaded {label}: {', '.join(sorted(applied))}")


_override_path = os.environ.get("PINN_CONFIG_OVERRIDE_PATH", "").strip()
if _override_path:
    override_file = Path(_override_path)
    if not override_file.is_absolute():
        override_file = (ROOT / override_file).resolve()
    if override_file.exists():
        _apply_override_namespace(runpy.run_path(str(override_file)), label=str(override_file))
    else:
        print(f"[WARN] PINN_CONFIG_OVERRIDE_PATH not found: {override_file}")
else:
    # Allow override from optional config_override.py
    try:
        from config_override import *  # type: ignore # noqa: F401,F403

        print("Loaded config_override.py")
    except ImportError:
        pass

_finalize_pressure_weights()
_finalize_encoder_depths()

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
