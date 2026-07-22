"""Run pre-trained models on one freshly-built predict_web domain.

Supports two model kinds, exposed through one `predict_all()` entry point so the
web app does not need to know which architecture is loaded:

  - "single":  one checkpoint (e.g. the no-physics baseline). Global + ROI both
               run through the same network — the legacy code path.
  - "cascade": a frozen Stage-1 conditioner (hybrid OR grid_unet) plus a
               Stage-2 ROI refiner. Globals come from Stage-1 alone; ROIs come
               from Stage-2 with a Stage-1 background sampled at the query
               points and a residual added on top.

Add a new .pth + one entry in MODEL_REGISTRY and it shows up in the dropdown.
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Model registry
#
# Each entry has a "kind" field:
#   - "single":  use `file` + `model_kind` + `arch`. One checkpoint, legacy path.
#   - "cascade": use `stage1` and `stage2` sub-entries. Stage-1 is the frozen
#                conditioner (hybrid or grid_unet); Stage-2 is always a
#                CascadeStage2Refiner. Stage-2 checkpoints from recent runs
#                already carry an embedded `config_snapshot`, so no `arch`
#                override is needed for them. Stage-1 checkpoints predating
#                that mechanism use the `arch` dict to pin the architecture.
# ---------------------------------------------------------------------------
def _resolve_ckpt_path(file_str: str) -> Path:
    """Resolve a registry `file` value to an actual checkpoint path.

    Plain filenames live in predict_web/checkpoints/. Entries containing a
    path separator are resolved relative to the repo root, so the registry can
    reference training checkpoints under results/ without copying ~250 MB.
    """
    p = Path(str(file_str))
    if p.is_absolute():
        return p
    if "/" in str(file_str) or "\\" in str(file_str):
        return SCRIPTS_DIR.parent / p
    return CHECKPOINT_DIR / p


# The two final 292-domain physics-informed cascades: the models used in the
# paper and served by the web app. Each `file` is a plain filename resolved to
# webapp/checkpoints/ (fetched from the model release; see checkpoints/README).
# The hybrid is the default: it is fast on CPU, which suits a hosted deployment.
_ALL_MODELS = [
    {
        "id": "hybrid-cascade-292d-pinn",
        "label": "Hybrid (cascade, 292d, physics)",
        "tier": "balanced",
        "description": (
            "Final hybrid cascade trained on the full 292-domain split with "
            "y-reflection augmentation and physics losses. Best aggregate accuracy and fastest inference; "
            "smoother between closely spaced grid structures than the 3D UNet."
        ),
        "kind": "cascade",
        "stage1": {
            "file": "hybrid-stage1-292d.pth",
            "model_kind": "hybrid",
            "arch": {
                "GLOBAL_ENCODER_DEPTH": 5,
                "GLOBAL_ENCODER_DILATIONS": [1, 2, 4, 8, 16, 32],
                "STRUCTURE_ENCODER_INPUT_MODE": "basic",
            },
        },
        "stage2": {"file": "hybrid-stage2-292d.pth"},
        "default": True,
    },
    {
        "id": "grid-unet-cascade-292d-pinn",
        "label": "3D UNet (cascade, 292d, physics)",
        "tier": "accurate",
        "description": (
            "Final grid U-Net cascade trained on the full 292-domain split with "
            "y-reflection augmentation and physics losses. Better at inter-structure high-speed corridors, but "
            "slow on CPU (tiled 3D inference; wants a GPU)."
        ),
        "kind": "cascade",
        "stage1": {
            "file": "grid-unet-stage1-292d.pth",
            "model_kind": "grid_unet",
            "arch": {
                "GRID_UNET_BASE_WIDTH": 32,
                "GRID_UNET_LEVELS": 4,
                "GRID_UNET_DROPOUT": 0.0,
                "GRID_UNET_ROI_STRUCTURE_MODE": "context_v2",
                "GRID_UNET_USE_TERRAIN_CONTEXT": True,
                "GRID_UNET_TERRAIN_CONTEXT_WIDTH": 32,
                "GRID_UNET_TERRAIN_CONTEXT_DEPTH": 5,
                "GRID_UNET_TERRAIN_CONTEXT_DILATIONS": [1, 2, 4, 8, 16, 32],
                "GLOBAL_PATCH_SHAPE": [32, 32, 16],
                "STRUCTURE_ENCODER_INPUT_MODE": "context_v2",
            },
        },
        "stage2": {"file": "grid-unet-stage2-292d.pth"},
        "default": False,
    },
]


def _entry_available(m: dict) -> bool:
    """An entry is offered only when all of its checkpoint files exist."""
    try:
        if m.get("kind") == "cascade":
            return (
                _resolve_ckpt_path(m["stage1"]["file"]).exists()
                and _resolve_ckpt_path(m["stage2"]["file"]).exists()
            )
        return _resolve_ckpt_path(m["file"]).exists()
    except Exception:
        return False


def _model_selection() -> tuple:
    """Optional per-deployment override of which models are offered.

    Precedence: the PINN_WEBAPP_MODELS env var (comma-separated ids) wins;
    otherwise webapp/models.yaml (`enabled: [...]`, optional `default: id`).
    Returns (enabled_ids or None, default_id or None); None means "offer all".
    A CPU-only host can serve just the fast hybrid without touching code.
    """
    env = os.environ.get("PINN_WEBAPP_MODELS", "").strip()
    if env:
        return [s.strip() for s in env.split(",") if s.strip()], None
    cfg = Path(__file__).resolve().parent / "models.yaml"
    if cfg.exists():
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(cfg.read_text()) or {}
            enabled = data.get("enabled")
            enabled = list(enabled) if enabled else None
            return enabled, data.get("default")
        except Exception:
            pass
    return None, None


_AVAILABLE = [m for m in _ALL_MODELS if _entry_available(m)]
_ENABLED_IDS, _DEFAULT_OVERRIDE = _model_selection()
if _ENABLED_IDS is not None:
    _selected = [m for m in _AVAILABLE if m["id"] in _ENABLED_IDS]
    MODEL_REGISTRY = _selected or _AVAILABLE  # bad filter -> fall back to all
else:
    MODEL_REGISTRY = _AVAILABLE
if not MODEL_REGISTRY:  # pragma: no cover - mis-installed checkpoints
    raise RuntimeError(
        "No model checkpoints found. Expected the four .pth files under "
        "webapp/checkpoints/ (run fetch_checkpoints.py)."
    )

DEFAULT_MODEL_ID = next(
    (m["id"] for m in MODEL_REGISTRY if m["id"] == _DEFAULT_OVERRIDE),
    next(
        (m["id"] for m in MODEL_REGISTRY if m.get("default")),
        MODEL_REGISTRY[0]["id"],
    ),
)


def list_models() -> list:
    """Public, UI-facing view of the registry (no internal arch details)."""
    return [
        {
            "id": m["id"],
            "label": m["label"],
            "tier": m.get("tier", ""),
            "description": m.get("description", ""),
            "default": bool(m.get("default")),
        }
        for m in MODEL_REGISTRY
    ]


def get_model_entry(model_id: Optional[str]) -> dict:
    """Resolve a model id to its registry entry, falling back to the default."""
    if model_id:
        for m in MODEL_REGISTRY:
            if m["id"] == model_id:
                return m
    for m in MODEL_REGISTRY:
        if m["id"] == DEFAULT_MODEL_ID:
            return m
    return MODEL_REGISTRY[0]


# Backward-compat: external callers / _model_name() still import this symbol.
def _default_ckpt_path() -> Path:
    entry = get_model_entry(None)
    if entry.get("kind") == "cascade":
        return _resolve_ckpt_path(entry["stage2"]["file"])
    return _resolve_ckpt_path(entry["file"])


DEFAULT_CHECKPOINT = _default_ckpt_path()


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _torch_load(checkpoint_path: Path, device: str) -> dict:
    with torch.serialization.safe_globals([]):
        try:
            return torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        except TypeError:
            return torch.load(str(checkpoint_path), map_location=device)


def _build_with_snapshot(snapshot: Optional[dict], device: str, model_kind: str):
    """Build a network with the right architecture, restoring config afterwards.

    Uses `model_runtime.applied_config_snapshot`: temporarily applies the
    uppercase keys in `snapshot` to the `config` module, rebuilds the model,
    then restores baseline. Safe to call multiple times in one process.
    """
    from model_runtime import applied_config_snapshot  # type: ignore
    from models import create_model  # type: ignore

    with applied_config_snapshot(snapshot):
        model = create_model(device=device, model_kind=model_kind)
    return model


def _resolve_snapshot(ckpt: dict, arch: Optional[dict]) -> dict:
    """Pick the snapshot dict the model was trained with, falling back to arch.

    Recent training runs embed `config_snapshot` in the checkpoint (either at
    the top level or inside `train_config`). For older checkpoints we synthesise
    a snapshot from the registry's `arch` override.
    """
    if isinstance(ckpt, dict):
        snap = ckpt.get("config_snapshot")
        if isinstance(snap, dict) and snap:
            return dict(snap)
        tc = ckpt.get("train_config")
        if isinstance(tc, dict):
            snap = tc.get("config_snapshot")
            if isinstance(snap, dict) and snap:
                return dict(snap)
    return dict(arch or {})


@dataclass
class _Stage:
    model: torch.nn.Module
    scalers: object
    ckpt: dict
    model_kind: str
    snapshot: dict
    file: str


@dataclass
class ModelStack:
    """One loaded model (single) or a Stage-1 + Stage-2 pair (cascade)."""

    entry_id: str
    kind: str  # "single" or "cascade"
    primary: _Stage
    conditioner: Optional[_Stage] = None  # set for cascades; Stage-1
    grid_global_cache: OrderedDict = field(default_factory=OrderedDict)

    @property
    def model(self) -> torch.nn.Module:
        return self.primary.model

    @property
    def scalers(self):
        return self.primary.scalers

    @property
    def ckpt(self) -> dict:
        return self.primary.ckpt


def _load_stage(spec: dict, device: str) -> _Stage:
    """Load one .pth, rebuild model with the right arch, return everything."""
    ckpt_path = _resolve_ckpt_path(spec["file"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = _torch_load(ckpt_path, device)
    snapshot = _resolve_snapshot(ckpt, spec.get("arch"))
    model_kind = spec.get("model_kind") or (
        ckpt.get("train_config", {}).get("model_kind") if isinstance(ckpt, dict) else "hybrid"
    ) or "hybrid"
    if model_kind == "cascade_stage2":
        # Stage-2 refiner: the model_kind in config is "cascade".
        model_kind = "cascade"
    model = _build_with_snapshot(snapshot, device=device, model_kind=model_kind)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return _Stage(
        model=model,
        scalers=ckpt["scalers"],
        ckpt=ckpt,
        model_kind=model_kind,
        snapshot=snapshot,
        file=spec["file"],
    )


# Process-level cache: loading a cascade re-reads ~70-200 MB from disk and
# rebuilds the nets; with the cache, repeated /predict calls (and every wind
# rose sector) reuse the already-loaded stack. Bounded to 2 stacks so toggling
# between the two final families stays warm without hoarding RAM.
_STACK_CACHE: "OrderedDict[tuple, ModelStack]" = OrderedDict()
_STACK_CACHE_MAX = 2
_STACK_LOCK = None  # created lazily to keep import light


def _stack_lock():
    global _STACK_LOCK
    if _STACK_LOCK is None:
        import threading
        _STACK_LOCK = threading.Lock()
    return _STACK_LOCK


def load_model_stack(model_id: Optional[str] = None, device: str = "cpu") -> ModelStack:
    """Build (or fetch from cache) the requested model and load weights + scalers."""
    entry = get_model_entry(model_id)
    key = (entry["id"], str(device))
    with _stack_lock():
        cached = _STACK_CACHE.get(key)
        if cached is not None:
            _STACK_CACHE.move_to_end(key)
            # Bound the per-stack global-volume cache (one full-domain float32
            # volume per predicted domain) so long sessions don't hoard RAM.
            while len(cached.grid_global_cache) > 4:
                cached.grid_global_cache.popitem(last=False)
            return cached
        stack = _build_model_stack(entry, device)
        _STACK_CACHE[key] = stack
        while len(_STACK_CACHE) > _STACK_CACHE_MAX:
            _STACK_CACHE.popitem(last=False)
        return stack


def _build_model_stack(entry: dict, device: str) -> ModelStack:
    if entry.get("kind") == "cascade":
        # Stage-1 first (the conditioner). Build with its arch — it may predate
        # the embedded-snapshot mechanism, in which case `arch` is the only
        # source of the depth-5 / grid-UNet knobs.
        stage1 = _load_stage(entry["stage1"], device)
        # Stage-2 next. Its checkpoint embeds a full snapshot, so `arch` is
        # not needed; passing a default empty arch is fine.
        stage2 = _load_stage(entry["stage2"], device)
        # Stage-2 is the refiner that is queried for ROI prediction; Stage-1
        # is the frozen conditioner.
        return ModelStack(
            entry_id=entry["id"],
            kind="cascade",
            primary=stage2,
            conditioner=stage1,
        )
    # Single-stage
    stage = _load_stage(entry, device)
    return ModelStack(entry_id=entry["id"], kind="single", primary=stage)


# Backward-compat shim — older callers still expect (model, scalers, ckpt).
def load_model_and_scalers(
    model_id: Optional[str] = None,
    device: str = "cpu",
    *,
    checkpoint_path: Optional[Path] = None,
):
    """Legacy interface. For cascades returns the Stage-2 refiner triple."""
    if checkpoint_path is not None:
        # Explicit path: assume single-stage, use legacy `arch` lookup by filename.
        entry = next(
            (m for m in MODEL_REGISTRY if (m.get("file") == checkpoint_path.name)),
            None,
        )
        if entry is None or entry.get("kind") != "single":
            # Best-effort default arch
            entry = {"file": checkpoint_path.name, "kind": "single", "model_kind": "hybrid", "arch": {}}
        ckpt = _torch_load(checkpoint_path, device)
        snapshot = _resolve_snapshot(ckpt, entry.get("arch"))
        model = _build_with_snapshot(snapshot, device=device, model_kind=entry.get("model_kind", "hybrid"))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt["scalers"], ckpt
    stack = load_model_stack(model_id, device=device)
    return stack.primary.model, stack.primary.scalers, stack.primary.ckpt


# ---------------------------------------------------------------------------
# Inference: shared helpers
# ---------------------------------------------------------------------------
def _resolve_device(device: Optional[str]) -> str:
    requested = str(device or os.environ.get("PINN_DEVICE", "auto")).strip().lower()
    if requested in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"PINN_DEVICE={requested!r}, but PyTorch cannot access CUDA. "
            "Check that the container has a GPU and a CUDA-enabled PyTorch wheel."
        )
    return requested


def runtime_device_info() -> dict:
    """Small JSON-safe runtime summary used by health checks and deployment tests."""
    resolved = _resolve_device(None)
    info = {
        "requested": os.environ.get("PINN_DEVICE", "auto"),
        "resolved": resolved,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch": str(torch.__version__),
    }
    if resolved.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(torch.device(resolved))
        info["cuda_runtime"] = str(torch.version.cuda)
    return info


def _global_bundle(cfd_dir: Path):
    from data_loader import _load_grid_bundle  # type: ignore
    return _load_grid_bundle(Path(cfd_dir), kind="global", roi_name=None, parent_name=None)


def _roi_bundles(cfd_dir: Path) -> list:
    """Discover ROI subdirs through the global meta.json's `roi_paths` list."""
    from data_loader import _load_grid_bundle  # type: ignore
    gmeta = json.loads((Path(cfd_dir) / "meta.json").read_text())
    rel_paths = gmeta.get("roi_paths") or []
    out = []
    for rel in rel_paths:
        roi_dir = Path(cfd_dir) / rel
        if (roi_dir / "meta.json").exists() and (roi_dir / "phi_wall.npy").exists():
            out.append((rel, _load_grid_bundle(roi_dir, kind="roi", roi_name=roi_dir.name, parent_name=Path(cfd_dir).name)))
    return out


def _widen_roi_fluid_mask(roi_bundle) -> np.ndarray:
    """Viz-grade fluid mask: every cell above terrain and outside structures.

    The empty-export pipeline shrinks `is_fluid` to a small subset of the ROI;
    we want the model queried over the full ROI for plots and 3D views.
    """
    meta = roi_bundle.meta if isinstance(roi_bundle.meta, dict) else {}
    bounds = meta.get("bounds") if isinstance(meta, dict) else None
    isf = np.asarray(roi_bundle.is_fluid)
    if not bounds or len(bounds) < 6:
        return isf > 0.5
    nx, ny, nz = isf.shape
    x = np.linspace(float(bounds[0]), float(bounds[1]), nx, dtype=np.float32)
    y = np.linspace(float(bounds[2]), float(bounds[3]), ny, dtype=np.float32)
    z_levels = meta.get("z_levels")
    if z_levels is not None and len(z_levels) == nz:
        z = np.asarray(z_levels, dtype=np.float32)
    else:
        z = np.linspace(float(bounds[4]), float(bounds[5]), nz, dtype=np.float32)
    elev = np.asarray(roi_bundle.terrain_raw.get("elevation"), dtype=np.float32)
    if elev.shape == (ny, nx):
        elev_ij = elev.T
    elif elev.shape == (nx, ny):
        elev_ij = elev
    else:
        return isf > 0.5
    above = z[None, None, :] > (elev_ij[:, :, None] + 1e-6)
    solid = np.zeros_like(above)
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
    finite_flow = np.isfinite(np.asarray(roi_bundle.flow)).all(axis=-1)
    return (above & ~solid & finite_flow)


# ---------------------------------------------------------------------------
# Inference: single-stage hybrid path (legacy)
# ---------------------------------------------------------------------------
def _run_global_hybrid(model, bundle, scalers, device: str, pred_batch_size: int) -> np.ndarray:
    from config import GLOBAL_INPUT_COLS  # type: ignore
    from data_loader import iter_fullgrid_predictions, terrain_tensor  # type: ignore
    from losses import apply_output_constraints_from_scaled_inputs  # type: ignore

    terrain = terrain_tensor(bundle).to(device)
    pred_flow = np.full(bundle.flow.shape, np.nan, dtype=np.float32)
    with torch.no_grad():
        for idx, x_scaled, _y, xy_local, _zr, _pw in iter_fullgrid_predictions(
            bundle, x_scaler=scalers.x_scaler_global, chunk_size=int(pred_batch_size), include_phi_wall=False,
        ):
            x_dev = x_scaled.to(device)
            raw = model.forward_global(terrain, x_dev, xy_local.to(device))
            _, pred_phys = apply_output_constraints_from_scaled_inputs(
                raw,
                x_batch=x_dev,
                x_scaler=scalers.x_scaler_global,
                input_cols=GLOBAL_INPUT_COLS,
                y_scaler=scalers.y_scaler,
                hard_ground_bc=False,
            )
            pred_flow.reshape(-1, pred_flow.shape[-1])[idx] = pred_phys.cpu().numpy()
    return pred_flow


def _run_roi_hybrid(model, global_bundle, roi_bundle, scalers, device: str, pred_batch_size: int) -> np.ndarray:
    """Single-stage hybrid ROI inference. Mirrors training._evaluate_case_roi."""
    from config import ROI_INPUT_COLS  # type: ignore
    from data_loader import iter_fullgrid_predictions, terrain_tensor  # type: ignore
    from losses import apply_output_constraints_from_scaled_inputs  # type: ignore

    gterr = terrain_tensor(global_bundle).to(device)
    rterr = terrain_tensor(roi_bundle).to(device)
    x_idx = ROI_INPUT_COLS.index("x")
    y_idx = ROI_INPUT_COLS.index("y")
    x_min = float(scalers.x_scaler_roi.data_min_[x_idx])
    x_max = float(scalers.x_scaler_roi.data_max_[x_idx])
    y_min = float(scalers.x_scaler_roi.data_min_[y_idx])
    y_max = float(scalers.x_scaler_roi.data_max_[y_idx])
    gx0, gx1, gy0, gy1, _, _ = global_bundle.bounds

    pred_flow = np.full(roi_bundle.flow.shape, np.nan, dtype=np.float32)
    isf_orig = roi_bundle.is_fluid
    widened = _widen_roi_fluid_mask(roi_bundle)
    roi_bundle.is_fluid = widened.astype(np.float32)
    try:
      with torch.no_grad():
        for idx, x_scaled, _y, xy_local, _zr, _pw in iter_fullgrid_predictions(
            roi_bundle, x_scaler=scalers.x_scaler_roi, chunk_size=int(pred_batch_size), include_phi_wall=True,
        ):
            x_dev = x_scaled.to(device)
            x_phys = x_scaled[:, x_idx].numpy() * (x_max - x_min) + x_min
            y_phys = x_scaled[:, y_idx].numpy() * (y_max - y_min) + y_min
            xy_global = torch.as_tensor(
                np.stack(
                    [
                        2.0 * ((x_phys - gx0) / max(gx1 - gx0, 1e-6)) - 1.0,
                        2.0 * ((y_phys - gy0) / max(gy1 - gy0, 1e-6)) - 1.0,
                    ],
                    axis=1,
                ),
                dtype=torch.float32,
            )
            raw = model.forward_roi(gterr, rterr, x_dev, xy_global.to(device), xy_local.to(device))
            _, pred_phys = apply_output_constraints_from_scaled_inputs(
                raw,
                x_batch=x_dev,
                x_scaler=scalers.x_scaler_roi,
                input_cols=ROI_INPUT_COLS,
                y_scaler=scalers.y_scaler,
                hard_ground_bc=False,
            )
            pred_flow.reshape(-1, pred_flow.shape[-1])[idx] = pred_phys.cpu().numpy()
    finally:
        roi_bundle.is_fluid = isf_orig
    return pred_flow


# ---------------------------------------------------------------------------
# Inference: cascade path (Stage-1 + Stage-2)
#
# Reuses the training-time helpers: same code path the model was selected on.
# ---------------------------------------------------------------------------
def _make_conditioner(stack: ModelStack):
    """Wrap Stage-1 as a CascadeConditioner that training helpers understand."""
    from training import CascadeConditioner  # type: ignore

    s1 = stack.conditioner
    if s1 is None:
        raise RuntimeError("cascade inference requires a Stage-1 conditioner")
    return CascadeConditioner(
        model=s1.model,
        scalers=s1.scalers,
        checkpoint_path=s1.file,
        config_snapshot=s1.snapshot,
        grid_global_cache=stack.grid_global_cache,
    )


def _run_global_cascade(stack: ModelStack, bundle, device: str, pred_batch_size: int) -> np.ndarray:
    """Global field comes from Stage-1 alone — Stage-2 is ROI-only."""
    s1 = stack.conditioner
    if s1 is None:
        raise RuntimeError("cascade inference requires a Stage-1 conditioner")
    if s1.model_kind == "grid_unet":
        from data_loader import terrain_tensor  # type: ignore
        from training import _predict_global_grid_unet_volume  # type: ignore

        terr = terrain_tensor(bundle).to(device)
        from model_runtime import applied_config_snapshot  # type: ignore
        with applied_config_snapshot(s1.snapshot):
            patch_shape = tuple(int(v) for v in s1.snapshot.get("GLOBAL_PATCH_SHAPE", [32, 32, 16]))
            pred_flow = _predict_global_grid_unet_volume(
                s1.model,
                bundle,
                s1.scalers,
                device=device,
                hard_ground_bc=False,
                patch_shape=patch_shape,
                terr=terr,
            )
        # Mask cells outside fluid: keep NaN for solid/below-terrain.
        isf = np.asarray(bundle.is_fluid) > 0.5
        pred_flow[~isf] = np.nan
        return pred_flow
    # Hybrid Stage-1: same as the legacy single-stage global path.
    return _run_global_hybrid(s1.model, bundle, s1.scalers, device, pred_batch_size)


def _run_roi_cascade(stack: ModelStack, global_bundle, roi_bundle, device: str, pred_batch_size: int) -> np.ndarray:
    """Stage-2 refiner with the Stage-1 background sampled at each ROI point.

    Dispatches to the grid-refiner tile path when Stage-2 is the local 3D
    UNet ROI refiner (`cascade_stage2_refiner_kind == "grid_unet"`).
    """
    stage2 = stack.primary
    s2_model = stage2.model
    if bool(getattr(s2_model, "uses_cascade_grid_refiner", False)):
        return _run_roi_cascade_grid(stack, global_bundle, roi_bundle, device)

    from config import ROI_INPUT_COLS  # type: ignore
    from data_loader import iter_fullgrid_predictions, terrain_tensor  # type: ignore
    from losses import apply_output_constraints_from_scaled_inputs  # type: ignore
    from model_runtime import applied_config_snapshot  # type: ignore
    from training import (  # type: ignore
        _cascade_conditioner_predict_on_roi_inputs,
    )

    s2_scalers = stage2.scalers
    conditioner = _make_conditioner(stack)

    gterr = terrain_tensor(global_bundle).to(device)
    rterr = terrain_tensor(roi_bundle).to(device)
    sterr = _structure_tensor(roi_bundle, stage2.snapshot, device=device)

    pred_flow = np.full(roi_bundle.flow.shape, np.nan, dtype=np.float32)
    isf_orig = roi_bundle.is_fluid
    widened = _widen_roi_fluid_mask(roi_bundle)
    roi_bundle.is_fluid = widened.astype(np.float32)
    try:
        with applied_config_snapshot(stage2.snapshot):
            with torch.no_grad():
                # Stage-2 expects its own encoder forward built with the Stage-2 snapshot.
                rfeat = s2_model.encode_roi(rterr)
                sfeat = s2_model.encode_structure(sterr) if sterr is not None else None
                for idx, x_scaled, _y, xy_local, _zr, _pw in iter_fullgrid_predictions(
                    roi_bundle, x_scaler=s2_scalers.x_scaler_roi,
                    chunk_size=int(pred_batch_size), include_phi_wall=True,
                ):
                    x_dev = x_scaled.to(device)
                    xy_local_dev = xy_local.to(device)
                    # Stage-1 background sampled at these ROI points, rescaled to
                    # Stage-2's y-scaler. Uses the cache on `conditioner`.
                    bg_scaled, _bg_phys, _gt, _gfeat = _cascade_conditioner_predict_on_roi_inputs(
                        conditioner,
                        global_bundle,
                        x_scaled_roi=x_dev,
                        x_scaler_roi=s2_scalers.x_scaler_roi,
                        target_y_scaler=s2_scalers.y_scaler,
                        device=device,
                        hard_ground_bc=False,
                        gterr=gterr,
                    )
                    raw_resid_scaled = s2_model.forward_roi_from_encoded(
                        rfeat,
                        x_dev,
                        xy_local_dev,
                        bg_scaled,
                        s_feat=sfeat,
                    )
                    pred_scaled, pred_phys = apply_output_constraints_from_scaled_inputs(
                        raw_resid_scaled + bg_scaled,
                        x_batch=x_dev,
                        x_scaler=s2_scalers.x_scaler_roi,
                        input_cols=ROI_INPUT_COLS,
                        y_scaler=s2_scalers.y_scaler,
                        hard_ground_bc=False,
                    )
                    pred_flow.reshape(-1, pred_flow.shape[-1])[idx] = pred_phys.cpu().numpy()
    finally:
        roi_bundle.is_fluid = isf_orig
    return pred_flow


def _roi_grid_patch_batch_size() -> int:
    """How many same-shape ROI patches to push through the Stage-2 3D UNet in one
    forward. Overridable via env for tuning per GPU; defaults to 16. Falls back
    automatically on CUDA OOM (see `_run_roi_cascade_grid`)."""
    import os

    try:
        v = int(os.environ.get("PREDICT_WEB_ROI_PATCH_BATCH", "16"))
    except (TypeError, ValueError):
        v = 16
    return max(1, v)


def _run_roi_cascade_grid(stack: ModelStack, global_bundle, roi_bundle, device: str) -> np.ndarray:
    """Grid Stage-2 refiner path: tile the ROI volume, run a 3D UNet refiner on
    each patch, blend with a Hanning window.

    Mirrors `_evaluate_case_roi_cascade_grid` from training.py without the metric
    accumulation, but processes the tiles in *batches* of identical-shape patches
    so the Stage-2 3D UNet runs on the GPU at high utilisation instead of one
    small forward at a time. The frozen Stage-1 background is sampled once per
    batch (one trilinear-sampling call over all the batch's points) rather than
    once per tile.

    The result is numerically identical to the per-tile loop: the refiner is a
    3D CNN evaluated in ``eval`` mode (batch-invariant), the background sampling
    is point-wise, and the overlap/Hanning blend are unchanged. Only the *number*
    of Python/GPU round-trips changes.
    """
    from model_runtime import applied_config_snapshot  # type: ignore
    import data_loader as _dl  # type: ignore  # live module: snapshot re-syncs its globals
    from data_loader import (  # type: ignore
        resolve_structure_channel_mode,
        _structure_model_channels_for_mode,
        _feature_rows,
        _scale_inputs,
    )
    from training import (  # type: ignore
        _cascade_conditioner_predict_on_roi_inputs,
        _iter_case_tiles,
        _blend_window_3d,
    )
    from losses import apply_output_constraints_from_scaled_inputs  # type: ignore
    from config import ROI_INPUT_COLS  # type: ignore

    stage2 = stack.primary
    s2_model = stage2.model
    s2_scalers = stage2.scalers
    conditioner = _make_conditioner(stack)
    snapshot = stage2.snapshot

    patch_shape = tuple(int(v) for v in snapshot.get("ROI_PATCH_SHAPE", [32, 32, 16]))
    pred_acc = np.zeros(roi_bundle.flow.shape, dtype=np.float32)
    weight_acc = np.zeros(roi_bundle.flow.shape[:3], dtype=np.float32)

    # Conditioner-background state threaded across batches (global terrain tensor;
    # the global volume itself is cached by name inside the conditioner).
    bgterr_box = [None]

    # Populated once inside the config-snapshot context (below): the full-ROI
    # scaled-input volume and structure channels, sliced per patch.
    vol_box = {"x": None, "struct": None}

    def _precompute_roi_volume():
        """Build the full-ROI scaled-input volume and structure channels ONCE.

        `extract_patch_batch` is a per-cell function, so computing the inputs for
        the whole ROI grid here and slicing per patch is byte-identical to calling
        it per tile -- but it avoids recomputing the full-ROI structure channels
        (and re-deriving features / unused targets) on every one of the hundreds
        of overlapping tiles, which dominated runtime. Must run inside the
        Stage-2 config snapshot so the structure-channel mode matches the model.
        """
        nx, ny, nz = roi_bundle.flow.shape[:3]
        ii, jj, kk = np.meshgrid(
            np.arange(nx, dtype=np.int64),
            np.arange(ny, dtype=np.int64),
            np.arange(nz, dtype=np.int64),
            indexing="ij",
        )
        x_rows = _feature_rows(
            roi_bundle, ii.reshape(-1), jj.reshape(-1), kk.reshape(-1), include_phi_wall=True
        )
        vol_box["x"] = _scale_inputs(x_rows, scaler=s2_scalers.x_scaler_roi).view(nx, ny, nz, -1)

        structure_mode = resolve_structure_channel_mode(_dl.GRID_UNET_ROI_STRUCTURE_MODE)
        if structure_mode != "none":
            sm = _structure_model_channels_for_mode(roi_bundle, structure_mode)
            if sm.shape[0] > 0:
                vol_box["struct"] = sm

    def _build_patch_volume(i0, j0, k0, shape):
        """Slice the precomputed volume to reproduce extract_patch_batch outputs.

        Returns (x_scaled_flat, x_volume_scaled) identical to the per-tile path.
        """
        px, py, pz = shape
        x_vol_full = vol_box["x"]
        struct_model = vol_box["struct"]
        sub = x_vol_full[i0:i0 + px, j0:j0 + py, k0:k0 + pz, :]
        x_scaled = sub.reshape(-1, sub.shape[-1])
        x_volume = sub.permute(3, 0, 1, 2).contiguous()
        if struct_model is not None:
            struct_patch = np.transpose(struct_model[:, j0:j0 + py, i0:i0 + px], (0, 2, 1))
            struct_volume = np.repeat(struct_patch[:, :, :, None], pz, axis=3)
            x_volume = torch.cat(
                [x_volume, torch.as_tensor(struct_volume, dtype=torch.float32)], dim=0
            )
        return x_scaled, x_volume

    def _run_batch(tiles_batch):
        """Predict a list of same-shape tiles in a single UNet forward.

        Returns a list of (i0, j0, k0, shape, pred_patch_np). Splits and retries
        on CUDA OOM so a too-large batch never aborts the whole prediction.
        """
        if not tiles_batch:
            return []
        shape = tiles_batch[0][3]
        sx, sy, sz = shape
        n_pts = sx * sy * sz

        built = [_build_patch_volume(i0, j0, k0, shape) for (i0, j0, k0, _shape) in tiles_batch]
        k = len(built)

        x_cat = torch.cat([b[0] for b in built], dim=0)
        # Stage-1 background sampled for every point in the batch at once.
        bg_scaled_flat, _bg_phys, gterr, _gfeat = _cascade_conditioner_predict_on_roi_inputs(
            conditioner,
            global_bundle,
            x_scaled_roi=x_cat.to(device),
            x_scaler_roi=s2_scalers.x_scaler_roi,
            target_y_scaler=s2_scalers.y_scaler,
            device=device,
            hard_ground_bc=False,
            gterr=bgterr_box[0],
        )
        bgterr_box[0] = gterr

        c_out = bg_scaled_flat.shape[-1]
        bg_volume = (
            bg_scaled_flat.view(k, sx, sy, sz, c_out).permute(0, 4, 1, 2, 3).contiguous()
        )
        x_volume = torch.stack([b[1] for b in built], dim=0).to(device)

        try:
            raw_resid_volume = s2_model.forward_roi_grid(x_volume, bg_volume)
        except RuntimeError as exc:  # pragma: no cover - depends on GPU memory
            if "out of memory" in str(exc).lower() and k > 1:
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                mid = k // 2
                return _run_batch(tiles_batch[:mid]) + _run_batch(tiles_batch[mid:])
            raise

        raw_resid_flat = raw_resid_volume.permute(0, 2, 3, 4, 1).reshape(k * n_pts, c_out)
        _pred_scaled, pred_phys = apply_output_constraints_from_scaled_inputs(
            raw_resid_flat + bg_scaled_flat,
            x_batch=x_cat.to(device),
            x_scaler=s2_scalers.x_scaler_roi,
            input_cols=ROI_INPUT_COLS,
            y_scaler=s2_scalers.y_scaler,
            hard_ground_bc=False,
        )
        pred_np = pred_phys.detach().cpu().numpy().reshape(k, sx, sy, sz, -1)
        return [
            (i0, j0, k0, shape, pred_np[idx])
            for idx, (i0, j0, k0, _shape) in enumerate(tiles_batch)
        ]

    isf_orig = roi_bundle.is_fluid
    widened = _widen_roi_fluid_mask(roi_bundle)
    roi_bundle.is_fluid = widened.astype(np.float32)
    try:
        with applied_config_snapshot(snapshot):
            with torch.no_grad():
                # Build the full-ROI input volume + structure channels once,
                # inside the snapshot so the structure mode matches the model.
                _precompute_roi_volume()
                # Group tiles by shape (only edge tiles differ) so each forward
                # stacks identically-sized patches.
                tiles_by_shape: dict = {}
                for i0, j0, k0, shape in _iter_case_tiles(roi_bundle, patch_shape, overlap_fraction=0.5):
                    tiles_by_shape.setdefault(shape, []).append((i0, j0, k0, shape))

                max_batch = _roi_grid_patch_batch_size()
                for shape, tiles in tiles_by_shape.items():
                    for start in range(0, len(tiles), max_batch):
                        for i0, j0, k0, shp, pred_patch in _run_batch(tiles[start:start + max_batch]):
                            sx, sy, sz = shp
                            blend = _blend_window_3d(shp)
                            pred_acc[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz, :] += pred_patch * blend[..., None]
                            weight_acc[i0:i0 + sx, j0:j0 + sy, k0:k0 + sz] += blend
    finally:
        roi_bundle.is_fluid = isf_orig

    pred_flow = pred_acc / np.maximum(weight_acc[..., None], 1.0e-6)
    pred_flow[weight_acc <= 0.0] = np.nan
    # Mask cells outside fluid using the WIDENED mask, matching the
    # point-MLP cascade path (where iter_fullgrid_predictions fills only
    # widened-fluid cells). Using the original strict mask here would
    # over-NaN cells in dense multistructure grids and break 2D contours,
    # glyphs, and streamline seeding.
    isf = np.asarray(widened) > 0.5
    pred_flow[~isf] = np.nan
    return pred_flow


def _structure_tensor(roi_bundle, snapshot: dict, *, device: str) -> Optional[torch.Tensor]:
    """Encode the structure context channels expected by the Stage-2 refiner.

    The structure encoder mode (basic / context_v2 / context_v3) is determined
    by the Stage-2 snapshot. We have to apply the snapshot when calling
    `_structure_model_channels` because it reads STRUCTURE_ENCODER_INPUT_MODE
    at module level.
    """
    from model_runtime import applied_config_snapshot  # type: ignore
    from data_loader import _structure_model_channels_for_mode  # type: ignore

    mode = str(snapshot.get("STRUCTURE_ENCODER_INPUT_MODE", "basic")).lower()
    if mode == "none":
        return None
    with applied_config_snapshot(snapshot):
        ch = _structure_model_channels_for_mode(roi_bundle, mode)
    if ch.size == 0:
        return None
    # Channels-first batch of shape (1, C, ny, nx) on device.
    t = torch.as_tensor(ch, dtype=torch.float32, device=device).unsqueeze(0)
    return t


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def predict_global(
    cfd_dir: Path,
    *,
    model_id: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None,
    pred_batch_size: int = 200_000,
) -> dict:
    """Run global prediction only (no ROIs). Mostly for CLI / unit tests."""
    dev = _resolve_device(device)
    if checkpoint_path is not None:
        model, scalers, _ = load_model_and_scalers(model_id, device=dev, checkpoint_path=checkpoint_path)
        bundle = _global_bundle(Path(cfd_dir))
        pred_flow = _run_global_hybrid(model, bundle, scalers, dev, pred_batch_size)
        return {"pred_flow": pred_flow, "bundle": bundle, "scalers": scalers}
    stack = load_model_stack(model_id, device=dev)
    bundle = _global_bundle(Path(cfd_dir))
    if stack.kind == "cascade":
        pred_flow = _run_global_cascade(stack, bundle, dev, pred_batch_size)
        # For globals, "scalers" returned is Stage-1's (the model that produced
        # the field). Plot code only needs y-scale info, both stages share the
        # same input cols for the global so this is consistent.
        scalers = stack.conditioner.scalers if stack.conditioner else stack.primary.scalers
    else:
        pred_flow = _run_global_hybrid(stack.primary.model, bundle, stack.primary.scalers, dev, pred_batch_size)
        scalers = stack.primary.scalers
    return {"pred_flow": pred_flow, "bundle": bundle, "scalers": scalers}


def predict_all(
    cfd_dir: Path,
    *,
    model_id: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None,
    pred_batch_size: int = 200_000,
) -> dict:
    """Run both global and (if present) ROI predictions, returning everything."""
    dev = _resolve_device(device)
    if checkpoint_path is not None:
        # Legacy path: single-stage hybrid from explicit file.
        model, scalers, _ = load_model_and_scalers(model_id, device=dev, checkpoint_path=checkpoint_path)
        global_bundle = _global_bundle(Path(cfd_dir))
        pred_flow = _run_global_hybrid(model, global_bundle, scalers, dev, pred_batch_size)
        roi_items = _roi_bundles(Path(cfd_dir))
        roi_preds, roi_bundles = {}, {}
        for rel_label, rb in roi_items:
            key = Path(rel_label).name
            roi_bundles[key] = rb
            roi_preds[key] = _run_roi_hybrid(model, global_bundle, rb, scalers, dev, pred_batch_size)
        return {
            "pred_flow": pred_flow,
            "bundle": global_bundle,
            "roi_preds": roi_preds,
            "roi_bundles": roi_bundles,
            "scalers": scalers,
        }

    stack = load_model_stack(model_id, device=dev)
    global_bundle = _global_bundle(Path(cfd_dir))

    if stack.kind == "cascade":
        pred_flow = _run_global_cascade(stack, global_bundle, dev, pred_batch_size)
        roi_items = _roi_bundles(Path(cfd_dir))
        roi_preds, roi_bundles = {}, {}
        for rel_label, rb in roi_items:
            key = Path(rel_label).name
            roi_bundles[key] = rb
            roi_preds[key] = _run_roi_cascade(stack, global_bundle, rb, dev, pred_batch_size)
        # Scalers exposed downstream are Stage-2's (where ROI metrics live);
        # Stage-1's global scalers are accessible via stack.conditioner.scalers
        # if a caller needs them. Plot code uses ROI scalers for per-ROI plots
        # and global y-scaler info that both stages share.
        return {
            "pred_flow": pred_flow,
            "bundle": global_bundle,
            "roi_preds": roi_preds,
            "roi_bundles": roi_bundles,
            "scalers": stack.primary.scalers,
        }

    # Single-stage
    model = stack.primary.model
    scalers = stack.primary.scalers
    pred_flow = _run_global_hybrid(model, global_bundle, scalers, dev, pred_batch_size)
    roi_items = _roi_bundles(Path(cfd_dir))
    roi_preds, roi_bundles = {}, {}
    for rel_label, rb in roi_items:
        key = Path(rel_label).name
        roi_bundles[key] = rb
        roi_preds[key] = _run_roi_hybrid(model, global_bundle, rb, scalers, dev, pred_batch_size)
    return {
        "pred_flow": pred_flow,
        "bundle": global_bundle,
        "roi_preds": roi_preds,
        "roi_bundles": roi_bundles,
        "scalers": scalers,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run pretrained model on one CFD binary case dir.")
    ap.add_argument("--cfd-dir", required=True, help="Holds meta.json/terrain.npz/flow.npz [+ roi/].")
    ap.add_argument("--model", default=None,
                    help=f"Model id from the registry. Default: {DEFAULT_MODEL_ID}. "
                         f"Available: {', '.join(m['id'] for m in MODEL_REGISTRY)}")
    ap.add_argument("--checkpoint", default=None, help="Explicit .pth path (overrides --model). Single-stage only.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=200_000)
    ap.add_argument("--rois", action="store_true", help="Also run ROI predictions.")
    ap.add_argument("--save", default=None, help="Save pred_flow.npy (global only).")
    args = ap.parse_args()

    if args.rois:
        out = predict_all(
            Path(args.cfd_dir),
            model_id=args.model,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            device=args.device,
            pred_batch_size=int(args.batch_size),
        )
        pf = out["pred_flow"]
        print(f"global pred_flow shape={pf.shape}")
        umag = np.linalg.norm(pf[..., :3], axis=-1)
        print(f"  Umag: {np.nanmin(umag):.2f}..{np.nanmax(umag):.2f} m/s")
        for k, v in out["roi_preds"].items():
            umag = np.linalg.norm(v[..., :3], axis=-1)
            print(f"ROI {k}: shape={v.shape}  Umag: {np.nanmin(umag):.2f}..{np.nanmax(umag):.2f} m/s")
    else:
        out = predict_global(
            Path(args.cfd_dir),
            model_id=args.model,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            device=args.device,
            pred_batch_size=int(args.batch_size),
        )
        pf = out["pred_flow"]
        print(f"pred_flow shape={pf.shape}")
        umag = np.linalg.norm(pf[..., :3], axis=-1)
        print(f"Umag: {np.nanmin(umag):.2f}..{np.nanmax(umag):.2f} m/s")

    if args.save:
        np.save(args.save, out["pred_flow"])
        print(f"Saved {args.save}")
