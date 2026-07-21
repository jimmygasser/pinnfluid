"""Helpers for reconstructing models from checkpoints with saved config snapshots."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Iterator

import torch


def _snapshot_uppercase(module) -> dict:
    return {k: getattr(module, k) for k in dir(module) if k.isupper()}


def _sync_consumer_module_from_config(cfg_mod, consumer_mod) -> None:
    """Re-bind uppercase config names in a consumer module that did
    `from config import FOO` at import time. Such imports bind values, so
    later writes to `config.FOO` do not propagate. The web app holds
    multiple snapshots in one process, so this sync is required."""
    for key in list(vars(consumer_mod)):
        if key.isupper() and hasattr(cfg_mod, key):
            setattr(consumer_mod, key, getattr(cfg_mod, key))


# Backwards-compatible alias for any external callers.
_sync_model_module_from_config = _sync_consumer_module_from_config


def _sync_all_consumers(cfg_mod) -> None:
    import sys
    import models as models_mod  # type: ignore
    _sync_consumer_module_from_config(cfg_mod, models_mod)
    # data_loader, losses, and training also do `from config import FOO`.
    # Sync any of them that are already loaded; ignore the rest.
    for name in ("data_loader", "losses", "training"):
        mod = sys.modules.get(name)
        if mod is not None:
            _sync_consumer_module_from_config(cfg_mod, mod)


@contextmanager
def applied_config_snapshot(snapshot: dict | None) -> Iterator[None]:
    import config as cfg_mod  # type: ignore

    baseline = _snapshot_uppercase(cfg_mod)
    try:
        if snapshot:
            for key, value in baseline.items():
                setattr(cfg_mod, key, value)
            for key, value in snapshot.items():
                if str(key).isupper():
                    setattr(cfg_mod, str(key), value)
            if hasattr(cfg_mod, "_finalize_pressure_weights"):
                cfg_mod._finalize_pressure_weights()
            if hasattr(cfg_mod, "_finalize_encoder_depths"):
                cfg_mod._finalize_encoder_depths()
            _sync_all_consumers(cfg_mod)
        yield
    finally:
        for key, value in baseline.items():
            setattr(cfg_mod, key, value)
        if hasattr(cfg_mod, "_finalize_pressure_weights"):
            cfg_mod._finalize_pressure_weights()
        if hasattr(cfg_mod, "_finalize_encoder_depths"):
            cfg_mod._finalize_encoder_depths()
        _sync_all_consumers(cfg_mod)


def load_checkpoint_payload(checkpoint_path: Path | str, device: str) -> dict:
    ckpt_path = Path(checkpoint_path)
    with torch.serialization.safe_globals([]):
        try:
            return torch.load(str(ckpt_path), map_location=device, weights_only=False)
        except TypeError:
            return torch.load(str(ckpt_path), map_location=device)


def _sibling_run_config_snapshot(checkpoint_path: Path | str) -> dict:
    ckpt_path = Path(checkpoint_path)
    candidates = [
        ckpt_path.parent.parent / "run_config.json",
        ckpt_path.parent / "run_config.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        snapshot = payload.get("config_snapshot")
        if isinstance(snapshot, dict) and snapshot:
            return snapshot
    return {}


def load_model_bundle_from_checkpoint(checkpoint_path: Path | str, device: str):
    from models import create_model  # type: ignore

    ckpt = load_checkpoint_payload(checkpoint_path, device)
    train_cfg = ckpt.get("train_config") if isinstance(ckpt, dict) else {}
    model_kind = str((train_cfg or {}).get("model_kind", "hybrid")).strip().lower()
    snapshot = ckpt.get("config_snapshot") or (train_cfg or {}).get("config_snapshot") or {}
    if not snapshot:
        snapshot = _sibling_run_config_snapshot(checkpoint_path)
        if snapshot and isinstance(ckpt, dict):
            ckpt["config_snapshot"] = snapshot
            if isinstance(train_cfg, dict):
                train_cfg["config_snapshot"] = snapshot
    with applied_config_snapshot(snapshot):
        model = create_model(device=device, model_kind=model_kind)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("scalers"), ckpt
