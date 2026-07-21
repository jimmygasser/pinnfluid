#!/usr/bin/env python3
"""Build a y-mirrored CFD training root without modifying the canonical data.

The output root contains:
  - symlinks to the original split cases;
  - real y-mirrored copies for train cases only, named ``ymir__<case>``;
  - a split JSON whose train list is original train + mirrored train, while
    val/test stay byte-for-byte on the original case names.

The mirror transform is physical reflection across the case y-midline:
  y' = y_min + y_max - y
  Ux' = Ux, Uy' = -Uy, Uz' = Uz, p' = p

Terrain elevation is flipped in y and slope/aspect are recomputed so the model
sees a self-consistent mirrored terrain, not just a copied raster channel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np


CATEGORIES = ("complexterrain_only", "singlestructures", "multistructures")
MIRROR_PREFIX = "ymir__"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def case_category_map(data_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for cat in CATEGORIES:
        root = data_root / cat
        if not root.exists():
            continue
        for p in root.iterdir():
            if p.is_dir() and (p / "meta.json").exists():
                out[p.name] = cat
    return out


def rel_symlink(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(rel, target_is_directory=True)


def terrain_channels_from_elevation(elevation: np.ndarray, *, dx: float, dy: float, template: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    elev = np.asarray(elevation, dtype=np.float32)
    dzdy, dzdx = np.gradient(elev, float(dy), float(dx))
    slope = np.degrees(np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))).astype(np.float32)
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) % 360.0).astype(np.float32)
    aspect[slope < 1.0e-6] = 0.0
    out: dict[str, np.ndarray] = {
        "elevation": elev.astype(np.float32, copy=False),
        "slope": slope,
        "aspect": aspect,
    }
    for key, arr in template.items():
        if key in out:
            continue
        if np.asarray(arr).ndim >= 1 and np.asarray(arr).shape[0] == elev.shape[0]:
            out[key] = np.flip(np.asarray(arr), axis=0).astype(np.float32, copy=False)
    return out


def mirror_bounds_y(bounds_list: list, *, y0: float, y1: float) -> list:
    out = []
    for item in bounds_list or []:
        if not isinstance(item, dict):
            out.append(item)
            continue
        copied = json.loads(json.dumps(item))
        mn = copied.get("min")
        mx = copied.get("max")
        if isinstance(mn, list) and isinstance(mx, list) and len(mn) >= 2 and len(mx) >= 2:
            old_min_y = float(mn[1])
            old_max_y = float(mx[1])
            mn[1] = float(y0 + y1 - old_max_y)
            mx[1] = float(y0 + y1 - old_min_y)
        out.append(copied)
    return out


def mirror_wind_from(value) -> float:
    try:
        return float((-float(value)) % 360.0)
    except Exception:
        return float("nan")


def mirror_meta(
    meta: dict,
    *,
    new_name: str,
    source_name: str,
    mirror_y_bounds: tuple[float, float] | None = None,
) -> dict:
    out = json.loads(json.dumps(meta))
    bounds = out.get("bounds") or [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    if mirror_y_bounds is None:
        y0 = float(bounds[2])
        y1 = float(bounds[3])
    else:
        y0, y1 = mirror_y_bounds
    if isinstance(bounds, list) and len(bounds) >= 4:
        old_min_y = float(bounds[2])
        old_max_y = float(bounds[3])
        bounds[2] = float(y0 + y1 - old_max_y)
        bounds[3] = float(y0 + y1 - old_min_y)
        out["bounds"] = bounds
    out["case_name"] = new_name
    out["augmentation"] = {
        "type": "y_mirror",
        "source_case_name": source_name,
        "y_reflection_bounds": [y0, y1],
    }
    if isinstance(out.get("source_case_dir"), str):
        out["source_case_dir"] = out["source_case_dir"]
    if isinstance(out.get("structure_bounds"), list):
        out["structure_bounds"] = mirror_bounds_y(out["structure_bounds"], y0=y0, y1=y1)
    abl = out.get("ABL")
    if isinstance(abl, dict):
        flow = abl.get("flowDir")
        if isinstance(flow, list) and len(flow) >= 2:
            flow = list(flow)
            flow[1] = float(-float(flow[1]))
            abl["flowDir"] = flow
        if "wind_from_deg" in abl:
            abl["wind_from_deg"] = mirror_wind_from(abl.get("wind_from_deg"))
    return out


def mirror_case(
    src: Path,
    dst: Path,
    *,
    new_name: str,
    overwrite: bool,
    mirror_y_bounds: tuple[float, float] | None = None,
) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    meta = read_json(src / "meta.json")
    if mirror_y_bounds is None:
        bounds = meta.get("bounds") or [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        mirror_y_bounds = (float(bounds[2]), float(bounds[3]))
    grid_spacing = meta.get("grid_spacing") or [1.0, 1.0, None]
    dx = float(grid_spacing[0] or 1.0)
    dy = float(grid_spacing[1] or 1.0)

    terrain_in = {k: np.asarray(v) for k, v in np.load(src / "terrain.npz", allow_pickle=False).items()}
    elev_m = np.flip(np.asarray(terrain_in["elevation"], dtype=np.float32), axis=0)
    terrain_out = terrain_channels_from_elevation(elev_m, dx=dx, dy=dy, template=terrain_in)
    np.savez_compressed(dst / "terrain.npz", **terrain_out)

    flow_in = {k: np.asarray(v) for k, v in np.load(src / "flow.npz", allow_pickle=False).items()}
    flow_out: dict[str, np.ndarray] = {}
    for key, arr in flow_in.items():
        mirrored = np.flip(np.asarray(arr), axis=1)
        if key == "Uy":
            mirrored = -mirrored
        flow_out[key] = mirrored.astype(arr.dtype, copy=False)
    np.savez_compressed(dst / "flow.npz", **flow_out)

    for name in ("nut.npy", "phi_wall.npy"):
        p = src / name
        if p.exists():
            arr = np.load(p, allow_pickle=False)
            np.save(dst / name, np.flip(arr, axis=1).astype(arr.dtype, copy=False))

    write_json(
        dst / "meta.json",
        mirror_meta(
            meta,
            new_name=new_name,
            source_name=src.name,
            mirror_y_bounds=mirror_y_bounds,
        ),
    )

    roi_root = src / "roi"
    if roi_root.exists():
        for roi_dir in sorted(p for p in roi_root.iterdir() if p.is_dir()):
            mirror_case(
                roi_dir,
                dst / "roi" / roi_dir.name,
                new_name=new_name,
                overwrite=overwrite,
                mirror_y_bounds=mirror_y_bounds,
            )


def split_cases(split: dict) -> Iterable[str]:
    for key in ("train", "val", "test"):
        for name in split.get(key, []):
            yield str(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", default="data/cfd")
    ap.add_argument("--output-root", default="data/cfd_ymirror")
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--output-split", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mirror-prefix", default=MIRROR_PREFIX)
    args = ap.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    split_path = Path(args.split_json)
    out_split_path = Path(args.output_split)
    split = read_json(split_path)
    categories = case_category_map(source_root)

    missing = [name for name in split_cases(split) if name not in categories]
    if missing:
        raise SystemExit(f"Missing {len(missing)} split cases under {source_root}: {missing[:20]}")

    # Link all original split cases into the mirror root.
    for name in sorted(set(split_cases(split))):
        cat = categories[name]
        rel_symlink(source_root / cat / name, output_root / cat / name, overwrite=bool(args.overwrite))

    mirror_names: list[str] = []
    for name in split.get("train", []):
        cat = categories[str(name)]
        mirror_name = f"{args.mirror_prefix}{name}"
        mirror_case(
            source_root / cat / str(name),
            output_root / cat / mirror_name,
            new_name=mirror_name,
            overwrite=bool(args.overwrite),
        )
        mirror_names.append(mirror_name)

    out_split = json.loads(json.dumps(split))
    out_split["train"] = [str(v) for v in split.get("train", [])] + mirror_names
    out_split["val"] = [str(v) for v in split.get("val", [])]
    out_split["test"] = [str(v) for v in split.get("test", [])]
    out_split["notes"] = (
        str(split.get("notes", ""))
        + f"\nY-mirror augmentation: train includes {len(mirror_names)} mirrored copies "
        f"with prefix {args.mirror_prefix!r}; val/test unchanged."
    ).strip()
    out_split["augmentation"] = {
        "type": "y_mirror",
        "source_split": str(split_path),
        "source_root": str(source_root),
        "data_root": str(output_root),
        "mirror_prefix": str(args.mirror_prefix),
        "mirrored_train_cases": int(len(mirror_names)),
    }
    write_json(out_split_path, out_split)

    print(f"[YMIRROR] linked originals: {len(set(split_cases(split)))}")
    print(f"[YMIRROR] mirrored train cases: {len(mirror_names)}")
    print(f"[YMIRROR] data root: {output_root}")
    print(f"[YMIRROR] split: {out_split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
