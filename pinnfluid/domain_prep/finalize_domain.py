#!/usr/bin/env python3
"""Finalize a domain for OpenFOAM: shift z so ground starts at 0.

Reads ground.stl (and optionally structure STLs) from a dem prep directory,
shifts all z-coordinates by -z_min, and writes the result into the target
data_preparation directory.

Usage:
  # Single terrain-only domain
  python3 scripts/finalize_domain.py \
      --prep-dir dem/01_flat_plain/prep \
      --out-dir data_preparation/complexterrain_only/01_flat_plain

  # All terrain-only domains at once
  python3 scripts/finalize_domain.py --batch-terrain

  # Terrain + structure domain (shifts both together)
  python3 scripts/finalize_domain.py \
      --prep-dir dem/05_downhill_steep/prep \
      --out-dir data_preparation/complexterrain_structures/05_downhill_steep \
      --structure-stls domains/my_case/structures_placed.stl
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    from stl import mesh as stl_mesh
except ImportError:
    stl_mesh = None


def shift_stl_z(input_path: str, output_path: str, z_offset: float) -> dict:
    """Shift all z-coordinates in an STL by z_offset. Returns bounds info."""
    if stl_mesh is None:
        raise ImportError("numpy-stl is required: pip install numpy-stl")

    m = stl_mesh.Mesh.from_file(input_path)
    verts = m.vectors.reshape(-1, 3)

    z_min_before = float(verts[:, 2].min())
    z_max_before = float(verts[:, 2].max())

    verts[:, 2] += z_offset
    m.vectors = verts.reshape(-1, 3, 3)
    m.save(output_path)

    z_min_after = float(verts[:, 2].min())
    z_max_after = float(verts[:, 2].max())

    return {
        "z_min_before": z_min_before,
        "z_max_before": z_max_before,
        "z_min_after": z_min_after,
        "z_max_after": z_max_after,
    }


def finalize_domain(
    prep_dir: str,
    out_dir: str,
    structure_stls=None,
) -> None:
    """Finalize one domain: shift z to 0, copy to output."""
    prep_dir = Path(prep_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read z_range from transform.json
    transform_json = prep_dir / "transform.json"
    if not transform_json.exists():
        print(f"  ERROR: {transform_json} not found", file=sys.stderr)
        return

    with open(transform_json) as f:
        meta = json.load(f)

    z_range = meta.get("z_range", [0, 0])
    z_min = float(z_range[0])
    z_offset = -z_min

    print(f"  z_range: [{z_range[0]:.1f}, {z_range[1]:.1f}]  "
          f"-> shift by {z_offset:+.1f}m")

    # Shift ground STL
    ground_src = prep_dir / "ground.stl"
    if not ground_src.exists():
        ground_src = prep_dir / "terrain.stl"
    if not ground_src.exists():
        print(f"  ERROR: No ground.stl or terrain.stl in {prep_dir}", file=sys.stderr)
        return

    tri_dir = out_dir / "constant" / "triSurface"
    tri_dir.mkdir(parents=True, exist_ok=True)

    ground_dst = tri_dir / "ground.stl"
    info = shift_stl_z(str(ground_src), str(ground_dst), z_offset)
    print(f"  ground.stl: z=[{info['z_min_before']:.1f}, {info['z_max_before']:.1f}] "
          f"-> [{info['z_min_after']:.1f}, {info['z_max_after']:.1f}]")

    # Shift structure STLs if provided
    if structure_stls:
        for stl_path in structure_stls:
            stl_path = Path(stl_path)
            if not stl_path.exists():
                print(f"  WARN: Structure STL not found: {stl_path}", file=sys.stderr)
                continue
            dst = tri_dir / "structure.stl"
            info = shift_stl_z(str(stl_path), str(dst), z_offset)
            print(f"  {stl_path.name}: z=[{info['z_min_before']:.1f}, {info['z_max_before']:.1f}] "
                  f"-> [{info['z_min_after']:.1f}, {info['z_max_after']:.1f}]")

    # Write updated transform.json with z_offset info
    meta_out = dict(meta)
    meta_out["z_offset_applied"] = z_offset
    meta_out["z_range_final"] = [z_range[0] + z_offset, z_range[1] + z_offset]
    meta_out["domain_size"] = [
        meta["domain_size"][0],
        meta["domain_size"][1],
        z_range[1] - z_range[0],
    ]

    with open(tri_dir / "transform.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    # Copy dem_final.tif for reference
    dem_final = prep_dir / "dem_final.tif"
    if dem_final.exists():
        shutil.copy2(str(dem_final), str(tri_dir / "dem_final.tif"))

    print(f"  -> {out_dir}/")


def batch_terrain_only(root: Path) -> None:
    """Process all dem/**/prep/ into data_preparation/complexterrain_only/."""
    dem_root = root / "dem"
    out_root = root / "data_preparation" / "complexterrain_only"

    cases = sorted([d for d in dem_root.iterdir() if d.is_dir() and (d / "prep").is_dir()])
    print(f"Batch: {len(cases)} terrain-only domains\n")

    for case_dir in cases:
        name = case_dir.name
        prep = case_dir / "prep"
        out = out_root / name
        print(f"=== {name} ===")
        finalize_domain(str(prep), str(out))
        print()

    print(f"Done. {len(cases)} domains in {out_root}/")


def main():
    p = argparse.ArgumentParser(description="Finalize domain: shift z to 0, copy STLs")
    p.add_argument("--prep-dir", type=str, default=None,
                   help="Path to dem/<case>/prep directory")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (e.g. data_preparation/complexterrain_only/<case>)")
    p.add_argument("--structure-stls", nargs="*", default=None,
                   help="Optional structure STL files to shift along with terrain")
    p.add_argument("--batch-terrain", action="store_true",
                   help="Process all dem/*/prep/ into data_preparation/complexterrain_only/")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent.parent

    if args.batch_terrain:
        batch_terrain_only(root)
    elif args.prep_dir and args.out_dir:
        finalize_domain(args.prep_dir, args.out_dir, args.structure_stls)
    else:
        print("ERROR: provide --batch-terrain or --prep-dir + --out-dir", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
