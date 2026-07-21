# pinnfluid — physics-informed wind and pressure surrogate over terrain and structures

A steady-RANS surrogate that predicts the 3D wind and pressure field around
built structures embedded in real terrain, in seconds instead of hours. The
model is a two-stage cascade: a Stage-1 background over the full domain at
about 30 m resolution, refined by a Stage-2 network over the region of interest
at about 0.5 m resolution. Two architectures are provided:

- **Hybrid cascade** — a point-wise MLP head with a 2D CNN terrain encoder and
  Fourier features. Fast, smooth, best aggregate accuracy.
- **3D U-Net cascade** — a full 3D grid network. Best at resolving the
  high-speed corridors between closely spaced structures.

Both are trained on 292 idealised OpenFOAM domains with physics-informed losses
(continuity and momentum residuals) added to a supervised objective.

> This is the public research code accompanying the PhD work of Jimmy Gasser
> (EPFL, CRYOS). It is meant to reproduce the paper's results and to run the
> interactive prediction app, not as a production engineering tool.

## Status and roadmap

This is an active research project in a beta phase. Both the models and the
structure library are improved on a rolling basis, so results and available
geometry will change over time. Ongoing directions include adding turbulence
quantities to the output, improving accuracy, wider real-terrain forcing, and
real-world validation.

The structure library shipped here is a starting set of primitives. If you need
a specific structure that is not included, get in touch and it can often be
added: **jimmy.gasser@epfl.ch**.

## Repository layout

```
pinnfluid/                 core package (training + inference stack)
  config.py                all hyperparameters and paths
  data_loader.py           dataset assembly, scalers, grid/point bundles
  models.py                hybrid and grid U-Net architectures
  training.py              training loops, cascade logic, evaluation
  losses.py                supervised + physics (continuity/momentum) losses
  model_runtime.py         rebuild a model from a checkpoint's config snapshot
  physics_grid.py          finite-difference operators for physics losses
  utils.py                 misc helpers (seeding, logging, optional wandb)
  main.py                  training entry point (see "Reproducing")
  domain_prep/             swisstopo DEM download, domain building
  input_prep/              OpenFOAM case export helpers
  vis/                     figure/report generation
  configs/                 the exact YAML configs for the two final models
  splits/                  the 292-domain train/val/test split
  webapp/                  interactive prediction app (BETA — see its README)
    checkpoints/           model weights go here (fetched from the release)
requirements.txt
LICENSE
CITATION.cff
DATA.md                    where the CFD + STL datasets live and how to place them
```

Modules are imported flat (e.g. `from models import create_model`), so run
scripts from inside `pinnfluid/` or add it to `PYTHONPATH`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CPU-only PyTorch build runs the hybrid model comfortably. The 3D U-Net is
much faster on a GPU.

## Run the prediction app (BETA)

```bash
# 1. fetch the four checkpoints into pinnfluid/webapp/checkpoints/
python pinnfluid/webapp/fetch_checkpoints.py     # or download them manually

# 2. launch
python pinnfluid/webapp/app.py                   # serves http://127.0.0.1:8779
```

Pick a Swiss location (or import a DEM), place structures, predict the field,
inspect plots and 3D views, export VTK/NPZ/PDF. See
[`pinnfluid/webapp/README.md`](pinnfluid/webapp/README.md) for the full
feature list and the important security/deployment caveats.

## Reproducing the two final models

The datasets are hosted separately (see [`DATA.md`](DATA.md)). Once the CFD
data is placed under `data/cfd/` and the split is in `pinnfluid/splits/`:

```bash
cd pinnfluid
# Stage-1 backgrounds
python main.py --config configs/iter4_struct292_hybrid_cascade.yaml
python main.py --config configs/iter4_struct292_unet_cascade.yaml
# Stage-2 physics-informed refiners (both final models)
python main.py --config configs/iter4_struct292_physics_stage2_extra.yaml
```

Each checkpoint stores the exact config it was trained with, so inference and
evaluation rebuild the network without needing the YAML.

## Data and weights

- **Datasets** (OpenFOAM CFD fields + structure STLs): see [`DATA.md`](DATA.md).
- **Model weights** (4 checkpoints, ~220 MB): fetched into
  `pinnfluid/webapp/checkpoints/`. They are not tracked in git; see that
  folder's README.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)). Confirm the choice
of license with your PhD supervisor and the EPFL Technology Transfer Office
before the public release, and check the licenses of the swisstopo DEM tiles
and any bundled STL geometry.

## Citation

If you use this code or the models, please cite the accompanying paper (see
[`CITATION.cff`](CITATION.cff)).
