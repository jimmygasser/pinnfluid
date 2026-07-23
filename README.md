# pinnfluid — physics-informed wind and pressure surrogate over terrain and structures

A steady-RANS surrogate that predicts the 3D wind and pressure field around
built structures embedded in real terrain, in seconds instead of hours. The
model is a two-stage cascade. A Stage 1 background over the full domain at
about 30 m resolution, refined by a Stage 2 network over the region of interest
at about 0.5 m resolution. Two architectures are provided:

- **Hybrid cascade** — a point-wise MLP head with a 2D CNN terrain encoder and
  Fourier features. Fast, smooth, best aggregate accuracy.
- **3D U-Net cascade** — a full 3D grid network. Best at resolving the
  high-speed corridors between closely spaced structures.

Both are trained on 292 idealised OpenFOAM domains with y-reflection data
augmentation and physics-informed losses (continuity and turbulent-viscosity
momentum residuals) added to a supervised objective.

On the frozen 18-domain test set, the final checkpoints reach the following
mean normalised RMSE values:

| model | global U | global p | global p (gauge) | ROI U | ROI p | ROI p (gauge) |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid | 0.157 | 0.247 | 0.202 | 0.169 | 0.122 | 0.089 |
| 3D U-Net | 0.177 | 0.233 | 0.185 | 0.176 | 0.130 | 0.086 |

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
  run_manifest.py          sequential Stage-1/Stage-2 YAML launcher
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
tests/                     release and web-app regression tests
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

## Tests

Run the release regression suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Run the prediction app (BETA)

```bash
# 1. fetch the four checkpoints into pinnfluid/webapp/checkpoints/
python pinnfluid/webapp/fetch_checkpoints.py     # or download them manually

# 2. launch
python pinnfluid/webapp/app.py                   # serves http://127.0.0.1:8779
```

Pick a Swiss location (or import a DEM), place structures interactively, predict the field,
inspect plots and 3D views, export VTK/NPZ/report PDF. See
[`pinnfluid/webapp/README.md`](pinnfluid/webapp/README.md) for the full
feature list and the important security/deployment caveats.

## Reproducing the two final models

The training dataset is not yet currently distributed publicly (see
[`DATA.md`](DATA.md)). Once the CFD data is available under `data/cfd/`, create
the mirrored training root:

```bash
python pinnfluid/input_prep/make_y_mirror_cfd.py \
  --source-root data/cfd \
  --output-root data/cfd_ymirror \
  --split-json pinnfluid/splits/recommended_292domains_struct_al_full.json \
  --output-split pinnfluid/splits/recommended_292domains_struct_al_full_ymirror.json

# Each manifest trains Stage 1 and then its Stage-2 refiner.
python pinnfluid/run_manifest.py \
  --sweep-yaml pinnfluid/configs/final_hybrid_292_ymirror.yaml
python pinnfluid/run_manifest.py \
  --sweep-yaml pinnfluid/configs/final_grid_unet_292_ymirror.yaml
```

The mirrored copies are used only for training; validation and test retain the
original cases. Each checkpoint stores its config and fitted scalers, so
inference rebuilds the network without needing the YAML.

## Data and weights

- **Datasets** (OpenFOAM CFD fields + structure STLs): see [`DATA.md`](DATA.md).
- **Model weights** (4 checkpoints, ~220 MB): fetched into
  `pinnfluid/webapp/checkpoints/`. They are not tracked in git; see that
  folder's README.

## License

Released under the MIT License (see [`LICENSE`](LICENSE)). External terrain
data and third-party geometry remain subject to their respective licences.

## Citation

If you use this code or the models, cite the software using
[`CITATION.cff`](CITATION.cff). A paper citation will be added after publication.
