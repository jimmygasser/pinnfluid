# Data

The training CFD fields are **not** stored in this git repository. They will
live in a public data archive (see "Where to get it" below) and are downloaded
once into the layout the code expects. The lightweight STL primitive library is
included under `single_stl/`.

## What the dataset contains

- **CFD fields** — one folder per simulated domain, grouped into three
  categories:
  - `complexterrain_only/` — terrain, no structures
  - `singlestructures/`    — one structure on terrain
  - `multistructures/`     — several structures on terrain

  Each case folder holds:
  | file          | contents                                                        |
  |---------------|-----------------------------------------------------------------|
  | `terrain.npz` | terrain elevation, slope/aspect, roughness, domain metadata     |
  | `flow.npz`    | the steady RANS solution (velocity components, pressure)        |
  | `nut.npy`     | turbulent viscosity field                                       |
  | `meta.json`   | inflow speed/direction, reference height, roughness, grid info  |

  Cases that contain structures (`singlestructures/`, `multistructures/`) also
  have a `roi/` subfolder with one refined region of interest per structure or
  cluster (`roi/roi_000/`, `roi/roi_001/`, ...). Each ROI holds the same
  `terrain.npz` / `flow.npz` / `meta.json` at the fine ~0.5 m resolution, in the
  same coordinate frame as the parent domain. These are what the Stage-2 refiner
  is trained and evaluated on. Terrain-only cases have no `roi/`.


- **Structure geometry** — the STL primitive library (`single_stl/`): panels,
  cones, cubes, cylinders, concentrators, etc. used to build the structures.

## Expected on-disk layout

Place the downloaded data so the paths in `pinnfluid/config.py` resolve
(`DATA_CFD_ROOT = <repo>/data/cfd`):

```
data/
  cfd/
    complexterrain_only/<case>/{terrain.npz, flow.npz, nut.npy, meta.json}
    singlestructures/  <case>/...
    multistructures/   <case>/...
single_stl/            *.stl        (structure primitive library)
pinnfluid/splits/recommended_292domains_struct_al_full.json
```

`data/` is git-ignored. The 292-domain split that selects the train/val/test
cases and the STL library both ship with the code.

## Final-model y-reflection augmentation

The final models use each of the 256 training domains together with a physical
reflection across the domain y-midline. Validation and test are not augmented.
Build this derived root without modifying `data/cfd`:

```bash
python pinnfluid/input_prep/make_y_mirror_cfd.py \
  --source-root data/cfd \
  --output-root data/cfd_ymirror \
  --split-json pinnfluid/splits/recommended_292domains_struct_al_full.json \
  --output-split pinnfluid/splits/recommended_292domains_struct_al_full_ymirror.json
```

The output contains relative symlinks to the 292 original cases and real
mirrored copies of the 256 training cases. The mirrored split therefore has
512 training entries and the unchanged 18 validation and 18 test entries.

## Where to get it

The CFD training dataset is not currently distributed publicly. The pretrained
checkpoints and prediction app can be used without it. Access and archival
information will be added here if the dataset is released.
