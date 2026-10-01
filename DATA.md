# Data

The training CFD fields are **not** stored in this git repository. They are
kept in a public data archive (see "Where to get it" below) and are downloaded
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
  | `terrain.npz` | terrain elevation, slope and aspect                             |
  | `flow.npz`    | the steady RANS solution (velocity components, pressure) and the fluid mask |
  | `nut.npy`     | turbulent viscosity field                                       |
  | `meta.json`   | inflow speed/direction, reference height, roughness length, grid info |

  The aerodynamic roughness length is constant in this dataset
  (`z0 = 0.1` m for every domain). It is therefore stored once per case in
  `meta.json` and not as a map in `terrain.npz`.

  Cases that contain structures (`singlestructures/`, `multistructures/`) also
  have a `roi/` subfolder with one refined region of interest per structure or
  cluster (`roi/roi_000/`, `roi/roi_001/`, ...). Each ROI holds the same
  `terrain.npz` / `flow.npz` / `nut.npy` / `meta.json` at the fine ~0.5 m
  resolution, in the same coordinate frame as the parent domain, plus
  `phi_wall.npy`, the signed distance to the nearest structure surface. These
  are what the Stage-2 refiner is trained and evaluated on. Terrain-only cases
  have no `roi/`.


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

The dataset is archived on EnviDat under the DOI
[10.16904/envidat.810](https://doi.org/10.16904/envidat.810) (CC BY 4.0).
If you use it, please cite the dataset together with the paper.

It is distributed as three zip archives, one per category, so that a single
category can be downloaded on its own:

| archive                                  | cases | download size | unpacked |
|------------------------------------------|------:|--------------:|---------:|
| `pinnfluid_cfd_complexterrain_only.zip`  |   108 |    0.8 GB  |   1.1 GB |
| `pinnfluid_cfd_multistructures.zip`      |    93 |    1.3 GB  |   1.5 GB |
| `pinnfluid_cfd_singlestructures.zip`     |    91 |    14.9 GB  |    17 GB |

These are the 292 domains of the split shipped with the code (256 training,
18 validation, 18 test). Each archive unpacks to `cfd/<category>/<case>/...`,
so extract all of them inside the `data/` folder of the repository:

```bash
mkdir -p data
unzip pinnfluid_cfd_complexterrain_only.zip -d data/
unzip pinnfluid_cfd_multistructures.zip     -d data/
unzip pinnfluid_cfd_singlestructures.zip    -d data/
```

This gives the `data/cfd/` layout shown above. On Windows, extract the three
archives into the same `data` folder with the file explorer or 7-Zip. The
single-structure archive is larger than 4 GB and needs a tool that supports
the Zip64 format, which all current ones do.

The pretrained checkpoints and the prediction app can be used without the
dataset.

### Notes on the archived fields

- The mirrored training copies used for the final models are not part of the
  archive. They are derived data and are generated locally with the command
  given in "Final-model y-reflection augmentation".
- For 23 of the hand-designed terrain-only cases (numbers 08 to 40), the flow
  fields were reflected across the domain y-midline after export so that they
  are consistent with the terrain. This is an exact symmetry of the governing
  equations. The archived fields are the corrected ones, which are those used
  to train and evaluate the published models.
