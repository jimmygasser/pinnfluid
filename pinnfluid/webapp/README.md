# webapp — interactive wind-field prediction app (BETA)

**Live application:** [pinnfluid.ch](https://pinnfluid.ch)

> **Beta / test phase.** This app is a research demo, not a validated
> engineering tool. Its experimental loads, pressures and snow indicators are
> pre-design screening estimates, not code-verified design values. Do not use
> it as the sole basis for any real design decision. Displayed pressures are
> relative, with the global fluid-domain mean set to zero for each prediction.

A local web app around the pinnfluid surrogate: pick a Swiss location (or
import a DEM), place structures, predict the 3D wind and pressure field in
seconds to minutes, inspect plots and 3D views, and export VTK / NPZ / PDF.

## Run

```bash
# from the repo root, after installing requirements.txt and fetching weights:
python pinnfluid/webapp/app.py                 # serves http://127.0.0.1:8779
python pinnfluid/webapp/app.py --port 9000 --no-browser
```

CPU-only works; CUDA is used automatically when available. The hybrid model is
fast on CPU; the 3D U-Net is much slower on CPU and is best run on a GPU.

## Models

`inference.MODEL_REGISTRY` exposes the two final 292-domain, y-reflection
augmented, physics-informed cascades:

- **Hybrid (cascade, 292d, physics)** — default; fastest, CPU-friendly, best
  aggregate accuracy.
- **3D UNet (cascade, 292d, physics)** — more accurate at inter-structure jets,
  but slow on CPU (wants a GPU).

Each entry needs its two checkpoints under `checkpoints/`; entries whose files
are missing are hidden automatically. See `checkpoints/README.md`.

## Features

- **Interactive map** — the quick-look result, embedded as the first thing shown
  after a prediction. A top-down Plotly heatmap of wind speed at a selectable
  height above ground (terrain-following), plus relative pressure and terrain
  altitude, with faint terrain contours. Oriented north-up to match the PDF
  report. Hover for exact values, zoom and pan; refined ROIs are overlaid at
  their true position.
- **Prediction jobs with progress** — `/predict` returns a `job_id`; the UI
  polls `/job_status`. One heavy job runs at a time (others queue).
- **Wind rose** — one full prediction per direction (4/8/12/16 sectors or a
  custom list), a binned-colour rose, per-direction loads, a combined
  multi-direction PDF, and 3D viewers with a direction selector.
- **Experimental loads** — per-structure, pressure-only force estimates from
  integration over the STL mesh (with overturning moment and projected frontal
  area); ISA air density at the site elevation. Components in the horizontal
  plane use the current wind-aligned frame. These are screening values, not
  validated design loads.
- **Uncertainty map** — optional second run with the other model family
  (hybrid vs U-Net); |ΔU| disagreement maps and stats.
- **Snow drift indicator** — heuristic near-surface wind-speed thresholding
  (deposition / neutral / erosion). A screening layer, not a transport model.

## Configuration

Environment variables (all optional):

| variable                | default | effect                                                        |
|-------------------------|---------|---------------------------------------------------------------|
| `PINN_WEBAPP_MODELS`    | (all)   | comma-separated model ids to offer, e.g. the hybrid only      |
| `PINN_WEBAPP_MAX_RUNS_PER_SESSION` | 10 | the web app keeps each browser session's most recently used runs (a wind rose is one run; 0 disables) |
| `PINN_WEBAPP_MAX_GB`    | 15      | all saved runs together stay under this size in GB, least recently used first (0 disables) |
| `PINN_WEBAPP_MAX_RUNS`  | 40      | `results_io.enforce_results_retention` only (callers other than the web app; 0 disables) |
| `PINN_WEBAPP_MAX_ACTIVE_JOBS` | 0 | reject jobs once this many are queued/running (0 disables)    |
| `PINN_WEBAPP_RATE_LIMIT_JOBS` | 0 | process-wide job submissions per window (0 disables)          |
| `PINN_WEBAPP_RATE_LIMIT_PREP` | 0 | terrain downloads/uploads/builds per window (0 disables)      |
| `PINN_WEBAPP_RATE_LIMIT_WINDOW` | 3600 | submission-limit window in seconds                       |
| `PINN_WEBAPP_MAX_ROSE_SECTORS` | 16 | maximum directions in one wind-rose job                    |
| `PINN_WEBAPP_MAX_REQUEST_MB` | 32 | maximum JSON request body                                   |
| `PINN_WEBAPP_MAX_UPLOAD_MB` | 20 | maximum decoded custom GeoTIFF                              |
| `PINN_WEBAPP_MAX_DEM_PIXELS` | 4000000 | maximum pixels of a custom GeoTIFF, read from its header before decoding (0 disables) |
| `PINN_WEBAPP_MAX_DOMAIN_M` | 3000 | maximum terrain width and height                            |
| `PINN_WEBAPP_MAX_SINGLE_STRUCTURES` | 10 | maximum individually placed structures                 |
| `PINN_WEBAPP_ENABLE_RUN_INDEX` | 0 | expose the shared `/runs` index when set to 1              |
| `PINN_WEBAPP_STEP_TIMEOUT_S` | 900 | time limit of one terrain step (dem_prep, place_structure, finalize_domain) |
| `PINN_WEBAPP_SOCKET_TIMEOUT_S` | 60 | close a client connection silent for this long (0 disables) |
| `PINN_WEBAPP_JOB_TTL_S` | 3600 | forget finished jobs after this many seconds (at most 200 are kept) |
| `PINN_WEBAPP_WORK_TTL_S` | 21600 | remove DEMs, left-over workspaces and wind-rose summaries unused for this long (0 disables) |
| `PINN_WEBAPP_MAX_DEMS` | 50 | keep at most this many DEM directories, newest first (0 disables) |
| `PINN_WEBAPP_MAX_EXPORT_BUILDS` | 2 | VTK/NPZ export archives being built at once, all sessions (429 beyond; 0 disables) |
| `PINN_WEBAPP_MAX_EXPORT_BUILDS_PER_SESSION` | 1 | the same for one session |
| `PINN_DEVICE`           | auto    | force `cpu` or `cuda`                                         |
| `PINN_WEBAPP_GITHUB_URL`| project repository | the "About / GitHub" link in the app header      |
| `PINN_WEBAPP_HOST`      | 127.0.0.1 | bind address; set `0.0.0.0` in a container                   |
| `PORT`                  | 8779    | listen port (Cloud Run / Docker set this automatically)       |
| `PINN_WEBAPP_WORK_DIR` | (checkout) | absolute dir for everything the app writes: `dem/`, `workspace/`, `results/` (the source tree can then be read-only) |
| `PINN_WEBAPP_RESULTS_DIR` | (local) | saved-runs dir; point at a mounted volume in a container (overrides `<work dir>/results`) |
| `PINN_WEBAPP_BASE_PATH` | (root) | URL prefix behind a gateway, e.g. `/pinnfluid`: the app answers only under it (404 elsewhere) |
| `PINN_WEBAPP_SECURE_COOKIE` | 0 | set to 1 behind HTTPS: the session cookie gets the `Secure` flag |

`models.yaml` (next to `app.py`) is the file equivalent of `PINN_WEBAPP_MODELS`
and also sets the default model. On a CPU-only host, serve the fast hybrid only:

```bash
PINN_WEBAPP_MODELS=hybrid-cascade-292d-pinn python pinnfluid/webapp/app.py
```

## Storage

Each saved run keeps ~50–100 MB under `webapp/results/<domain>/` (inputs +
predictions + plots + PDF). DEM downloads are cached under `dem/`. With
`PINN_WEBAPP_WORK_DIR` set, `dem/`, `workspace/` and `results/` are below that
directory instead and nothing is written into the checkout. Old runs are
pruned automatically: each browser session keeps its `PINN_WEBAPP_MAX_RUNS_PER_SESSION`
most recently used runs, and all runs stay within `PINN_WEBAPP_MAX_GB` (least
recently used first; the run just produced and runs in use are never pruned). Both `results/`
and `workspace/` are git-ignored. The 3D viewers and the map load plotly.js from
the app itself (`static/plotly.min.js`, a relative URL like every other URL
the app renders, so the pages also work under `PINN_WEBAPP_BASE_PATH`), so
they work offline and under a strict CSP. The page loads Leaflet and leaflet-draw from
`static/vendor/` the same way; those files are fetched at deployment (see
`static/vendor/README.md`), or `PINN_WEBAPP_LEAFLET_CDN=1` loads them from the
unpkg CDN.

## Security / deployment status

Bound to `127.0.0.1` by default and **no authentication**. Domain names are
letters, digits, `-`, `_` and `.` (at most 64, starting with a letter or digit),
download names follow RFC 6266, and structure types must be shipped
`single_stl/*.stl` files. Uploaded DEMs are opened as GeoTIFF only (never VRT or
another GDAL format), with the GDAL defaults `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`
and `CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.none`, and their pixel count is checked
before decoding. DEM tiles are fetched only from `https://data.geo.admin.ch`.
HTML endpoints escape user input, request and DEM sizes are bounded, and terrain/prediction operations have optional
process-wide admission limits. Each browser gets a random session id in an
HttpOnly cookie (`pinnfluid_session`, scoped to the base path); its domains are
stored under a namespace derived from it, so results, terrain and jobs
(`/job_status`) of one session are not reachable from another, even with the same
domain name (they answer 404). The past-run index, disabled by default, lists the
session's own runs only. Results live as long as the server keeps them
(`PINN_WEBAPP_MAX_RUNS_PER_SESSION`, `PINN_WEBAPP_MAX_GB`). There is no
CSRF protection and the in-process limits are not DDoS-grade. Terrain steps,
tile downloads (`data.geo.admin.ch` only) and idle connections have time limits.
Clients see validation messages, or a generic message with a reference whose
traceback is in the server log; responses carry no server paths. Follow
`DEPLOY.md` before exposing the app publicly; use authentication or Cloud Armor
if stronger per-user enforcement becomes necessary.
Checkpoints are loaded with `torch.load(weights_only=False)` — only load
checkpoints you trust. See the project's `DEPLOY.md` for the container and
Cloud Run deployment procedure and the remaining public-hosting caveats.

## Units

The plots, interactive map, 3D views, and PDF report show **relative pressure**
in Pa. One constant is used throughout each prediction: the arithmetic mean
over valid global fluid cells is set to zero. Wind-rose sectors are independent
predictions and therefore each use their own reference. The Pa conversion uses
ISA air density evaluated at the site's mean altitude, stated in the report.

Internally the model outputs kinematic pressure (p/rho, m²/s², the OpenFOAM
convention). Saved fields and VTK/NPZ exports retain that unmodified raw model
output so downstream tools can choose their own pressure reference and density.
