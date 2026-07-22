# webapp — interactive wind-field prediction app (BETA)

> **Beta / test phase.** This app is a research demo, not a validated
> engineering tool. Its loads, pressures and snow indicators are pre-design
> screening estimates, not code-verified design values. Do not use it as the
> sole basis for any real design decision.

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
  height above ground (terrain-following), plus surface pressure and terrain
  altitude, with faint terrain contours. Oriented north-up to match the PDF
  report. Hover for exact values, zoom and pan; refined ROIs are overlaid at
  their true position.
- **Prediction jobs with progress** — `/predict` returns a `job_id`; the UI
  polls `/job_status`. One heavy job runs at a time (others queue).
- **Wind rose** — one full prediction per direction (4/8/12/16 sectors or a
  custom list), a binned-colour rose, per-direction loads, a combined
  multi-direction PDF, and 3D viewers with a direction selector.
- **Loads** — per-structure forces by surface-pressure integration over the STL
  mesh (with overturning moment and projected frontal area); ISA air density at
  the site elevation. Pre-design estimates only.
- **Uncertainty map** — optional second run with the other model family
  (hybrid vs U-Net); |ΔU| disagreement maps and stats.
- **Snow drift indicator** — heuristic near-surface wind-speed thresholding
  (deposition / neutral / erosion). A screening layer, not a transport model.
- **Past runs** — `/runs` lists saved results with PDF / 3D / export links.

## Configuration

Environment variables (all optional):

| variable                | default | effect                                                        |
|-------------------------|---------|---------------------------------------------------------------|
| `PINN_WEBAPP_MODELS`    | (all)   | comma-separated model ids to offer, e.g. the hybrid only      |
| `PINN_WEBAPP_MAX_RUNS`  | 40      | keep at most this many saved runs (0 disables)                |
| `PINN_WEBAPP_MAX_GB`    | 15      | keep saved runs under this size budget in GB (0 disables)     |
| `PINN_WEBAPP_MAX_ACTIVE_JOBS` | 0 | reject jobs once this many are queued/running (0 disables)    |
| `PINN_WEBAPP_RATE_LIMIT_JOBS` | 0 | process-wide job submissions per window (0 disables)          |
| `PINN_WEBAPP_RATE_LIMIT_WINDOW` | 3600 | submission-limit window in seconds                       |
| `PINN_DEVICE`           | auto    | force `cpu` or `cuda`                                         |
| `PINN_WEBAPP_GITHUB_URL`| (placeholder) | the "About / GitHub" link in the app header            |
| `PINN_WEBAPP_HOST`      | 127.0.0.1 | bind address; set `0.0.0.0` in a container                   |
| `PORT`                  | 8779    | listen port (Cloud Run / Docker set this automatically)       |
| `PINN_WEBAPP_RESULTS_DIR` | (local) | saved-runs dir; point at a mounted volume in a container     |

`models.yaml` (next to `app.py`) is the file equivalent of `PINN_WEBAPP_MODELS`
and also sets the default model. On a CPU-only host, serve the fast hybrid only:

```bash
PINN_WEBAPP_MODELS=hybrid-cascade-292d-pinn python pinnfluid/webapp/app.py
```

## Storage

Each saved run keeps ~50–100 MB under `webapp/results/<domain>/` (inputs +
predictions + plots + PDF). DEM downloads are cached under `dem/`. Old runs are
pruned automatically to stay within `PINN_WEBAPP_MAX_RUNS` and
`PINN_WEBAPP_MAX_GB` (the run just produced is never pruned). Both `results/`
and `workspace/` are git-ignored. The 3D viewers and the map load plotly.js from
the app itself (`/static/plotly.min.js`), so they work offline and under a
strict CSP.

## Security / deployment status

Bound to `127.0.0.1` by default and **no authentication**. Domain names are
sanitised against path traversal and HTML endpoints escape user input. The
optional process-wide job limit prevents an unbounded prediction queue, but
there is no CSRF protection and it is not DDoS-grade rate limiting. Follow
`DEPLOY.md` before exposing the app publicly; use authentication or Cloud Armor
if stronger per-user enforcement becomes necessary.
Checkpoints are loaded with `torch.load(weights_only=False)` — only load
checkpoints you trust. See the project's `DEPLOY.md` for the container and
Cloud Run deployment procedure and the remaining public-hosting caveats.

## Units

The plots, the interactive map, and the PDF report show pressure in **Pa**,
gauge-referenced to the outlet. The conversion uses ISA air density evaluated at
the site's mean altitude (so it adapts with elevation), stated in the report.

Internally the model outputs kinematic pressure (p/rho, m²/s², the OpenFOAM
convention); only the raw VTK/NPZ exports keep those kinematic units, so
downstream tools can apply their own density.
