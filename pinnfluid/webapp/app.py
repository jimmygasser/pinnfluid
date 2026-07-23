"""Interactive app for running pinnfluid on a user-selected domain.

Three-step UX (mirrors the vocabulary the user thinks in):
    1. Confirm terrain         → /download_dem (swisstopo tiles → cropped DEM)
    2. Confirm structure(s)    → /build_inputs (STLs + terrain.npz + flow.npz)
    3. Predict flow field      → /predict     (inference + plots + cleanup)

Plots land as individual PNGs in webapp/results/<domain>/plots/
and are also streamed inline to the browser for preview + per-file download.
A separate zip endpoint bundles all of them.
"""

from __future__ import annotations

import argparse
import base64 as _b64
import binascii
import copy
import hashlib
import html as _html
import http.server
import io
import json
import math
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
import webbrowser
import zipfile
import zlib
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
ROOT = SCRIPTS_DIR.parent
# Saved runs live here. Override with PINN_WEBAPP_RESULTS_DIR to point at a
# mounted persistent volume (e.g. a Cloud Storage bucket) when deploying in a
# container whose local disk is ephemeral.
RESULTS_DIR = Path(os.environ.get(
    "PINN_WEBAPP_RESULTS_DIR", Path(__file__).resolve().parent / "results"))

# Public repository linked from the app header; deployments may override it.
GITHUB_URL = os.environ.get(
    "PINN_WEBAPP_GITHUB_URL", "https://github.com/jimmygasser/pinnfluid"
)

for extra in (SCRIPTS_DIR, SCRIPTS_DIR / "domain_prep", SCRIPTS_DIR / "input_prep"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from domain_builder import (  # type: ignore  noqa: E402
    HTML_TEMPLATE as BUILDER_HTML,
    download_tiles,
    lv95_to_wgs84_point,
    merge_and_crop,
    query_stac_tiles,
    register_custom_dem,
    wgs84_point_to_lv95,
    wgs84_to_lv95,
)

from build_inputs import build_inputs, cleanup_case, _case_paths  # type: ignore  noqa: E402
from export_npz import export_prediction_npz  # type: ignore  noqa: E402
from export_vtk import export_prediction_vtk  # type: ignore  noqa: E402
from inference import (  # type: ignore  noqa: E402
    DEFAULT_CHECKPOINT,
    MODEL_REGISTRY,
    get_model_entry,
    list_models,
    predict_all,
    runtime_device_info,
)
from plots import (  # type: ignore  noqa: E402
    generate_prediction_report,
    plot_disagreement,
    plot_roi_disagreement,
)
from report import (  # type: ignore  noqa: E402
    compute_summary_stats,
    write_pdf_report,
    write_rose_pdf_report,
)
from results_io import (  # type: ignore  noqa: E402
    has_saved_inputs,
    load_saved_inputs,
    save_inputs_and_predictions,
)
from view_3d import write_3d_html, write_structure_3d_html  # type: ignore  noqa: E402
from interactive_map import write_map_html  # type: ignore  noqa: E402
from pressure_reference import (  # type: ignore  noqa: E402
    global_pressure_reference_kinematic,
    presentation_prediction,
)


# ---------------------------------------------------------------------------
# Name sanitisation (single chokepoint for every user-supplied domain name)
# ---------------------------------------------------------------------------
def _safe_name(raw) -> str:
    """Sanitise a user-supplied domain name before it touches the filesystem.

    Mirrors build_inputs._case_paths' replacement rules and additionally
    rejects traversal-shaped names, so `?domain=../../...` cannot escape
    RESULTS_DIR / dem/ regardless of which endpoint received it.
    """
    s = str(raw or "").strip().replace("/", "_").replace("\\", "_")
    if not s or s.startswith(".") or ".." in s:
        raise ValueError(f"invalid domain name: {raw!r}")
    return s


_PLOTLY_JS_CACHE: Optional[bytes] = None


def _plotly_js_bytes() -> bytes:
    """plotly.js served same-origin at /static/plotly.min.js.

    Uses the exact bundle that ships with the installed plotly, so it always
    matches the figures. Serving it ourselves (instead of the CDN) keeps the 3D
    viewers and the map working offline and under a strict content-security policy.
    """
    global _PLOTLY_JS_CACHE
    if _PLOTLY_JS_CACHE is None:
        from plotly.offline import get_plotlyjs  # type: ignore
        _PLOTLY_JS_CACHE = get_plotlyjs().encode("utf-8")
    return _PLOTLY_JS_CACHE


# ---------------------------------------------------------------------------
# Background jobs (predict / wind rose) with progress polling
# ---------------------------------------------------------------------------
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
# One heavy computation at a time: jobs queue on this gate instead of racing
# over the GPU / workspace dirs.
_PREDICT_GATE = threading.Semaphore(1)
_JOB_SUBMISSIONS: list[float] = []
_PREP_SUBMISSIONS: list[float] = []
_PREP_ACTIVE = 0


def _env_nonnegative_int(name: str, default: int = 0) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return max(0, default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_mebibytes(name: str, default: int) -> int:
    return max(1, _env_nonnegative_int(name, default)) * 1024 * 1024


def _active_jobs_locked() -> int:
    return sum(j.get("state") in ("queued", "running") for j in _JOBS.values())


def _prep_admit() -> tuple[bool, str, int]:
    """Reserve one synchronous terrain/input-preparation operation."""
    global _PREP_ACTIVE
    max_active = _env_nonnegative_int("PINN_WEBAPP_MAX_ACTIVE_JOBS", 0)
    rate_ops = _env_nonnegative_int("PINN_WEBAPP_RATE_LIMIT_PREP", 0)
    window = _env_nonnegative_int("PINN_WEBAPP_RATE_LIMIT_WINDOW", 3600)
    now = time.time()
    with _JOBS_LOCK:
        if max_active and _active_jobs_locked() + _PREP_ACTIVE >= max_active:
            return False, "Another terrain or prediction operation is already running.", 30
        if rate_ops and window:
            cutoff = now - window
            _PREP_SUBMISSIONS[:] = [t for t in _PREP_SUBMISSIONS if t >= cutoff]
            if len(_PREP_SUBMISSIONS) >= rate_ops:
                retry_after = max(1, int(_PREP_SUBMISSIONS[0] + window - now) + 1)
                return False, "The public terrain-processing limit has been reached. Try again later.", retry_after
            _PREP_SUBMISSIONS.append(now)
        _PREP_ACTIVE += 1
    return True, "", 0


def _prep_release() -> None:
    global _PREP_ACTIVE
    with _JOBS_LOCK:
        _PREP_ACTIVE = max(0, _PREP_ACTIVE - 1)


def _validate_compute_request(body: dict) -> None:
    """Enforce public-app bounds even when browser validation is bypassed."""
    def bounded(name: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(body.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
        return value

    try:
        domain_size = int(body.get("domain_size", 1000))
    except (TypeError, ValueError) as exc:
        raise ValueError("domain_size must be numeric") from exc
    if domain_size not in {500, 1000, 2000, 3000}:
        raise ValueError("domain_size must be one of 500, 1000, 2000 or 3000 m")
    bounded("wind_from", 270.0, 0.0, 360.0)
    bounded("uref", 10.0, 1.0, 30.0)
    bounded("zref", 20.0, 5.0, 100.0)
    bounded("z0", 0.1, 0.001, 2.0)
    if body.get("z_top_offset") is not None:
        bounded("z_top_offset", 300.0, 50.0, 1000.0)

    structures = body.get("structures") or []
    max_single = _env_nonnegative_int("PINN_WEBAPP_MAX_SINGLE_STRUCTURES", 10)
    if not isinstance(structures, list):
        raise ValueError("structures must be a list")
    if max_single and len(structures) > max_single:
        raise ValueError(f"at most {max_single} individual structures are allowed")
    for structure in structures:
        if not isinstance(structure, dict):
            raise ValueError("each structure must be an object")
        try:
            yaw = float(structure.get("yaw", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("structure yaw must be numeric") from exc
        if not math.isfinite(yaw):
            raise ValueError("structure yaw must be finite")

    sampling_points = body.get("sampling_points") or []
    if not isinstance(sampling_points, list) or len(sampling_points) > 10:
        raise ValueError("at most 10 sampling points are allowed")

    grid = body.get("grid")
    if grid is not None:
        if not isinstance(grid, dict):
            raise ValueError("grid must be an object")
        try:
            rows = int(grid.get("rows", 0))
            cols = int(grid.get("cols", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("grid dimensions must be integers") from exc
        if not (1 <= rows <= 10 and 1 <= cols <= 10):
            raise ValueError("structure grids are limited to 10 x 10")
        for key in ("spacing_x", "spacing_y"):
            try:
                spacing = float(grid.get(key, 3.0))
            except (TypeError, ValueError) as exc:
                raise ValueError("grid spacing must be numeric") from exc
            if not math.isfinite(spacing) or not 1.0 <= spacing <= 20.0:
                raise ValueError("grid spacing must be between 1 and 20 m")
        for key in ("grid_yaw", "struct_yaw"):
            try:
                yaw = float(grid.get(key, 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError("grid yaw values must be numeric") from exc
            if not math.isfinite(yaw):
                raise ValueError("grid yaw values must be finite")


def _validate_dem_bounds(body: dict) -> tuple[float, float, float, float]:
    west = float(body["west"])
    south = float(body["south"])
    east = float(body["east"])
    north = float(body["north"])
    vals = (west, south, east, north)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("DEM bounds must be finite")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("invalid WGS84 DEM bounds")
    e_min, n_min, e_max, n_max = wgs84_to_lv95(west, south, east, north)
    max_domain = float(_env_nonnegative_int("PINN_WEBAPP_MAX_DOMAIN_M", 3000))
    # The browser requests a 1.5x DEM footprint so the square model domain
    # remains covered after wind-alignment rotation. Permit up to 1.6x because
    # the browser constructs that box approximately in WGS84 and the projected
    # LV95 extent varies slightly across Switzerland.
    max_extent = 1.6 * max_domain
    if e_max - e_min > max_extent or n_max - n_min > max_extent:
        raise ValueError(
            f"DEM extent exceeds the {max_extent:.0f} m preparation limit "
            f"for a {max_domain:.0f} m model domain"
        )
    return west, south, east, north


def _job_admit_and_create(kind: str) -> tuple[Optional[str], str, int]:
    """Apply process-local cost guards and atomically reserve a heavy job.

    Cloud Run is deployed with one instance, so a global limit is deliberate:
    it cannot be bypassed by changing a client IP header. Values of zero disable
    the corresponding guard. The counters reset when the instance scales down.
    """
    max_active = _env_nonnegative_int("PINN_WEBAPP_MAX_ACTIVE_JOBS", 0)
    rate_jobs = _env_nonnegative_int("PINN_WEBAPP_RATE_LIMIT_JOBS", 0)
    window = _env_nonnegative_int("PINN_WEBAPP_RATE_LIMIT_WINDOW", 3600)
    now = time.time()
    with _JOBS_LOCK:
        active = _active_jobs_locked() + _PREP_ACTIVE
        if max_active and active >= max_active:
            return None, "A prediction is already running. Try again when it finishes.", 30

        if rate_jobs and window:
            cutoff = now - window
            _JOB_SUBMISSIONS[:] = [t for t in _JOB_SUBMISSIONS if t >= cutoff]
            if len(_JOB_SUBMISSIONS) >= rate_jobs:
                retry_after = max(1, int(_JOB_SUBMISSIONS[0] + window - now) + 1)
                return None, "The public prediction limit has been reached. Try again later.", retry_after

        if rate_jobs and window:
            _JOB_SUBMISSIONS.append(now)
        jid = uuid.uuid4().hex[:12]
        _JOBS[jid] = {
            "id": jid, "kind": kind, "state": "queued",
            "message": "queued…", "created": time.time(),
            "result": None, "error": None,
        }
    return jid, "", 0


def _job_update(jid: str, **kw) -> None:
    with _JOBS_LOCK:
        if jid in _JOBS:
            _JOBS[jid].update(kw)


def _job_get(jid: str) -> Optional[dict]:
    with _JOBS_LOCK:
        j = _JOBS.get(jid)
        return dict(j) if j else None


def _job_start(jid: str, fn: Callable[[Callable[[str], None]], dict]) -> None:
    """Run `fn(progress_cb)` on a worker thread, serialised by _PREDICT_GATE."""

    def _progress(msg: str) -> None:
        _job_update(jid, message=str(msg))

    def _worker() -> None:
        with _PREDICT_GATE:
            _job_update(jid, state="running", message="starting…")
            try:
                result = fn(_progress)
                _job_update(jid, state="done", result=result, message="done")
            except Exception as e:  # noqa: BLE001 - job boundary
                traceback.print_exc()
                _job_update(jid, state="error", error=f"{type(e).__name__}: {e}", message="error")

    threading.Thread(target=_worker, daemon=True).start()


def _invalidate_3d_cache(name: str) -> None:
    """Drop cached 3D viewer HTML so a re-predict never serves stale views."""
    base = RESULTS_DIR / name
    if not base.exists():
        return
    for pattern in ("view_3d*.html", "map.html"):
        for p in base.glob(pattern):
            try:
                p.unlink()
            except OSError:
                pass


def _companion_model_id(primary_id: Optional[str]) -> Optional[str]:
    """The 'other family' cascade used for the disagreement/uncertainty map."""
    prim = get_model_entry(primary_id)
    prim_is_hybrid = "hybrid" in prim["id"]
    for m in MODEL_REGISTRY:
        if m.get("kind") != "cascade":
            continue
        if ("hybrid" in m["id"]) != prim_is_hybrid:
            return m["id"]
    return None


# ---------------------------------------------------------------------------
# HTML patching (start from domain_builder.HTML_TEMPLATE, surgically modify)
# ---------------------------------------------------------------------------
def _build_html() -> str:
    html = BUILDER_HTML
    html = html.replace(
        "Domain Builder — pinn_terr_struc",
        "pinnfluid - Wind and pressure prediction",
    )
    _info_link = (
        f'<a href="{_html.escape(GITHUB_URL)}" target="_blank" rel="noopener" '
        'title="Source code, documentation and models on GitHub" '
        'style="font-size:13px; font-weight:normal; text-decoration:none; '
        'vertical-align:middle; margin-left:10px; color:#fff; opacity:0.9;">'
        '&#9432; About / GitHub</a>'
    )
    html = html.replace(
        "<h2>Domain Builder</h2>",
        f"<h2>pinnfluid{_info_link}</h2>", 1,
    )
    html = html.replace(
        "Terrain + structure domain creation for pinn_terr_struc",
        "Interactive wind and pressure prediction over complex terrain and structures.<br>"
        "Found a bug or need another structure or feature? "
        '<a href="mailto:jimmy.gasser@epfl.ch" style="color:#fff;">'
        "jimmy.gasser@epfl.ch</a>",
        1,
    )

    # Step 1: rename the DEM button.
    html = html.replace("Download DEM tiles", "Confirm terrain", 1)

    # Step 2a: Confirm button inside the Single-structures section. The help
    # line after the flattenGround checkbox is a unique anchor for that section.
    # Starts blue (btn-primary = "action needed") and turns green on success.
    html = html.replace(
        '<div class="help">Levels terrain at structure footprint (+2m margin).</div>',
        '<div class="help">Levels terrain at structure footprint (+2m margin).</div>\n'
        '      <button class="btn btn-primary" id="btnConfirmSingle" onclick="confirmStructures()" style="margin-top:10px;">Confirm structure(s)</button>',
        1,
    )

    # Step 2b: Confirm button inside the Structure-grid section, anchored on
    # the "Flatten ground under entire grid" label that's unique to it.
    html = html.replace(
        '<label for="flattenGrid">Flatten ground under entire grid</label>\n      </div>',
        '<label for="flattenGrid">Flatten ground under entire grid</label>\n'
        '      </div>\n'
        '      <button class="btn btn-primary" id="btnConfirmGrid" onclick="confirmStructures()" style="margin-top:10px;">Confirm structure(s)</button>',
        1,
    )

    # Step 3: the original btnBuild becomes the Predict button (now step 6 — the
    # sampling-points section is injected as step 5 below). Keep the id so the
    # template's terrain-ready gating continues to work.
    html = html.replace(
        '<div class="section-title">5. Build domain</div>',
        '<div class="section-title">6. Predict</div>',
        1,
    )
    # Step 5: sampling-points section, injected just before the Build/Predict
    # section so it reads as its own numbered step after the structure steps.
    html = html.replace(
        '<!-- BUILD -->',
        _sampling_section_html() + '\n\n  <!-- BUILD -->',
        1,
    )
    html = html.replace(
        "Build domain (DEM + structures → STL)",
        "Predict flow field",
        1,
    )
    html = html.replace(
        'Creates ground.stl + structure.stl in data_preparation/, shifted to z=0.',
        'If you placed structures, click <b>Confirm structure(s)</b> first. '
        'Otherwise, Predict will build inputs automatically.',
        1,
    )
    html = html.replace(
        'onclick="buildDomain()"',
        'onclick="runPredict()"',
        1,
    )

    # Step 3b: model picker, injected just before the Predict button so it is
    # the last choice the user makes before predicting. Optional — defaults to
    # the recommended model if untouched.
    html = html.replace(
        '<button class="btn btn-success" id="btnBuild"',
        _model_picker_html() + _extra_controls_html()
        + '\n    <button class="btn btn-success" id="btnBuild"',
        1,
    )

    # Give sampling-point placement priority on map clicks when its mode is on,
    # so a click drops a sampling point instead of a structure/grid (and leaves
    # any placed structures/grids untouched). Patched into the builder's own
    # click handler so domain_builder.py stays generic.
    html = html.replace(
        '    // Grid mode: each click replaces the grid position\n'
        '    if (document.getElementById(\'enableGrid\').checked) {',
        '    if (typeof samplingActive === \'function\' && samplingActive()) {\n'
        '      addSamplingAtLatLng(e.latlng.lat, e.latlng.lng);\n'
        '      return;\n'
        '    }\n'
        '    // Grid mode: each click replaces the grid position\n'
        '    if (document.getElementById(\'enableGrid\').checked) {',
        1,
    )

    html = html.replace("</body>", _PREDICT_SCRIPT + _SAMPLING_SCRIPT + "\n</body>", 1)
    return html


def _model_picker_html() -> str:
    """Server-rendered <select> built from the inference model registry."""
    models = list_models()
    default_id = next((m["id"] for m in models if m["default"]), models[0]["id"])
    opts = []
    for m in models:
        sel = " selected" if m["id"] == default_id else ""
        label = m["label"] + (" — default" if m["default"] else "")
        opts.append(
            f'<option value="{m["id"]}"{sel}>{label}</option>'
        )
    desc_map = {m["id"]: m["description"] for m in models}
    desc_json = json.dumps(desc_map)
    default_desc = desc_map.get(default_id, "")
    return (
        '<div class="field" style="margin-bottom:10px;">\n'
        '      <label for="modelSelect">Model</label>\n'
        '      <select id="modelSelect" onchange="onModelChange()" '
        'style="width:100%; padding:7px; border-radius:6px; border:1px solid #ccc;">\n'
        f'        {"".join(opts)}\n'
        '      </select>\n'
        f'      <div class="help" id="modelDesc" style="margin-top:4px;">{default_desc}</div>\n'
        f'      <script>window._MODEL_DESCS = {desc_json};</script>\n'
        '    </div>'
    )


def _extra_controls_html() -> str:
    """Uncertainty + wind-rose options, injected just above the Predict button."""
    max_sectors = max(2, _env_nonnegative_int("PINN_WEBAPP_MAX_ROSE_SECTORS", 16))
    sector_options = [n for n in (4, 8, 12, 16) if n <= max_sectors]
    if not sector_options:
        sector_options = [max_sectors]
    sector_html = "".join(
        f'<option{" selected" if n == min(8, max(sector_options)) else ""}>{n}</option>'
        for n in sector_options
    )
    past_runs = (
        '      <div style="margin-top:8px;"><a href="/runs" target="_blank" '
        'style="font-size:12px;">Past runs</a></div>\n'
        if _env_flag("PINN_WEBAPP_ENABLE_RUN_INDEX", False)
        else ""
    )
    return (
        '<div class="field" style="margin-bottom:10px;">\n'
        '      <label style="display:flex;gap:6px;align-items:center;cursor:pointer;">'
        '<input type="checkbox" id="optUncert" style="width:auto;"> '
        'Uncertainty map (runs both model families, ≈2× time)</label>\n'
        '      <label style="display:flex;gap:6px;align-items:center;cursor:pointer;margin-top:6px;">'
        '<input type="checkbox" id="optRose" style="width:auto;" '
        'onchange="document.getElementById(\'roseOpts\').style.display=this.checked?\'block\':\'none\';"> '
        'Wind rose (multi-direction sweep)</label>\n'
        '      <div id="roseOpts" style="display:none;margin-top:6px;padding-left:22px;">\n'
        '        <label for="roseSectors">Sectors</label>\n'
        '        <select id="roseSectors" style="width:90px;padding:5px;border-radius:6px;border:1px solid #ccc;">\n'
        f'          {sector_html}\n'
        '        </select>\n'
        '        <label for="roseDirs" style="margin-top:6px;">Custom directions '
        '(&deg; wind-from, comma-separated &mdash; overrides sectors)</label>\n'
        '        <input id="roseDirs" type="text" placeholder="e.g. 0, 90, 180" '
        'oninput="document.getElementById(\'roseSectors\').disabled = !!this.value.trim();" '
        'style="width:100%;padding:6px;border-radius:6px;border:1px solid #ccc;">\n'
        '        <div class="help">One full prediction per direction — expect minutes '
        '(structure grids: much longer). Full artifacts (plots, PDF, 3D, exports) '
        'are kept for every direction (~50–100 MB each); a combined '
        'multi-direction PDF and 3D direction selectors are produced at the end. '
        '<b>Structures remain fixed as placed for the Step 1 wind direction; only '
        'the inflow direction changes.</b> Evenly spaced sectors start at the Step 1 '
        'direction. With custom directions, the first valid direction is marked '
        'governing. With evenly spaced sectors, the direction with the highest '
        'near-ground wind speed is governing.</div>\n'
        '      </div>\n'
        + past_runs +
        '    </div>'
    )


def _sampling_section_html() -> str:
    """Optional sampling-point placement (predict app only) as its own numbered
    section between the structures and Predict steps. Mirrors the structure
    section: tick to enable, then click the map or enter CRS coords. Up to 10
    points; each yields a vertical wind-speed profile + a height/|U| table in the
    report and a cross on the terrain plot."""
    return (
        '  <!-- SAMPLING POINTS -->\n'
        '  <div class="section">\n'
        '    <div class="section-title">5. Sampling points (optional)</div>\n'
        '    <div class="checkbox-row">\n'
        '      <input type="checkbox" id="optSampling"/>\n'
        '      <label for="optSampling">Add sampling point(s)</label>\n'
        '    </div>\n'
        '    <div id="samplingOptions" style="display:none;">\n'
        '      <div class="help">While ticked, <b>click the map</b> to drop a sampling point '
        '(after terrain is ready), or enter CRS coords below. Up to 10. Each gives a vertical '
        'wind-speed profile and a height/|U| table in the report; a cross marks it on the '
        'terrain plot.</div>\n'
        '      <div class="row" style="margin-top:6px;">\n'
        '        <div><input type="text" id="sampleE" placeholder="E (LV95)"/></div>\n'
        '        <div><input type="text" id="sampleN" placeholder="N (LV95)"/></div>\n'
        '        <div><button class="btn btn-secondary" onclick="addSamplingManual()" style="margin-top:0">Add</button></div>\n'
        '      </div>\n'
        '      <div id="sampling-list" style="margin-top:4px;"></div>\n'
        '      <button class="btn btn-secondary" onclick="clearSamplingPoints()" style="margin-top:4px">Clear sampling points</button>\n'
        '    </div>\n'
        '  </div>'
    )


_PREDICT_SCRIPT = r"""
<div id="predict-panel" style="
  position:fixed; left:0; top:0; width:calc(100% - 380px); height:100vh;
  background:rgba(255,255,255,0.97); z-index:2000; display:none;
  overflow-y:auto; padding:20px 28px; box-shadow:inset 0 0 24px rgba(0,0,0,0.06);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
    <h2 style="margin:0; font-size:22px; color:#0d47a1;">Wind Flow Prediction Report</h2>
    <button class="btn" style="width:auto; background:#e0e0e0;" onclick="closePredictPanel()">Close</button>
  </div>
  <div id="predict-meta" style="font-size:12px; color:#555; margin-bottom:12px;"></div>
  <div id="predict-actions" style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:18px;">
    <a id="mapBtn"    class="rprt-btn" style="background:#00897b;" href="#" target="_blank" title="Interactive 2D map (Plotly): top-down wind-speed / relative-pressure / terrain heatmap. Hover for exact values, zoom and pan.">🗺️ Interactive map</a>
    <a id="dlPdfBtn"  class="rprt-btn" style="background:#c62828;" href="#">📄 Download PDF report</a>
    <a id="dlVtkBtn"  class="rprt-btn" style="background:#1565c0;" href="#" title="Full 3D flow field as VTK structured grid (open in ParaView).">📦 Export VTK (.vtk.zip)</a>
    <a id="dlNpzBtn"  class="rprt-btn" style="background:#2e7d32;" href="#" title="Full 3D flow field as NumPy compressed arrays.">🧮 Export NumPy (.npz.zip)</a>
    <a id="view3dBtn" class="rprt-btn" style="background:#6a1b9a;" href="#" target="_blank" title="Interactive 3D viewer (Plotly): terrain + structures + wind glyphs + streamtubes.">🌐 Open 3D view</a>
    <a id="view3dStrBtn" class="rprt-btn" style="background:#8e24aa;" href="#" target="_blank" title="3D view focused on the structure(s): structure and ground coloured by relative pressure, with light streamlines.">🏛️ Open 3D structure view</a>
    <a id="dlPlotsBtn" class="rprt-btn" style="background:#37474f;" href="#">🖼️ Plots only (.zip)</a>
  </div>
  <iframe id="predict-map" title="Interactive wind and pressure map"
          style="width:100%; height:72vh; border:1px solid #ddd; border-radius:6px; margin-bottom:8px; display:none;"></iframe>
  <div id="predict-map-hint" style="font-size:11px; color:#888; margin-bottom:16px; display:none;">
    Use the <b>View</b> dropdown (top-left of the map) to switch height above ground, relative pressure, or terrain. Hover for exact values; drag to zoom. Summary numbers and static plots are in the PDF report.
  </div>
  <div id="predict-values-toggle" style="margin:6px 0;">
    <button class="rprt-btn" style="background:#607d8b;" onclick="toggleValues()">Show summary values ▾</button>
  </div>
  <div id="predict-stats" style="display:none;"></div>
  <div id="predict-plot-toggle" style="margin:12px 0 6px 0;">
    <button class="rprt-btn" style="background:#607d8b;" onclick="togglePlots()">Show plots ▾</button>
  </div>
  <div id="predict-plots" style="display:none;"></div>
</div>
<style>
.rprt-btn {
  display:inline-block; padding:8px 14px; color:white; text-decoration:none;
  border-radius:4px; font-size:13px; font-weight:500; border:none; cursor:pointer;
}
.rprt-btn:hover { filter:brightness(1.1); }
#predict-stats table { border-collapse:collapse; font-size:13px; }
#predict-stats th { text-align:left; background:#f5f5f5; padding:8px 12px; border-bottom:2px solid #ddd; font-weight:600; color:#37474f; }
#predict-stats td { padding:8px 12px; border-bottom:1px solid #eee; }
#predict-stats td.num { text-align:right; font-family:'SFMono-Regular',Consolas,monospace; }
#predict-stats h3 { margin:18px 0 6px 0; color:#0d47a1; font-size:15px; }
</style>
<script>
var lastInputsDomain = null;   // most recent domain whose inputs are built
var lastPredictDomain = null;  // most recent domain whose prediction finished

function closePredictPanel() {
  document.getElementById('predict-panel').style.display = 'none';
}

function _fmtLoc(loc) {
  if (!loc) return 'n/a';
  var parts = ['x=' + Math.round(loc.x_m) + ' m',
               'y=' + Math.round(loc.y_m) + ' m',
               'z_rel=' + (loc.z_rel_m != null ? loc.z_rel_m.toFixed(1) : '?') + ' m'];
  if (loc.lat != null && loc.lng != null) {
    parts.push('(' + loc.lat.toFixed(5) + ', ' + loc.lng.toFixed(5) + ')');
  }
  return parts.join(', ');
}
function _row(label, value, loc) {
  var locStr = loc ? _fmtLoc(loc) : '';
  return '<tr><td>'+label+'</td><td class="num">'+value+'</td><td>'+locStr+'</td></tr>';
}
function renderStats(stats, payload) {
  if (!stats || !stats.global) {
    return '<p style="color:#999; font-style:italic;">No stats available.</p>';
  }
  var g = stats.global, abl = stats.abl || {}, rois = stats.rois || {};
  var html = '';

  // Conditions block
  html += '<h3>Inflow conditions</h3>';
  html += '<table>';
  html += '<tr><th style="width:240px;">Quantity</th><th>Value</th></tr>';
  html += '<tr><td>Reference wind speed (Uref)</td><td class="num">'+(abl.Uref_mps||0).toFixed(2)+' m/s</td></tr>';
  html += '<tr><td>Reference height (Zref)</td><td class="num">'+(abl.Zref_m||0).toFixed(1)+' m</td></tr>';
  html += '<tr><td>Roughness length (z₀)</td><td class="num">'+(abl.z0_m||0).toFixed(4)+' m</td></tr>';
  var fd = abl.flowDir || [0,0,0];
  html += '<tr><td>Flow direction (x,y,z)</td><td class="num">('+fd[0].toFixed(2)+', '+fd[1].toFixed(2)+', '+fd[2].toFixed(2)+')</td></tr>';
  var air = stats.air || {};
  if (air.rho_kg_m3) {
    html += '<tr><td>Air density (ρ)</td><td class="num">'+air.rho_kg_m3.toFixed(3)+' kg/m³'
         + ' <span style="color:#888;font-family:inherit;">(ISA at site elevation '
         + (air.site_elevation_m_asl != null ? air.site_elevation_m_asl.toFixed(0) : '?')
         + ' m a.s.l. — used for all Pa values and loads)</span></td></tr>';
  }
  html += '</table>';

  // Wind speed
  html += '<h3>Wind speed</h3>';
  html += '<table>';
  html += '<tr><th style="width:280px;">Quantity</th><th style="width:120px;">Value</th><th>Location</th></tr>';
  if (g.max_umag) {
    html += _row('Max wind speed', g.max_umag.value_mps.toFixed(2)+' m/s', g.max_umag);
  }
  if (g.max_umag_near_ground_z_rel_le_10m) {
    html += _row('Max wind near ground (z<sub>rel</sub> ≤ 10 m)',
                 g.max_umag_near_ground_z_rel_le_10m.value_mps.toFixed(2)+' m/s',
                 g.max_umag_near_ground_z_rel_le_10m);
  }
  if (g.mean_umag_at_zref) {
    var mu = g.mean_umag_at_zref;
    html += '<tr><td>Mean wind at Z<sub>ref</sub>+terrain (z = '+mu.actual_z_m.toFixed(0)+' m)</td>'
         + '<td class="num">'+mu.value_mps.toFixed(2)+' m/s</td><td>—</td></tr>';
  }
  html += '</table>';

  // Relative pressure
  html += '<h3>Relative pressure</h3>';
  html += '<div style="font-size:11px;color:#666;margin:-4px 0 7px;">'
       + 'One pressure reference is used throughout this prediction: the global fluid-domain mean is set to zero.</div>';
  html += '<table>';
  html += '<tr><th style="width:280px;">Quantity</th><th style="width:120px;">Value</th><th>Location</th></tr>';
  if (g.max_p) html += _row('Max relative pressure', g.max_p.value_pa.toFixed(2)+' Pa', g.max_p);
  if (g.min_p) html += _row('Min relative pressure', g.min_p.value_pa.toFixed(2)+' Pa', g.min_p);
  html += '</table>';

  // Domain
  html += '<h3>Global domain</h3>';
  html += '<table>';
  html += '<tr><th style="width:280px;">Quantity</th><th>Value</th></tr>';
  if (g.grid_shape) html += '<tr><td>Grid shape</td><td class="num">'+g.grid_shape.join(' × ')+'</td></tr>';
  if (g.n_fluid_cells != null) html += '<tr><td>Fluid cells</td><td class="num">'+g.n_fluid_cells.toLocaleString()+' ('+(100*g.fluid_fraction).toFixed(1)+'%)</td></tr>';
  if (g.bounds_m) {
    html += '<tr><td>Bounds (x, y, z) [m]</td><td class="num">x∈['+g.bounds_m.x.map(function(v){return v.toFixed(0);}).join(',')+'] y∈['+g.bounds_m.y.map(function(v){return v.toFixed(0);}).join(',')+'] z∈['+g.bounds_m.z.map(function(v){return v.toFixed(0);}).join(',')+']</td></tr>';
  }
  if (g.terrain_elev_range_m) html += '<tr><td>Terrain elevation [m]</td><td class="num">'+g.terrain_elev_range_m[0].toFixed(0)+' .. '+g.terrain_elev_range_m[1].toFixed(0)+'</td></tr>';
  html += '</table>';

  // ROIs (per-structure, only if present)
  var roiKeys = Object.keys(rois);
  if (roiKeys.length > 0) {
    html += '<h3>Region(s) of interest (around structures)</h3>';
    roiKeys.forEach(function(k) {
      var r = rois[k];
      html += '<h4 style="margin:10px 0 4px 0; color:#37474f; font-size:13px;">'+k+'</h4>';
      html += '<table>';
      html += '<tr><th style="width:280px;">Quantity</th><th style="width:120px;">Value</th><th>Location</th></tr>';
      if (r.grid_shape) html += '<tr><td>Grid</td><td class="num">'+r.grid_shape.join(' × ')+'</td><td>'+(r.n_fluid_cells||0).toLocaleString()+' fluid cells</td></tr>';
      if (r.max_umag) html += _row('Max wind speed', r.max_umag.value_mps.toFixed(2)+' m/s', r.max_umag);
      if (r.max_p) html += _row('Max relative pressure', r.max_p.value_pa.toFixed(2)+' Pa', r.max_p);
      if (r.min_p) html += _row('Min relative pressure', r.min_p.value_pa.toFixed(2)+' Pa', r.min_p);
      if (r.max_p_near_wall) html += _row('Max relative pressure on structure surface',
                                          r.max_p_near_wall.value_pa.toFixed(2)+' Pa',
                                          r.max_p_near_wall);
      if (r.p_near_wall_range_pa) {
        html += '<tr><td>Relative pressure on structure surface (range)</td><td class="num">'
             + r.p_near_wall_range_pa.min.toFixed(2)+' .. '+r.p_near_wall_range_pa.max.toFixed(2)+' Pa</td><td>—</td></tr>';
      }
      html += '</table>';

      // Per-structure estimated wind loads (forces integrated over AABB face
      // mean pressures). PDF report carries the same numbers in a wider
      // formatted table; here we show a compact inline version.
      var forces = r.per_structure_forces || [];
      if (forces.length > 0) {
        html += '<h4 style="margin:10px 0 4px 0; color:#37474f; font-size:13px;">Experimental wind-load estimates (' + forces.length + ' structure' + (forces.length>1?'s':'') + ')</h4>';
        html += '<table>';
        html += '<tr><th style="width:36px;">#</th><th style="width:90px;">Fx local [N]</th><th style="width:90px;">Fy local [N]</th><th style="width:90px;">Fz [N]</th><th style="width:90px;">|F_drag| [N]</th><th style="width:80px;">Cd</th><th>Frontal A [m²]</th></tr>';
        var _fmt = function(v, digits) {
          return (v != null && isFinite(v)) ? Number(v).toFixed(digits) : '—';
        };
        forces.forEach(function(f, i) {
          var F = f.F_N || [null, null, null];
          var sidx = (f.structure_index != null) ? f.structure_index : (i + 1);
          html += '<tr>'
               + '<td class="num" style="font-weight:600; color:#37474f;">'+sidx+'</td>'
               + '<td class="num">'+_fmt(F[0], 1)+'</td>'
               + '<td class="num">'+_fmt(F[1], 1)+'</td>'
               + '<td class="num">'+_fmt(F[2], 1)+'</td>'
               + '<td class="num">'+_fmt(f.F_drag_N != null ? Math.abs(f.F_drag_N) : null, 1)+'</td>'
               + '<td class="num">'+_fmt(f.Cd, 2)+'</td>'
               + '<td class="num">'+_fmt(f.frontal_area_m2, 1)+'</td>'
               + '</tr>';
        });
        html += '</table>';
        var meth = (forces[0] && forces[0].method === 'surface_integration')
          ? 'Surface-pressure integration over the structure mesh.'
          : 'First-order estimate from AABB face-mean pressures.';
        var rhoStr = (air.rho_kg_m3 ? air.rho_kg_m3.toFixed(3) : '1.225');
        html += '<div style="font-size:11px; color:#888; margin-top:4px;">'
              + meth + ' Cd = F<sub>drag</sub> / (½·ρ·U<sub>ref</sub>²·A<sub>frontal</sub>). '
              + 'Fx/Fy use the wind-aligned frame for the current direction. Forces use ρ = '
              + rhoStr + ' kg/m³ (ISA at site elevation). Experimental pressure-only screening estimate; '
              + 'not a validated design load.</div>';
      }
    });
  }

  // Sampling points: wind speed |U| at standard heights above ground.
  var sps = stats.sampling_points || [];
  if (sps.length > 0) {
    html += '<h3>Sampling points</h3>';
    // Heights are shared across points; derive the column list from the first
    // in-domain point that has a profile.
    var hsrc = null;
    sps.forEach(function(sp){ if (!hsrc && sp.in_domain && sp.heights && sp.heights.length) hsrc = sp.heights; });
    if (hsrc) {
      html += '<table><tr><th style="width:120px;">z above ground</th>';
      sps.forEach(function(sp){ html += '<th>'+(sp.label||'')+' &nbsp;|U| [m/s]</th>'; });
      html += '</tr>';
      hsrc.forEach(function(h, ri){
        html += '<tr><td>'+h.z_rel_m.toFixed(0)+' m</td>';
        sps.forEach(function(sp){
          var u = (sp.in_domain && sp.heights && sp.heights[ri]) ? sp.heights[ri].u_mps : null;
          html += '<td class="num">'+(u!=null ? u.toFixed(2) : '—')+'</td>';
        });
        html += '</tr>';
      });
      // Column-max row.
      html += '<tr><td style="font-weight:600;">column max</td>';
      sps.forEach(function(sp){
        var v = (sp.in_domain && sp.col_max_u_mps!=null)
          ? sp.col_max_u_mps.toFixed(2)+' <span style="color:#888;">@'+(sp.col_max_u_zrel_m!=null?sp.col_max_u_zrel_m.toFixed(0):'?')+'m</span>'
          : '—';
        html += '<td class="num">'+v+'</td>';
      });
      html += '</tr></table>';
    }
    // Per-point locations + any out-of-domain notes.
    var locLines = sps.map(function(sp){
      var bits = [];
      if (sp.x_m != null) bits.push('x='+Math.round(sp.x_m)+' m, y='+Math.round(sp.y_m||0)+' m');
      if (sp.lat != null && sp.lng != null) bits.push('('+sp.lat.toFixed(5)+', '+sp.lng.toFixed(5)+')');
      if (sp.in_domain === false) bits.push('<span style="color:#c62828;">outside domain</span>');
      return '<b>'+(sp.label||'')+'</b>: '+bits.join(' · ');
    });
    html += '<div style="font-size:11px; color:#666; margin-top:4px;">'+locLines.join('<br>')+'</div>';
  }

  return html;
}

function renderPlotsSection(payload, container) {
  container.innerHTML = '';
  var keys = payload.plot_order || Object.keys(payload.plots || {});
  keys.forEach(function(key) {
    if (!payload.plots || !payload.plots[key]) return;
    var item = payload.plots[key];
    var wrap = document.createElement('div');
    wrap.style.margin = '14px 0 28px 0';
    var h = document.createElement('h3');
    h.textContent = item.title || key.replace(/_/g,' ');
    h.style.margin = '0 0 6px 0';
    h.style.fontSize = '14px';
    h.style.color = '#333';
    wrap.appendChild(h);
    var img = document.createElement('img');
    img.src = 'data:image/png;base64,' + item.png_base64;
    img.style.maxWidth = '100%';
    img.style.border = '1px solid #ddd';
    img.style.borderRadius = '4px';
    wrap.appendChild(img);
    container.appendChild(wrap);
  });
}

function togglePlots() {
  var p = document.getElementById('predict-plots');
  var btn = document.querySelector('#predict-plot-toggle button');
  if (p.style.display === 'none') {
    p.style.display = 'block';
    btn.innerHTML = 'Hide plots ▴';
  } else {
    p.style.display = 'none';
    btn.innerHTML = 'Show plots ▾';
  }
}

function renderReport(payload) {
  var panel = document.getElementById('predict-panel');
  var meta  = document.getElementById('predict-meta');
  var statsDiv = document.getElementById('predict-stats');
  var plots = document.getElementById('predict-plots');
  var stats = payload.stats || {};
  var domain = payload.domain_name || '';
  var modelName = (stats && stats.model_name) || 'best.pth';

  var lines = [];
  lines.push('Domain: <b>'+domain+'</b>');
  lines.push('Model: <code>'+modelName+'</code>');
  if (payload.grid_shape) lines.push('Grid: '+payload.grid_shape.join(' × '));
  if (payload.elapsed_s != null) lines.push('Total time: '+payload.elapsed_s.toFixed(1)+' s');
  meta.innerHTML = lines.join(' &nbsp;·&nbsp; ');

  // Buttons (always wired; download endpoints check existence on the server)
  document.getElementById('dlPdfBtn').href   = '/download_pdf?domain=' + encodeURIComponent(domain);
  document.getElementById('dlVtkBtn').href   = '/download_vtk?domain=' + encodeURIComponent(domain);
  document.getElementById('dlNpzBtn').href   = '/download_npz?domain=' + encodeURIComponent(domain);
  document.getElementById('dlPlotsBtn').href = '/download_zip?domain=' + encodeURIComponent(domain);
  // 3D view: always available — endpoint generates on first hit (lazy).
  document.getElementById('mapBtn').href = '/map?domain=' + encodeURIComponent(domain);
  document.getElementById('view3dBtn').href = '/view_3d?domain=' + encodeURIComponent(domain);
  // Structure view only makes sense when the case has at least one structure.
  var nStr = parseInt((payload && payload.n_structures) || 0);
  var hasStruct = nStr > 0;
  var strBtn = document.getElementById('view3dStrBtn');
  strBtn.href = '/view_3d_structure?domain=' + encodeURIComponent(domain);
  strBtn.style.display = hasStruct ? '' : 'none';

  // Primary view: the interactive map, embedded as the first result.
  var mapFrame = document.getElementById('predict-map');
  mapFrame.src = '/map?domain=' + encodeURIComponent(domain);
  mapFrame.style.display = 'block';
  document.getElementById('predict-map-hint').style.display = 'block';

  // Summary values + static plots stay collapsed (they live in the PDF report).
  statsDiv.innerHTML = renderStats(stats, payload);
  statsDiv.style.display = 'none';
  document.querySelector('#predict-values-toggle button').innerHTML = 'Show summary values ▾';
  renderPlotsSection(payload, plots);
  plots.style.display = 'none';
  document.querySelector('#predict-plot-toggle button').innerHTML = 'Show plots ▾';

  panel.style.display = 'block';
}

function toggleValues() {
  var s = document.getElementById('predict-stats');
  var b = document.querySelector('#predict-values-toggle button');
  if (s.style.display === 'none') { s.style.display = 'block'; b.innerHTML = 'Hide summary values ▴'; }
  else { s.style.display = 'none'; b.innerHTML = 'Show summary values ▾'; }
}

function _collectBody() {
  var name = document.getElementById('domainName').value.trim();
  var domSize = parseInt(document.getElementById('domainSize').value);
  var windFrom = parseInt(document.getElementById('windFrom').value);
  var flat = document.getElementById('flatTerrain').checked;
  var structData = structures.map(function(s) {
    return {lat:s.lat, lng:s.lng, type:s.type, yaw:s.yaw,
            crs_x:s.crs_x, crs_y:s.crs_y};
  });
  var uref = parseFloat(document.getElementById('uref').value) || 10;
  var zref = parseFloat(document.getElementById('zref').value) || 20;
  var z0 = parseFloat(document.getElementById('z0').value) || 0.1;
  var flattenGround = document.getElementById('flattenGround').checked;
  var samplingData = (typeof samplingPoints !== 'undefined' ? samplingPoints : []).map(function(s) {
    return {lat:s.lat, lng:s.lng, crs_x:s.crs_x, crs_y:s.crs_y, label:s.label};
  });
  return {
    domain_name: name,
    domain_size: domSize,
    wind_from: windFrom,
    flat_terrain: flat,
    structures: structData,
    sampling_points: samplingData,
    center: terrainCenter,
    uref: uref, zref: zref, z0: z0,
    flatten_ground: flattenGround,
    z_top_offset: document.getElementById('customZoffset').checked ?
      parseFloat(document.getElementById('zOffsetVal').value) : null,
    grid: document.getElementById('enableGrid').checked ? {
      type: document.getElementById('gridStructType').value,
      rows: parseInt(document.getElementById('gridRows').value),
      cols: parseInt(document.getElementById('gridCols').value),
      spacing_x: parseFloat(document.getElementById('gridSpacingX').value),
      spacing_y: parseFloat(document.getElementById('gridSpacingY').value),
      struct_yaw: parseFloat(document.getElementById('gridStructYaw').value) || 0,
      grid_yaw: parseFloat(document.getElementById('gridYaw').value) || 0,
      center: gridCenter,
      flatten: document.getElementById('flattenGrid').checked
    } : null
  };
}

function _getConfirmBtns() {
  return [document.getElementById('btnConfirmSingle'),
          document.getElementById('btnConfirmGrid')].filter(Boolean);
}

function _setConfirmBtnsDisabled(d) {
  _getConfirmBtns().forEach(function(b){ b.disabled = d; });
}

function _setConfirmBtnsState(state) {
  // state: 'pending' (blue) or 'ready' (green)
  _getConfirmBtns().forEach(function(b){
    if (state === 'ready') {
      b.classList.remove('btn-primary'); b.classList.add('btn-success');
    } else {
      b.classList.remove('btn-success'); b.classList.add('btn-primary');
    }
  });
}

// Step 2: build STLs + binary inputs. Triggered explicitly from the Confirm
// buttons inside sections 3/4.
function confirmStructures() {
  var body = _collectBody();
  if (!body.domain_name) { setStatus('Enter a domain name','err'); return; }
  setStatus('Confirming structure(s) — building STLs and model inputs…','busy');
  _setConfirmBtnsDisabled(true);
  document.getElementById('btnBuild').disabled = true;
  fetch('/build_inputs', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  }).then(r=>r.json()).then(function(d) {
    _setConfirmBtnsDisabled(false);
    if (d.success) {
      lastInputsDomain = body.domain_name;
      document.getElementById('btnBuild').disabled = false;
      _setConfirmBtnsState('ready');   // confirm button → green
      var gs = (d.grid_shape || []).join(' × ');
      setStatus('Structures confirmed. Inputs ready ('+gs+'). Now click Predict.','ok');
    } else {
      _setConfirmBtnsState('pending');
      setStatus('Build-inputs error: ' + d.error, 'err');
    }
  }).catch(function(e) {
    _setConfirmBtnsDisabled(false);
    _setConfirmBtnsState('pending');
    setStatus('Build-inputs error: '+e, 'err');
  });
}

// Step 3: Predict. If structures are ticked and inputs aren't built yet,
// refuse and tell the user to Confirm first. If no structures ticked, build
// inputs implicitly then predict (single chained call — cleaner UX).
function onModelChange() {
  var sel = document.getElementById('modelSelect');
  var d = document.getElementById('modelDesc');
  if (sel && d && window._MODEL_DESCS) { d.textContent = window._MODEL_DESCS[sel.value] || ''; }
}

function _pollJob(jobId, onDone) {
  var timer = setInterval(function() {
    fetch('/job_status?id=' + encodeURIComponent(jobId)).then(r=>r.json()).then(function(j) {
      if (j.state === 'queued' || j.state === 'running') {
        setStatus(j.message || 'working…', 'busy');
        return;
      }
      clearInterval(timer);
      document.getElementById('btnBuild').disabled = false;
      if (j.state === 'done' && j.result && j.result.success) {
        onDone(j.result);
      } else {
        setStatus('Error: ' + (j.error || (j.result && j.result.error) || 'unknown'), 'err');
      }
    }).catch(function(e) {
      clearInterval(timer);
      document.getElementById('btnBuild').disabled = false;
      setStatus('Error: ' + e, 'err');
    });
  }, 1500);
}

function renderRose(payload) {
  var panel = document.getElementById('predict-panel');
  var meta  = document.getElementById('predict-meta');
  var statsDiv = document.getElementById('predict-stats');
  var plots = document.getElementById('predict-plots');
  var domain = payload.domain_name || '';
  var worst = payload.worst_domain || '';

  var lines = [];
  lines.push('Domain: <b>'+domain+'</b>');
  lines.push('Wind rose: '+ (payload.sectors||[]).length + ' sectors');
  lines.push('Model: <code>'+(payload.model_name||'')+'</code>');
  if (payload.geometry_reference_dir_deg != null) {
    lines.push('Fixed-layout reference: '+payload.geometry_reference_dir_deg.toFixed(0)+'°');
  }
  if (payload.elapsed_s != null) lines.push('Total time: '+payload.elapsed_s.toFixed(1)+' s');
  meta.innerHTML = lines.join(' &nbsp;·&nbsp; ');

  // Combined multi-direction PDF + 3D wrappers (direction selector) live at
  // the BASE domain; raw VTK/NPZ exports default to the governing direction.
  document.getElementById('dlPdfBtn').href   = '/download_pdf?domain=' + encodeURIComponent(domain);
  document.getElementById('dlVtkBtn').href   = '/download_vtk?domain=' + encodeURIComponent(worst);
  document.getElementById('dlNpzBtn').href   = '/download_npz?domain=' + encodeURIComponent(worst);
  document.getElementById('dlPlotsBtn').href = '/download_zip?domain=' + encodeURIComponent(worst);
  document.getElementById('mapBtn').href = '/map_rose?domain=' + encodeURIComponent(domain);
  document.getElementById('view3dBtn').href  = '/view_3d_rose?domain=' + encodeURIComponent(domain);
  var strBtn = document.getElementById('view3dStrBtn');
  strBtn.href = '/view_3d_structure_rose?domain=' + encodeURIComponent(domain);
  strBtn.style.display = (payload.n_structures > 0) ? '' : 'none';

  var html = '';
  if (payload.rose_png_base64) {
    html += '<div style="margin:8px 0 16px 0;"><img style="max-width:720px;width:100%;border:1px solid #ddd;border-radius:4px;" src="data:image/png;base64,' + payload.rose_png_base64 + '"></div>';
  }
  html += '<div style="margin:6px 0 12px;padding:8px 10px;background:#eef5fb;border-left:3px solid #1976d2;font-size:12px;">'
       + 'The terrain frame is rotated for each inflow, but all structures retain the physical orientation shown for the Step 1 wind direction ('
       + (payload.geometry_reference_dir_deg != null ? payload.geometry_reference_dir_deg.toFixed(0)+'°' : 'reference') + '). '
       + (payload.governing_rule || '') + '</div>';
  html += '<h3>Per-sector summary</h3>';
  html += '<table><tr><th>Wind from [°]</th><th>Max |U| near ground [m/s]</th><th>Mean |U| at Z<sub>ref</sub> [m/s]</th><th>Max experimental |F<sub>drag</sub>| [N]</th><th>Max relative suction [Pa]</th><th></th></tr>';
  (payload.sectors || []).forEach(function(s) {
    var isWorst = (s.domain === worst);
    var dq = encodeURIComponent(s.domain);
    var links = '<a href="/download_pdf?domain='+dq+'" target="_blank">PDF</a> · '
              + '<a href="/map?domain='+dq+'" target="_blank">map</a> · '
              + '<a href="/view_3d?domain='+dq+'" target="_blank">3D</a> · '
              + '<a href="/download_zip?domain='+dq+'">plots</a>'
              + (isWorst ? ' <b style="color:#e65100;">governing</b>' : '');
    html += '<tr'+(isWorst?' style="background:#fff3e0;font-weight:600;"':'')+'>'
         + '<td class="num">'+s.dir_deg.toFixed(0)+'</td>'
         + '<td class="num">'+(s.max_u_near_ground!=null?s.max_u_near_ground.toFixed(2):'—')+'</td>'
         + '<td class="num">'+(s.mean_u_zref!=null?s.mean_u_zref.toFixed(2):'—')+'</td>'
         + '<td class="num">'+(s.max_drag_N!=null?Math.abs(s.max_drag_N).toFixed(1):'—')+'</td>'
         + '<td class="num">'+(s.max_suction_pa!=null?s.max_suction_pa.toFixed(1):'—')+'</td>'
         + '<td>'+links+'</td>'
         + '</tr>';
  });
  html += '</table>';
  html += '<div style="font-size:11px;color:#888;margin-top:6px;">Each sector uses its own pressure reference, with its global fluid-domain mean set to zero. Full artifacts are kept for every direction (~50–100 MB each). The PDF button above is the combined multi-direction report; per-direction PDFs are in the table.</div>';
  statsDiv.innerHTML = html;
  statsDiv.style.display = 'block';
  document.querySelector('#predict-values-toggle button').innerHTML = 'Hide summary values ▴';
  var mapFrame = document.getElementById('predict-map');
  mapFrame.removeAttribute('src');
  mapFrame.style.display = 'none';
  document.getElementById('predict-map-hint').style.display = 'none';
  renderPlotsSection(payload, plots);
  plots.style.display = 'none';
  document.querySelector('#predict-plot-toggle button').innerHTML = 'Show plots ▾';
  panel.style.display = 'block';
}

function runPredict() {
  var body = _collectBody();
  var msel = document.getElementById('modelSelect');
  if (msel) { body.model = msel.value; }
  body.uncertainty = !!(document.getElementById('optUncert') && document.getElementById('optUncert').checked);
  var roseOn = !!(document.getElementById('optRose') && document.getElementById('optRose').checked);
  if (roseOn) {
    body.rose_sectors = parseInt(document.getElementById('roseSectors').value) || 8;
    var customDirs = (document.getElementById('roseDirs') || {value:''}).value.trim();
    if (customDirs) { body.rose_directions = customDirs; }
  }
  if (!body.domain_name) { setStatus('Enter a domain name','err'); return; }
  var singleOn = document.getElementById('enableSingle').checked;
  var gridOn   = document.getElementById('enableGrid').checked;
  var structsOn = singleOn || gridOn;
  var inputsReady = (lastInputsDomain === body.domain_name);

  if (structsOn && !inputsReady && !roseOn) {
    setStatus('Click Confirm structure(s) first.','err');
    return;
  }

  document.getElementById('btnBuild').disabled = true;
  var endpoint = roseOn ? '/predict_rose' : '/predict';
  setStatus(roseOn ? 'Starting wind-rose sweep…' : 'Starting prediction…', 'busy');
  fetch(endpoint, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  }).then(r=>r.json()).then(function(d) {
    if (!d.job_id) {
      document.getElementById('btnBuild').disabled = false;
      setStatus('Error: ' + (d.error || 'no job id'), 'err');
      return;
    }
    _pollJob(d.job_id, function(result) {
      lastPredictDomain = body.domain_name;
      setStatus('Done (' + (result.elapsed_s||0).toFixed(1) + ' s).', 'ok');
      if (roseOn) { renderRose(result); } else { renderReport(result); }
    });
  }).catch(function(e) {
    document.getElementById('btnBuild').disabled = false;
    setStatus('Error: '+e, 'err');
  });
}
</script>
"""


# Sampling-point placement JS (predict app only). Reuses the builder globals
# (map, _latlngToCrs/_crsToLatlng, customDemActive, terrainLocked, setStatus).
# The builder's map-click handler is patched in _build_html to call
# addSamplingAtLatLng first when sampling mode is active, so a click drops a
# sampling point without also placing a structure (and structures/grids stay
# intact — sampling mode is non-destructive).
_SAMPLING_SCRIPT = r"""
<script>
var samplingPoints = [];            // {lat, lng, crs_x, crs_y, label}
var MAX_SAMPLING_POINTS = 10;
var samplingMarkers = L.layerGroup().addTo(map);
var samplingIcon = L.divIcon({
  className:'',
  html:'<div style="font-size:20px;line-height:14px;color:#00e5ff;'
     + 'text-shadow:0 0 2px #01303a,0 0 3px #01303a;font-weight:bold;">&#10010;</div>',
  iconSize:[16,16], iconAnchor:[8,8]
});

function samplingActive() {
  var el = document.getElementById('optSampling');
  return !!(el && el.checked);
}

function addSamplingAtLatLng(lat, lng) {
  if (samplingPoints.length >= MAX_SAMPLING_POINTS) {
    setStatus('Max ' + MAX_SAMPLING_POINTS + ' sampling points reached.', 'err');
    return;
  }
  var crs = (typeof _latlngToCrs === 'function') ? _latlngToCrs(lat, lng) : null;
  var label = 'SP' + (samplingPoints.length + 1);
  var entry = {lat:lat, lng:lng, label:label};
  if (crs) { entry.crs_x = crs[0]; entry.crs_y = crs[1]; }
  samplingPoints.push(entry);
  renderSamplingMarkers();
  renderSamplingList();
  var locStr = crs ? 'CRS=(' + crs[0].toFixed(0) + ', ' + crs[1].toFixed(0) + ')'
                   : '(' + lat.toFixed(5) + ', ' + lng.toFixed(5) + ')';
  setStatus('Sampling point ' + label + ' at ' + locStr, 'ok');
}

function addSamplingManual() {
  var e = parseFloat(document.getElementById('sampleE').value);
  var n = parseFloat(document.getElementById('sampleN').value);
  if (isNaN(e) || isNaN(n)) { setStatus('Enter valid E/N coordinates','err'); return; }
  if (typeof customDemActive === 'function' && customDemActive()) {
    var ll = (typeof _crsToLatlng === 'function') ? _crsToLatlng(e, n) : null;
    if (!ll) { setStatus('Custom DEM not loaded','err'); return; }
    addSamplingAtLatLng(ll[0], ll[1]);
    return;
  }
  fetch('/lv95_to_wgs84?e='+e+'&n='+n).then(r=>r.json()).then(function(d) {
    if (d.lat && d.lng) { addSamplingAtLatLng(d.lat, d.lng); }
    else { setStatus('Coordinate conversion failed','err'); }
  });
}

function clearSamplingPoints() {
  samplingPoints = [];
  samplingMarkers.clearLayers();
  renderSamplingList();
}

function removeSampling(i) {
  samplingPoints.splice(i, 1);
  samplingPoints.forEach(function(s, idx){ s.label = 'SP' + (idx + 1); });
  renderSamplingMarkers();
  renderSamplingList();
}

function renderSamplingMarkers() {
  samplingMarkers.clearLayers();
  samplingPoints.forEach(function(s) {
    samplingMarkers.addLayer(
      L.marker([s.lat, s.lng], {icon:samplingIcon})
        .bindPopup(s.label + '<br>(' + s.lat.toFixed(5) + ', ' + s.lng.toFixed(5) + ')')
    );
  });
}

function renderSamplingList() {
  var html = '';
  samplingPoints.forEach(function(s, i) {
    html += '<div class="struct-item"><span>' + s.label +
      '<br><small>lat=' + s.lat.toFixed(5) + ' lng=' + s.lng.toFixed(5) + '</small></span>' +
      '<span class="remove" onclick="removeSampling(' + i + ')">&times;</span></div>';
  });
  var el = document.getElementById('sampling-list');
  if (el) el.innerHTML = html || '<div style="color:#999; font-size:11px;">No sampling points yet</div>';
}

// Toggle only shows/hides the panel. Placement priority is handled in the
// patched map-click handler, so this is non-destructive to structures/grids.
(function(){
  var opt = document.getElementById('optSampling');
  if (opt) opt.addEventListener('change', function() {
    document.getElementById('samplingOptions').style.display = this.checked ? 'block' : 'none';
    if (this.checked) {
      setStatus('Sampling mode on — click the map to drop points (structures are kept).', '');
    }
  });
})();
</script>
"""


# ---------------------------------------------------------------------------
# Pipeline stages (one per HTTP endpoint)
# ---------------------------------------------------------------------------
def _coerce_body_for_build(body: dict) -> dict:
    """Whitelist + cast fields shared by /build_inputs and /predict."""
    return dict(
        domain_name=str(body["domain_name"]).strip(),
        domain_size=float(body["domain_size"]),
        wind_from=float(body["wind_from"]),
        flat_terrain=bool(body.get("flat_terrain", False)),
        structures=list(body.get("structures", []) or []),
        center_latlng=body.get("center"),
        uref=float(body.get("uref", 10.0)),
        zref=float(body.get("zref", 20.0)),
        z0=float(body.get("z0", 0.1)),
        flatten_ground=bool(body.get("flatten_ground", True)),
        grid=body.get("grid"),
        z_top_offset=body.get("z_top_offset"),
    )


def _body_signature(body: dict) -> str:
    """Stable hash of every build-relevant field of the request body.

    Stored next to the built inputs; /predict compares it so a changed wind
    direction / Uref / structure list after Confirm can never be silently
    predicted with stale inputs — the inputs are rebuilt instead.
    """
    kw = _coerce_body_for_build(body)
    blob = json.dumps(kw, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _run_build_inputs(body: dict) -> dict:
    t0 = time.time()
    kw = _coerce_body_for_build(body)
    name = _safe_name(kw["domain_name"])
    kw["domain_name"] = name
    # Fresh workspace for this domain — but keep the DEM that Confirm terrain
    # just downloaded into dem/<name>/; build_domain needs dem_cropped.tif.
    try:
        cleanup_case(name, keep_dem=True)
    except Exception:
        pass
    inputs = build_inputs(**kw)
    # Record what these inputs were built from (see _body_signature).
    try:
        (Path(inputs["cfd_dir"]) / "build_signature.txt").write_text(_body_signature(body))
    except Exception:
        pass
    # Peek at the exported grid_shape so the UI can show it.
    import json as _json
    meta_path = Path(inputs["cfd_dir"]) / "meta.json"
    grid_shape = _json.loads(meta_path.read_text())["grid_shape"] if meta_path.exists() else []
    return {
        "success": True,
        "domain_name": name,
        "case_dir": str(inputs["case_dir"]),
        "cfd_dir": str(inputs["cfd_dir"]),
        "grid_shape": list(grid_shape),
        "elapsed_s": float(time.time() - t0),
    }


def _model_name(model_id: Optional[str] = None) -> str:
    """Human-readable model identifier shown in the report panel."""
    try:
        entry = get_model_entry(model_id)
        if entry.get("kind") == "cascade":
            fname = Path(entry["stage2"]["file"]).name
        else:
            fname = Path(entry.get("file", "?")).name
        return f'{entry["label"]} [{fname}]'
    except Exception:
        pass
    return "unknown-model"


def _exports_state(name: str) -> dict:
    """Indicate which on-demand exports are reachable for this domain."""
    base = RESULTS_DIR / name
    return {
        "pdf": (base / "report.pdf").exists(),
        "vtk_zip": has_saved_inputs(RESULTS_DIR, name),
        "npz_zip": has_saved_inputs(RESULTS_DIR, name),
        "view_3d_html": (base / "view_3d.html").exists(),
    }


_MAX_SAMPLING_POINTS = 10


def _sampling_points_to_local(body: dict, transform_meta: Optional[dict]) -> list:
    """Project user-placed sampling points into the domain-local (x, y) frame.

    The browser sends sampling points with lat/lng and, in custom-DEM mode,
    CRS coords (`crs_x`/`crs_y`, the DEM CRS). On the Swiss path map clicks do
    NOT carry CRS (the client's _latlngToCrs only works for custom DEMs), so we
    convert lat/lng → LV95 here, exactly like structure placement does. The flow
    grid lives in a local frame centred on the transform's `pivot_xy`, rotated by
    `theta_math_deg`, spanning [0, final_W] x [0, final_H]. This is the inverse
    of report._local_to_lv95. Points with no usable coords or no transform get no
    local coords (they are later treated as out-of-domain). Capped at
    _MAX_SAMPLING_POINTS.
    """
    import math
    pts = (body.get("sampling_points") or [])[:_MAX_SAMPLING_POINTS]
    pivot = (transform_meta or {}).get("pivot_xy")
    theta = math.radians(float((transform_meta or {}).get("theta_math_deg", 0.0) or 0.0))
    ds = (transform_meta or {}).get("domain_size") or []
    W = float((transform_meta or {}).get("final_W", ds[0] if len(ds) > 0 else 0.0) or 0.0)
    H = float((transform_meta or {}).get("final_H", ds[1] if len(ds) > 1 else 0.0) or 0.0)
    out = []
    for idx, sp in enumerate(pts):
        if not isinstance(sp, dict):
            continue
        rec = {
            "label": str(sp.get("label") or f"SP{idx + 1}"),
            "lat": sp.get("lat"),
            "lng": sp.get("lng"),
            "crs_x": sp.get("crs_x"),
            "crs_y": sp.get("crs_y"),
        }
        e, n = sp.get("crs_x"), sp.get("crs_y")
        # Swiss path: no client CRS → derive LV95 from lat/lng (same as structures).
        if (e is None or n is None) and sp.get("lat") is not None and sp.get("lng") is not None:
            try:
                e, n = wgs84_point_to_lv95(float(sp["lng"]), float(sp["lat"]))
                rec["crs_x"], rec["crs_y"] = float(e), float(n)
            except Exception:
                e, n = None, None
        if pivot and len(pivot) >= 2 and e is not None and n is not None:
            de = float(e) - float(pivot[0])
            dn = float(n) - float(pivot[1])
            rec["x"] = de * math.cos(theta) + dn * math.sin(theta) + 0.5 * W
            rec["y"] = -de * math.sin(theta) + dn * math.cos(theta) + 0.5 * H
        out.append(rec)
    return out


def _canonical_plot_order_names() -> list:
    """Canonical plot-filename order shared by the web panel and PDF report:
    terrain → global → ROI → sampling → extras (uncertainty)."""
    from plots import _GLOBAL_PLOTS, _ROI_PLOTS  # type: ignore
    names = [fname for (_k, _fn, fname, _t) in _GLOBAL_PLOTS]
    names += [fname for (_k, _fn, fname, _t) in _ROI_PLOTS]
    names += ["sampling_profiles.png", "uncertainty_global.png", "uncertainty_roi.png"]
    return names


def _order_plot_paths(paths) -> list:
    """Sort plot PNG paths into the canonical display order; unknown filenames
    sort last (alphabetically). Used wherever plots are collected from a glob
    (e.g. the wind-rose per-direction sets) so the order matches the live panel."""
    order = _canonical_plot_order_names()
    idx = {n: i for i, n in enumerate(order)}
    return sorted(paths, key=lambda p: (idx.get(Path(p).name, len(order)), Path(p).name))


def _run_predict(body: dict, progress: Optional[Callable[[str], None]] = None) -> dict:
    t0 = time.time()

    def _p(msg: str) -> None:
        if progress:
            progress(msg)

    name = _safe_name(body.get("domain_name"))
    body = dict(body, domain_name=name)
    paths = _case_paths(name)
    cfd_dir = paths["cfd_dir"]
    case_dir = paths["case_dir"]

    # Inputs missing (cleaned after the previous predict) or built from a
    # different form state (wind/Uref/structures changed after Confirm):
    # rebuild from THIS body so predictions can never use stale inputs.
    sig_path = cfd_dir / "build_signature.txt"
    inputs_ok = (cfd_dir / "meta.json").exists()
    if inputs_ok and sig_path.exists():
        inputs_ok = sig_path.read_text().strip() == _body_signature(body)
    if not inputs_ok:
        _p("building model inputs…")
        _run_build_inputs(body)

    model_id = str(body.get("model") or "").strip() or None
    _p(f"running {_model_name(model_id)}…")
    infer_out = predict_all(cfd_dir, model_id=model_id)
    pressure_reference_kinematic = global_pressure_reference_kinematic(
        infer_out["pred_flow"],
        infer_out["bundle"].is_fluid,
    )
    display_out = presentation_prediction(
        infer_out,
        pressure_reference_kinematic,
    )
    t_infer = time.time() - t0

    # transform.json from the workspace (pre-cleanup) — LV95 georeferencing +
    # the z shift needed for the site air density below.
    transform_meta = None
    tform_ws = case_dir / "constant" / "triSurface" / "transform.json"
    if tform_ws.exists():
        try:
            transform_meta = json.loads(tform_ws.read_text())
        except Exception:
            transform_meta = None

    # Site air density (ISA at mean terrain elevation): all Pa conversions in
    # stats/plots/PDF/3D use this for the rest of the run.
    import units as _units
    import numpy as _np
    try:
        z_off = float((transform_meta or {}).get("z_offset_applied", 0.0) or 0.0)
        site_elev = float(_np.nanmean(infer_out["bundle"].terrain_raw["elevation"])) - z_off
        rho = _units.set_air_density_for_elevation(site_elev)
        print(f"[predict_web] site elevation ~{site_elev:.0f} m a.s.l. -> rho={rho:.3f} kg/m3", flush=True)
    except Exception:
        _units.set_air_density(_units.RHO_AIR_SEA_LEVEL)

    # User-placed sampling points → domain-local coords (for profile plots,
    # terrain crosses and the report height/|U| table). Optional; empty if none.
    sampling_points_local = _sampling_points_to_local(body, transform_meta)

    _p("generating plots…")
    plots_dir = RESULTS_DIR / name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_report = generate_prediction_report(
        cfd_dir,
        display_out["pred_flow"],
        out_dir=plots_dir,
        roi_pred_flows=display_out.get("roi_preds") or None,
        sampling_points=sampling_points_local or None,
    )
    t_plots = time.time() - t0 - t_infer

    # Optional two-family disagreement (uncertainty proxy).
    uncert_stats = None
    if bool(body.get("uncertainty")):
        companion = _companion_model_id(model_id)
        if companion:
            _p(f"uncertainty: running {_model_name(companion)}…")
            try:
                infer_b = predict_all(cfd_dir, model_id=companion)
                label_a = get_model_entry(model_id)["label"]
                label_b = get_model_entry(companion)["label"]
                _saved, png = plot_disagreement(
                    cfd_dir, infer_out["pred_flow"], infer_b["pred_flow"],
                    label_a=label_a, label_b=label_b,
                    out_path=plots_dir / "uncertainty_global.png",
                )
                plot_report["uncertainty_global"] = {
                    "path": str(plots_dir / "uncertainty_global.png"),
                    "filename": "uncertainty_global.png",
                    "title": "Model disagreement (uncertainty proxy)",
                    "png_base64": _b64.b64encode(png).decode("ascii"),
                }
                roi_a = infer_out.get("roi_preds") or {}
                roi_b = infer_b.get("roi_preds") or {}
                if roi_a and roi_b:
                    res = plot_roi_disagreement(
                        cfd_dir, roi_a, roi_b, label_a=label_a, label_b=label_b,
                        out_path=plots_dir / "uncertainty_roi.png",
                    )
                    if res is not None:
                        _saved, png = res
                        plot_report["uncertainty_roi"] = {
                            "path": str(plots_dir / "uncertainty_roi.png"),
                            "filename": "uncertainty_roi.png",
                            "title": "ROI model disagreement (uncertainty proxy)",
                            "png_base64": _b64.b64encode(png).decode("ascii"),
                        }
                ua = _np.linalg.norm(infer_out["pred_flow"][..., :3], axis=-1)
                ub = _np.linalg.norm(infer_b["pred_flow"][..., :3], axis=-1)
                both = _np.isfinite(ua) & _np.isfinite(ub)
                d = _np.abs(ua - ub)[both]
                uncert_stats = {
                    "model_b": _model_name(companion),
                    "global_mean_dU_mps": float(_np.mean(d)) if d.size else None,
                    "global_p99_dU_mps": float(_np.quantile(d, 0.99)) if d.size else None,
                }
                roi_d = []
                for k, ra in roi_a.items():
                    rb = roi_b.get(k)
                    if rb is None:
                        continue
                    va = _np.linalg.norm(ra[..., :3], axis=-1)
                    vb = _np.linalg.norm(rb[..., :3], axis=-1)
                    m = _np.isfinite(va) & _np.isfinite(vb)
                    if m.any():
                        roi_d.append(_np.abs(va - vb)[m])
                if roi_d:
                    allroi = _np.concatenate(roi_d)
                    uncert_stats["roi_mean_dU_mps"] = float(_np.mean(allroi))
                    uncert_stats["roi_p99_dU_mps"] = float(_np.quantile(allroi, 0.99))
            except Exception as e:
                print(f"[predict_web] uncertainty warning: {e}", flush=True)
                traceback.print_exc()

    # Persist inputs + predictions BEFORE cleanup so on-demand exports and the
    # 3D viewer can be (re)generated later without rerunning the model. Stale
    # 3D viewer HTML from a previous run of the same name is dropped first.
    _invalidate_3d_cache(name)
    selection_path = ROOT / "dem" / name / "selection.json"
    try:
        save_inputs_and_predictions(
            RESULTS_DIR,
            name,
            case_dir=case_dir,
            cfd_dir=cfd_dir,
            predict_out=infer_out,
            pressure_reference_kinematic=pressure_reference_kinematic,
            selection_path=selection_path if selection_path.exists() else None,
        )
    except Exception as e:
        print(f"[predict_web] save_inputs warning: {e}", flush=True)
        traceback.print_exc()
    t_save = time.time() - t0 - t_infer - t_plots

    _p("computing stats + report…")
    structure_stl = case_dir / "constant" / "triSurface" / "structure.stl"
    runtime_s = float(time.time() - t0)
    stats: dict = {}
    try:
        stats = compute_summary_stats(
            infer_out["bundle"],
            display_out["pred_flow"],
            roi_bundles=infer_out.get("roi_bundles"),
            roi_pred_flows=display_out.get("roi_preds"),
            model_name=_model_name(model_id),
            transform_meta=transform_meta,
            runtime_s=runtime_s,
            structure_stl_path=structure_stl if structure_stl.exists() else None,
            sampling_points=sampling_points_local or None,
            pressure_reference_kinematic=pressure_reference_kinematic,
        )
        if uncert_stats:
            stats["uncertainty"] = uncert_stats
        with (RESULTS_DIR / name / "stats.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"[predict_web] stats warning: {e}", flush=True)
        traceback.print_exc()
    t_stats = time.time() - t0 - t_infer - t_plots - t_save

    # Pre-generate the PDF report (stats page + plots). Cheap, ~1 s.
    # Use the plot_report ORDER (terrain → global → ROI → sampling → extras),
    # NOT an alphabetical glob, so the PDF page order matches the web "Show
    # plots" panel exactly.
    pdf_path = RESULTS_DIR / name / "report.pdf"
    try:
        plot_paths = [Path(v["path"]) for v in plot_report.values()
                      if v.get("path") and Path(v["path"]).exists()]
        if not plot_paths:  # defensive fallback
            plot_paths = _order_plot_paths(plots_dir.glob("*.png"))
        write_pdf_report(pdf_path, domain_name=name, stats=stats, plot_paths=plot_paths)
    except Exception as e:
        print(f"[predict_web] PDF warning: {e}", flush=True)
        traceback.print_exc()

    # Clean intermediates but KEEP the DEM: a re-predict with new wind/Uref
    # only needs an input rebuild, not a re-download of terrain tiles.
    try:
        cleanup_case(name, keep_dem=True)
    except Exception as e:
        print(f"[predict_web] cleanup warning: {e}", flush=True)

    bundle = infer_out["bundle"]
    n_structures = int((infer_out.get("roi_bundles") or {}).__len__()) if infer_out.get("roi_bundles") else 0
    # Prefer the meta's n_structures (the real placed count) over the ROI
    # count, which can be 1 even for a grid of 36.
    try:
        bmeta = bundle.meta if isinstance(bundle.meta, dict) else {}
        n_structures = int(bmeta.get("n_structures") or n_structures)
    except Exception:
        pass
    return {
        "success": True,
        "domain_name": name,
        "n_structures": n_structures,
        "grid_shape": list(bundle.flow.shape[:3]),
        "saved_dir": str(plots_dir),
        "plots": plot_report,
        "plot_order": list(plot_report.keys()),
        "stats": stats,
        "exports": _exports_state(name),
        "elapsed_s": float(time.time() - t0),
        "timings_s": {
            "inference": float(t_infer),
            "plots": float(t_plots),
            "save_inputs": float(t_save),
            "stats_pdf": float(t_stats),
        },
    }


def _sector_score(rec: dict) -> float:
    """Governing-sector ranking for automatic roses: near-ground wind speed."""
    return float(rec.get("max_u_near_ground") or 0.0)


def _select_governing_sector(records: list[dict], *, custom_directions: bool) -> dict:
    if not records:
        raise ValueError("cannot select a governing sector from an empty list")
    if custom_directions:
        return records[0]
    return max(records, key=_sector_score)


def _rose_figure(records: list, *, worst_domain: str) -> bytes:
    """Polar wind-rose, met. convention (N up, clockwise). Bar length AND
    colour = max near-ground wind per direction; the governing direction is
    outlined."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    dirs = [float(r["dir_deg"]) for r in records]
    vals = [float(r.get("max_u_near_ground") or 0.0) for r in records]

    # Bar width: smallest circular gap (so custom lists like [0, 90] never
    # overlap), capped at 25 degrees so few-direction roses keep slim
    # arrow-like petals instead of near-half-circle wedges.
    if len(dirs) > 1:
        s = sorted(d % 360.0 for d in dirs)
        gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)] + [360.0 - s[-1] + s[0]]
        width = math.radians(max(min(gaps) * 0.8, 4.0))
    else:
        width = math.radians(30.0)
    width = min(width, math.radians(25.0))

    # Classic wind-rose speed bins. Each petal is STACKED radially: the
    # innermost rings (0-4 m/s) are blue, then green, yellow, orange, red —
    # so the radial ring labels double as the colour key (no legend needed).
    bins = [(0.0, 4.0, "#3b6fb6"), (4.0, 8.0, "#4caf50"), (8.0, 12.0, "#f5d327"),
            (12.0, 16.0, "#f59427"), (16.0, float("inf"), "#d32f2f")]

    fig = plt.figure(figsize=(7.2, 6.6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for r, v in zip(records, vals):
        th = math.radians(float(r["dir_deg"]))
        bottom = 0.0
        for lo, hi, c in bins:
            top = min(float(v), hi)
            if top <= bottom:
                break
            ax.bar([th], [top - bottom], width=width, bottom=bottom,
                   color=c, edgecolor="white", linewidth=0.5, alpha=0.95)
            bottom = top
        if r.get("domain") == worst_domain and v > 0:
            ax.bar([th], [v], width=width, bottom=0.0, fill=False,
                   edgecolor="#222222", linewidth=2.2)
    ax.set_rlabel_position(112.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(alpha=0.4)
    ax.set_title("Max near-ground wind speed by wind direction [m/s]\n"
                 "(ring labels in m/s; black outline = governing direction)",
                 fontsize=11)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _parse_custom_rose_directions(raw) -> list:
    """Parse custom directions while preserving the user's written order."""
    dirs: list = []
    seen = set()
    if raw:
        parts = raw.replace(";", ",").split(",") if isinstance(raw, str) else list(raw)
        for p in parts:
            try:
                direction = round(float(p) % 360.0, 1)
            except (TypeError, ValueError):
                continue
            if direction not in seen:
                dirs.append(direction)
                seen.add(direction)
    return dirs


def _parse_rose_directions(body: dict) -> list:
    """Custom directions win; otherwise split uniformly from Step 1 wind."""
    max_sectors = max(2, _env_nonnegative_int("PINN_WEBAPP_MAX_ROSE_SECTORS", 16))
    custom = _parse_custom_rose_directions(body.get("rose_directions"))
    if custom:
        if len(custom) > max_sectors:
            raise ValueError(f"wind roses are limited to {max_sectors} directions")
        return custom
    n_sectors = max(2, min(max_sectors, int(body.get("rose_sectors") or 8)))
    reference = float(body.get("wind_from", 0.0)) % 360.0
    return [
        round((reference + i * 360.0 / n_sectors) % 360.0, 1)
        for i in range(n_sectors)
    ]


def _normalise_yaw(yaw_deg: float) -> float:
    """Canonical signed yaw, preserving physical orientation modulo 360."""
    return (float(yaw_deg) + 180.0) % 360.0 - 180.0


def _rose_sector_body(
    body: dict,
    *,
    sector_name: str,
    sector_wind_from: float,
    geometry_reference_wind_from: float,
) -> dict:
    """Build one rose request while keeping structures fixed geographically.

    DEM preparation rotates the terrain into a wind-aligned local frame. A
    yaw entered in the UI is expressed in that local frame, so reusing it for
    another inflow would rotate the physical structure with the wind. Adding
    `(sector - reference)` to every local yaw exactly cancels that frame
    rotation. Structure/grid centres are already CRS coordinates and therefore
    need no compensation.
    """
    sec_body = copy.deepcopy(body)
    sec_body["domain_name"] = sector_name
    sec_body["wind_from"] = float(sector_wind_from)
    sec_body.pop("rose_sectors", None)
    sec_body.pop("rose_directions", None)
    sec_body["uncertainty"] = False

    yaw_delta = float(sector_wind_from) - float(geometry_reference_wind_from)
    for structure in sec_body.get("structures") or []:
        structure["yaw"] = _normalise_yaw(float(structure.get("yaw", 0.0)) + yaw_delta)
    grid = sec_body.get("grid")
    if isinstance(grid, dict):
        grid["grid_yaw"] = _normalise_yaw(float(grid.get("grid_yaw", 0.0)) + yaw_delta)
        grid["struct_yaw"] = _normalise_yaw(float(grid.get("struct_yaw", 0.0)) + yaw_delta)
    return sec_body


def _run_rose(body: dict, progress: Optional[Callable[[str], None]] = None) -> dict:
    """Multi-direction sweep with a fixed geographic structure layout.

    Each sector reuses the already-downloaded DEM of the base domain (copied
    under the sector's name, removed afterwards), while full result artifacts
    are retained for every direction.
    """
    t0 = time.time()

    def _p(msg: str) -> None:
        if progress:
            progress(msg)

    name = _safe_name(body.get("domain_name"))
    custom_dirs = _parse_custom_rose_directions(body.get("rose_directions"))
    dirs = _parse_rose_directions(body)
    n_dirs = len(dirs)
    model_id = str(body.get("model") or "").strip() or None
    base_dem = ROOT / "dem" / name
    geometry_reference_dir = float(body.get("wind_from", dirs[0])) % 360.0
    governing_rule = (
        "Custom directions: the first valid direction is user-selected as governing."
        if custom_dirs else
        "Evenly spaced sectors start at the Step 1 direction; the direction with the highest near-ground wind speed is governing."
    )

    records: list = []
    for i, d in enumerate(dirs):
        sector_name = f"{name}_r{int(round(d)) % 360:03d}"
        _p(f"direction {d:.0f}° ({i + 1}/{n_dirs})…")
        sector_dem = ROOT / "dem" / sector_name
        try:
            if base_dem.exists() and not sector_dem.exists():
                shutil.copytree(base_dem, sector_dem)
            sec_body = _rose_sector_body(
                body,
                sector_name=sector_name,
                sector_wind_from=d,
                geometry_reference_wind_from=geometry_reference_dir,
            )
            out = _run_predict(sec_body, progress=lambda m, _d=d, _i=i: _p(f"direction {_d:.0f}° ({_i + 1}/{n_dirs}): {m}"))

            g = (out.get("stats") or {}).get("global") or {}
            rois = (out.get("stats") or {}).get("rois") or {}
            max_drag = None
            max_suction = None
            for r in rois.values():
                for f in (r.get("per_structure_forces") or []):
                    fd = f.get("F_drag_N")
                    if fd is not None and (max_drag is None or abs(fd) > abs(max_drag)):
                        max_drag = fd
                mn = (r.get("min_p") or {}).get("value_pa")
                if mn is not None and (max_suction is None or mn < max_suction):
                    max_suction = mn
            rec = {
                "dir_deg": float(d),
                "domain": sector_name,
                "max_u": ((g.get("max_umag") or {}).get("value_mps")),
                "max_u_near_ground": ((g.get("max_umag_near_ground_z_rel_le_10m") or {}).get("value_mps")),
                "mean_u_zref": ((g.get("mean_umag_at_zref") or {}).get("value_mps")),
                "max_drag_N": max_drag,
                "max_suction_pa": max_suction,
                "n_structures": int(out.get("n_structures") or 0),
            }
            records.append(rec)

            # Full artifacts (plots, PDF, 3D inputs, exports) are kept for
            # EVERY direction so per-direction reports and 3D views work.
            # Disk: ~50-100 MB per direction under predict_web/results/.
        finally:
            # Remove the sector's workspace + DEM copy regardless of outcome.
            try:
                cleanup_case(sector_name)
            except Exception:
                pass

    if not records:
        raise RuntimeError("wind rose produced no sectors")
    worst = _select_governing_sector(records, custom_directions=bool(custom_dirs))

    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rose_png = _rose_figure(records, worst_domain=worst["domain"])
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots" / "wind_rose.png").write_bytes(rose_png)
    summary = {
        "domain_name": name,
        "model_name": _model_name(model_id),
        "n_directions": n_dirs,
        "directions_deg": dirs,
        "sectors": records,
        "worst_domain": worst["domain"],
        "worst_dir_deg": worst["dir_deg"],
        "geometry_reference_dir_deg": geometry_reference_dir,
        "governing_rule": governing_rule,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with (out_dir / "rose_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots payload for the report panel: the rose first, then EVERY
    # direction's full plot set (the same images go into the combined PDF).
    plots_payload: dict = {
        "wind_rose": {
            "filename": "wind_rose.png",
            "title": "Wind rose (max near-ground wind by direction)",
            "png_base64": _b64.b64encode(rose_png).decode("ascii"),
        }
    }
    sections: list = []
    for rec in records:
        sec_dir = RESULTS_DIR / rec["domain"]
        governing = rec["domain"] == worst["domain"]
        gov_tag = " (governing)" if governing else ""
        plot_paths = _order_plot_paths((sec_dir / "plots").glob("*.png")) if (sec_dir / "plots").exists() else []
        for p in plot_paths:
            plots_payload[f"d{int(round(rec['dir_deg'])) % 360:03d}_{p.stem}"] = {
                "filename": p.name,
                "title": f"Direction {rec['dir_deg']:.0f}°{gov_tag} — {p.stem.replace('_', ' ')}",
                "png_base64": _b64.b64encode(p.read_bytes()).decode("ascii"),
            }
        sec_stats = {}
        try:
            sp = sec_dir / "stats.json"
            if sp.exists():
                sec_stats = json.loads(sp.read_text())
        except Exception:
            sec_stats = {}
        sections.append({
            "label": rec["domain"],
            "dir_deg": rec["dir_deg"],
            "governing": governing,
            "stats": sec_stats,
            "plot_paths": plot_paths,
        })

    # Combined multi-direction PDF at the BASE domain (title page per direction).
    try:
        write_rose_pdf_report(
            out_dir / "report.pdf",
            domain_name=name,
            summary=summary,
            rose_png_path=out_dir / "plots" / "wind_rose.png",
            sections=sections,
        )
    except Exception as e:
        print(f"[predict_web] rose PDF warning: {e}", flush=True)
        traceback.print_exc()

    return {
        "success": True,
        "domain_name": name,
        "model_name": _model_name(model_id),
        "sectors": records,
        "worst_domain": worst["domain"],
        "worst_dir_deg": worst["dir_deg"],
        "geometry_reference_dir_deg": geometry_reference_dir,
        "governing_rule": governing_rule,
        "n_structures": int(worst.get("n_structures") or 0),
        "rose_png_base64": _b64.b64encode(rose_png).decode("ascii"),
        "plots": plots_payload,
        "plot_order": list(plots_payload.keys()),
        "elapsed_s": float(time.time() - t0),
    }


def _rose_direction_wrapper_html(
    name: str, *, endpoint: str, heading: str, detail: str
) -> str:
    """Direction selector shared by wind-rose map and 3D result pages."""
    p = RESULTS_DIR / name / "rose_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"No rose summary for '{name}'")
    summary = json.loads(p.read_text())
    secs = summary.get("sectors") or []
    if not secs:
        raise FileNotFoundError(f"rose summary for '{name}' has no directions")
    worst_dom = summary.get("worst_domain")
    opts = []
    for r in secs:
        gov = " — governing" if r.get("domain") == worst_dom else ""
        opts.append(
            f'<option value="{_html.escape(str(r.get("domain", "")), quote=True)}">'
            f'{float(r.get("dir_deg", 0)):.0f}°{gov}</option>'
        )
    first = quote(str(secs[0]["domain"]))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{_html.escape(heading)} — {_html.escape(name)} (wind rose)</title>"
        "<style>body{margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif;}"
        "#bar{padding:8px 14px;background:#0d47a1;color:#fff;display:flex;gap:10px;align-items:center;}"
        "select{padding:5px;border-radius:5px;border:none;font-size:14px;}"
        "#viewer{position:relative;height:calc(100vh - 46px);}"
        "#loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;"
        "background:#fff;color:#555;font-size:14px;z-index:2;}"
        "iframe{border:none;width:100%;height:100%;}</style></head><body>"
        f"<div id='bar'><b>{_html.escape(heading)} — {_html.escape(name)}</b>"
        "<label for='dirSel'>Wind direction:</label>"
        "<select id='dirSel' onchange='loadDirection(this.value)'>"
        + "".join(opts) +
        f"</select><span style='font-size:12px;opacity:0.85;'>{_html.escape(detail)}</span></div>"
        "<div id='viewer'><div id='loading'>Loading selected direction…</div>"
        f"<iframe id='v' src='{endpoint}?domain={first}' "
        "onload=\"document.getElementById('loading').style.display='none'\"></iframe></div>"
        "<script>function loadDirection(domain){var l=document.getElementById('loading');"
        "l.style.display='flex';document.getElementById('v').src='"
        + endpoint
        + "?domain='+encodeURIComponent(domain);}</script>"
        "</body></html>"
    )


def _rose_3d_wrapper_html(name: str, *, structure: bool) -> str:
    endpoint = "/view_3d_structure" if structure else "/view_3d"
    heading = "3D structure view" if structure else "3D view"
    return _rose_direction_wrapper_html(
        name,
        endpoint=endpoint,
        heading=heading,
        detail="Each direction renders on first selection.",
    )


def _rose_map_wrapper_html(name: str) -> str:
    return _rose_direction_wrapper_html(
        name,
        endpoint="/map",
        heading="Interactive map",
        detail="Select a wind direction to compare wind, pressure and terrain.",
    )


def _runs_index_html() -> str:
    """Server-rendered 'past runs' index over RESULTS_DIR."""
    rows = []
    dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()] if RESULTS_DIR.exists() else []
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs:
        stats = {}
        try:
            sp = d / "stats.json"
            if sp.exists():
                stats = json.loads(sp.read_text())
        except Exception:
            stats = {}
        name_esc = _html.escape(d.name)
        name_url = quote(d.name)
        g = stats.get("global") or {}
        max_u = (g.get("max_umag") or {}).get("value_mps")
        links = []
        if (d / "report.pdf").exists():
            links.append(f'<a href="/download_pdf?domain={name_url}">PDF</a>')
        if has_saved_inputs(RESULTS_DIR, d.name):
            links.append(f'<a href="/view_3d?domain={name_url}" target="_blank">3D</a>')
            links.append(f'<a href="/download_vtk?domain={name_url}">VTK</a>')
        if (d / "plots").exists():
            links.append(f'<a href="/download_zip?domain={name_url}">plots</a>')
        rose = " 🌹" if (d / "rose_summary.json").exists() else ""
        max_u_cell = f"{max_u:.2f} m/s" if max_u is not None else "—"
        rows.append(
            "<tr>"
            f"<td>{name_esc}{rose}</td>"
            f"<td>{_html.escape(str(stats.get('generated_at', '')))}</td>"
            f"<td>{_html.escape(str(stats.get('model_name', '')))}</td>"
            f"<td style='text-align:right;'>{max_u_cell}</td>"
            f"<td>{' · '.join(links) or '—'}</td>"
            "</tr>"
        )
    body = "".join(rows) or "<tr><td colspan='5' style='color:#888;'>no runs yet</td></tr>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>predict_web — past runs</title>"
        "<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:36px;}"
        "table{border-collapse:collapse;font-size:14px;}"
        "th{background:#f5f5f5;text-align:left;padding:8px 14px;border-bottom:2px solid #ddd;}"
        "td{padding:8px 14px;border-bottom:1px solid #eee;}a{color:#1565c0;}</style></head><body>"
        "<h1>Past runs</h1>"
        "<table><tr><th>Domain</th><th>Generated</th><th>Model</th><th>Max |U|</th><th>Artifacts</th></tr>"
        f"{body}</table>"
        "<p style='color:#888;font-size:12px;margin-top:18px;'>🌹 = wind-rose sweep. "
        "Each saved run keeps ~50–100 MB under predict_web/results/&lt;domain&gt;/ — delete folders there to reclaim space.</p>"
        "</body></html>"
    )


_VTK_README = """\
VTK export — pinn_terr_struc surrogate prediction
==================================================

Files in this archive:
  <domain>_global.vtk     Predicted (Ux, Uy, Uz, p, is_fluid) on the global grid
  <domain>_<roi>.vtk      Predicted fields on each Region of Interest grid
  ground.stl              Terrain surface (so you have spatial context)
  structure.stl           Structure surfaces (panels, turbines, ...) - if any

How to use in ParaView:
  1. File -> Open -> select all the files (Ctrl-click), then "Apply" each.
  2. The .vtk files are STRUCTURED_GRID datasets that include the *full*
     bounding cube. Each cell carries an `is_fluid` scalar (1 = fluid,
     0 = inside the terrain or a structure). To hide the cube:
         Filters -> Threshold -> Scalars: is_fluid -> Min=1 -> Max=1 -> Apply
  3. To draw streamlines on the predicted velocity field:
         Filters -> Stream Tracer -> Vectors: U -> seed type: Point Cloud
  4. The raw model pressure field is in the scalar `p`; magnitude is in `Umag`.

Units:
  U / Umag are in m/s. `p` is the RAW KINEMATIC model output (m^2/s^2, the
  OpenFOAM simpleFoam convention). The app's maps, plots, 3D views and PDF show
  relative pressure after subtracting the global fluid-cell mean; this export
  deliberately preserves the unmodified output. Multiply `p` by air density
  to get Pa.

Local frame:
  All coordinates are in the local domain frame (metres from the SW corner).
  Same frame for every file in the archive, so they overlay correctly.
"""


def _render_vtk_zip(name: str) -> bytes:
    if not has_saved_inputs(RESULTS_DIR, name):
        raise FileNotFoundError(f"No saved prediction for '{name}'")
    saved = load_saved_inputs(RESULTS_DIR, name)
    vtk_dir = RESULTS_DIR / name / "vtk"
    info = export_prediction_vtk(
        vtk_dir,
        domain_name=name,
        global_bundle=saved["bundle"],
        global_pred_flow=saved["pred_flow"],
        roi_bundles=saved.get("roi_bundles"),
        roi_pred_flows=saved.get("roi_preds"),
    )
    if not info["files"]:
        skipped_summary = ", ".join(
            f"{s['label']} ({s['n_points']/1e6:.1f}M pts)" for s in info["skipped"]
        )
        raise RuntimeError(f"No VTK files produced (all skipped: {skipped_summary or 'none'})")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in info["files"]:
            zf.write(p, arcname=Path(p).name)
        # Bundle the terrain + structure STLs so the user can overlay them
        # in ParaView and immediately see where the cube interior corresponds
        # to terrain vs fluid.
        case_tri = RESULTS_DIR / name / "inputs" / "case" / "triSurface"
        for stl_name in ("ground.stl", "structure.stl"):
            stl_path = case_tri / stl_name
            if stl_path.exists():
                zf.write(stl_path, arcname=stl_name)
        # README explaining the layout + how to filter to fluid only.
        zf.writestr("README.txt", _VTK_README)
    buf.seek(0)
    return buf.read()


def _render_npz_zip(name: str) -> bytes:
    if not has_saved_inputs(RESULTS_DIR, name):
        raise FileNotFoundError(f"No saved prediction for '{name}'")
    saved = load_saved_inputs(RESULTS_DIR, name)
    npz_dir = RESULTS_DIR / name / "npz"
    info = export_prediction_npz(
        npz_dir,
        domain_name=name,
        global_bundle=saved["bundle"],
        global_pred_flow=saved["pred_flow"],
        roi_bundles=saved.get("roi_bundles"),
        roi_pred_flows=saved.get("roi_preds"),
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in info["files"]:
            zf.write(p, arcname=Path(p).name)
    buf.seek(0)
    return buf.read()


def _read_pdf_bytes(name: str) -> bytes:
    pdf_path = RESULTS_DIR / name / "report.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"No report.pdf for '{name}'")
    return pdf_path.read_bytes()


def _ensure_map_html(name: str) -> Path:
    """Generate the interactive 2D wind/pressure map on demand (cached on disk)."""
    out_path = RESULTS_DIR / name / "map.html"
    if out_path.exists():
        return out_path
    if not has_saved_inputs(RESULTS_DIR, name):
        raise FileNotFoundError(f"No saved prediction for '{name}'")
    cfd_dir = RESULTS_DIR / name / "inputs" / "cfd"
    saved = load_saved_inputs(RESULTS_DIR, name)
    display = presentation_prediction(
        saved,
        saved["pressure_reference_kinematic"],
    )
    write_map_html(
        out_path,
        case_dir=cfd_dir,
        domain_name=name,
        pred_flow=display["pred_flow"],
        roi_pred_flows=display.get("roi_preds"),
    )
    return out_path


def _ensure_3d_html(name: str) -> Path:
    """Generate the standalone 3D Plotly HTML on demand (cached on disk)."""
    out_path = RESULTS_DIR / name / "view_3d.html"
    if out_path.exists():
        return out_path
    if not has_saved_inputs(RESULTS_DIR, name):
        raise FileNotFoundError(f"No saved prediction for '{name}'")
    saved = load_saved_inputs(RESULTS_DIR, name)
    saved = presentation_prediction(
        saved,
        saved["pressure_reference_kinematic"],
    )
    structure_stl = RESULTS_DIR / name / "inputs" / "case" / "triSurface" / "structure.stl"
    write_3d_html(
        out_path,
        saved_inputs=saved,
        domain_name=name,
        structure_stl_path=structure_stl if structure_stl.exists() else None,
    )
    return out_path


def _list_saved_roi_labels(name: str) -> list[str]:
    rois_dir = RESULTS_DIR / name / "inputs" / "cfd" / "roi"
    if not rois_dir.exists():
        return []
    return sorted(p.name for p in rois_dir.iterdir() if p.is_dir() and (p / "meta.json").exists())


def _ensure_structure_3d_html(name: str, roi_label: Optional[str] = None) -> Path:
    """Generate the structure-focused 3D Plotly HTML on demand.

    When `roi_label` is given, the cache key includes it so multi-ROI cases
    keep one HTML per ROI.
    """
    suffix = f"_{roi_label}" if roi_label else ""
    out_path = RESULTS_DIR / name / f"view_3d_structure{suffix}.html"
    if out_path.exists():
        return out_path
    if not has_saved_inputs(RESULTS_DIR, name):
        raise FileNotFoundError(f"No saved prediction for '{name}'")
    saved = load_saved_inputs(RESULTS_DIR, name)
    saved = presentation_prediction(
        saved,
        saved["pressure_reference_kinematic"],
    )
    structure_stl = RESULTS_DIR / name / "inputs" / "case" / "triSurface" / "structure.stl"
    write_structure_3d_html(
        out_path,
        saved_inputs=saved,
        domain_name=name,
        structure_stl_path=structure_stl if structure_stl.exists() else None,
        roi_label=roi_label,
    )
    return out_path


def _structure_roi_chooser_html(name: str, labels: list[str]) -> str:
    """Render a tiny chooser page when multiple ROIs exist."""
    name_esc = _html.escape(name)
    name_url = quote(name)
    rows = "".join(
        f'<li><a href="/view_3d_structure?domain={name_url}&roi={quote(lbl)}" '
        f'style="display:inline-block;padding:8px 14px;margin:4px 0;background:#8e24aa;'
        f'color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
        f'🏛️ {_html.escape(lbl)}</a></li>'
        for lbl in labels
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<title>Choose ROI — {name_esc}</title>'
        '<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:40px;}'
        'h1{color:#222;}ul{list-style:none;padding:0;}</style>'
        '</head><body>'
        f'<h1>Choose ROI for 3D structure view</h1>'
        f'<p>Domain: <b>{name_esc}</b>. Pick which structure ROI to focus on:</p>'
        f'<ul>{rows}</ul>'
        f'<p style="color:#777;font-size:13px;margin-top:24px;">'
        f'(Multistructure cases automatically use one ROI covering all '
        f'structures; you should only see this chooser when several '
        f'independent single-structure ROIs were generated.)</p>'
        '</body></html>'
    )


def _zip_plots(domain_name: str) -> bytes:
    plots_dir = RESULTS_DIR / domain_name / "plots"
    if not plots_dir.exists():
        raise FileNotFoundError(f"No plots for domain '{domain_name}'")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(plots_dir.glob("*.png")):
            zf.write(p, arcname=p.name)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "pinnfluid"
    sys_version = ""
    dem_root = str(ROOT / "dem")
    page_html: str = ""

    def log_message(self, fmt, *args):
        pass

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "same-origin")
        super().end_headers()

    def _json(self, payload: dict, *, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> Optional[dict]:
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            self._json({
                "success": False,
                "error": "chunked request bodies are not supported",
            }, status=400)
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.close_connection = True
            self._json({"success": False, "error": "Content-Length is required"}, status=411)
            return None
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            self._json({"success": False, "error": "invalid Content-Length"}, status=400)
            return None
        if length < 0:
            self._json({"success": False, "error": "invalid Content-Length"}, status=400)
            return None
        max_bytes = _env_mebibytes("PINN_WEBAPP_MAX_REQUEST_MB", 32)
        if length > max_bytes:
            self.close_connection = True
            self._json({
                "success": False,
                "error": f"request body exceeds the {max_bytes // (1024 * 1024)} MiB limit",
            }, status=413)
            return None
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        if len(raw) != length:
            self._json({"success": False, "error": "incomplete request body"}, status=400)
            return None
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json({"success": False, "error": "request body must be valid JSON"}, status=400)
            return None
        if not isinstance(payload, dict):
            self._json({"success": False, "error": "request JSON must be an object"}, status=400)
            return None
        return payload

    def _admit_preparation(self) -> bool:
        ok, message, retry_after = _prep_admit()
        if ok:
            return True
        self._json({
            "success": False,
            "error": message,
            "retry_after": retry_after,
        }, status=429)
        return False

    def _send_path(
        self,
        path: Path,
        *,
        content_type: str,
        content_disposition: Optional[str] = None,
        allow_gzip: bool = True,
    ) -> None:
        """Stream a file with HTTP chunking, optionally gzip-compressed.

        Plotly 3D pages can exceed Cloud Run's 32 MiB limit for buffered
        HTTP/1 responses. Chunking marks the response as streamed; gzip keeps
        transfer and browser parsing costs reasonable without changing the
        cached HTML artifact.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        use_gzip = allow_gzip and "gzip" in self.headers.get("Accept-Encoding", "").lower()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if content_disposition:
            self.send_header("Content-Disposition", content_disposition)
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        compressor = (
            zlib.compressobj(level=5, wbits=16 + zlib.MAX_WBITS)
            if use_gzip else None
        )

        def _chunk(data: bytes) -> None:
            if not data:
                return
            self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
            self.wfile.write(data)
            self.wfile.write(b"\r\n")

        try:
            with path.open("rb") as src:
                while True:
                    raw = src.read(1024 * 1024)
                    if not raw:
                        break
                    _chunk(compressor.compress(raw) if compressor else raw)
            if compressor:
                _chunk(compressor.flush())
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_GET(self):
        path = urlparse(self.path).path
        # Cloud Run reserves /healthz at its frontend, so /status is the
        # externally reachable diagnostic endpoint. Keep the old path for
        # local compatibility.
        if path in ("/status", "/healthz"):
            self._json({
                "status": "ok",
                "models": [m["id"] for m in list_models()],
                "runtime": runtime_device_info(),
            })
            return
        if path == "/static/plotly.min.js":
            body = _plotly_js_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/", "/index.html"):
            body = self.page_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/lv95_to_wgs84":
            qs = parse_qs(urlparse(self.path).query)
            try:
                e = float(qs["e"][0])
                n = float(qs["n"][0])
                lng, lat = lv95_to_wgs84_point(e, n)
                self._json({"lat": lat, "lng": lng})
            except Exception as ex:
                self._json({"error": str(ex)}, status=400)
            return
        if path == "/job_status":
            qs = parse_qs(urlparse(self.path).query)
            jid = unquote(qs.get("id", [""])[0]).strip()
            job = _job_get(jid)
            if job is None:
                self._json({"error": "unknown job", "state": "error"}, status=404)
            else:
                self._json(job)
            return
        if path in ("/map_rose", "/view_3d_rose", "/view_3d_structure_rose"):
            qs = parse_qs(urlparse(self.path).query)
            domain = unquote(qs.get("domain", [""])[0]).strip()
            try:
                if not domain:
                    raise ValueError("missing ?domain=")
                domain = _safe_name(domain)
                if path == "/map_rose":
                    html = _rose_map_wrapper_html(domain)
                else:
                    html = _rose_3d_wrapper_html(
                        domain, structure=path.endswith("structure_rose")
                    )
                data = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as ex:
                self._json({"success": False, "error": str(ex)}, status=404)
            return
        if path == "/runs" and _env_flag("PINN_WEBAPP_ENABLE_RUN_INDEX", False):
            try:
                data = _runs_index_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as ex:
                traceback.print_exc()
                self._json({"success": False, "error": str(ex)}, status=500)
            return
        if path == "/download_zip":
            qs = parse_qs(urlparse(self.path).query)
            domain = unquote(qs.get("domain", [""])[0]).strip()
            try:
                if not domain:
                    raise ValueError("missing ?domain=")
                domain = _safe_name(domain)
                data = _zip_plots(domain)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{domain}_plots.zip"',
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as ex:
                self._json({"success": False, "error": str(ex)}, status=404)
            return
        if path == "/download_vtk":
            qs = parse_qs(urlparse(self.path).query)
            domain = unquote(qs.get("domain", [""])[0]).strip()
            try:
                if not domain:
                    raise ValueError("missing ?domain=")
                domain = _safe_name(domain)
                data = _render_vtk_zip(domain)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{domain}_vtk.zip"',
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as ex:
                self._json({"success": False, "error": str(ex)}, status=404)
            return
        if path == "/download_npz":
            qs = parse_qs(urlparse(self.path).query)
            domain = unquote(qs.get("domain", [""])[0]).strip()
            try:
                if not domain:
                    raise ValueError("missing ?domain=")
                domain = _safe_name(domain)
                data = _render_npz_zip(domain)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{domain}_npz.zip"',
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as ex:
                self._json({"success": False, "error": str(ex)}, status=404)
            return
        if path == "/download_pdf":
            qs = parse_qs(urlparse(self.path).query)
            domain = unquote(qs.get("domain", [""])[0]).strip()
            try:
                if not domain:
                    raise ValueError("missing ?domain=")
                domain = _safe_name(domain)
                data = _read_pdf_bytes(domain)
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{domain}_report.pdf"',
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as ex:
                self._json({"success": False, "error": str(ex)}, status=404)
            return
        if path == "/map":
            qs = parse_qs(urlparse(self.path).query)
            domain = unquote(qs.get("domain", [""])[0]).strip()
            try:
                if not domain:
                    raise ValueError("missing ?domain=")
                domain = _safe_name(domain)
                self._send_path(
                    _ensure_map_html(domain),
                    content_type="text/html; charset=utf-8",
                )
            except Exception as ex:
                traceback.print_exc()
                self._json({"success": False, "error": str(ex)}, status=500)
            return
        if path == "/view_3d" or path == "/download_3d":
            qs = parse_qs(urlparse(self.path).query)
            domain = unquote(qs.get("domain", [""])[0]).strip()
            try:
                if not domain:
                    raise ValueError("missing ?domain=")
                domain = _safe_name(domain)
                html_path = _ensure_3d_html(domain)
                disposition = (
                    f'attachment; filename="{domain}_view_3d.html"'
                    if path == "/download_3d" else None
                )
                self._send_path(
                    html_path,
                    content_type="text/html; charset=utf-8",
                    content_disposition=disposition,
                    allow_gzip=path != "/download_3d",
                )
            except Exception as ex:
                traceback.print_exc()
                self._json({"success": False, "error": str(ex)}, status=500)
            return
        if path == "/view_3d_structure" or path == "/download_3d_structure":
            qs = parse_qs(urlparse(self.path).query)
            domain = unquote(qs.get("domain", [""])[0]).strip()
            roi_label = unquote(qs.get("roi", [""])[0]).strip() or None
            try:
                if not domain:
                    raise ValueError("missing ?domain=")
                domain = _safe_name(domain)
                labels = _list_saved_roi_labels(domain)
                # Multi-ROI without explicit roi=...: show chooser page.
                if roi_label is None and len(labels) > 1:
                    html = _structure_roi_chooser_html(domain, labels).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return
                if roi_label is not None and roi_label not in labels:
                    raise ValueError(f"unknown roi '{roi_label}' for domain '{domain}'")
                html_path = _ensure_structure_3d_html(domain, roi_label=roi_label)
                disposition = None
                if path == "/download_3d_structure":
                    fn_suffix = f"_{roi_label}" if roi_label else ""
                    disposition = (
                        f'attachment; filename="{domain}_view_3d_structure{fn_suffix}.html"'
                    )
                self._send_path(
                    html_path,
                    content_type="text/html; charset=utf-8",
                    content_disposition=disposition,
                    allow_gzip=path != "/download_3d_structure",
                )
            except Exception as ex:
                traceback.print_exc()
                self._json({"success": False, "error": str(ex)}, status=500)
            return
        self.send_error(404)

    def do_POST(self):
        body = self._read_json_body()
        if body is None:
            return

        if self.path == "/download_dem":
            admitted = False
            try:
                west, south, east, north = _validate_dem_bounds(body)
                name = _safe_name(body.get("domain_name", "unnamed"))
                res = body.get("resolution", "2")
                if str(res) != "2":
                    raise ValueError("only the 2 m public DEM resolution is supported")
                if not self._admit_preparation():
                    return
                admitted = True
                output_dir = os.path.join(self.dem_root, name)

                print(f"\n[DEM] {name}: W={west:.5f} S={south:.5f} E={east:.5f} N={north:.5f}", flush=True)
                tiles = query_stac_tiles(west, south, east, north, resolution=res)
                if not tiles:
                    self._json({"success": False, "error": "No tiles found"})
                    return
                downloaded, failed = download_tiles(tiles, output_dir)
                if downloaded == 0:
                    self._json({"success": False, "error": "All downloads failed"})
                    return
                e_min, n_min, e_max, n_max = wgs84_to_lv95(west, south, east, north)
                _, w_m, h_m = merge_and_crop(output_dir, e_min, n_min, e_max, n_max)
                center_e = (e_min + e_max) / 2
                center_n = (n_min + n_max) / 2
                sel = {
                    "center_lv95": {"E": round(center_e, 1), "N": round(center_n, 1)},
                    "domain_name": name,
                }
                with open(os.path.join(output_dir, "selection.json"), "w") as f:
                    json.dump(sel, f, indent=2)
                self._json({"success": True, "dem_size": f"{w_m:.0f}x{h_m:.0f} m", "downloaded": downloaded})
            except Exception as e:
                print(f"[ERROR] {e}", flush=True)
                traceback.print_exc()
                self._json({"success": False, "error": str(e)}, status=400)
            finally:
                if admitted:
                    _prep_release()
            return

        if self.path == "/upload_dem":
            admitted = False
            try:
                name = _safe_name(body.get("domain_name", ""))
                data_b64 = body.get("data_base64") or ""
                if not data_b64:
                    raise ValueError("missing data_base64 (base64-encoded GeoTIFF)")
                max_upload = _env_mebibytes("PINN_WEBAPP_MAX_UPLOAD_MB", 20)
                estimated_size = (len(data_b64) * 3) // 4
                if estimated_size > max_upload + 3:
                    raise ValueError(
                        f"uploaded DEM exceeds the {max_upload // (1024 * 1024)} MiB limit"
                    )
                try:
                    dem_bytes = _b64.b64decode(data_b64, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError("data_base64 is not valid base64") from exc
                if len(dem_bytes) < 1000:
                    raise ValueError("uploaded file is too small to be a valid DEM")
                if len(dem_bytes) > max_upload:
                    raise ValueError(
                        f"uploaded DEM exceeds the {max_upload // (1024 * 1024)} MiB limit"
                    )
                if not self._admit_preparation():
                    return
                admitted = True
                # Wipe any prior dem/<name>/ so a stale Swiss DEM doesn't shadow.
                try:
                    cleanup_case(name)
                except Exception:
                    pass
                info = register_custom_dem(name, dem_bytes, dem_root=self.dem_root)
                self._json({
                    "success": True,
                    "width_m": info["width_m"],
                    "height_m": info["height_m"],
                    "bounds_crs": info["bounds_crs"],
                    "crs": info["crs"],
                    "png_base64": info["png_base64"],
                    "dem_size": f"{info['width_m']:.0f}x{info['height_m']:.0f} m (custom)",
                })
            except Exception as e:
                traceback.print_exc()
                self._json({"success": False, "error": str(e)}, status=400)
            finally:
                if admitted:
                    _prep_release()
            return

        if self.path == "/build_inputs":
            name = str(body.get("domain_name", "")).strip()
            admitted = False
            try:
                _safe_name(name)
                _validate_compute_request(body)
                if not self._admit_preparation():
                    return
                admitted = True
                self._json(_run_build_inputs(body))
            except Exception as e:
                traceback.print_exc()
                if name:
                    # Keep the DEM: one transient build error must not force a
                    # full terrain re-download.
                    try: cleanup_case(name, keep_dem=True)
                    except Exception: pass
                self._json({
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }, status=400)
            finally:
                if admitted:
                    _prep_release()
            return

        if self.path in ("/predict", "/predict_rose"):
            is_rose = self.path == "/predict_rose"
            try:
                name = _safe_name(body.get("domain_name"))
                _validate_compute_request(body)
                if is_rose:
                    _parse_rose_directions(body)
            except ValueError as e:
                self._json({"success": False, "error": str(e)}, status=400)
                return
            kind = "rose" if is_rose else "predict"
            jid, message, retry_after = _job_admit_and_create(kind)
            if jid is None:
                self._json({
                    "success": False,
                    "error": message,
                    "retry_after": retry_after,
                }, status=429)
                return
            runner = _run_rose if is_rose else _run_predict

            def _fn(progress, _body=body, _runner=runner, _name=name):
                try:
                    return _runner(_body, progress=progress)
                except Exception:
                    try:
                        cleanup_case(_name, keep_dem=True)
                    except Exception:
                        pass
                    raise

            _job_start(jid, _fn)
            self._json({"job_id": jid})
            return

        self.send_error(404)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="predict_web local server")
    ap.add_argument("--host", default=os.environ.get("PINN_WEBAPP_HOST", "127.0.0.1"),
                    help="Bind address. Local default 127.0.0.1; use 0.0.0.0 to "
                         "serve inside a container (Cloud Run / Docker). Also "
                         "reads PINN_WEBAPP_HOST.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8779")),
                    help="Port (falls back to the PORT env var, then 8779).")
    ap.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser.")
    args = ap.parse_args()

    # Containers can see the host CPU count even when cgroup quotas grant far
    # fewer vCPUs. Explicitly matching PyTorch's thread pools to the deployed
    # CPU allocation avoids severe oversubscription during 3D U-Net inference.
    torch_threads = int(os.environ.get("PINN_WEBAPP_TORCH_THREADS", "0") or 0)
    if torch_threads > 0:
        import torch
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(torch_threads)
        except RuntimeError:
            pass
        print(f"[predict_web] PyTorch CPU threads: {torch_threads}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Handler.page_html = _build_html()

    server = http.server.ThreadingHTTPServer((args.host, int(args.port)), Handler)
    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    url = f"http://{shown_host}:{args.port}/"
    print(f"[predict_web] Serving at {url}  (bind {args.host}:{args.port})")
    print(f"[predict_web] Results root: {RESULTS_DIR}")
    print(f"[predict_web] Runtime: {runtime_device_info()}")
    # Only auto-open a browser for a local bind, never inside a container.
    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        def _open():
            try:
                if "microsoft" in os.uname().release.lower():
                    os.system(f'cmd.exe /c start {url}')
                else:
                    webbrowser.open(url)
            except Exception:
                pass
        threading.Timer(0.6, _open).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[predict_web] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
