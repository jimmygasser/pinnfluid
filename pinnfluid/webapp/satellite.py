"""SWISSIMAGE ground texture for the result views.

One aerial image per run is fetched from the swisstopo WMS in LV95 for the
axis-aligned bounding box of the (wind-rotated) domain, cached next to the run
and reused by the 3D views (vertex colours of a terrain mesh) and by the 2D
results map (background image). Everything degrades to the previous grey /
colour-mapped ground when the run is not in Switzerland, the WMS is
unreachable, or the transform metadata is incomplete.

Frame convention (must match report._local_to_lv95 / dem_prep):
    local = R(+theta) (p - pivot) + centre,   centre = (W/2, H/2)
    p     = pivot + R(-theta) (local - centre)
with theta = transform_meta['theta_math_deg'] (CCW, from LV95 east).
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

WMS_URL = "https://wms.geo.admin.ch/"
WMS_LAYER = "ch.swisstopo.swissimage"
# Plausible LV95 extent of Switzerland (E, N); anything outside is not covered.
_LV95_E = (2_450_000.0, 2_850_000.0)
_LV95_N = (1_050_000.0, 1_310_000.0)
_CACHE_JPG = "satellite_lv95.jpg"
_CACHE_META = "satellite_lv95.json"
_NEG_CACHE = "satellite_unavailable.json"
_NEG_TTL_S = 3600.0          # retry a failed fetch after an hour
_TARGET_M_PER_PX = 2.0       # terrain vertices are >= 7.5 m apart in the 3D view
_MAX_PX = 2600               # ~1.5 MB JPEG worst case; WMS allows up to 10000
_TIMEOUT_S = 45.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _frame(transform_meta: Optional[dict]):
    """(pivot_E, pivot_N, theta_rad, W, H) or None if incomplete."""
    if not transform_meta:
        return None
    pivot = transform_meta.get("pivot_xy")
    if not pivot or len(pivot) < 2:
        return None
    ds = transform_meta.get("domain_size") or []
    W = float(transform_meta.get("final_W", ds[0] if len(ds) > 0 else 0.0))
    H = float(transform_meta.get("final_H", ds[1] if len(ds) > 1 else 0.0))
    if W <= 0 or H <= 0:
        return None
    theta = math.radians(float(transform_meta.get("theta_math_deg", 0.0)))
    return float(pivot[0]), float(pivot[1]), theta, W, H


def is_lv95(transform_meta: Optional[dict]) -> bool:
    """True when the run's CRS is LV95 (or unknown) AND the pivot lies in Switzerland."""
    fr = _frame(transform_meta)
    if fr is None:
        return False
    crs = str(transform_meta.get("crs") or "")
    if crs and not any(tag in crs for tag in ("LV95", "2056", "CH1903+")):
        return False
    e, n = fr[0], fr[1]
    return _LV95_E[0] <= e <= _LV95_E[1] and _LV95_N[0] <= n <= _LV95_N[1]


def local_to_lv95(x, y, transform_meta: dict):
    """Vectorised local (x, y) [m from the SW corner] -> LV95 (E, N)."""
    pe, pn, th, W, H = _frame(transform_meta)
    cx = np.asarray(x, dtype=np.float64) - 0.5 * W
    cy = np.asarray(y, dtype=np.float64) - 0.5 * H
    c, s = math.cos(th), math.sin(th)
    return pe + (cx * c + cy * s), pn + (-cx * s + cy * c)


def lv95_to_local(e, n, transform_meta: dict):
    """Vectorised LV95 (E, N) -> local (x, y): local = R(+theta)(p - pivot) + centre."""
    pe, pn, th, W, H = _frame(transform_meta)
    de = np.asarray(e, dtype=np.float64) - pe
    dn = np.asarray(n, dtype=np.float64) - pn
    c, s = math.cos(th), math.sin(th)
    return 0.5 * W + (de * c - dn * s), 0.5 * H + (de * s + dn * c)


def domain_lv95_bbox(transform_meta: dict, x_coords, y_coords, pad_m: float = 20.0):
    """Axis-aligned LV95 bbox (Emin, Nmin, Emax, Nmax) of the terrain grid extent."""
    x0, x1 = float(np.min(x_coords)), float(np.max(x_coords))
    y0, y1 = float(np.min(y_coords)), float(np.max(y_coords))
    xs = np.array([x0, x1, x1, x0]); ys = np.array([y0, y0, y1, y1])
    E, N = local_to_lv95(xs, ys, transform_meta)
    return (float(E.min()) - pad_m, float(N.min()) - pad_m,
            float(E.max()) + pad_m, float(N.max()) + pad_m)


# ---------------------------------------------------------------------------
# Image container
# ---------------------------------------------------------------------------
class SatelliteImage:
    def __init__(self, img, bbox, layer: str = WMS_LAYER):
        self.img = img.convert("RGB")
        self.bbox = tuple(float(v) for v in bbox)   # Emin, Nmin, Emax, Nmax
        self.layer = layer
        self._arr = None

    @property
    def array(self) -> np.ndarray:
        if self._arr is None:
            self._arr = np.asarray(self.img, dtype=np.uint8)   # (rows, cols, 3), row 0 = north
        return self._arr

    def sample_rgb(self, E, N) -> np.ndarray:
        """Nearest-pixel RGB at LV95 points; (..., 3) uint8. Outside -> mid grey."""
        e0, n0, e1, n1 = self.bbox
        a = self.array
        h, w = a.shape[:2]
        E = np.asarray(E, dtype=np.float64); N = np.asarray(N, dtype=np.float64)
        col = np.floor((E - e0) / max(e1 - e0, 1e-9) * w).astype(np.int64)
        row = np.floor((n1 - N) / max(n1 - n0, 1e-9) * h).astype(np.int64)
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        out = np.full(E.shape + (3,), 128, dtype=np.uint8)
        out[inside] = a[row[inside], col[inside]]
        return out

    def data_uri(self, quality: int = 82) -> str:
        buf = io.BytesIO()
        self.img.save(buf, format="JPEG", quality=quality, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    # --- consumers -------------------------------------------------------
    def terrain_vertex_rgb(self, x1d, y1d, transform_meta: dict) -> np.ndarray:
        """RGB (ny, nx, 3) for a terrain grid given by 1-D local x and y axes."""
        X, Y = np.meshgrid(np.asarray(x1d, dtype=np.float64), np.asarray(y1d, dtype=np.float64))
        E, N = local_to_lv95(X, Y, transform_meta)
        return self.sample_rgb(E, N)

    def layout_image(self, transform_meta: dict, rotpt, angle_deg: float, cx: float, cy: float) -> dict:
        """Plotly `layout.images` entry for the north-up 2D map.

        `rotpt(px, py, angle, cx, cy)` is the map's own display rotation so the
        image goes through exactly the same chain as the field layers:
        LV95 corner -> local frame -> display rotation.
        """
        e0, n0, e1, n1 = self.bbox
        Es = np.array([e0, e1, e1, e0]); Ns = np.array([n0, n0, n1, n1])
        lx, ly = lv95_to_local(Es, Ns, transform_meta)
        pts = [rotpt(float(px), float(py), angle_deg, cx, cy) for px, py in zip(lx, ly)]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return dict(
            source=self.data_uri(), xref="x", yref="y",
            x=min(xs), y=max(ys), sizex=max(xs) - min(xs), sizey=max(ys) - min(ys),
            xanchor="left", yanchor="top", sizing="stretch", layer="below", opacity=1.0,
        )


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------
def _wms_size(bbox, m_per_px: float = _TARGET_M_PER_PX, max_px: int = _MAX_PX):
    e0, n0, e1, n1 = bbox
    w = (e1 - e0) / m_per_px; h = (n1 - n0) / m_per_px
    scale = min(1.0, max_px / max(w, h, 1.0))
    return max(64, int(round(w * scale))), max(64, int(round(h * scale)))


def fetch_swissimage(bbox, *, m_per_px: float = _TARGET_M_PER_PX,
                     timeout: float = _TIMEOUT_S) -> Optional[SatelliteImage]:
    from PIL import Image
    w, h = _wms_size(bbox, m_per_px)
    q = urllib.parse.urlencode({
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER, "STYLES": "", "CRS": "EPSG:2056",
        "BBOX": ",".join(f"{v:.1f}" for v in bbox),
        "WIDTH": str(w), "HEIGHT": str(h), "FORMAT": "image/jpeg",
    })
    req = urllib.request.Request(WMS_URL + "?" + q, headers={"User-Agent": "pinnfluid-webapp/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ctype = resp.headers.get("Content-Type", "")
        data = resp.read()
    if not ctype.startswith("image/"):
        raise RuntimeError(f"WMS returned {ctype!r}: {data[:200]!r}")
    return SatelliteImage(Image.open(io.BytesIO(data)), bbox)


def _cache_names(tag: Optional[str]):
    suffix = f"_{tag}" if tag else ""
    return (f"satellite_lv95{suffix}.jpg", f"satellite_lv95{suffix}.json", f"satellite_unavailable{suffix}.json")


def load_cached(run_dir: Path, tag: Optional[str] = None) -> Optional[SatelliteImage]:
    from PIL import Image
    run_dir = Path(run_dir)
    jpg_name, meta_name, _ = _cache_names(tag)
    jpg, meta = run_dir / jpg_name, run_dir / meta_name
    if not (jpg.exists() and meta.exists()):
        return None
    try:
        m = json.loads(meta.read_text())
        return SatelliteImage(Image.open(jpg), m["bbox"], m.get("layer", WMS_LAYER))
    except Exception:
        return None


def get_satellite_for_run(run_dir: Path, transform_meta: Optional[dict],
                          x_coords, y_coords, *, log=None, tag: Optional[str] = None,
                          m_per_px: float = _TARGET_M_PER_PX) -> Optional[SatelliteImage]:
    """Cached SWISSIMAGE covering the given local grid extent; fetched once.

    `tag` keys a separate cache entry (e.g. a sharper image for one ROI) and
    `m_per_px` sets the requested ground resolution (SWISSIMAGE is 0.1 m
    natively; 2 m is plenty for a 30 m terrain grid, 0.25 m for a structure
    ROI). Returns None when the run is outside Switzerland or the fetch fails.
    """
    run_dir = Path(run_dir)
    cached = load_cached(run_dir, tag)
    if cached is not None:
        return cached
    if not is_lv95(transform_meta):
        return None
    jpg_name, meta_name, neg_name = _cache_names(tag)
    neg = run_dir / neg_name
    if neg.exists():
        try:
            if time.time() - float(json.loads(neg.read_text()).get("t", 0.0)) < _NEG_TTL_S:
                return None
        except Exception:
            pass
    bbox = domain_lv95_bbox(transform_meta, x_coords, y_coords, pad_m=max(5.0, 10.0 * m_per_px))
    try:
        sat = fetch_swissimage(bbox, m_per_px=m_per_px)
    except Exception as exc:  # offline, WMS down, outside coverage, ...
        if log:
            log(f"satellite{'/' + tag if tag else ''}: fetch failed ({exc}); "
                "views fall back to colour-mapped ground")
        try:
            neg.write_text(json.dumps({"t": time.time(), "error": str(exc)[:300]}))
        except OSError:
            pass
        return None
    try:
        sat.img.save(run_dir / jpg_name, format="JPEG", quality=90)
        (run_dir / meta_name).write_text(json.dumps(
            {"bbox": list(sat.bbox), "layer": sat.layer, "width": sat.img.width,
             "height": sat.img.height, "m_per_px": m_per_px,
             "source": "swisstopo WMS (EPSG:2056)"}))
        if neg.exists():
            neg.unlink()
    except OSError:
        pass
    return sat
