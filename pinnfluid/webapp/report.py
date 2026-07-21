"""Stats computation + PDF report generation for predict_web.

`compute_summary_stats` ingests a global GridBundle + predicted flow (and
optionally per-ROI bundles + predictions) and returns a JSON-serialisable
dict with engineering-relevant numbers: max wind speed and where, max/min
pressure and where, near-ground wind, mean wind at Zref, and per-ROI
near-wall pressure stats.

`write_pdf_report` renders that dict (plus a list of plot PNG paths) into a
multi-page PDF using matplotlib's PdfPages — page 1 is a stats summary,
pages 2..N embed each PNG as a full-page image.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# matplotlib + PIL are already in the stack via plots.py — no new deps.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


# Standard heights above ground (m) reported in the sampling-point table.
SAMPLING_HEIGHTS_M = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def _argmax_3d(arr: np.ndarray, valid: np.ndarray):
    if not bool(np.any(valid)):
        return None
    masked = np.where(valid, arr, -np.inf)
    flat_idx = int(np.argmax(masked))
    nx, ny, nz = arr.shape
    i, rem = divmod(flat_idx, ny * nz)
    j, k = divmod(rem, nz)
    return int(i), int(j), int(k), float(arr[i, j, k])


def _argmin_3d(arr: np.ndarray, valid: np.ndarray):
    if not bool(np.any(valid)):
        return None
    masked = np.where(valid, arr, np.inf)
    flat_idx = int(np.argmin(masked))
    nx, ny, nz = arr.shape
    i, rem = divmod(flat_idx, ny * nz)
    j, k = divmod(rem, nz)
    return int(i), int(j), int(k), float(arr[i, j, k])


def _phys_location(bundle, i: int, j: int, k: int) -> dict:
    x = float(bundle.x_coords[i])
    y = float(bundle.y_coords[j])
    z = float(bundle.z_levels[k])
    elev = float(np.asarray(bundle.terrain_raw['elevation'])[j, i])
    return {
        'index': [i, j, k],
        'x_m': x, 'y_m': y, 'z_m': z,
        'z_rel_m': max(0.0, z - elev),
        'terrain_elev_m': elev,
    }


def _local_to_lv95(x_local: float, y_local: float, transform_meta: Optional[dict]):
    """Convert local-domain (x, y) [metres from SW corner] to LV95 (E, N).

    The local frame is centred on `pivot_xy` (LV95) with the +x axis rotated by
    `theta_math_deg` from LV95 east. Domain spans `[0, final_W] x [0, final_H]`,
    so the SW corner is at pivot - (W/2, H/2) after rotation.
    Returns None if metadata is incomplete.
    """
    if not transform_meta:
        return None
    try:
        import math
        pivot = transform_meta.get('pivot_xy')
        if not pivot or len(pivot) < 2:
            return None
        theta_deg = float(transform_meta.get('theta_math_deg', 0.0))
        ds = transform_meta.get('domain_size') or []
        W = float(transform_meta.get('final_W', ds[0] if len(ds) > 0 else 0.0))
        H = float(transform_meta.get('final_H', ds[1] if len(ds) > 1 else 0.0))
        cx = x_local - 0.5 * W
        cy = y_local - 0.5 * H
        theta = math.radians(theta_deg)
        de = cx * math.cos(theta) - cy * math.sin(theta)
        dn = cx * math.sin(theta) + cy * math.cos(theta)
        return float(pivot[0]) + de, float(pivot[1]) + dn
    except Exception:
        return None


def _augment_location_with_geo(loc: Optional[dict], transform_meta: Optional[dict]) -> Optional[dict]:
    if loc is None or not transform_meta:
        return loc
    en = _local_to_lv95(loc.get('x_m', 0.0), loc.get('y_m', 0.0), transform_meta)
    if en is None:
        return loc
    e, n = en
    loc['lv95_E'] = float(e)
    loc['lv95_N'] = float(n)
    try:
        import sys
        from pathlib import Path as _P
        here = _P(__file__).resolve().parent.parent
        if str(here / 'domain_prep') not in sys.path:
            sys.path.insert(0, str(here / 'domain_prep'))
        from domain_builder import lv95_to_wgs84_point  # type: ignore
        lng, lat = lv95_to_wgs84_point(e, n)
        loc['lng'] = float(lng)
        loc['lat'] = float(lat)
    except Exception:
        pass
    return loc


def _per_structure_forces(roi_bundle, pred_flow_roi: np.ndarray, abl: dict) -> list:
    """Per-structure integrated drag / lift estimate.

    Approximates the surface pressure integral around each structure AABB by:
      1. For each of the 6 axis-aligned faces, average the predicted gauge
         pressure over the cells in a 1 m halo *just outside* that face.
      2. Net force = sum over 6 faces of (-p_face * area * n_outward).
      3. Drag = F dot flow_direction; lift_z = F_z.
      4. Cd uses the projected frontal area for the actual flow direction.

    Notes:
      - Pressure values stored on `pred_flow` are kinematic (m^2/s^2). We
        multiply by `RHO_AIR` here so all reported forces are in Newtons.
      - The outlet-referenced pressure constant cancels on a closed body, so
        absolute reference doesn't affect the net force estimate.
      - The model's pressure accuracy on multistructure interiors is currently
        weak (see project notes). Treat Cd values for closely-packed grids as
        rough first estimates.
    """
    from units import RHO_AIR  # type: ignore  noqa: E402
    meta = roi_bundle.meta if isinstance(roi_bundle.meta, dict) else {}
    sb_list = meta.get('structure_bounds') or []
    if not sb_list:
        return []

    x = np.asarray(roi_bundle.x_coords, dtype=np.float32)
    y = np.asarray(roi_bundle.y_coords, dtype=np.float32)
    z = np.asarray(roi_bundle.z_levels, dtype=np.float32)
    p_pa = RHO_AIR * np.asarray(pred_flow_roi[..., 3], dtype=np.float32)

    uref = float(abl.get('Uref', abl.get('Uref_mps', 1.0)) or 1.0)
    flow_dir = np.asarray(abl.get('flowDir', [1.0, 0.0, 0.0]), dtype=np.float64)
    n_fd = float(np.linalg.norm(flow_dir))
    if n_fd > 1e-6:
        flow_dir = flow_dir / n_fd
    else:
        flow_dir = np.array([1.0, 0.0, 0.0])

    halo = 1.0
    out = []
    # enumerate index (1-based) is the structure number shown both in this
    # table and as the centred label on the structure boxes in the plots, so
    # they stay in sync. We keep numbering by position in sb_list even when a
    # box is skipped below, matching the plot's box enumeration.
    for s_idx, sb in enumerate(sb_list):
        try:
            xmin, ymin, zmin = (float(v) for v in sb['min'])
            xmax, ymax, zmax = (float(v) for v in sb['max'])
        except Exception:
            continue
        a_yz = max(0.0, (ymax - ymin) * (zmax - zmin))
        a_xz = max(0.0, (xmax - xmin) * (zmax - zmin))
        a_xy = max(0.0, (xmax - xmin) * (ymax - ymin))

        m_x_xmin_halo = (x >= xmin - halo) & (x < xmin)
        m_x_xmax_halo = (x > xmax) & (x <= xmax + halo)
        m_y_ymin_halo = (y >= ymin - halo) & (y < ymin)
        m_y_ymax_halo = (y > ymax) & (y <= ymax + halo)
        m_z_zmin_halo = (z >= zmin - halo) & (z < zmin)
        m_z_zmax_halo = (z > zmax) & (z <= zmax + halo)
        m_x_in = (x >= xmin) & (x <= xmax)
        m_y_in = (y >= ymin) & (y <= ymax)
        m_z_in = (z >= zmin) & (z <= zmax)

        def _avg(mx, my, mz):
            sub = p_pa[np.ix_(mx, my, mz)] if (mx.any() and my.any() and mz.any()) else np.array([])
            if sub.size == 0:
                return float('nan')
            finite = np.isfinite(sub)
            if not finite.any():
                return float('nan')
            return float(np.mean(sub[finite]))

        p_neg_x = _avg(m_x_xmin_halo, m_y_in, m_z_in)
        p_pos_x = _avg(m_x_xmax_halo, m_y_in, m_z_in)
        p_neg_y = _avg(m_x_in, m_y_ymin_halo, m_z_in)
        p_pos_y = _avg(m_x_in, m_y_ymax_halo, m_z_in)
        p_neg_z = _avg(m_x_in, m_y_in, m_z_zmin_halo)
        p_pos_z = _avg(m_x_in, m_y_in, m_z_zmax_halo)

        def _nz(v):
            return 0.0 if (v is None or not np.isfinite(v)) else float(v)

        # F = -∮ p n_out dA → for each pair (low, high) on axis a:
        #   F_a = (p_low - p_high) * A_perp_to_a
        Fx = (_nz(p_neg_x) - _nz(p_pos_x)) * a_yz
        Fy = (_nz(p_neg_y) - _nz(p_pos_y)) * a_xz
        Fz = (_nz(p_neg_z) - _nz(p_pos_z)) * a_xy
        F_drag = Fx * flow_dir[0] + Fy * flow_dir[1] + Fz * flow_dir[2]
        a_frontal = abs(flow_dir[0]) * a_yz + abs(flow_dir[1]) * a_xz + abs(flow_dir[2]) * a_xy
        denom = 0.5 * RHO_AIR * uref * uref * max(a_frontal, 1e-6)
        cd = float(F_drag) / denom if denom > 0 else float('nan')

        # JSON has no NaN/Inf — substitute None so the stats payload is valid.
        def _j(v):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            return fv if np.isfinite(fv) else None

        out.append({
            'structure_index': int(s_idx + 1),
            'label': sb.get('label', ''),
            'aabb_min_m': [_j(xmin), _j(ymin), _j(zmin)],
            'aabb_max_m': [_j(xmax), _j(ymax), _j(zmax)],
            'frontal_area_m2': _j(a_frontal),
            'face_p_pa': {
                'neg_x': _j(p_neg_x), 'pos_x': _j(p_pos_x),
                'neg_y': _j(p_neg_y), 'pos_y': _j(p_pos_y),
                'neg_z': _j(p_neg_z), 'pos_z': _j(p_pos_z),
            },
            'F_N': [_j(Fx), _j(Fy), _j(Fz)],
            'F_drag_N': _j(F_drag),
            'F_lift_z_N': _j(Fz),
            'Cd': _j(cd),
        })
    return out


def _surface_integrated_forces(
    roi_bundle,
    pred_flow_roi: np.ndarray,
    abl: dict,
    structure_stl_path,
) -> list:
    """Per-structure force/moment from pressure integration over the STL mesh.

    For every mesh face: sample the predicted gauge pressure just outside the
    surface (face centroid + outward normal x offset, retried at 2-3 offsets),
    then F = sum(-p * n * A) and M = sum(r x f) about the structure AABB base
    centre. Faces whose probes land in solid/NaN cells are skipped and reported
    via `area_coverage`. Far more faithful than the AABB-halo estimate (tilted
    panels, real frontal area, suction sides), but still based on the predicted
    pressure field at 0.5 m resolution — treat as pre-design loads, not
    code-verified design values. Shear stress is not included.
    """
    from units import RHO_AIR  # type: ignore  noqa: E402
    try:
        import trimesh  # type: ignore
        from scipy.interpolate import RegularGridInterpolator  # type: ignore
    except Exception:
        return []

    meta = roi_bundle.meta if isinstance(roi_bundle.meta, dict) else {}
    sb_list = meta.get('structure_bounds') or []
    if not sb_list or structure_stl_path is None or not Path(structure_stl_path).exists():
        return []

    try:
        mesh = trimesh.load(str(structure_stl_path), force="mesh")
        centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
        normals = np.asarray(mesh.face_normals, dtype=np.float64)
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
    except Exception:
        return []
    if centroids.shape[0] == 0:
        return []

    x = np.asarray(roi_bundle.x_coords, dtype=np.float64)
    y = np.asarray(roi_bundle.y_coords, dtype=np.float64)
    z = np.asarray(roi_bundle.z_levels, dtype=np.float64)
    p_kin = np.asarray(pred_flow_roi[..., 3], dtype=np.float64)
    interp = RegularGridInterpolator(
        (x, y, z), p_kin, method="linear", bounds_error=False, fill_value=np.nan
    )
    dx = float(x[1] - x[0]) if len(x) > 1 else 0.5
    offsets = (1.2 * dx, 2.0 * dx, 3.0 * dx)

    uref = float(abl.get('Uref', abl.get('Uref_mps', 1.0)) or 1.0)
    flow_dir = np.asarray(abl.get('flowDir', [1.0, 0.0, 0.0]), dtype=np.float64)
    n_fd = float(np.linalg.norm(flow_dir))
    flow_dir = flow_dir / n_fd if n_fd > 1e-6 else np.array([1.0, 0.0, 0.0])

    def _j(v):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        return fv if np.isfinite(fv) else None

    pad = 0.15  # m tolerance when assigning mesh faces to structure AABBs
    out = []
    for s_idx, sb in enumerate(sb_list):
        try:
            mn = np.asarray([float(v) for v in sb['min']], dtype=np.float64)
            mx = np.asarray([float(v) for v in sb['max']], dtype=np.float64)
        except Exception:
            continue
        inside = np.all((centroids >= mn - pad) & (centroids <= mx + pad), axis=1)
        if not inside.any():
            continue
        c = centroids[inside]
        n = normals[inside]
        a = areas[inside]

        # Sample p at increasing outward offsets; keep the first finite value.
        p_face = np.full(c.shape[0], np.nan, dtype=np.float64)
        for off in offsets:
            todo = ~np.isfinite(p_face)
            if not todo.any():
                break
            p_face[todo] = interp(c[todo] + n[todo] * off)
        ok = np.isfinite(p_face)
        area_total = float(a.sum())
        area_ok = float(a[ok].sum())
        if area_ok <= 0.0:
            continue

        p_pa = RHO_AIR * p_face[ok]
        f_faces = -p_pa[:, None] * n[ok] * a[ok][:, None]  # N per face
        F = f_faces.sum(axis=0)
        base_centre = np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, mn[2]])
        M = np.cross(c[ok] - base_centre[None, :], f_faces).sum(axis=0)

        F_drag = float(F @ flow_dir)
        # True frontal area: projected area of windward-facing faces.
        ndotd = n[ok] @ flow_dir
        a_frontal = float(np.sum(a[ok][ndotd < 0.0] * (-ndotd[ndotd < 0.0])))
        denom = 0.5 * RHO_AIR * uref * uref * max(a_frontal, 1e-6)
        cd = F_drag / denom if denom > 0 else float('nan')

        out.append({
            'structure_index': int(s_idx + 1),
            'label': sb.get('label', ''),
            'method': 'surface_integration',
            'rho_kg_m3': _j(RHO_AIR),
            'n_faces': int(ok.sum()),
            'area_total_m2': _j(area_total),
            'area_coverage': _j(area_ok / max(area_total, 1e-9)),
            'frontal_area_m2': _j(a_frontal),
            'F_N': [_j(F[0]), _j(F[1]), _j(F[2])],
            'F_drag_N': _j(F_drag),
            'F_lift_z_N': _j(F[2]),
            'M_base_Nm': [_j(M[0]), _j(M[1]), _j(M[2])],
            'M_overturning_Nm': _j(float(np.linalg.norm(M[:2]))),
            'Cd': _j(cd),
            'aabb_min_m': [_j(mn[0]), _j(mn[1]), _j(mn[2])],
            'aabb_max_m': [_j(mx[0]), _j(mx[1]), _j(mx[2])],
        })
    return out


def _sampling_point_profiles(
    bundle,
    pred_flow: np.ndarray,
    sampling_points: list,
    *,
    transform_meta: Optional[dict] = None,
    heights=SAMPLING_HEIGHTS_M,
) -> list:
    """Per sampling point: wind speed |U| at a set of heights above ground.

    Each `sampling_points` entry carries local-domain coords `x`, `y` (m) and a
    `label`. We pick the nearest grid column, build the (z_rel, |U|) profile over
    fluid cells, and linearly interpolate |U| at each requested height. Heights
    outside the available column range report `None`. Points outside the domain
    are returned with `in_domain: False`.
    """
    x = np.asarray(bundle.x_coords, dtype=np.float64)
    y = np.asarray(bundle.y_coords, dtype=np.float64)
    z = np.asarray(bundle.z_levels, dtype=np.float64)
    elev = np.asarray(bundle.terrain_raw['elevation'], dtype=np.float64)  # (ny, nx)
    is_fluid = (bundle.is_fluid > 0.5)
    Umag = np.linalg.norm(pred_flow[..., :3], axis=-1)
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())

    out = []
    for idx, sp in enumerate(sampling_points or []):
        xl, yl = sp.get('x'), sp.get('y')
        label = sp.get('label') or f'SP{idx + 1}'
        rec: dict = {'label': label}
        if xl is not None and yl is not None:
            rec['x_m'] = float(xl)
            rec['y_m'] = float(yl)
        # Carry geo references straight through from the placement (browser map).
        for k in ('lat', 'lng', 'crs_x', 'crs_y'):
            if sp.get(k) is not None:
                rec[k] = float(sp[k])

        if xl is None or yl is None or not (x0 <= float(xl) <= x1 and y0 <= float(yl) <= y1):
            rec['in_domain'] = False
            out.append(rec)
            continue

        i = int(np.argmin(np.abs(x - float(xl))))
        j = int(np.argmin(np.abs(y - float(yl))))
        terr = float(elev[j, i])
        col = Umag[i, j, :]
        fluid = is_fluid[i, j, :] & np.isfinite(col)
        zrel = z - terr
        keep = fluid & np.isfinite(zrel)
        zr = zrel[keep]
        vv = col[keep]
        rec['in_domain'] = True
        rec['terrain_elev_m'] = terr
        # lat/lng/CRS already come from the browser placement; only fall back to
        # deriving them from the local x,y when the client did not supply them.
        if rec.get('lat') is None and transform_meta:
            geo = _augment_location_with_geo({'x_m': float(xl), 'y_m': float(yl)}, transform_meta) or {}
            for k in ('lat', 'lng', 'lv95_E', 'lv95_N'):
                if geo.get(k) is not None:
                    rec[k] = geo[k]
        hts = []
        if zr.size:
            order = np.argsort(zr)
            zr = zr[order]
            vv = vv[order]
            for h in heights:
                if h < float(zr[0]) - 1e-6 or h > float(zr[-1]) + 1e-6:
                    u = None
                else:
                    u = float(np.interp(float(h), zr, vv))
                hts.append({'z_rel_m': float(h), 'u_mps': u})
            rec['col_max_u_mps'] = float(np.max(vv))
            rec['col_max_u_zrel_m'] = float(zr[int(np.argmax(vv))])
        rec['heights'] = hts
        out.append(rec)
    return out


def compute_summary_stats(
    bundle,
    pred_flow: np.ndarray,
    *,
    roi_bundles: Optional[dict] = None,
    roi_pred_flows: Optional[dict] = None,
    model_name: str = 'best.pth',
    transform_meta: Optional[dict] = None,
    runtime_s: Optional[float] = None,
    structure_stl_path=None,
    sampling_points: Optional[list] = None,
) -> dict:
    """Engineering-flavoured stats: max wind, max/min pressure, locations, ABL.

    `transform_meta`: optional dict with LV95 origin (lv95_origin_E/N) used to
    convert domain-local locations to LV95 / lat-lng.
    """
    is_fluid = (bundle.is_fluid > 0.5)
    finite = np.isfinite(pred_flow).all(axis=-1)
    valid = is_fluid & finite

    Umag = np.linalg.norm(pred_flow[..., :3], axis=-1)
    p = pred_flow[..., 3]
    abl = bundle.meta.get('ABL', {}) if isinstance(bundle.meta, dict) else {}
    x0, x1, y0, y1, z0, z1 = bundle.bounds

    def _loc(triplet, *, value_key: str, value: float) -> dict:
        if triplet is None:
            return None
        i, j, k, v = triplet
        loc = {value_key: float(value)}
        loc.update(_phys_location(bundle, i, j, k))
        return _augment_location_with_geo(loc, transform_meta)

    max_u_t = _argmax_3d(Umag, valid)
    max_u = _loc(max_u_t, value_key='value_mps', value=max_u_t[3] if max_u_t else 0.0)

    # Near-ground (z_rel <= 10 m)
    elev = np.asarray(bundle.terrain_raw['elevation']).T  # (nx, ny)
    z_rel = bundle.z_levels[None, None, :] - elev[:, :, None]  # (nx, ny, nz)
    near_ground = (z_rel >= 0.0) & (z_rel <= 10.0)
    max_u_ng_t = _argmax_3d(Umag, valid & near_ground)
    max_u_ng = _loc(max_u_ng_t, value_key='value_mps', value=max_u_ng_t[3] if max_u_ng_t else 0.0)

    # Convert kinematic (m^2/s^2) -> Pa for display labels (see units.RHO_AIR).
    from units import RHO_AIR  # type: ignore  noqa: E402
    max_p_t = _argmax_3d(p, valid)
    max_p_info = _loc(max_p_t, value_key='value_pa', value=(RHO_AIR * max_p_t[3]) if max_p_t else 0.0)
    min_p_t = _argmin_3d(p, valid)
    min_p_info = _loc(min_p_t, value_key='value_pa', value=(RHO_AIR * min_p_t[3]) if min_p_t else 0.0)

    # Mean wind at z = mean_terrain + Zref (the ABL reference height above ground)
    zref = float(abl.get('Zref', 20.0))
    elev_mean = float(np.nanmean(bundle.terrain_raw['elevation']))
    target_z = elev_mean + zref
    k_zref = int(np.argmin(np.abs(bundle.z_levels - target_z)))
    actual_z = float(bundle.z_levels[k_zref])
    valid_zref = valid[..., k_zref]
    mean_u_zref = float(np.mean(Umag[..., k_zref][valid_zref])) if bool(np.any(valid_zref)) else float('nan')

    # ROI stats
    rois_out = {}
    if roi_bundles and roi_pred_flows:
        for label, rb in roi_bundles.items():
            rf = roi_pred_flows.get(label)
            if rf is None:
                continue
            rUmag = np.linalg.norm(rf[..., :3], axis=-1)
            rp = rf[..., 3]
            rvalid = (rb.is_fluid > 0.5) & np.isfinite(rf).all(axis=-1)
            rinfo: dict = {
                'grid_shape': list(rb.flow.shape[:3]),
                'n_fluid_cells': int(np.sum(rvalid)),
            }
            mu_t = _argmax_3d(rUmag, rvalid)
            if mu_t is not None:
                i, j, k, v = mu_t
                rinfo['max_umag'] = {'value_mps': v, **_phys_location(rb, i, j, k)}
                _augment_location_with_geo(rinfo['max_umag'], transform_meta)
            mp_t = _argmax_3d(rp, rvalid)
            if mp_t is not None:
                i, j, k, v = mp_t
                rinfo['max_p'] = {'value_pa': RHO_AIR * v, **_phys_location(rb, i, j, k)}
                _augment_location_with_geo(rinfo['max_p'], transform_meta)
            np_t = _argmin_3d(rp, rvalid)
            if np_t is not None:
                i, j, k, v = np_t
                rinfo['min_p'] = {'value_pa': RHO_AIR * v, **_phys_location(rb, i, j, k)}
                _augment_location_with_geo(rinfo['min_p'], transform_meta)

            if rb.phi_wall is not None:
                near_wall = np.abs(rb.phi_wall) <= 1.0
                near_wall_valid = rvalid & near_wall
                mwp_t = _argmax_3d(rp, near_wall_valid)
                if mwp_t is not None:
                    i, j, k, v = mwp_t
                    rinfo['max_p_near_wall'] = {'value_pa': RHO_AIR * v, **_phys_location(rb, i, j, k)}
                    _augment_location_with_geo(rinfo['max_p_near_wall'], transform_meta)
                if bool(np.any(near_wall_valid)):
                    rinfo['p_near_wall_range_pa'] = {
                        'min': float(RHO_AIR * np.min(rp[near_wall_valid])),
                        'max': float(RHO_AIR * np.max(rp[near_wall_valid])),
                    }
            # Per-structure loads: surface integration over the STL when
            # available, AABB-halo first-order estimate as fallback.
            forces: list = []
            try:
                forces = _surface_integrated_forces(rb, rf, abl, structure_stl_path)
            except Exception:
                forces = []
            if not forces:
                try:
                    forces = _per_structure_forces(rb, rf, abl)
                    for f in forces:
                        f.setdefault('method', 'aabb_halo')
                except Exception:
                    forces = []
            rinfo['per_structure_forces'] = forces
            rois_out[str(label)] = rinfo

    import units as _units  # current display density (set per run from site elevation)
    z_off = float((transform_meta or {}).get('z_offset_applied', 0.0) or 0.0)
    site_elev = float(np.nanmean(bundle.terrain_raw['elevation'])) - z_off

    sampling_out = []
    if sampling_points:
        try:
            sampling_out = _sampling_point_profiles(
                bundle, pred_flow, sampling_points, transform_meta=transform_meta
            )
        except Exception:
            sampling_out = []

    return {
        'model_name': str(model_name),
        'runtime_s': float(runtime_s) if runtime_s is not None else None,
        'air': {
            'rho_kg_m3': float(_units.RHO_AIR),
            'site_elevation_m_asl': float(site_elev),
        },
        'global': {
            'grid_shape': list(bundle.flow.shape[:3]),
            'bounds_m': {
                'x': [float(x0), float(x1)],
                'y': [float(y0), float(y1)],
                'z': [float(z0), float(z1)],
            },
            'n_fluid_cells': int(np.sum(valid)),
            'fluid_fraction': float(np.mean(valid)),
            'terrain_elev_range_m': [
                float(np.nanmin(bundle.terrain_raw['elevation'])) - float((transform_meta or {}).get('z_offset_applied', 0.0) or 0.0),
                float(np.nanmax(bundle.terrain_raw['elevation'])) - float((transform_meta or {}).get('z_offset_applied', 0.0) or 0.0),
            ],
            'max_umag': max_u,
            'max_umag_near_ground_z_rel_le_10m': max_u_ng,
            'max_p': max_p_info,
            'min_p': min_p_info,
            'mean_umag_at_zref': {
                'zref_m': zref,
                'k_index': k_zref,
                'actual_z_m': actual_z,
                'value_mps': mean_u_zref,
            },
        },
        'rois': rois_out,
        'sampling_points': sampling_out,
        'abl': {
            'Uref_mps': float(abl.get('Uref', 0.0)),
            'Zref_m': float(abl.get('Zref', 0.0)),
            'z0_m': float(abl.get('z0', 0.0)),
            'flowDir': [float(v) for v in abl.get('flowDir', [1.0, 0.0, 0.0])],
        },
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    }


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
A4_WIDTH = 8.27
A4_HEIGHT = 11.69


def _fmt_loc(loc: Optional[dict]) -> str:
    if not loc:
        return 'n/a'
    parts = [f"x={loc['x_m']:.0f} m", f"y={loc['y_m']:.0f} m", f"z_rel={loc['z_rel_m']:.1f} m"]
    if 'lat' in loc and 'lng' in loc:
        parts.append(f"lat={loc['lat']:.5f}, lng={loc['lng']:.5f}")
    return ', '.join(parts)


def _stats_lines(stats: dict, domain_name: str) -> list[str]:
    g = stats.get('global', {})
    abl = stats.get('abl', {})
    out: list[str] = []

    out.append(f"Domain:   {domain_name}")
    out.append(f"Model:    {stats.get('model_name', '?')}")
    out.append(f"Run time: {stats.get('runtime_s', 0):.1f} s" if stats.get('runtime_s') else "")
    out.append(f"Generated: {stats.get('generated_at', '')}")
    air = stats.get('air') or {}
    if air.get('rho_kg_m3'):
        out.append(
            f"Air density: {air['rho_kg_m3']:.3f} kg/m3 "
            f"(ISA at site elevation {air.get('site_elevation_m_asl', 0):.0f} m a.s.l.)"
        )
    out.append("")

    out.append("--- Inflow conditions (ABL) ---")
    out.append(f"  Uref:   {abl.get('Uref_mps', 0):.2f} m/s")
    out.append(f"  Zref:   {abl.get('Zref_m', 0):.1f} m")
    out.append(f"  z0:     {abl.get('z0_m', 0):.4f} m")
    fd = abl.get('flowDir', [0, 0, 0])
    out.append(f"  Flow direction: ({fd[0]:+.2f}, {fd[1]:+.2f}, {fd[2]:+.2f})")
    out.append("")

    out.append("--- Global domain ---")
    gs = g.get('grid_shape', [])
    out.append(f"  Grid:   {' x '.join(str(s) for s in gs)}  ({g.get('n_fluid_cells', 0):,} fluid cells)")
    bx = g.get('bounds_m', {})
    if bx:
        out.append(f"  Bounds: x={bx.get('x', [])} m, y={bx.get('y', [])} m")
    elev = g.get('terrain_elev_range_m', [])
    if elev:
        out.append(f"  Terrain elevation: {elev[0]:.1f} .. {elev[1]:.1f} m")
    out.append("")

    out.append("--- Wind speed ---")
    if g.get('max_umag'):
        out.append(f"  Max wind:               {g['max_umag']['value_mps']:.2f} m/s")
        out.append(f"      at  {_fmt_loc(g['max_umag'])}")
    if g.get('max_umag_near_ground_z_rel_le_10m'):
        loc = g['max_umag_near_ground_z_rel_le_10m']
        out.append(f"  Max wind (z_rel<=10m):  {loc['value_mps']:.2f} m/s")
        out.append(f"      at  {_fmt_loc(loc)}")
    mu_z = g.get('mean_umag_at_zref', {})
    if mu_z:
        out.append(f"  Mean wind at Zref+terrain ({mu_z.get('actual_z_m', 0):.0f}m): {mu_z.get('value_mps', float('nan')):.2f} m/s")
    out.append("")

    out.append("--- Pressure ---")
    if g.get('max_p'):
        out.append(f"  Max pressure:           {g['max_p']['value_pa']:+.2f} Pa")
        out.append(f"      at  {_fmt_loc(g['max_p'])}")
    if g.get('min_p'):
        out.append(f"  Min pressure:           {g['min_p']['value_pa']:+.2f} Pa")
        out.append(f"      at  {_fmt_loc(g['min_p'])}")
    out.append("")

    rois = stats.get('rois', {})
    if rois:
        out.append("--- ROI(s) ---")
        for label, r in rois.items():
            out.append(f"  [{label}]")
            gs = r.get('grid_shape', [])
            out.append(f"    Grid: {' x '.join(str(s) for s in gs)}  ({r.get('n_fluid_cells', 0):,} fluid cells)")
            if r.get('max_umag'):
                out.append(f"    Max wind:                  {r['max_umag']['value_mps']:.2f} m/s")
            if r.get('max_p'):
                out.append(f"    Max pressure (load):       {r['max_p']['value_pa']:+.2f} Pa")
            if r.get('min_p'):
                out.append(f"    Min pressure (suction):    {r['min_p']['value_pa']:+.2f} Pa")
            if r.get('max_p_near_wall'):
                out.append(f"    Max pressure on structure: {r['max_p_near_wall']['value_pa']:+.2f} Pa  at z_rel={r['max_p_near_wall']['z_rel_m']:.1f}m")
            if r.get('p_near_wall_range_pa'):
                rng = r['p_near_wall_range_pa']
                out.append(f"    Pressure on structure (Pa): min={rng['min']:+.2f}, max={rng['max']:+.2f}")
            forces = r.get('per_structure_forces') or []
            if forces:
                def _ff(v, w, prec):
                    if v is None or not np.isfinite(float(v)):
                        return ('—').rjust(w)
                    return f"{float(v):+{w}.{prec}f}"
                out.append("")
                method = (forces[0].get('method') or 'aabb_halo')
                out.append("    Estimated wind loads"
                           + ("  [surface pressure integration over STL mesh]"
                              if method == 'surface_integration' else
                              "  [first-order AABB face estimate]"))
                out.append(f"      {'#':>3s} {'Fx [N]':>10s} {'Fy [N]':>10s} {'Fz [N]':>10s} {'|Fdrag| [N]':>12s} {'M_ovt [Nm]':>11s} {'Cd':>7s}")
                for i, f in enumerate(forces):
                    fx, fy, fz = (f.get('F_N') or [None] * 3)
                    fd = f.get('F_drag_N')
                    cd = f.get('Cd')
                    movt = f.get('M_overturning_Nm')
                    sidx = f.get('structure_index')
                    if sidx is None:
                        sidx = i + 1
                    out.append(
                        f"      {int(sidx):>3d} {_ff(fx, 10, 1)} {_ff(fy, 10, 1)} {_ff(fz, 10, 1)} {_ff(fd, 12, 1)} {_ff(movt, 11, 1)} {_ff(cd, 7, 2)}"
                    )
                if method == 'surface_integration':
                    out.append("      F = -sum(p n dA) over mesh faces (gauge p sampled just off-surface);")
                    out.append("      M_ovt about the structure base centre; shear stress not included.")
                    cov = [f.get('area_coverage') for f in forces if f.get('area_coverage') is not None]
                    if cov:
                        out.append(f"      Surface coverage of pressure samples: {100.0 * min(cov):.0f}-{100.0 * max(cov):.0f}% of mesh area.")
                else:
                    out.append("      Forces from AABB face-mean pressures (fallback path).")
                out.append("      Cd = F_drag / (0.5*rho*Uref^2*A_frontal). Pre-design estimate, not a code-verified design load.")
                out.append(
                    f"      Forces use rho = {air.get('rho_kg_m3', 1.225):.3f} kg/m3 "
                    f"(ISA at site elevation {air.get('site_elevation_m_asl', 0):.0f} m a.s.l.)."
                )
            out.append("")

    sps = stats.get('sampling_points') or []
    if sps:
        out.append("--- Sampling points ---")
        out.append("  Wind speed |U| (m/s) vs height above ground at each placed point.")
        for sp in sps:
            loc_bits = []
            if sp.get('x_m') is not None:
                loc_bits.append(f"x={sp['x_m']:.0f} m, y={sp.get('y_m', 0):.0f} m")
            if sp.get('lat') is not None and sp.get('lng') is not None:
                loc_bits.append(f"lat={sp['lat']:.5f}, lng={sp['lng']:.5f}")
            out.append(f"  [{sp.get('label', '?')}]  " + "  ".join(loc_bits))
            if not sp.get('in_domain', True):
                out.append("    (outside domain — no profile available)")
                out.append("")
                continue
            heights = sp.get('heights') or []
            if heights:
                out.append(f"      {'z_rel [m]':>10s}  {'|U| [m/s]':>10s}")
                for h in heights:
                    u = h.get('u_mps')
                    us = '—' if u is None else f"{u:.2f}"
                    out.append(f"      {h.get('z_rel_m', 0):>10.0f}  {us:>10s}")
            if sp.get('col_max_u_mps') is not None:
                out.append(
                    f"      column max |U| = {sp['col_max_u_mps']:.2f} m/s "
                    f"at z_rel = {sp.get('col_max_u_zrel_m', 0):.0f} m"
                )
            out.append("")

    return [s for s in out if s is not None]


# Monospace stats text: how many lines safely fit on one A4 portrait page.
# A4 portrait at fontsize 8.6 with 1.25 leading fits ~70 lines from y=0.92 down
# to the bottom margin; 58 is a conservative cap that never clips.
_STATS_LINES_PER_PAGE = 58


def _render_stats_pages(pdf, lines, *, title: str, title_font: float = 20,
                        body_font: float = 8.6, footer: Optional[str] = None) -> None:
    """Render monospace stats text across as many A4 pages as needed.

    Page 1 carries the big title; each overflow page gets a small "(continued)"
    header so long stats blocks (many ROIs and/or sampling points) flow onto a
    2nd/3rd page automatically instead of running off the bottom edge.
    """
    lines = list(lines)
    n_per = max(1, _STATS_LINES_PER_PAGE)
    chunks = [lines[i:i + n_per] for i in range(0, len(lines), n_per)] or [[]]
    n_pages = len(chunks)
    for pi, chunk in enumerate(chunks):
        fig, ax = plt.subplots(figsize=(A4_WIDTH, A4_HEIGHT))
        ax.axis('off')
        if pi == 0:
            ax.text(0.05, 0.97, title, transform=ax.transAxes, fontsize=title_font,
                    fontweight='bold', va='top', color='#0d47a1')
        else:
            ax.text(0.05, 0.97, f"{title}  (continued {pi + 1}/{n_pages})",
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    va='top', color='#0d47a1')
        ax.text(0.05, 0.92, '\n'.join(chunk), transform=ax.transAxes,
                fontsize=body_font, family='monospace', va='top', linespacing=1.25)
        if footer and pi == n_pages - 1:
            ax.text(0.05, 0.02, footer, transform=ax.transAxes, fontsize=7,
                    style='italic', color='#666')
        pdf.savefig(fig)
        plt.close(fig)


def _pdf_image_page(pdf, plot_path, *, title: Optional[str] = None) -> None:
    try:
        img = Image.open(plot_path)
    except Exception:
        return
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
    ax = fig.add_subplot(111)
    ax.imshow(np.asarray(img))
    ax.set_title(title if title is not None else Path(plot_path).stem.replace('_', ' '))
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def write_rose_pdf_report(out_path, *, domain_name: str, summary: dict,
                          rose_png_path=None, sections: list) -> Path:
    """Combined wind-rose report: cover + rose + one block per direction.

    `sections`: list of dicts {label, dir_deg, governing, stats, plot_paths}.
    Each block = full-page title ("WIND DIRECTION i: d°"), the direction's
    stats page, then its plots.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        # Cover: per-direction summary table.
        fig, ax = plt.subplots(figsize=(A4_WIDTH, A4_HEIGHT))
        ax.axis('off')
        ax.text(0.05, 0.97, "Wind Rose Prediction Report", transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top', color='#0d47a1')
        lines = [
            f"Domain:    {domain_name}",
            f"Model:     {summary.get('model_name', '?')}",
            f"Generated: {summary.get('generated_at', '')}",
            f"Directions: {summary.get('n_directions', len(sections))}",
            "",
            f"{'dir':>6s} {'maxU ng':>9s} {'meanU zref':>11s} {'max|Fdrag|':>11s} {'max suction':>12s}",
        ]
        for rec in (summary.get('sectors') or []):
            def _f(v, w, p):
                return ('—').rjust(w) if v is None else f"{float(v):{w}.{p}f}"
            gov = '  << governing' if rec.get('domain') == summary.get('worst_domain') else ''
            lines.append(
                f"{rec.get('dir_deg', 0):>5.0f}° {_f(rec.get('max_u_near_ground'), 9, 2)} "
                f"{_f(rec.get('mean_u_zref'), 11, 2)} {_f(rec.get('max_drag_N'), 11, 1)} "
                f"{_f(rec.get('max_suction_pa'), 12, 1)}{gov}"
            )
        lines += ["", "Full artifacts (3D views, exports, per-direction PDF) are kept",
                  "for every direction under results/<domain>_rXXX/."]
        ax.text(0.05, 0.90, '\n'.join(lines), transform=ax.transAxes,
                fontsize=9, family='monospace', va='top')
        pdf.savefig(fig)
        plt.close(fig)

        if rose_png_path is not None and Path(rose_png_path).exists():
            _pdf_image_page(pdf, rose_png_path, title='Wind rose')

        for i, sec in enumerate(sections):
            # Title page for this direction.
            fig, ax = plt.subplots(figsize=(A4_WIDTH, A4_HEIGHT))
            ax.axis('off')
            gov = "\n(governing direction)" if sec.get('governing') else ""
            ax.text(0.5, 0.55, f"WIND DIRECTION {i + 1}\n\n{sec.get('dir_deg', 0):.0f}°{gov}",
                    transform=ax.transAxes, fontsize=30, fontweight='bold',
                    ha='center', va='center', color='#0d47a1')
            pdf.savefig(fig)
            plt.close(fig)
            # Stats page(s) — auto-paginated like the single-domain report.
            stats = sec.get('stats') or {}
            if stats:
                _render_stats_pages(
                    pdf, _stats_lines(stats, sec.get('label', domain_name)),
                    title=f"Direction {sec.get('dir_deg', 0):.0f}° — summary",
                    title_font=16, body_font=8.0,
                )
            for plot_path in (sec.get('plot_paths') or []):
                _pdf_image_page(
                    pdf, plot_path,
                    title=f"{sec.get('dir_deg', 0):.0f}° — {Path(plot_path).stem.replace('_', ' ')}",
                )
    return out_path


def write_pdf_report(out_path, *, domain_name: str, stats: dict, plot_paths: list) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_paths = [Path(p) for p in (plot_paths or []) if Path(p).exists()]

    with PdfPages(out_path) as pdf:
        # Stats — auto-paginated so long ROI / sampling-point tables never clip.
        _render_stats_pages(
            pdf, _stats_lines(stats, domain_name),
            title="Wind Flow Prediction Report",
            footer="Generated by predict_web (pinn_terr_struc surrogate). "
                   "All locations are in the local domain frame.",
        )

        # Following pages: each plot as a page (A4 portrait, fit-to-page)
        for plot_path in plot_paths:
            try:
                img = Image.open(plot_path)
            except Exception:
                continue
            fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
            ax = fig.add_subplot(111)
            ax.imshow(np.asarray(img))
            ax.set_title(plot_path.stem.replace('_', ' '))
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    return out_path
