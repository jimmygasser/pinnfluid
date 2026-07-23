"""Standalone 3D viewer (Plotly) for a predicted flow field.

Self-contained HTML page with:
  - terrain Surface coloured either by elevation or by **relative pressure**
    (toggle button on the figure)
  - structure meshes (Mesh3d parsed from structure.stl)
  - velocity glyphs (Cone) at 4 pre-rendered densities; switchable via slider
  - **streamlines** (constant-thickness Scatter3d lines from RK2 integration);
    two coordinated sliders for **height above terrain** and **count**
  - persistent Viridis colorbar for |U| via a dummy trace

All wind plots use the **Viridis** colourscale. Layer toggles via the legend
(`legendgroup`s consolidate cones / streamlines into one entry each).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
_TERRAIN_MAX_CELLS_SIDE = 400

GLYPH_DENSITY_LEVELS = [1000, 3000, 9000, 25000]
DEFAULT_GLYPH_IDX = 1  # 3k

STREAMLINE_HEIGHTS_M = [2, 10, 20, 30, 50, 100, 200]
STREAMLINE_COUNTS = [10, 20, 30, 40, 50]
DEFAULT_HEIGHT_IDX = 1   # 10 m
DEFAULT_COUNT_IDX = 1    # 20 lines

_STREAM_MAX_STEPS = 200
_STREAM_LINE_WIDTH = 4
_STREAM_DS_FACTOR = 0.6  # step length as a fraction of mean grid spacing
# When a streamline drops below the local terrain, snap it back to (terrain +
# clearance) instead of killing it — lets paths "slide along" steep ground if
# the predicted velocity has a downward component near the surface. Hard-kill
# only after N consecutive snaps (genuine pile-up at a stagnation point).
_STREAM_MAX_CONSECUTIVE_GROUND_SNAPS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _downsample_terrain(elev: np.ndarray, x: np.ndarray, y: np.ndarray):
    ny, nx = elev.shape
    sx = max(1, nx // _TERRAIN_MAX_CELLS_SIDE)
    sy = max(1, ny // _TERRAIN_MAX_CELLS_SIDE)
    if sx == 1 and sy == 1:
        return elev, x, y, sx, sy
    return elev[::sy, ::sx], x[::sx], y[::sy], sx, sy


def _surface_hover_text(*fields: np.ndarray, formatter) -> np.ndarray:
    """Preformat Plotly Surface/Mesh3d hover labels.

    Some Plotly.js Surface builds expose z but leave surfacecolor/customdata
    template variables literal in the tooltip. Supplying text avoids that
    browser-dependent path and also lets pressure views show real altitude.
    """
    arrays = [np.asarray(field) for field in fields]
    if not arrays:
        return np.empty(0, dtype=object)
    shape = arrays[0].shape
    if any(arr.shape != shape for arr in arrays[1:]):
        raise ValueError("hover fields must have matching shapes")
    flat = [arr.reshape(-1) for arr in arrays]
    labels = []
    for values in zip(*flat):
        if all(np.isfinite(v) for v in values):
            labels.append(formatter(*(float(v) for v in values)))
        else:
            labels.append("")
    return np.asarray(labels, dtype=object).reshape(shape)


def _wind_glyph_indices(valid: np.ndarray, n_target: int) -> np.ndarray:
    flat = np.flatnonzero(valid.reshape(-1))
    if flat.size == 0 or flat.size <= n_target:
        return flat
    step = int(np.ceil(flat.size / n_target))
    return flat[::step]


def _viz_fluid_mask(bundle) -> np.ndarray:
    """Widened (viz-only) fluid mask: every cell above terrain and outside the
    structure AABBs. Matches plots._rebuild_pred_fluid_mask so 3D glyphs cover
    the same domain as the 2D plots, rather than the narrow truth-bundle mask.
    """
    nx, ny, nz = bundle.flow.shape[:3]
    meta = bundle.meta if isinstance(bundle.meta, dict) else {}
    bounds = meta.get('bounds')
    isf = np.asarray(bundle.is_fluid)
    if not bounds or len(bounds) < 6:
        return isf > 0.5
    x = np.asarray(bundle.x_coords, dtype=np.float32)
    y = np.asarray(bundle.y_coords, dtype=np.float32)
    z_levels = meta.get('z_levels')
    if z_levels is not None and len(z_levels) == nz:
        z = np.asarray(z_levels, dtype=np.float32)
    else:
        z = np.asarray(bundle.z_levels, dtype=np.float32)
    elev = np.asarray(bundle.terrain_raw.get('elevation'), dtype=np.float32)
    if elev.shape == (ny, nx):
        elev_ij = elev.T
    elif elev.shape == (nx, ny):
        elev_ij = elev
    else:
        return isf > 0.5
    above_ground = z[None, None, :] > (elev_ij[:, :, None] + 1e-6)
    solid = np.zeros((nx, ny, nz), dtype=bool)
    for sb in meta.get('structure_bounds') or []:
        try:
            xmin, ymin, zmin = (float(v) for v in sb['min'])
            xmax, ymax, zmax = (float(v) for v in sb['max'])
        except Exception:
            continue
        solid |= (
            (x[:, None, None] >= xmin) & (x[:, None, None] <= xmax)
            & (y[None, :, None] >= ymin) & (y[None, :, None] <= ymax)
            & (z[None, None, :] >= zmin) & (z[None, None, :] <= zmax)
        )
    return above_ground & ~solid


def _ground_pressure_field(bundle, pred_flow: np.ndarray) -> np.ndarray:
    """Pressure at the lowest predicted cell above each (i, j). (ny, nx).

    Gates by `np.isfinite(pred_flow)` instead of `bundle.is_fluid` so the
    field follows the inference coverage (widened by predict_web) rather
    than the truth bundle's narrow empty-export mask. Falls back to
    treating "all-zero" cells as un-predicted for stale saves that wrote
    zeros instead of NaNs.
    """
    nx, ny, nz = bundle.flow.shape[:3]
    finite = np.isfinite(pred_flow).all(axis=-1)
    non_zero = (np.asarray(pred_flow, dtype=np.float32) != 0.0).any(axis=-1)
    valid = finite & non_zero                              # (nx, ny, nz)
    p_field = pred_flow[..., 3]
    first_k = np.argmax(valid, axis=-1)                    # 0 if all False
    has_any = valid.any(axis=-1)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    p_at_first = p_field[ii, jj, first_k].astype(np.float32)
    p_at_first[~has_any] = np.nan
    return p_at_first.T  # (ny, nx) for Plotly Surface ordering


def _ground_speed_field(bundle, pred_flow: np.ndarray) -> np.ndarray:
    """|U| at the lowest predicted cell above each (i, j). Returns (ny, nx)."""
    nx, ny, nz = bundle.flow.shape[:3]
    finite = np.isfinite(pred_flow).all(axis=-1)
    non_zero = (np.asarray(pred_flow, dtype=np.float32) != 0.0).any(axis=-1)
    valid = finite & non_zero
    umag = np.linalg.norm(np.asarray(pred_flow[..., :3], dtype=np.float32), axis=-1)
    first_k = np.argmax(valid, axis=-1)
    has_any = valid.any(axis=-1)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    u_at_first = umag[ii, jj, first_k].astype(np.float32)
    u_at_first[~has_any] = np.nan
    return u_at_first.T  # (ny, nx)


try:  # keep the threshold in sync with the 2D snow plots
    from plots import SNOW_TRANSPORT_THRESHOLD_MPS as _SNOW_T_MPS  # type: ignore
except Exception:  # pragma: no cover
    _SNOW_T_MPS = 5.0

# Discrete 3-class colorscale: deposition-prone / neutral / erosion.
_SNOW_COLORSCALE = [
    [0.0, '#8ab4e8'], [1 / 3, '#8ab4e8'],
    [1 / 3, '#ececec'], [2 / 3, '#ececec'],
    [2 / 3, '#d9885a'], [1.0, '#d9885a'],
]


def _snow_class_field(speed_yx: np.ndarray, threshold: float = _SNOW_T_MPS) -> np.ndarray:
    """0 = deposition-prone, 1 = neutral, 2 = erosion; NaN preserved."""
    cls = np.full(speed_yx.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(speed_yx)
    cls[finite & (speed_yx < 0.5 * threshold)] = 0.0
    cls[finite & (speed_yx >= 0.5 * threshold) & (speed_yx <= threshold)] = 1.0
    cls[finite & (speed_yx > threshold)] = 2.0
    return cls


def _snow_surface_trace(x, y, z_ds, speed_ds, *, lighting, lightposition, visible=False):
    """Terrain Surface coloured by the heuristic snow drift classes."""
    import plotly.graph_objects as go
    snow_ds = _snow_class_field(speed_ds)
    return go.Surface(
        x=x, y=y, z=z_ds,
        surfacecolor=snow_ds,
        text=_surface_hover_text(
            speed_ds,
            formatter=lambda speed: (
                f'Snow indicator (heuristic) - near-ground |U| = {speed:.1f} m/s'
            ),
        ),
        colorscale=_SNOW_COLORSCALE,
        cmin=-0.5, cmax=2.5,
        opacity=1.0, showscale=True,
        colorbar=dict(
            title='Snow drift', x=0.0, xanchor='left', len=0.4, y=0.5,
            tickvals=[0, 1, 2],
            ticktext=['deposition', 'neutral', 'erosion'],
        ),
        lighting=lighting, lightposition=lightposition,
        contours=dict(z=dict(show=False)),
        name='Terrain (snow indicator)',
        legendgroup='terrain', showlegend=False,
        hoverinfo='text',
        visible=visible,
    )


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------
def _terrain_traces(bundle, pred_flow, *, z_offset_applied: float = 0.0):
    """Three Surface traces: coloured by elevation (default), relative pressure,
    or the heuristic snow drift indicator.

    `z_offset_applied` is the *additive* shift recorded by domain_builder:
    `domain_z = real_z + z_offset_applied` (z_offset_applied is typically
    negative — e.g. -802 — so the terrain min lands at z=0 in the domain
    frame). Real elevation is therefore `real = domain - z_offset_applied`,
    same convention as the z-axis tick labels (`z_label_shift = -z_offset_applied`).
    Surface position (z) stays in domain frame so structures align; only the
    colormap is shifted into real elevations for display.
    """
    import plotly.graph_objects as go
    elev_raw = np.asarray(bundle.terrain_raw['elevation'])     # (ny, nx)
    elev_real = elev_raw - float(z_offset_applied)             # for color only
    elev_ds, x, y, sx, sy = _downsample_terrain(
        elev_raw,
        np.asarray(bundle.x_coords, dtype=np.float32),
        np.asarray(bundle.y_coords, dtype=np.float32),
    )
    elev_real_ds = elev_real[::sy, ::sx]
    from units import RHO_AIR  # type: ignore  noqa: E402
    p_ground = RHO_AIR * _ground_pressure_field(bundle, pred_flow)  # Pa, (ny, nx)
    p_ground_ds = p_ground[::sy, ::sx]
    finite_p = np.isfinite(p_ground_ds)
    if bool(np.any(finite_p)):
        p_lim = float(np.nanmax(np.abs(p_ground_ds[finite_p])))
    else:
        p_lim = 1.0

    common_lighting = dict(ambient=0.55, diffuse=0.85, specular=0.12,
                           roughness=0.7, fresnel=0.1)
    common_lightpos = dict(x=50_000, y=50_000, z=100_000)

    elev_trace = go.Surface(
        x=x, y=y, z=elev_ds,
        surfacecolor=elev_real_ds,
        colorscale='earth',
        cmin=float(np.nanmin(elev_real)),
        cmax=float(np.nanmax(elev_real)),
        opacity=1.0, showscale=True,
        colorbar=dict(title='Elevation (m)', x=0.0, xanchor='left', len=0.4, y=0.5),
        lighting=common_lighting, lightposition=common_lightpos,
        contours=dict(z=dict(show=False)),
        name='Terrain (elevation)',
        legendgroup='terrain', showlegend=True,
        text=_surface_hover_text(
            elev_real_ds,
            formatter=lambda z: f'Terrain elevation: {z:.1f} m',
        ),
        hoverinfo='text',
        visible=True,
    )
    pressure_trace = go.Surface(
        x=x, y=y, z=elev_ds,
        surfacecolor=p_ground_ds,
        colorscale='RdBu_r',
        cmin=-p_lim, cmax=p_lim,
        opacity=1.0, showscale=True,
        colorbar=dict(title='Relative p (Pa)', x=0.0, xanchor='left', len=0.4, y=0.5),
        lighting=common_lighting, lightposition=common_lightpos,
        contours=dict(z=dict(show=False)),
        name='Terrain (relative pressure)',
        legendgroup='terrain', showlegend=False,   # belongs to same legend item
        text=_surface_hover_text(
            elev_real_ds, p_ground_ds,
            formatter=lambda z, p: (
                f'Terrain elevation: {z:.1f} m; relative p = {p:+.2f} Pa'
            ),
        ),
        hoverinfo='text',
        visible=False,
    )
    speed_ground_ds = _ground_speed_field(bundle, pred_flow)[::sy, ::sx]
    snow_trace = _snow_surface_trace(
        x, y, elev_ds, speed_ground_ds,
        lighting=common_lighting, lightposition=common_lightpos, visible=False,
    )
    return [elev_trace, pressure_trace, snow_trace]


def _per_structure_base_shift(verts: np.ndarray, structure_bounds: list, bundle) -> np.ndarray:
    """Shift each connected structure component vertically so its base sits
    on the bundle's *displayed* terrain at the structure centroid.

    The global bundle's terrain raster (e.g. 16×16 for a 1 km case) is
    rendered by Plotly with bilinear interpolation, which smooths steep
    relief enough to leave the rendered surface ~several metres above or
    below the structure's actual z. Shifting each STL component closes
    that gap without changing the ROI/structure-view rendering (where the
    raster is fine enough).
    """
    if not structure_bounds:
        return verts
    try:
        from scipy.interpolate import RegularGridInterpolator
        elev = np.asarray(bundle.terrain_raw['elevation'], dtype=np.float32)
        # bundle.terrain_raw['elevation'] is (ny, nx)
        interp = RegularGridInterpolator(
            (np.asarray(bundle.y_coords, dtype=np.float32),
             np.asarray(bundle.x_coords, dtype=np.float32)),
            elev,
            bounds_error=False, fill_value=np.nan,
        )
    except Exception:
        return verts
    out = verts.copy()
    for sb in structure_bounds:
        try:
            xmin, ymin, zmin = (float(v) for v in sb['min'])
            xmax, ymax, zmax = (float(v) for v in sb['max'])
        except Exception:
            continue
        # Pick vertices inside this structure's xy bbox + a small margin.
        margin = 0.5
        mask = ((out[:, 0] >= xmin - margin) & (out[:, 0] <= xmax + margin)
                & (out[:, 1] >= ymin - margin) & (out[:, 1] <= ymax + margin))
        if not bool(mask.any()):
            continue
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        z_displayed = float(interp(np.array([[cy, cx]])))
        if not np.isfinite(z_displayed):
            continue
        z_actual = zmin
        delta = z_displayed - z_actual
        if abs(delta) > 0.1:
            out[mask, 2] += delta
    return out


def _structure_traces(structure_stl_path: Optional[Path], *,
                      bundle=None, align_to_terrain: bool = False):
    if not structure_stl_path or not Path(structure_stl_path).exists():
        return []
    try:
        import trimesh
        import plotly.graph_objects as go
    except Exception:
        return []
    try:
        mesh = trimesh.load_mesh(str(structure_stl_path))
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if len(verts) == 0 or len(faces) == 0:
            return []
        if align_to_terrain and bundle is not None:
            sb_list = bundle.meta.get('structure_bounds') if isinstance(bundle.meta, dict) else None
            if sb_list:
                verts = _per_structure_base_shift(verts, sb_list, bundle)
        return [go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='#6a1b9a', opacity=1.0, flatshading=True,
            lighting=dict(ambient=0.45, diffuse=0.85, specular=0.25, roughness=0.5),
            name='Structures', legendgroup='structures', showlegend=True,
            hovertemplate='Structure surface<extra></extra>',
        )]
    except Exception:
        return []


def _make_cone_trace(bundle, pred_flow, n_glyphs, *, name, visible, umag_max, sizeref):
    # The truth bundle's is_fluid can be very narrow (empty-export's phi_wall
    # sign convention). The predicted field is meaningful everywhere above
    # terrain and outside structures. Use the widened mask so glyphs fill the
    # full ROI / global domain — matches the 2D plot pipeline.
    is_fluid_wide = _viz_fluid_mask(bundle)
    finite = np.isfinite(pred_flow).all(axis=-1)
    valid = is_fluid_wide & finite
    flat = _wind_glyph_indices(valid, n_glyphs)
    nx, ny, nz = bundle.flow.shape[:3]
    if flat.size == 0:
        ii = jj = kk = np.zeros(0, dtype=np.int64)
        x = y = z = np.zeros(0, np.float32)
        U = np.zeros((0, 3), np.float32); Umag = np.zeros(0, np.float32)
    else:
        ii, rem = np.divmod(flat, ny * nz)
        jj, kk = np.divmod(rem, nz)
        x = bundle.x_coords[ii].astype(np.float32)
        y = bundle.y_coords[jj].astype(np.float32)
        z = bundle.z_levels[kk].astype(np.float32)
        U = pred_flow[ii, jj, kk, :3].astype(np.float32)
        Umag = np.linalg.norm(U, axis=-1)
    import plotly.graph_objects as go
    return go.Cone(
        x=x, y=y, z=z, u=U[:, 0], v=U[:, 1], w=U[:, 2],
        sizemode='absolute', sizeref=float(sizeref), anchor='tail',
        colorscale='Viridis', cmin=0.0, cmax=float(umag_max),
        showscale=False,
        name=name, legendgroup='glyphs', showlegend=False,
        hovertemplate='|U| = %{customdata:.2f} m/s<extra></extra>',
        customdata=Umag,
        visible=visible,
    )


def _build_uvw_interpolator(bundle, pred_flow):
    """Return a single RegularGridInterpolator that maps (x,y,z) -> (u,v,w).

    Treating the 3-component velocity as the *value* of the interpolant means
    one scipy call returns all three components, which is ~2x faster than
    three separate interpolators.
    """
    from scipy.interpolate import RegularGridInterpolator
    pf = np.where(np.isfinite(pred_flow), pred_flow, 0.0).astype(np.float32)
    x = bundle.x_coords.astype(np.float32)
    y = bundle.y_coords.astype(np.float32)
    z = bundle.z_levels.astype(np.float32)
    UVW = RegularGridInterpolator(
        (x, y, z), pf[..., :3],
        bounds_error=False, fill_value=0.0,
    )
    return UVW, x, y, z


def _integrate_streamlines_batch(seeds, UVW, x_grid, y_grid, z_grid, *,
                                  ds: float, max_steps: int,
                                  elev_interp=None, terrain_clearance_m: float = 0.5,
                                  ground_follow: bool = True,
                                  direction: str = 'forward'):
    """Vectorised forward RK2 integration of N streamlines simultaneously.

    `seeds`: (N, 3) array of starting (x, y, z).
    `UVW`: scipy RegularGridInterpolator returning (..., 3) for (u,v,w).
    `elev_interp`: optional scipy interpolator (y, x) -> elevation. When
        provided, a path is terminated as soon as its z drops below the local
        terrain elevation plus `terrain_clearance_m`. This prevents the
        plotter from drawing streamlines that visually descend into the
        ground on uphill terrain (which is otherwise a pure rendering
        artifact of forward integration ignoring the wall boundary).

    Returns a list of N (xs, ys, zs, speeds) tuples. Stalled / exited seeds
    return shorter histories (or None if they couldn't even start).
    """
    seeds = np.asarray(seeds, dtype=np.float32)
    N = seeds.shape[0]
    if N == 0:
        return []
    x0 = float(x_grid[0]); x1 = float(x_grid[-1])
    y0 = float(y_grid[0]); y1 = float(y_grid[-1])
    z0 = float(z_grid[0]); z1 = float(z_grid[-1])

    # History tensor: positions (T+1, N, 3) and speed at arrival (T+1, N).
    # We grow a "valid up to step k" counter for each seed.
    T = int(max_steps)
    pos_hist = np.full((T + 1, N, 3), np.nan, dtype=np.float32)
    sp_hist = np.zeros((T + 1, N), dtype=np.float32)
    pos_hist[0] = seeds
    # Initialise the seed-point speed sample. Without this the first stored
    # speed is 0, which Plotly colours dark-blue → ugly cap at every seed
    # (and a dark-blue band in the middle of "both" lines whose seam IS the
    # seed). Sampling UVW once at the seeds costs one scipy call.
    try:
        uvw_seed = UVW(seeds)
        sp_seed = np.sqrt(np.sum(uvw_seed * uvw_seed, axis=-1))
        sp_seed = np.where(np.isfinite(sp_seed), sp_seed, 0.0).astype(np.float32)
        sp_hist[0] = sp_seed
    except Exception:
        pass
    alive = np.ones(N, dtype=bool)
    last_step = np.zeros(N, dtype=np.int64)  # last step index where pos was written

    pos = seeds.copy()
    consec_snaps = np.zeros(N, dtype=np.int32)
    sign = -1.0 if str(direction).lower() == 'backward' else 1.0
    for step in range(T):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        p = pos[idx]                                    # (M, 3)
        # Single batched scipy call returns (M, 3)
        uvw = UVW(p)
        s = np.sqrt((uvw * uvw).sum(axis=-1))
        bad = (~np.isfinite(s)) | (s < 1e-3)
        if bad.any():
            alive[idx[bad]] = False
            keep = ~bad
            idx = idx[keep]; p = p[keep]; uvw = uvw[keep]; s = s[keep]
            if idx.size == 0:
                continue
        dt = sign * ds / s
        midp = p + 0.5 * uvw * dt[:, None]
        uvw_h = UVW(midp)
        sh = np.sqrt((uvw_h * uvw_h).sum(axis=-1))
        bad_h = (~np.isfinite(sh)) | (sh < 1e-3)
        if bad_h.any():
            alive[idx[bad_h]] = False
            keep = ~bad_h
            idx = idx[keep]; p = p[keep]; uvw_h = uvw_h[keep]; sh = sh[keep]
            if idx.size == 0:
                continue
        dt_h = sign * ds / sh
        new_p = p + uvw_h * dt_h[:, None]
        oob = ((new_p[:, 0] < x0) | (new_p[:, 0] > x1)
               | (new_p[:, 1] < y0) | (new_p[:, 1] > y1)
               | (new_p[:, 2] < z0) | (new_p[:, 2] > z1))
        if oob.any():
            alive[idx[oob]] = False
            keep = ~oob
            idx = idx[keep]; new_p = new_p[keep]; sh = sh[keep]
            if idx.size == 0:
                continue
        # Terrain interaction. Two modes:
        #   ground_follow=True (default for viz): when a step drops below
        #     local terrain, snap z up to (terrain + clearance) and let the
        #     horizontal velocity carry the streamline along the surface.
        #     Hard-kill only after MAX_CONSECUTIVE_GROUND_SNAPS to avoid
        #     infinite loops at stagnation points.
        #   ground_follow=False: legacy hard-kill on first contact.
        if elev_interp is not None and new_p.shape[0] > 0:
            yx = np.stack([new_p[:, 1], new_p[:, 0]], axis=-1)
            local_elev = elev_interp(yx).astype(np.float32)
            below = (new_p[:, 2] < local_elev + float(terrain_clearance_m))
            if below.any():
                if not ground_follow:
                    alive[idx[below]] = False
                    keep = ~below
                    idx = idx[keep]; new_p = new_p[keep]; sh = sh[keep]
                    if idx.size == 0:
                        continue
                else:
                    # Snap z to terrain + clearance for the below-set
                    new_p[below, 2] = local_elev[below] + float(terrain_clearance_m)
                    consec_snaps[idx[below]] += 1
                    consec_snaps[idx[~below]] = 0
                    stuck = consec_snaps[idx[below]] >= int(_STREAM_MAX_CONSECUTIVE_GROUND_SNAPS)
                    if bool(stuck.any()):
                        below_idx = idx[below][stuck]
                        alive[below_idx] = False
                        # Remove stuck rows from this step's update
                        mask_alive = np.ones(idx.size, dtype=bool)
                        below_pos = np.flatnonzero(below)
                        mask_alive[below_pos[stuck]] = False
                        idx = idx[mask_alive]
                        new_p = new_p[mask_alive]
                        sh = sh[mask_alive]
                        if idx.size == 0:
                            continue
            else:
                consec_snaps[idx] = 0
        # Vectorised record (no Python per-seed loop)
        pos_hist[step + 1, idx] = new_p
        sp_hist[step + 1, idx] = sh
        last_step[idx] = step + 1
        pos[idx] = new_p

    # Slice per-seed histories from the dense tensor.
    out = []
    for i in range(N):
        n_pts = int(last_step[i]) + 1
        if n_pts < 2:
            out.append(None)
            continue
        out.append((
            pos_hist[:n_pts, i, 0],
            pos_hist[:n_pts, i, 1],
            pos_hist[:n_pts, i, 2],
            sp_hist[:n_pts, i],
        ))
    return out


def _seed_positions(n_seeds: int, height_m: float, elev_interp,
                    flow_dir: np.ndarray, x_grid, y_grid,
                    *, side: str = 'inlet'):
    """Seed line at z = local_terrain(x, y) + height_m on a chosen face.

    `side` ∈ {'inlet', 'outlet', 'middle'}:
      - 'inlet'  : upwind face (forward streamlines start here)
      - 'outlet' : downwind face (backward streamlines start here)
      - 'middle' : centre of the flow axis (bi-directional integration)

    `elev_interp` is a scipy RegularGridInterpolator on (y, x) -> elevation.
    Using the *local* terrain elevation (not the mean) means "+2 m" is really
    2 m above the ground directly under each seed, which matters in
    mountainous terrain where the mean is far from the local elevation.
    """
    along_x = abs(flow_dir[0]) >= abs(flow_dir[1])
    side = str(side).lower()
    if along_x:
        inlet_x = float(x_grid[1] if flow_dir[0] > 0 else x_grid[-2])
        outlet_x = float(x_grid[-2] if flow_dir[0] > 0 else x_grid[1])
        if side == 'outlet':
            sx_val = outlet_x
        elif side == 'middle':
            sx_val = 0.5 * (inlet_x + outlet_x)
        else:
            sx_val = inlet_x
        sx_arr = np.full(n_seeds, sx_val, dtype=np.float32)
        sy_arr = np.linspace(y_grid[2], y_grid[-3], n_seeds, dtype=np.float32)
    else:
        inlet_y = float(y_grid[1] if flow_dir[1] > 0 else y_grid[-2])
        outlet_y = float(y_grid[-2] if flow_dir[1] > 0 else y_grid[1])
        if side == 'outlet':
            sy_val = outlet_y
        elif side == 'middle':
            sy_val = 0.5 * (inlet_y + outlet_y)
        else:
            sy_val = inlet_y
        sy_arr = np.full(n_seeds, sy_val, dtype=np.float32)
        sx_arr = np.linspace(x_grid[2], x_grid[-3], n_seeds, dtype=np.float32)
    seed_pts_yx = np.stack([sy_arr, sx_arr], axis=-1)
    local_elev = elev_interp(seed_pts_yx).astype(np.float32)
    sz_arr = local_elev + float(height_m)
    return np.stack([sx_arr, sy_arr, sz_arr], axis=-1)


def _concat_bidirectional_paths(paths_bwd, paths_fwd) -> list:
    """For each seed produce a single polyline = reversed backward + forward.

    Each `paths_*[i]` is `(xs, ys, zs, sp)` or `None`. Result list has the
    same length as the input pair; entries are concatenated polylines, or
    `None` if both sides failed.
    """
    out = []
    n = max(len(paths_bwd), len(paths_fwd))
    for i in range(n):
        b = paths_bwd[i] if i < len(paths_bwd) else None
        f = paths_fwd[i] if i < len(paths_fwd) else None
        if b is None and f is None:
            out.append(None); continue
        if b is None:
            out.append(f); continue
        if f is None:
            # Reverse the backward path so the line still starts upstream.
            out.append((b[0][::-1], b[1][::-1], b[2][::-1], b[3][::-1]))
            continue
        # Drop the duplicate seed point at the seam by skipping the first
        # element of the forward path.
        xs = np.concatenate([b[0][::-1], f[0][1:]])
        ys = np.concatenate([b[1][::-1], f[1][1:]])
        zs = np.concatenate([b[2][::-1], f[2][1:]])
        sp = np.concatenate([b[3][::-1], f[3][1:]])
        out.append((xs, ys, zs, sp))
    return out


def _streamline_trace_from_paths(paths, *, name: str, visible: bool, umag_max: float):
    """Combine a list of (xs, ys, zs, speeds) paths into one Scatter3d trace
    using None / 0.0 separators (None on xyz breaks the line, 0.0 placeholder
    on the colour array — Plotly's line.color rejects None)."""
    import plotly.graph_objects as go
    xs_all: list = []
    ys_all: list = []
    zs_all: list = []
    cs_all: list = []
    for entry in paths:
        if entry is None:
            continue
        xs, ys, zs, sp = entry
        if xs.size < 2:
            continue
        xs_all.extend(xs.tolist()); xs_all.append(None)
        ys_all.extend(ys.tolist()); ys_all.append(None)
        zs_all.extend(zs.tolist()); zs_all.append(None)
        cs_all.extend(sp.tolist()); cs_all.append(0.0)
    if not xs_all:
        return None
    return go.Scatter3d(
        x=xs_all, y=ys_all, z=zs_all,
        mode='lines',
        line=dict(
            color=cs_all,
            colorscale='Viridis',
            cmin=0.0, cmax=float(umag_max),
            width=_STREAM_LINE_WIDTH,
            showscale=False,
        ),
        connectgaps=False,
        name=name, legendgroup='streams', showlegend=False,
        hoverinfo='skip',
        visible=visible,
    )


def _colorbar_keeper(umag_max: float):
    """Always-visible dummy trace: provides the persistent |U| Viridis colorbar."""
    import plotly.graph_objects as go
    return go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(
            color=[0.0], colorscale='Viridis',
            cmin=0.0, cmax=float(umag_max),
            showscale=True, size=0.0001,
            colorbar=dict(title='|U| (m/s)', x=1.0, xanchor='right', len=0.4, y=0.5),
        ),
        showlegend=False, hoverinfo='skip',
        name='_colorbar_keeper',
    )


def _legend_handle(group: str, label: str, color: str):
    import plotly.graph_objects as go
    return go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        name=label, legendgroup=group, showlegend=True,
        hoverinfo='skip',
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def build_3d_figure(saved_inputs: dict, *, domain_name: str,
                    structure_stl_path: Optional[Path] = None):
    import plotly.graph_objects as go
    bundle = saved_inputs['bundle']
    pred_flow = saved_inputs['pred_flow']

    # --- Common scales ---
    is_fluid = (bundle.is_fluid > 0.5)
    finite = np.isfinite(pred_flow).all(axis=-1)
    valid = is_fluid & finite
    if bool(np.any(valid)):
        Umag_full = np.linalg.norm(pred_flow[valid][:, :3], axis=-1)
        umag_max = float(max(Umag_full.max(), 1e-3))
    else:
        umag_max = 1.0
    cone_sizeref = umag_max * 0.5

    # Grid spacing for streamline step size. Use ONLY the horizontal grid
    # spacing — streamlines travel mainly in xy and a stretched vertical
    # grid (median dz << dx) would force ds to ~1 m and exhaust the step
    # budget after a few hundred metres on big domains. Then size the
    # step budget to actually cross the domain twice over.
    dx = float(np.median(np.diff(bundle.x_coords))) if len(bundle.x_coords) > 1 else 1.0
    dy = float(np.median(np.diff(bundle.y_coords))) if len(bundle.y_coords) > 1 else 1.0
    ds = max(_STREAM_DS_FACTOR * min(dx, dy), 0.5)
    x0_b, x1_b, y0_b, y1_b, _, _ = bundle.bounds
    diag = float(np.hypot(x1_b - x0_b, y1_b - y0_b))
    max_steps_run = max(int(_STREAM_MAX_STEPS), int(2.0 * diag / max(ds, 0.1)))

    abl = bundle.meta.get('ABL', {}) if isinstance(bundle.meta, dict) else {}
    flow_dir = np.asarray(abl.get('flowDir', [1.0, 0.0, 0.0]), dtype=np.float32)
    elev_mean = float(np.nanmean(bundle.terrain_raw['elevation']))

    UVW, x_grid, y_grid, z_grid = _build_uvw_interpolator(bundle, pred_flow)

    # Elevation interpolator (y, x) -> z so seed heights are LOCAL terrain + h_m.
    from scipy.interpolate import RegularGridInterpolator
    elev_raw = np.asarray(bundle.terrain_raw['elevation'], dtype=np.float32)
    elev_interp = RegularGridInterpolator(
        (y_grid, x_grid), elev_raw,
        bounds_error=False, fill_value=elev_mean,
    )

    traces: list = []
    trace_groups: dict[str, list[int]] = {'glyphs': [], 'streams': []}

    # --- Terrain (elevation, relative pressure and snow traces) ---
    z_offset_applied = float(((saved_inputs.get('transform_meta') or {}).get('z_offset_applied')) or 0.0)
    terrain = _terrain_traces(bundle, pred_flow, z_offset_applied=z_offset_applied)
    terrain_idx = list(range(len(traces), len(traces) + len(terrain)))
    traces.extend(terrain)

    # --- Structures (aligned to the bundle's coarse terrain raster so
    #     they don't appear buried/floating on steep slopes) ---
    traces.extend(_structure_traces(structure_stl_path, bundle=bundle, align_to_terrain=True))

    # --- Cones (4 density levels) ---
    for idx, n in enumerate(GLYPH_DENSITY_LEVELS):
        t = _make_cone_trace(
            bundle, pred_flow, n,
            name=f'Wind ({n:,} glyphs)',
            visible=(idx == DEFAULT_GLYPH_IDX),
            umag_max=umag_max, sizeref=cone_sizeref,
        )
        trace_groups['glyphs'].append(len(traces))
        traces.append(t)

    # --- Streamlines: H x C x {fwd, bwd, both} pre-rendered traces ---
    # Forward seeds at the INLET face; backward seeds at the OUTLET face so
    # backward lines actually traverse the domain upstream. "Both" seeds
    # in the MIDDLE of the domain and integrates fwd + bwd from each seed;
    # the two halves are then concatenated into a SINGLE polyline per seed
    # (ParaView-style bi-directional integration).
    n_h = len(STREAMLINE_HEIGHTS_M)
    n_c = len(STREAMLINE_COUNTS)
    stream_index_grid_fwd: list[Optional[int]] = [None] * (n_h * n_c)
    stream_index_grid_bwd: list[Optional[int]] = [None] * (n_h * n_c)
    stream_index_grid_both: list[Optional[int]] = [None] * (n_h * n_c)

    seeds_inlet: list[np.ndarray] = []
    seeds_outlet: list[np.ndarray] = []
    seeds_middle: list[np.ndarray] = []
    span_per_combo: list[int] = []
    for h_m in STREAMLINE_HEIGHTS_M:
        for count in STREAMLINE_COUNTS:
            seeds_inlet.append(_seed_positions(int(count), float(h_m),
                                                elev_interp, flow_dir, x_grid, y_grid, side='inlet'))
            seeds_outlet.append(_seed_positions(int(count), float(h_m),
                                                 elev_interp, flow_dir, x_grid, y_grid, side='outlet'))
            seeds_middle.append(_seed_positions(int(count), float(h_m),
                                                 elev_interp, flow_dir, x_grid, y_grid, side='middle'))
            span_per_combo.append(int(count))
    inlet_concat = np.concatenate(seeds_inlet, axis=0) if seeds_inlet else np.zeros((0, 3), np.float32)
    outlet_concat = np.concatenate(seeds_outlet, axis=0) if seeds_outlet else np.zeros((0, 3), np.float32)
    middle_concat = np.concatenate(seeds_middle, axis=0) if seeds_middle else np.zeros((0, 3), np.float32)

    paths_fwd = _integrate_streamlines_batch(
        inlet_concat, UVW, x_grid, y_grid, z_grid,
        ds=ds, max_steps=max_steps_run,
        elev_interp=elev_interp, direction='forward',
    )
    paths_bwd = _integrate_streamlines_batch(
        outlet_concat, UVW, x_grid, y_grid, z_grid,
        ds=ds, max_steps=max_steps_run,
        elev_interp=elev_interp, direction='backward',
    )
    paths_mid_fwd = _integrate_streamlines_batch(
        middle_concat, UVW, x_grid, y_grid, z_grid,
        ds=ds, max_steps=max_steps_run,
        elev_interp=elev_interp, direction='forward',
    )
    paths_mid_bwd = _integrate_streamlines_batch(
        middle_concat, UVW, x_grid, y_grid, z_grid,
        ds=ds, max_steps=max_steps_run,
        elev_interp=elev_interp, direction='backward',
    )
    paths_both = _concat_bidirectional_paths(paths_mid_bwd, paths_mid_fwd)

    cur = 0; combo = 0
    for hi, h_m in enumerate(STREAMLINE_HEIGHTS_M):
        for ci, count in enumerate(STREAMLINE_COUNTS):
            n = span_per_combo[combo]
            paths_f = paths_fwd[cur:cur + n]
            paths_b = paths_bwd[cur:cur + n]
            paths_bo = paths_both[cur:cur + n]
            cur += n; combo += 1
            is_default = (hi == DEFAULT_HEIGHT_IDX) and (ci == DEFAULT_COUNT_IDX)
            tf = _streamline_trace_from_paths(
                paths_f,
                name=f'Streamlines fwd ({count} @ +{h_m} m)',
                visible=bool(is_default), umag_max=umag_max,
            )
            if tf is not None:
                stream_index_grid_fwd[hi * n_c + ci] = len(traces)
                trace_groups['streams'].append(len(traces))
                traces.append(tf)
            tb = _streamline_trace_from_paths(
                paths_b,
                name=f'Streamlines bwd ({count} @ +{h_m} m)',
                visible=False, umag_max=umag_max,
            )
            if tb is not None:
                stream_index_grid_bwd[hi * n_c + ci] = len(traces)
                trace_groups['streams'].append(len(traces))
                traces.append(tb)
            tboth = _streamline_trace_from_paths(
                paths_bo,
                name=f'Streamlines both ({count} @ +{h_m} m)',
                visible=False, umag_max=umag_max,
            )
            if tboth is not None:
                stream_index_grid_both[hi * n_c + ci] = len(traces)
                trace_groups['streams'].append(len(traces))
                traces.append(tboth)

    # --- Persistent |U| colorbar (always-visible dummy) ---
    traces.append(_colorbar_keeper(umag_max))

    # --- Legend handles (one clickable item per group) ---
    traces.append(_legend_handle('glyphs', 'Wind glyphs', '#fde725'))
    traces.append(_legend_handle('streams', 'Streamlines', '#5ec962'))

    fig = go.Figure(traces)

    # ----- Sliders (glyph density, streamline height, streamline count) -----
    glyph_idx = trace_groups['glyphs']
    stream_idx = trace_groups['streams']

    glyph_steps = []
    for i, n in enumerate(GLYPH_DENSITY_LEVELS):
        visible = [False] * len(glyph_idx); visible[i] = True if i < len(glyph_idx) else False
        glyph_steps.append(dict(
            method='restyle',
            args=[{'visible': visible}, glyph_idx],
            label=(f'{n//1000}k' if n >= 1000 else str(n)),
        ))

    # Streamline sliders use method='skip' — actual visibility update happens
    # via a small JS hook (see _streamline_js_hook) so the two sliders can
    # combine their states.
    height_steps = [dict(method='skip', label=f'{h} m') for h in STREAMLINE_HEIGHTS_M]
    count_steps = [dict(method='skip', label=f'{c}') for c in STREAMLINE_COUNTS]

    sliders = [
        dict(
            active=DEFAULT_GLYPH_IDX,
            x=0.05, y=0.04, len=0.27, xanchor='left', yanchor='top',
            currentvalue=dict(prefix='Glyph density: ', font=dict(size=12)),
            steps=glyph_steps, pad=dict(t=10, b=4),
            name='glyphs',
        ),
        dict(
            active=DEFAULT_HEIGHT_IDX,
            x=0.36, y=0.04, len=0.27, xanchor='left', yanchor='top',
            currentvalue=dict(prefix='Streamline height: ', font=dict(size=12)),
            steps=height_steps, pad=dict(t=10, b=4),
            name='stream_height',
        ),
        dict(
            active=DEFAULT_COUNT_IDX,
            x=0.68, y=0.04, len=0.27, xanchor='left', yanchor='top',
            currentvalue=dict(prefix='Streamline count: ', font=dict(size=12)),
            steps=count_steps, pad=dict(t=10, b=4),
            name='stream_count',
        ),
    ]

    # ----- Updatemenu (terrain colour mode toggle) -----
    terrain_buttons = [
        dict(
            label='Color: elevation', method='restyle',
            args=[{'visible': [True, False, False]}, terrain_idx],
        ),
        dict(
            label='Color: relative pressure', method='restyle',
            args=[{'visible': [False, True, False]}, terrain_idx],
        ),
        dict(
            label='Color: snow drift', method='restyle',
            args=[{'visible': [False, False, True]}, terrain_idx],
        ),
    ]
    # Streamline direction buttons use method='skip'; visibility is updated
    # by the JS hook based on the combined (height, count, direction) state.
    direction_buttons = [
        dict(label='→ forward', method='skip', args=[{'_direction': 'forward'}]),
        dict(label='← backward', method='skip', args=[{'_direction': 'backward'}]),
        dict(label='↔ both', method='skip', args=[{'_direction': 'both'}]),
    ]
    updatemenus = [
        dict(
            type='buttons', direction='right',
            buttons=terrain_buttons,
            x=0.05, y=1.02, xanchor='left', yanchor='bottom',
            showactive=True, active=0,
            bgcolor='#f0f0f0', bordercolor='#888',
        ),
        # Streamline direction (fwd/bwd/both) — placed at the bottom-right,
        # just above the streamline-count slider (which is at x=0.68, y=0.04).
        # Buttons are self-labeling with arrow glyphs so no extra annotation
        # is needed near them.
        dict(
            type='buttons', direction='right',
            buttons=direction_buttons,
            x=0.68, y=0.13, xanchor='left', yanchor='bottom',
            showactive=True, active=0,
            bgcolor='#eaf2ff', bordercolor='#0d47a1',
            name='stream_direction',
        ),
    ]

    # ----- Layout (incl. "Controls" annotation above the slider area) -----
    x0, x1, y0, y1, z0, z1 = bundle.bounds
    extent_x = max(float(x1 - x0), 1.0)
    extent_y = max(float(y1 - y0), 1.0)
    extent_z = max(float(z1 - z0), 1.0)
    norm = max(extent_x, extent_y)
    # Fixed aspect ratio derived once from the domain bounds. Using
    # `aspectmode='manual'` (rather than 'data') prevents Plotly from
    # rescaling the box when traces are toggled on/off — which would otherwise
    # make the terrain look stretched vertically when high-altitude streamlines
    # become visible.
    aspect = dict(x=extent_x / norm, y=extent_y / norm, z=extent_z / norm)
    # Z-axis labels: the model runs in the domain (z = 0 at terrain min)
    # frame, but for display we re-add z_offset_applied so the user sees
    # real-world elevations. Geometry stays in domain frame; only the
    # tick labels are shifted.
    z_offset_applied = float(((saved_inputs.get('transform_meta') or {}).get('z_offset_applied')) or 0.0)
    z_label_shift = -z_offset_applied  # real = domain - z_offset_applied
    if abs(z_label_shift) > 0.5:
        n_z_ticks = 6
        tickvals_z = np.linspace(float(z0), float(z1), n_z_ticks).tolist()
        ticktext_z = [f'{v + z_label_shift:.0f}' for v in tickvals_z]
        zaxis_cfg = dict(
            title='z (m, real)', range=[z0, z1], autorange=False,
            tickmode='array', tickvals=tickvals_z, ticktext=ticktext_z,
        )
    else:
        zaxis_cfg = dict(title='z (m)', range=[z0, z1], autorange=False)
    fig.update_layout(
        # Title sits in the top margin, centered horizontally. Original
        # `x=0.02` collided with the left-side elevation/ground-pressure
        # colorbars; the centered placement plus a healthy top margin gives
        # the title room to render without being clipped at the figure top.
        title=dict(text=f"3D view — {domain_name}", x=0.5, xanchor='center', y=0.97, yanchor='top'),
        scene=dict(
            xaxis=dict(title='x (m, local)', range=[x0, x1], autorange=False),
            yaxis=dict(title='y (m, local)', range=[y0, y1], autorange=False),
            zaxis=zaxis_cfg,
            aspectmode='manual',
            aspectratio=aspect,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
        # Top margin holds title + buttons + streamlines-mode label.
        margin=dict(l=0, r=0, b=140, t=90),
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            x=0.02, y=0.95,
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#888', borderwidth=1,
        ),
        sliders=sliders,
        updatemenus=updatemenus,
        annotations=[
            dict(
                text='<b>Controls</b>',
                xref='paper', yref='paper',
                x=0.02, y=0.18, xanchor='left', yanchor='top',
                showarrow=False, font=dict(size=14, color='#0d47a1'),
            ),
            # Inline label sitting just above the fwd/bwd/both buttons,
            # which were moved to the bottom-right of the figure.
            dict(
                text='<i>Streamlines: integration mode</i>',
                xref='paper', yref='paper',
                x=0.68, y=0.205, xanchor='left', yanchor='bottom',
                showarrow=False, font=dict(size=11, color='#0d47a1'),
            ),
        ],
    )

    # Pack metadata that the JS hook needs.
    meta_for_js = {
        'stream_indices': [int(s) for s in trace_groups['streams']],
        'stream_index_grid_fwd': [None if i is None else int(i) for i in stream_index_grid_fwd],
        'stream_index_grid_bwd': [None if i is None else int(i) for i in stream_index_grid_bwd],
        'stream_index_grid_both': [None if i is None else int(i) for i in stream_index_grid_both],
        'n_heights': n_h,
        'n_counts': n_c,
        'default_h_idx': DEFAULT_HEIGHT_IDX,
        'default_c_idx': DEFAULT_COUNT_IDX,
        'default_direction': 'forward',
    }
    return fig, meta_for_js


def _streamline_js_hook(meta: dict) -> str:
    """JS hook driving streamline visibility from (height, count, direction).

    Plotly slider/button changes for those three controls fire as 'skip'
    actions; this script catches them and computes which subset of the
    pre-rendered streamline traces should be visible.
    """
    import json as _json
    payload = _json.dumps({
        'all': meta['stream_indices'],
        'grid_fwd': meta['stream_index_grid_fwd'],
        'grid_bwd': meta['stream_index_grid_bwd'],
        'grid_both': meta.get('stream_index_grid_both') or [None] * (int(meta['n_heights']) * int(meta['n_counts'])),
        'n_h': meta['n_heights'],
        'n_c': meta['n_counts'],
        'h_idx': meta['default_h_idx'],
        'c_idx': meta['default_c_idx'],
        'dir': meta.get('default_direction', 'forward'),
    })
    return r"""
<script>
(function() {
  function init() {
    var divs = document.getElementsByClassName('plotly-graph-div');
    if (!divs || !divs.length) { setTimeout(init, 50); return; }
    var gd = divs[0];
    if (!gd._fullLayout) { setTimeout(init, 50); return; }
    var STATE = """ + payload + r""";
    var curH = STATE.h_idx;
    var curC = STATE.c_idx;
    var curDir = STATE.dir || 'forward';

    function targetsForCurrent() {
      var key = curH * STATE.n_c + curC;
      var targets = [];
      if (curDir === 'both') {
        // Use the single concatenated bi-directional polyline (seeded
        // mid-domain). Falls back to fwd+bwd union if not available.
        var bo = STATE.grid_both ? STATE.grid_both[key] : null;
        if (bo !== null && bo !== undefined) {
          targets.push(bo);
        } else {
          var fi = STATE.grid_fwd[key];
          if (fi !== null && fi !== undefined) targets.push(fi);
          var bi = STATE.grid_bwd[key];
          if (bi !== null && bi !== undefined) targets.push(bi);
        }
      } else if (curDir === 'backward') {
        var bi2 = STATE.grid_bwd[key];
        if (bi2 !== null && bi2 !== undefined) targets.push(bi2);
      } else {
        var fi2 = STATE.grid_fwd[key];
        if (fi2 !== null && fi2 !== undefined) targets.push(fi2);
      }
      return targets;
    }

    function applyVisibility() {
      var n = STATE.all.length;
      var visible = new Array(n).fill(false);
      var targets = targetsForCurrent();
      for (var k = 0; k < targets.length; k++) {
        var pos = STATE.all.indexOf(targets[k]);
        if (pos >= 0) visible[pos] = true;
      }
      Plotly.restyle(gd, {visible: visible}, STATE.all);
    }

    gd.on('plotly_sliderchange', function(e) {
      try {
        var name = (e && e.slider && e.slider.name) || '';
        var prefix = (e && e.slider && e.slider.currentvalue && e.slider.currentvalue.prefix) || '';
        if (name === 'stream_height' || prefix.indexOf('height') >= 0) {
          curH = e.slider.active; applyVisibility();
        } else if (name === 'stream_count' || prefix.indexOf('count') >= 0) {
          curC = e.slider.active; applyVisibility();
        }
      } catch (err) { console.warn('streamline slider hook:', err); }
    });

    gd.on('plotly_buttonclicked', function(e) {
      try {
        var label = (e && e.button && e.button.label) || '';
        if (label.indexOf('forward') >= 0) { curDir = 'forward'; applyVisibility(); }
        else if (label.indexOf('backward') >= 0) { curDir = 'backward'; applyVisibility(); }
        else if (label.indexOf('both') >= 0) { curDir = 'both'; applyVisibility(); }
      } catch (err) { console.warn('streamline button hook:', err); }
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>
"""


def write_3d_html(out_path: Path, *, saved_inputs: dict, domain_name: str,
                   structure_stl_path: Optional[Path] = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, meta = build_3d_figure(saved_inputs, domain_name=domain_name,
                                structure_stl_path=structure_stl_path)
    html = fig.to_html(
        include_plotlyjs='/static/plotly.min.js', full_html=True,  # served same-origin by app.py: offline + CSP safe
        config={'displayModeBar': True, 'scrollZoom': True},
    )
    html = html.replace('</body>', _streamline_js_hook(meta) + '\n</body>')
    out_path.write_text(html, encoding='utf-8')
    return out_path


# ---------------------------------------------------------------------------
# Structure-focused 3D view
# ---------------------------------------------------------------------------
def _pick_structure_focus_bundle(saved_inputs: dict):
    """Prefer the first ROI bundle (higher resolution near the structure)."""
    roi_bundles = saved_inputs.get('roi_bundles') or {}
    roi_preds = saved_inputs.get('roi_preds') or {}
    if roi_bundles:
        label = next(iter(roi_bundles))
        return roi_bundles[label], roi_preds.get(label, roi_bundles[label].flow)
    return saved_inputs['bundle'], saved_inputs['pred_flow']


def _pressure_interpolator(bundle, pred_flow):
    from scipy.interpolate import RegularGridInterpolator
    pf = np.where(np.isfinite(pred_flow), pred_flow, 0.0).astype(np.float32)
    return RegularGridInterpolator(
        (bundle.x_coords.astype(np.float32),
         bundle.y_coords.astype(np.float32),
         bundle.z_levels.astype(np.float32)),
        pf[..., 3],
        bounds_error=False,
        fill_value=np.nan,
    )


def _structure_pressure_vertices(structure_stl_path: Optional[Path], *,
                                    focus_bundle, focus_pred_flow,
                                    structure_bounds: Optional[list] = None):
    """Sample predicted pressure at each STL vertex. Returns (verts, faces,
    p_vert) or None if STL/trimesh unavailable.

    If `structure_bounds` is provided, the mesh is filtered to keep only
    faces whose centroid sits inside one of those AABBs (with a 2 m
    margin). Used to clip the structure mesh to the chosen ROI's
    structure(s) only.
    """
    if not structure_stl_path or not Path(structure_stl_path).exists():
        return None
    try:
        import trimesh
    except Exception:
        return None
    try:
        mesh = trimesh.load_mesh(str(structure_stl_path))
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if len(verts) == 0 or len(faces) == 0:
            return None
        if structure_bounds:
            face_centroids = verts[faces].mean(axis=1)
            keep = np.zeros(len(faces), dtype=bool)
            margin = 2.0
            for sb in structure_bounds:
                try:
                    xmin, ymin, zmin = (float(v) for v in sb['min'])
                    xmax, ymax, zmax = (float(v) for v in sb['max'])
                except Exception:
                    continue
                inside = (
                    (face_centroids[:, 0] >= xmin - margin) & (face_centroids[:, 0] <= xmax + margin)
                    & (face_centroids[:, 1] >= ymin - margin) & (face_centroids[:, 1] <= ymax + margin)
                    & (face_centroids[:, 2] >= zmin - margin) & (face_centroids[:, 2] <= zmax + margin)
                )
                keep |= inside
            if bool(keep.any()):
                faces = faces[keep]
                # Compact vertex array so Plotly doesn't carry stray vertices.
                used_v = np.unique(faces.ravel())
                if used_v.size < len(verts):
                    remap = -np.ones(len(verts), dtype=np.int32)
                    remap[used_v] = np.arange(used_v.size, dtype=np.int32)
                    verts = verts[used_v]
                    faces = remap[faces]
        p_interp = _pressure_interpolator(focus_bundle, focus_pred_flow)
        p_vert = p_interp(verts).astype(np.float32)
        nan = ~np.isfinite(p_vert)
        if nan.any():
            cx = float(np.mean(verts[:, 0]))
            cy = float(np.mean(verts[:, 1]))
            cz = float(np.mean(verts[:, 2]))
            dxy = verts[nan, :3] - np.array([cx, cy, cz], dtype=np.float32)
            n = np.linalg.norm(dxy[:, :2], axis=-1, keepdims=True) + 1e-6
            dxy[:, :2] *= (0.5 / n)
            p_off = p_interp(verts[nan] + dxy).astype(np.float32)
            p_vert[nan] = np.where(np.isfinite(p_off), p_off, 0.0)
        return verts, faces, p_vert
    except Exception:
        return None


def _focus_camera_bounds(saved_inputs: dict, *, roi_label: Optional[str] = None,
                          margin_m: float = 25.0):
    """Pick a tight camera range for the chosen ROI.

    Priority:
      1. The named ROI's bundle bounds (when `roi_label` is given) — this
         is the natural view for "open structure 3D view of roi_XXX"; the
         camera matches the focused terrain + streamlines exactly.
      2. The first ROI bundle's bounds.
      3. The structures' union from the global meta, padded.
    """
    roi_bundles = saved_inputs.get('roi_bundles') or {}
    if roi_label and roi_label in roi_bundles:
        return tuple(float(v) for v in roi_bundles[roi_label].bounds)
    global_bundle = saved_inputs['bundle']
    if roi_bundles:
        rb = next(iter(roi_bundles.values()))
        return tuple(float(v) for v in rb.bounds)
    meta = global_bundle.meta if isinstance(global_bundle.meta, dict) else {}
    boxes = meta.get('structure_bounds') if isinstance(meta, dict) else None
    if boxes:
        xs0, xs1, ys0, ys1, zs0, zs1 = [], [], [], [], [], []
        for sb in boxes:
            try:
                xs0.append(float(sb['min'][0])); xs1.append(float(sb['max'][0]))
                ys0.append(float(sb['min'][1])); ys1.append(float(sb['max'][1]))
                zs0.append(float(sb['min'][2])); zs1.append(float(sb['max'][2]))
            except Exception:
                continue
        if xs0:
            return (
                min(xs0) - margin_m, max(xs1) + margin_m,
                min(ys0) - margin_m, max(ys1) + margin_m,
                min(zs0) - margin_m, max(zs1) + 2 * margin_m,
            )
    return tuple(float(v) for v in global_bundle.bounds)


STRUCTURE_VIEW_STREAMLINE_HEIGHTS_M = [2, 5, 10, 15, 20]
STRUCTURE_VIEW_STREAMLINE_COUNTS = [10, 20, 30, 40, 50]


def _pick_structure_focus_bundle_for_roi(saved_inputs: dict, roi_label: Optional[str] = None):
    roi_bundles = saved_inputs.get('roi_bundles') or {}
    roi_preds = saved_inputs.get('roi_preds') or {}
    if roi_label and roi_label in roi_bundles:
        return roi_bundles[roi_label], roi_preds.get(roi_label, roi_bundles[roi_label].flow)
    if roi_bundles:
        label = next(iter(roi_bundles))
        return roi_bundles[label], roi_preds.get(label, roi_bundles[label].flow)
    return saved_inputs['bundle'], saved_inputs['pred_flow']


def _shared_pressure_limits(*arrays) -> float:
    """Compute a symmetric (around 0) limit using the 98th percentile of |p|.

    Using a percentile (instead of nanmax) prevents single outlier cells from
    saturating the colormap to ±100 Pa when 99% of the field is in ±10 Pa.
    """
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        v = np.asarray(arr, dtype=np.float32).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(np.abs(v))
    if not vals:
        return 1.0
    all_abs = np.concatenate(vals)
    if all_abs.size == 0:
        return 1.0
    return float(max(np.percentile(all_abs, 98.0), 1e-3))


# Wake-seeded recirculation streamlines in the structure view. Disabled for
# now (visually too cluttered); flip to True to bring them back — the seeding
# and integration code below is kept functional.
ENABLE_WAKE_STREAMLINES = False


def _wake_seed_positions(structure_bounds: list, x_grid, y_grid, z_grid,
                         *, per_struct: int = 24, cap: int = 144) -> np.ndarray:
    """Seeds inside the recirculation region behind each structure.

    Domains are pre-rotated so the flow is +x: 'behind' = downstream in +x.
    Box per structure: x in [x_max + 0.15 H, x_max + 2.5 H], y across the
    structure width, z from just above the base to ~1.3x the structure top.
    These seeds, integrated in BOTH directions, are what actually reveals
    the wake recirculation that inlet-line seeds almost always miss.
    """
    seeds: list = []
    x_hi = float(x_grid[-1]) if len(x_grid) else 0.0
    z_hi = float(z_grid[-1]) if len(z_grid) else 0.0
    for sb in structure_bounds or []:
        try:
            xmin, ymin, zmin = (float(v) for v in sb['min'])
            xmax, ymax, zmax = (float(v) for v in sb['max'])
        except Exception:
            continue
        H = max(zmax - zmin, 0.5)
        x0 = min(xmax + 0.15 * H, x_hi - 1e-3)
        x1 = min(xmax + 2.5 * H, x_hi)
        if x1 - x0 < 0.05:
            continue
        z0 = zmin + 0.1 * H
        z1 = min(zmax + 0.3 * H, z_hi)
        xs = np.linspace(x0, x1, 4, dtype=np.float32)
        ys = np.linspace(ymin, ymax, 3, dtype=np.float32)
        zs = np.linspace(z0, max(z1, z0 + 0.1), 2, dtype=np.float32)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        seeds.append(pts[:per_struct])
    if not seeds:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.concatenate(seeds, axis=0).astype(np.float32)
    if out.shape[0] > cap:
        keep = np.linspace(0, out.shape[0] - 1, cap).astype(int)
        out = out[keep]
    return out


def build_structure_3d_figure(saved_inputs: dict, *, domain_name: str,
                               structure_stl_path: Optional[Path] = None,
                               roi_label: Optional[str] = None):
    import plotly.graph_objects as go
    focus_bundle, focus_pred = _pick_structure_focus_bundle_for_roi(saved_inputs, roi_label)

    # |U| scale for cones / streamlines.
    isf = (focus_bundle.is_fluid > 0.5)
    finite = np.isfinite(focus_pred).all(axis=-1)
    valid = isf & finite
    if bool(np.any(valid)):
        Umag_full = np.linalg.norm(focus_pred[valid][:, :3], axis=-1)
        umag_max = float(max(Umag_full.max(), 1e-3))
    else:
        umag_max = 1.0
    cone_sizeref = umag_max * 0.5

    traces: list = []

    # Display pressure in Pa after the global fluid-domain mean was removed.
    from units import RHO_AIR  # type: ignore  noqa: E402

    # --- Ground relative pressure (for the shared pressure scale). ---
    p_ground = RHO_AIR * _ground_pressure_field(focus_bundle, focus_pred)

    # --- Structure vertices + per-vertex pressure. Filter the STL to only
    # the structure(s) belonging to the chosen ROI so we don't render
    # neighbouring structures floating in the empty terrain. ---
    focus_meta = focus_bundle.meta if isinstance(focus_bundle.meta, dict) else {}
    focus_structure_bounds = focus_meta.get('structure_bounds') or []
    struct_data = _structure_pressure_vertices(
        structure_stl_path,
        focus_bundle=focus_bundle, focus_pred_flow=focus_pred,
        structure_bounds=focus_structure_bounds or None,
    )
    if struct_data is not None:
        verts, faces, p_vert_kin = struct_data
        p_vert = RHO_AIR * p_vert_kin  # m^2/s^2 -> Pa
    else:
        verts = faces = p_vert = None

    # Single symmetric colour scale shared by terrain ground + structure.
    p_lim = _shared_pressure_limits(p_ground, p_vert)

    # --- Terrain (downsampled, coloured by relative pressure) ---
    elev_raw = np.asarray(focus_bundle.terrain_raw['elevation'])  # (ny, nx)
    elev_ds, x_ds, y_ds, sx, sy = _downsample_terrain(
        elev_raw,
        np.asarray(focus_bundle.x_coords, dtype=np.float32),
        np.asarray(focus_bundle.y_coords, dtype=np.float32),
    )
    p_ground_ds = p_ground[::sy, ::sx]
    z_off_struct = float(((saved_inputs.get('transform_meta') or {}).get('z_offset_applied')) or 0.0)
    elev_real_ds = (elev_raw - z_off_struct)[::sy, ::sx]
    common_lighting = dict(ambient=0.55, diffuse=0.85, specular=0.12,
                           roughness=0.7, fresnel=0.1)
    common_lightpos = dict(x=50_000, y=50_000, z=100_000)
    terrain_p_trace = go.Surface(
        x=x_ds, y=y_ds, z=elev_ds,
        surfacecolor=p_ground_ds,
        colorscale='RdBu_r',
        cmin=-p_lim, cmax=p_lim,
        opacity=1.0, showscale=True,
        colorbar=dict(title='Relative p (Pa)', x=0.0, xanchor='left',
                      len=0.5, y=0.5),
        lighting=common_lighting, lightposition=common_lightpos,
        contours=dict(z=dict(show=False)),
        name='Terrain (relative pressure)',
        legendgroup='terrain', showlegend=True,
        text=_surface_hover_text(
            elev_real_ds, p_ground_ds,
            formatter=lambda z, p: (
                f'Terrain elevation: {z:.1f} m; relative p = {p:+.2f} Pa'
            ),
        ),
        hoverinfo='text',
        visible=True,
    )
    traces.append(terrain_p_trace)
    # Ground colour alternatives (toggled by the "Ground:" buttons): elevation
    # and the heuristic snow drift indicator. Same z geometry, hidden initially.
    terrain_elev_trace = go.Surface(
        x=x_ds, y=y_ds, z=elev_ds,
        surfacecolor=elev_real_ds,
        colorscale='earth',
        cmin=float(np.nanmin(elev_raw - z_off_struct)),
        cmax=float(np.nanmax(elev_raw - z_off_struct)),
        opacity=1.0, showscale=True,
        colorbar=dict(title='Elevation (m)', x=0.0, xanchor='left', len=0.5, y=0.5),
        lighting=common_lighting, lightposition=common_lightpos,
        contours=dict(z=dict(show=False)),
        name='Terrain (elevation)',
        legendgroup='terrain', showlegend=False,
        text=_surface_hover_text(
            elev_real_ds,
            formatter=lambda z: f'Terrain elevation: {z:.1f} m',
        ),
        hoverinfo='text',
        visible=False,
    )
    traces.append(terrain_elev_trace)
    speed_ground_ds = _ground_speed_field(focus_bundle, focus_pred)[::sy, ::sx]
    traces.append(_snow_surface_trace(
        x_ds, y_ds, elev_ds, speed_ground_ds,
        lighting=common_lighting, lightposition=common_lightpos, visible=False,
    ))
    terrain_idx = [0, 1, 2]

    # --- Structure mesh: same colour scale, no separate colorbar ---
    if struct_data is not None:
        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=p_vert,
            intensitymode='vertex',
            colorscale='RdBu_r',
            cmin=-p_lim, cmax=p_lim,
            showscale=False,
            opacity=1.0,
            flatshading=False,
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.2, roughness=0.6),
            name='Structure (relative pressure)',
            legendgroup='structures', showlegend=True,
            text=_surface_hover_text(
                p_vert,
                formatter=lambda p: f'Structure relative p = {p:+.2f} Pa',
            ),
            hoverinfo='text',
        ))
    else:
        traces.extend(_structure_traces(structure_stl_path))

    # --- Per-structure max-load / max-suction markers ---
    #
    # The model output is noisy at exact face vertices (mesh corners pin
    # nearest-cell sampling). To pick a representative peak, we scan the
    # fluid volume within ~1 m of each wall (using phi_wall) inside the
    # structure's AABB and report the highest / lowest relative pressure
    # found there. The marker is placed at that physical cell.
    #
    # Each pair (max + min) is rendered as a *separate* legendgroup so
    # the user can hide the markers without hiding the structure mesh.
    try:
        ri = focus_bundle  # ROI bundle (in structure view, this IS the ROI)
        meta_struct = ri.meta if isinstance(ri.meta, dict) else {}
        sb_list = focus_structure_bounds or []
        # If no per-structure bounds, fall back to a single pair around all verts.
        if not sb_list and verts is not None:
            sb_list = [{
                'min': verts.min(axis=0).tolist(),
                'max': verts.max(axis=0).tolist(),
                'label': '',
            }]
        # Build coordinate grids once
        x_phys = np.asarray(ri.x_coords, dtype=np.float32)
        y_phys = np.asarray(ri.y_coords, dtype=np.float32)
        z_phys = np.asarray(ri.z_levels, dtype=np.float32)
        p_kin_3d = np.asarray(focus_pred[..., 3], dtype=np.float32)
        p_pa_3d = RHO_AIR * p_kin_3d  # Pa
        # Uref for Cp
        abl_struct = meta_struct.get('ABL', {}) if isinstance(meta_struct, dict) else {}
        u_ref = float(abl_struct.get('Uref', abl_struct.get('Uref_mps', 1.0)) or 1.0)
        from units import cp_from_kinematic  # type: ignore  noqa: E402
        # phi_wall: |phi| <= max_shell => inside the 1m shell around walls.
        phi = ri.phi_wall
        max_shell_m = 1.0
        if phi is not None:
            phi_arr = np.asarray(phi, dtype=np.float32)
            near_wall = (np.abs(phi_arr) <= max_shell_m)
        else:
            near_wall = np.ones_like(p_kin_3d, dtype=bool)
        is_fluid_wide = _viz_fluid_mask(ri)
        finite_pred = np.isfinite(focus_pred).all(axis=-1)
        # Per-structure search:
        mx_x = []; mx_y = []; mx_z = []; mx_text = []
        mn_x = []; mn_y = []; mn_z = []; mn_text = []
        margin = 1.0
        ii_grid, jj_grid, kk_grid = np.meshgrid(
            np.arange(p_kin_3d.shape[0]),
            np.arange(p_kin_3d.shape[1]),
            np.arange(p_kin_3d.shape[2]),
            indexing='ij',
        )
        for sb in sb_list:
            try:
                xmin, ymin, zmin = (float(v) for v in sb['min'])
                xmax, ymax, zmax = (float(v) for v in sb['max'])
            except Exception:
                continue
            # Box of cells inside the AABB + 1m halo:
            in_box = (
                (x_phys[:, None, None] >= xmin - margin) & (x_phys[:, None, None] <= xmax + margin)
                & (y_phys[None, :, None] >= ymin - margin) & (y_phys[None, :, None] <= ymax + margin)
                & (z_phys[None, None, :] >= zmin - margin) & (z_phys[None, None, :] <= zmax + margin)
            )
            valid = in_box & near_wall & is_fluid_wide & finite_pred
            if not bool(valid.any()):
                continue
            p_search = np.where(valid, p_pa_3d, np.nan)
            try:
                i_max = int(np.nanargmax(p_search))
                i_min = int(np.nanargmin(p_search))
            except ValueError:
                continue
            for i_flat, kind, x_acc, y_acc, z_acc, text_acc in (
                (i_max, 'max', mx_x, mx_y, mx_z, mx_text),
                (i_min, 'min', mn_x, mn_y, mn_z, mn_text),
            ):
                ii = ii_grid.flat[i_flat]
                jj = jj_grid.flat[i_flat]
                kk = kk_grid.flat[i_flat]
                xq = float(x_phys[ii]); yq = float(y_phys[jj]); zq = float(z_phys[kk])
                # The model output peak comes from a cell inside the 1m
                # halo, which can sit a few decimeters off the wall. For
                # display we snap the marker to the closest point on the
                # structure AABB surface so the diamond clearly anchors to
                # the structure rather than floating in the fluid.
                xx = float(np.clip(xq, xmin, xmax))
                yy = float(np.clip(yq, ymin, ymax))
                zz = float(np.clip(zq, zmin, zmax))
                p_pa_val = float(p_pa_3d[ii, jj, kk])
                cp_val = float(cp_from_kinematic(p_kin_3d[ii, jj, kk], u_ref))
                x_acc.append(xx); y_acc.append(yy); z_acc.append(zz)
                label = 'max relative pressure' if kind == 'max' else 'max relative suction'
                text_acc.append(f'{label}: {p_pa_val:+.1f} Pa  (Cp = {cp_val:+.2f})')
        if mx_x:
            traces.append(go.Scatter3d(
                x=mx_x + mn_x, y=mx_y + mn_y, z=mx_z + mn_z,
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=(['#b2182b'] * len(mx_x)) + (['#2166ac'] * len(mn_x)),
                    symbol='diamond',
                ),
                text=mx_text + mn_text,
                textposition='top center',
                textfont=dict(size=11, color='#222'),
                name='Relative pressure extrema (Pa, Cp)',
                # Separate legendgroup so unticking these doesn't hide the
                # structure mesh.
                legendgroup='pressure_markers', showlegend=True,
                hoverinfo='text',
            ))
    except Exception:
        pass

    # --- Wind glyph cones (4 density levels, slider-driven) ---
    trace_groups: dict[str, list[int]] = {'glyphs': [], 'streams': []}
    for idx, n in enumerate(GLYPH_DENSITY_LEVELS):
        t = _make_cone_trace(
            focus_bundle, focus_pred, n,
            name=f'Wind ({n:,} glyphs)',
            visible=(idx == DEFAULT_GLYPH_IDX),
            umag_max=umag_max, sizeref=cone_sizeref,
        )
        trace_groups['glyphs'].append(len(traces))
        traces.append(t)

    # --- Streamlines: heights x counts grid (terrain-clipped) ---
    dx = float(np.median(np.diff(focus_bundle.x_coords))) if len(focus_bundle.x_coords) > 1 else 1.0
    dy = float(np.median(np.diff(focus_bundle.y_coords))) if len(focus_bundle.y_coords) > 1 else 1.0
    ds = max(_STREAM_DS_FACTOR * min(dx, dy), 0.5)
    fx0_b, fx1_b, fy0_b, fy1_b, _, _ = focus_bundle.bounds
    diag_focus = float(np.hypot(fx1_b - fx0_b, fy1_b - fy0_b))
    max_steps_focus = max(int(_STREAM_MAX_STEPS), int(2.0 * diag_focus / max(ds, 0.1)))
    abl = focus_bundle.meta.get('ABL', {}) if isinstance(focus_bundle.meta, dict) else {}
    flow_dir = np.asarray(abl.get('flowDir', [1.0, 0.0, 0.0]), dtype=np.float32)
    from scipy.interpolate import RegularGridInterpolator
    elev_mean = float(np.nanmean(np.asarray(focus_bundle.terrain_raw['elevation'])))
    UVW, x_grid, y_grid, z_grid = _build_uvw_interpolator(focus_bundle, focus_pred)
    elev_interp = RegularGridInterpolator(
        (y_grid, x_grid),
        np.asarray(focus_bundle.terrain_raw['elevation'], dtype=np.float32),
        bounds_error=False, fill_value=elev_mean,
    )
    heights = STRUCTURE_VIEW_STREAMLINE_HEIGHTS_M
    counts = STRUCTURE_VIEW_STREAMLINE_COUNTS
    n_h = len(heights); n_c = len(counts)
    default_h_idx = 1  # 5 m
    default_c_idx = 1  # 20 lines
    stream_index_grid_fwd: list[Optional[int]] = [None] * (n_h * n_c)
    stream_index_grid_bwd: list[Optional[int]] = [None] * (n_h * n_c)
    stream_index_grid_both: list[Optional[int]] = [None] * (n_h * n_c)
    seeds_inlet: list[np.ndarray] = []
    seeds_outlet: list[np.ndarray] = []
    seeds_middle: list[np.ndarray] = []
    span_per_combo: list[int] = []
    for h_m in heights:
        for count in counts:
            seeds_inlet.append(_seed_positions(int(count), float(h_m),
                                                elev_interp, flow_dir, x_grid, y_grid, side='inlet'))
            seeds_outlet.append(_seed_positions(int(count), float(h_m),
                                                 elev_interp, flow_dir, x_grid, y_grid, side='outlet'))
            seeds_middle.append(_seed_positions(int(count), float(h_m),
                                                 elev_interp, flow_dir, x_grid, y_grid, side='middle'))
            span_per_combo.append(int(count))
    inlet_concat = np.concatenate(seeds_inlet, axis=0) if seeds_inlet else np.zeros((0, 3), np.float32)
    outlet_concat = np.concatenate(seeds_outlet, axis=0) if seeds_outlet else np.zeros((0, 3), np.float32)
    middle_concat = np.concatenate(seeds_middle, axis=0) if seeds_middle else np.zeros((0, 3), np.float32)
    paths_fwd = _integrate_streamlines_batch(
        inlet_concat, UVW, x_grid, y_grid, z_grid,
        ds=ds, max_steps=max_steps_focus,
        elev_interp=elev_interp, direction='forward',
    )
    paths_bwd = _integrate_streamlines_batch(
        outlet_concat, UVW, x_grid, y_grid, z_grid,
        ds=ds, max_steps=max_steps_focus,
        elev_interp=elev_interp, direction='backward',
    )
    paths_mid_fwd = _integrate_streamlines_batch(
        middle_concat, UVW, x_grid, y_grid, z_grid,
        ds=ds, max_steps=max_steps_focus,
        elev_interp=elev_interp, direction='forward',
    )
    paths_mid_bwd = _integrate_streamlines_batch(
        middle_concat, UVW, x_grid, y_grid, z_grid,
        ds=ds, max_steps=max_steps_focus,
        elev_interp=elev_interp, direction='backward',
    )
    paths_both = _concat_bidirectional_paths(paths_mid_bwd, paths_mid_fwd)
    cur = 0; combo = 0
    for hi, h_m in enumerate(heights):
        for ci, count in enumerate(counts):
            n = span_per_combo[combo]
            paths_f = paths_fwd[cur:cur + n]
            paths_b = paths_bwd[cur:cur + n]
            paths_bo = paths_both[cur:cur + n]
            cur += n; combo += 1
            is_default = (hi == default_h_idx) and (ci == default_c_idx)
            tf = _streamline_trace_from_paths(
                paths_f,
                name=f'Streamlines fwd ({count} @ +{h_m} m)',
                visible=bool(is_default), umag_max=umag_max,
            )
            if tf is not None:
                stream_index_grid_fwd[hi * n_c + ci] = len(traces)
                trace_groups['streams'].append(len(traces))
                traces.append(tf)
            tb = _streamline_trace_from_paths(
                paths_b,
                name=f'Streamlines bwd ({count} @ +{h_m} m)',
                visible=False, umag_max=umag_max,
            )
            if tb is not None:
                stream_index_grid_bwd[hi * n_c + ci] = len(traces)
                trace_groups['streams'].append(len(traces))
                traces.append(tb)
            tboth = _streamline_trace_from_paths(
                paths_bo,
                name=f'Streamlines both ({count} @ +{h_m} m)',
                visible=False, umag_max=umag_max,
            )
            if tboth is not None:
                stream_index_grid_both[hi * n_c + ci] = len(traces)
                trace_groups['streams'].append(len(traces))
                traces.append(tboth)

    # --- Wake streamlines (recirculation) ---
    # Seeds placed in the wake box behind each structure, integrated in BOTH
    # directions with a finer step, so closed/reversed wake flow shows up.
    # One trace, toggled from the legend; not driven by the H/C sliders.
    try:
        wake_seeds = (
            _wake_seed_positions(focus_structure_bounds, x_grid, y_grid, z_grid)
            if ENABLE_WAKE_STREAMLINES else np.zeros((0, 3), dtype=np.float32)
        )
        if wake_seeds.shape[0] > 0:
            ds_wake = max(0.5 * ds, 0.25)
            wf = _integrate_streamlines_batch(
                wake_seeds, UVW, x_grid, y_grid, z_grid,
                ds=ds_wake, max_steps=max_steps_focus,
                elev_interp=elev_interp, direction='forward')
            wb = _integrate_streamlines_batch(
                wake_seeds, UVW, x_grid, y_grid, z_grid,
                ds=ds_wake, max_steps=max_steps_focus,
                elev_interp=elev_interp, direction='backward')
            wt = _streamline_trace_from_paths(
                _concat_bidirectional_paths(wb, wf),
                name='Wake streamlines (recirculation)',
                visible=True, umag_max=umag_max)
            if wt is not None:
                wt.update(legendgroup='wake', showlegend=True)
                traces.append(wt)
    except Exception:
        pass

    # --- Persistent |U| Viridis colorbar (dummy) ---
    traces.append(_colorbar_keeper(umag_max))

    # --- Legend handles for glyph / stream groups ---
    traces.append(_legend_handle('glyphs', 'Wind glyphs', '#fde725'))
    traces.append(_legend_handle('streams', 'Streamlines', '#5ec962'))

    fig = go.Figure(traces)

    # ----- Sliders -----
    glyph_idx = trace_groups['glyphs']
    stream_idx = trace_groups['streams']

    glyph_steps = []
    for i, n in enumerate(GLYPH_DENSITY_LEVELS):
        visible = [False] * len(glyph_idx)
        if i < len(glyph_idx):
            visible[i] = True
        glyph_steps.append(dict(
            method='restyle',
            args=[{'visible': visible}, glyph_idx],
            label=(f'{n//1000}k' if n >= 1000 else str(n)),
        ))
    height_steps = [dict(method='skip', label=f'{h} m') for h in heights]
    count_steps = [dict(method='skip', label=f'{c}') for c in counts]

    sliders = [
        dict(
            active=DEFAULT_GLYPH_IDX,
            x=0.05, y=0.04, len=0.27, xanchor='left', yanchor='top',
            currentvalue=dict(prefix='Glyph density: ', font=dict(size=12)),
            steps=glyph_steps, pad=dict(t=10, b=4),
            name='glyphs',
        ),
        dict(
            active=default_h_idx,
            x=0.36, y=0.04, len=0.27, xanchor='left', yanchor='top',
            currentvalue=dict(prefix='Streamline height: ', font=dict(size=12)),
            steps=height_steps, pad=dict(t=10, b=4),
            name='stream_height',
        ),
        dict(
            active=default_c_idx,
            x=0.68, y=0.04, len=0.27, xanchor='left', yanchor='top',
            currentvalue=dict(prefix='Streamline count: ', font=dict(size=12)),
            steps=count_steps, pad=dict(t=10, b=4),
            name='stream_count',
        ),
    ]

    fx0, fx1, fy0, fy1, fz0, fz1 = _focus_camera_bounds(saved_inputs, roi_label=roi_label)
    extent_x = max(fx1 - fx0, 1.0)
    extent_y = max(fy1 - fy0, 1.0)
    extent_z = max(fz1 - fz0, 1.0)
    norm = max(extent_x, extent_y)
    aspect = dict(x=extent_x / norm, y=extent_y / norm, z=extent_z / norm)
    title_suffix = f' (ROI {roi_label})' if roi_label else ''
    # Z tick labels in REAL altitude (m a.s.l.), same convention as the global
    # view: geometry stays in the domain frame (z=0 at terrain min) so all
    # traces align; only the tick text is shifted by -z_offset_applied.
    z_label_shift = -z_off_struct
    if abs(z_label_shift) > 0.5:
        tickvals_z = np.linspace(float(fz0), float(fz1), 6).tolist()
        ticktext_z = [f'{v + z_label_shift:.0f}' for v in tickvals_z]
        zaxis_cfg = dict(title='z (m a.s.l.)', range=[fz0, fz1], autorange=False,
                         tickmode='array', tickvals=tickvals_z, ticktext=ticktext_z)
    else:
        zaxis_cfg = dict(title='z (m)', range=[fz0, fz1], autorange=False)
    fig.update_layout(
        title=dict(text=f"3D structure view — {domain_name}{title_suffix}",
                   x=0.5, xanchor='center', y=0.97, yanchor='top'),
        scene=dict(
            xaxis=dict(title='x (m)', range=[fx0, fx1], autorange=False),
            yaxis=dict(title='y (m)', range=[fy0, fy1], autorange=False),
            zaxis=zaxis_cfg,
            aspectmode='manual',
            aspectratio=aspect,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
        ),
        margin=dict(l=0, r=0, b=140, t=90),
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            x=0.02, y=0.95,
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#888', borderwidth=1,
        ),
        sliders=sliders,
        updatemenus=[
            # Ground colour mode — same placement as the global view.
            dict(
                type='buttons', direction='right',
                buttons=[
                    dict(label='Ground: relative pressure', method='restyle',
                         args=[{'visible': [True, False, False]}, terrain_idx]),
                    dict(label='Ground: elevation', method='restyle',
                         args=[{'visible': [False, True, False]}, terrain_idx]),
                    dict(label='Ground: snow drift', method='restyle',
                         args=[{'visible': [False, False, True]}, terrain_idx]),
                ],
                x=0.05, y=1.02, xanchor='left', yanchor='bottom',
                showactive=True, active=0,
                bgcolor='#f0f0f0', bordercolor='#888',
            ),
            # Streamline direction — bottom-right above the count slider,
            # same placement as the global view.
            dict(
                type='buttons', direction='right',
                buttons=[
                    dict(label='→ forward', method='skip', args=[{'_direction': 'forward'}]),
                    dict(label='← backward', method='skip', args=[{'_direction': 'backward'}]),
                    dict(label='↔ both', method='skip', args=[{'_direction': 'both'}]),
                ],
                x=0.68, y=0.13, xanchor='left', yanchor='bottom',
                showactive=True, active=0,
                bgcolor='#eaf2ff', bordercolor='#0d47a1',
                name='stream_direction',
            ),
        ],
        annotations=[
            dict(
                text='<b>Controls</b>',
                xref='paper', yref='paper',
                x=0.02, y=0.18, xanchor='left', yanchor='top',
                showarrow=False, font=dict(size=14, color='#0d47a1'),
            ),
            dict(
                text='<i>Streamlines: integration mode</i>',
                xref='paper', yref='paper',
                x=0.68, y=0.205, xanchor='left', yanchor='bottom',
                showarrow=False, font=dict(size=11, color='#0d47a1'),
            ),
        ],
    )

    meta_for_js = {
        'stream_indices': [int(s) for s in trace_groups['streams']],
        'stream_index_grid_fwd': [None if i is None else int(i) for i in stream_index_grid_fwd],
        'stream_index_grid_bwd': [None if i is None else int(i) for i in stream_index_grid_bwd],
        'stream_index_grid_both': [None if i is None else int(i) for i in stream_index_grid_both],
        'n_heights': n_h,
        'n_counts': n_c,
        'default_h_idx': default_h_idx,
        'default_c_idx': default_c_idx,
        'default_direction': 'forward',
    }
    return fig, meta_for_js


def write_structure_3d_html(out_path: Path, *, saved_inputs: dict, domain_name: str,
                             structure_stl_path: Optional[Path] = None,
                             roi_label: Optional[str] = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, meta = build_structure_3d_figure(
        saved_inputs, domain_name=domain_name,
        structure_stl_path=structure_stl_path, roi_label=roi_label,
    )
    html = fig.to_html(
        include_plotlyjs='/static/plotly.min.js', full_html=True,  # served same-origin by app.py: offline + CSP safe
        config={'displayModeBar': True, 'scrollZoom': True},
    )
    html = html.replace('</body>', _streamline_js_hook(meta) + '\n</body>')
    out_path.write_text(html, encoding='utf-8')
    return out_path
