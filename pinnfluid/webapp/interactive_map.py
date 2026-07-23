"""Interactive 2D wind / pressure map for a saved prediction.

A top-down plan view of the whole domain, drawn as a Plotly heatmap, oriented
the same way as the PDF report (north-up: the wind-aligned CFD frame is rotated
back by 270 - wind_from degrees). The full domain is the base layer; wherever a
region of interest was refined, its fine (0.5 m) field is overlaid at its true
position, so zooming into the structures reveals detail the coarse background
cannot show. A dropdown selects the height above ground (terrain-following) for
the wind-speed view, plus relative pressure and terrain altitude. Faint terrain
contours persist across every view. Hover reads the exact value.

Reuses the exact slicing the static PNG plots use, so the map and the PDF agree.
Served same-origin with the local plotly.js, so it works offline and under a
strict content-security policy.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

# Heights above ground (terrain-following) offered for the wind-speed view.
WIND_HEIGHTS_M = (2.0, 5.0, 10.0, 20.0, 40.0)
_ROI_HEIGHT_TOL_M = 3.0
# Narrow fields (e.g. a flat domain at 9-11 m/s) look falsely dramatic with a
# tight colour bar, so the wind scale is widened to at least this span.
WIND_MIN_SPAN_MS = 6.0


def _load_pred_flow(cfd_dir: Path) -> np.ndarray:
    with np.load(Path(cfd_dir) / "flow.npz") as f:
        return np.stack([f["Ux"], f["Uy"], f["Uz"], f["p"]], axis=-1)


def _altitude_offset(case_dir: Path) -> float:
    """Metres to add to the domain-frame terrain to recover true altitude
    (m above sea level). The CFD domain subtracts its lowest terrain point so it
    starts near zero; transform.json records that shift. 0.0 if unavailable."""
    import json
    tf = Path(case_dir).parent / "case" / "triSurface" / "transform.json"
    try:
        return -float(json.loads(tf.read_text()).get("z_offset_applied", 0.0))
    except Exception:
        return 0.0


def _broaden(vmin: float, vmax: float, min_span: float) -> tuple:
    """Widen a colour range to at least `min_span`, keeping it centred."""
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return vmin, vmax
    if (vmax - vmin) < min_span:
        mid = 0.5 * (vmin + vmax)
        return mid - 0.5 * min_span, mid + 0.5 * min_span
    return vmin, vmax


def _rotpt(px: float, py: float, angle_deg: float, cx: float, cy: float) -> tuple:
    """Rotate a point CCW by angle_deg about (cx, cy) — matches the PDF's map."""
    a = math.radians(angle_deg)
    dx, dy = px - cx, py - cy
    return (cx + dx * math.cos(a) - dy * math.sin(a),
            cy + dx * math.sin(a) + dy * math.cos(a))


def _disp(x, y, arr_xy, angle, cx, cy):
    """Downsample an [x, y] field and return (x1d, y1d, z_yx) for a heatmap,
    rotated CCW by `angle` degrees about (cx, cy) so the map is north-up."""
    from vis.domain_report import _downsample_map  # type: ignore
    xm, ym, ds = _downsample_map(x, y, arr_xy)
    z = np.asarray(ds, dtype=float).T  # [y, x]
    if abs(angle) < 1e-6:
        return xm, ym, z
    from scipy.ndimage import rotate as ndrotate  # type: ignore
    # scipy's array rotation runs opposite to the report's CCW display rotation
    # in this [y, x] frame, so negate; the world extent below still uses +angle.
    zr = ndrotate(z, -angle, axes=(1, 0), reshape=True, order=1,
                  cval=np.nan, mode="constant")
    corners = [(xm.min(), ym.min()), (xm.max(), ym.min()),
               (xm.max(), ym.max()), (xm.min(), ym.max())]
    rc = [_rotpt(px, py, angle, cx, cy) for px, py in corners]
    xs = [p[0] for p in rc]; ys = [p[1] for p in rc]
    x1d = np.linspace(min(xs), max(xs), zr.shape[1])
    y1d = np.linspace(min(ys), max(ys), zr.shape[0])
    return x1d, y1d, zr


def _box_path(x0, y0, x1, y1, angle, cx, cy) -> str:
    """SVG path (data coords) for a rectangle rotated about (cx, cy)."""
    pts = [_rotpt(x0, y0, angle, cx, cy), _rotpt(x1, y0, angle, cx, cy),
           _rotpt(x1, y1, angle, cx, cy), _rotpt(x0, y1, angle, cx, cy)]
    return "M " + " L ".join(f"{px:.2f},{py:.2f}" for px, py in pts) + " Z"


def _footprint_path(points, angle, cx, cy) -> str:
    """SVG path for an oriented structure footprint in the north-up frame."""
    pts = [_rotpt(float(p[0]), float(p[1]), angle, cx, cy) for p in points]
    return "M " + " L ".join(f"{px:.2f},{py:.2f}" for px, py in pts) + " Z"


def write_map_html(out_path: Path, *, case_dir: Path, domain_name: str,
                   pred_flow: "np.ndarray | None" = None,
                   roi_pred_flows: Optional[dict] = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import plotly.graph_objects as go  # type: ignore
    from plots import _common_axes, _map_layers_filled  # type: ignore
    from vis.domain_report import (  # type: ignore
        _display_rotation_deg, _map_layers, _map_quantiles, _surface_pressure_map,
        _terrain_xy,
    )
    from units import RHO_AIR  # type: ignore

    case_dir = Path(case_dir)
    if pred_flow is None:
        pred_flow = _load_pred_flow(case_dir)
    gctx = _common_axes(case_dir, pred_flow)
    gx, gy = gctx["x"], gctx["y"]
    alt_off = _altitude_offset(case_dir)
    angle = _display_rotation_deg(gctx["meta"])  # 270 - wind_from; north-up
    cx = 0.5 * (float(gx.min()) + float(gx.max()))
    cy = 0.5 * (float(gy.min()) + float(gy.max()))

    rois = []
    roi_root = case_dir / "roi"
    if roi_root.exists():
        for rdir in sorted(p for p in roi_root.iterdir() if p.is_dir()):
            if (rdir / "meta.json").exists() and (rdir / "flow.npz").exists():
                roi_flow = (roi_pred_flows or {}).get(rdir.name)
                if roi_flow is None:
                    roi_flow = _load_pred_flow(rdir)
                rois.append(_common_axes(rdir, roi_flow))

    wvmin, wvmax = _broaden(*_map_quantiles(
        _map_layers_filled(gctx["pred"], 10.0)["speed_mag"]), WIND_MIN_SPAN_MS)

    fig = go.Figure()
    views = []

    def _add(x, y, z, *, scale, vmin, vmax, showscale, name, unit, reverse=False):
        fig.add_trace(go.Heatmap(
            x=x, y=y, z=z, colorscale=scale, reversescale=reverse,
            zmin=vmin, zmax=vmax, zsmooth="best", showscale=showscale,
            colorbar=dict(title=unit) if showscale else None, name=name,
            hovertemplate="x = %{x:.0f} m<br>y = %{y:.0f} m<br>"
                          + unit.split(" ")[0] + " = %{z:.2f}<extra></extra>",
        ))
        return len(fig.data) - 1

    for h in WIND_HEIGHTS_M:
        idxs = []
        xm, ym, z = _disp(gx, gy, _map_layers_filled(gctx["pred"], h)["speed_mag"], angle, cx, cy)
        idxs.append(_add(xm, ym, z, scale="Viridis", vmin=wvmin, vmax=wvmax,
                         showscale=True, name=f"wind {h:.0f} m", unit="|U| [m/s]"))
        for rctx in rois:
            lay = _map_layers(rctx["pred"], h)
            az = np.asarray(lay.get("actual_zrel"), dtype=float)
            if not np.isfinite(az).any() or float(np.nanmedian(az)) < h - _ROI_HEIGHT_TOL_M:
                continue
            rxm, rym, rz = _disp(rctx["x"], rctx["y"], lay["speed_mag"], angle, cx, cy)
            idxs.append(_add(rxm, rym, rz, scale="Viridis", vmin=wvmin, vmax=wvmax,
                             showscale=False, name=f"wind ROI {h:.0f} m", unit="|U| [m/s]"))
        views.append((f"Wind speed - {h:.0f} m above ground", idxs,
                      f"{domain_name} - wind speed at {h:.0f} m above ground"))

    # Relative surface pressure.
    p_idxs = []
    g_p = RHO_AIR * _surface_pressure_map(gctx["pred"])
    pvv = float(np.nanmax(np.abs(g_p))) if np.isfinite(g_p).any() else 1.0
    xm, ym, z = _disp(gx, gy, g_p, angle, cx, cy)
    p_idxs.append(_add(xm, ym, z, scale="RdBu", vmin=-pvv, vmax=pvv, reverse=True,
                       showscale=True, name="relative pressure", unit="p_rel [Pa]"))
    for rctx in rois:
        rp = RHO_AIR * _surface_pressure_map(rctx["pred"])
        if not np.isfinite(rp).any():
            continue
        rxm, rym, rz = _disp(rctx["x"], rctx["y"], rp, angle, cx, cy)
        p_idxs.append(_add(rxm, rym, rz, scale="RdBu", vmin=-pvv, vmax=pvv, reverse=True,
                           showscale=False, name="relative pressure ROI", unit="p_rel [Pa]"))
    views.append(("Relative surface pressure", p_idxs,
                  f"{domain_name} - relative surface pressure "
                  "(global fluid-domain mean = 0)"))

    # Terrain altitude (true m a.s.l. when the offset is known).
    # _terrain_xy transposes the raw [y, x] elevation into the [x, y] frame the
    # flow fields (and the PDF plots) use. Without it the terrain view is
    # transposed, which looks like a 90-degree-plus-mirror rotation.
    unit_alt = "altitude [m]" if alt_off else "elevation [m]"
    t_idxs = []
    g_el = np.asarray(_terrain_xy(gctx["truth"]), dtype=float) + alt_off
    evmin = float(np.nanmin(g_el)) if np.isfinite(g_el).any() else 0.0
    evmax = float(np.nanmax(g_el)) if np.isfinite(g_el).any() else 1.0
    xm, ym, z = _disp(gx, gy, g_el, angle, cx, cy)
    t_idxs.append(_add(xm, ym, z, scale="Earth", vmin=evmin, vmax=evmax,
                       showscale=True, name="terrain", unit=unit_alt))
    for rctx in rois:
        r_el = np.asarray(_terrain_xy(rctx["truth"]), dtype=float) + alt_off
        if not np.isfinite(r_el).any():
            continue
        rxm, rym, rz = _disp(rctx["x"], rctx["y"], r_el, angle, cx, cy)
        t_idxs.append(_add(rxm, rym, rz, scale="Earth", vmin=evmin, vmax=evmax,
                           showscale=False, name="terrain ROI", unit=unit_alt))
    alt_note = " (m a.s.l.)" if alt_off else ""
    views.append(("Terrain elevation", t_idxs, f"{domain_name} - terrain altitude{alt_note}"))

    # Faint terrain contours (fine 20 m, heavier 100 m), persistent on every view.
    persistent = []
    ctx_c, cyy, czz = _disp(gx, gy, g_el, angle, cx, cy)
    if np.isfinite(czz).any():
        lo, hi = float(np.nanmin(g_el)), float(np.nanmax(g_el))
        f0 = math.floor(lo / 20.0) * 20.0
        b0 = math.floor(lo / 100.0) * 100.0
        for size, w, col, lbl in ((20.0, 0.5, "rgba(55,55,55,0.35)", False),
                                  (100.0, 1.4, "rgba(25,25,25,0.55)", True)):
            persistent.append(len(fig.data))
            fig.add_trace(go.Contour(
                x=ctx_c, y=cyy, z=czz, autocontour=False,
                contours=dict(start=(b0 if lbl else f0), end=hi, size=size,
                              coloring="none", showlabels=lbl,
                              labelfont=dict(size=9, color="rgba(30,30,30,0.7)")),
                line=dict(width=w, color=col), showscale=False, hoverinfo="skip",
                name=f"contours {size:.0f} m",
            ))

    # Structure footprints and ROI extents, rotated to match.
    shapes = []
    for sb in gctx["boxes"]:
        try:
            footprint = sb.get("footprint_xy")
            if isinstance(footprint, (list, tuple)) and len(footprint) >= 3:
                path = _footprint_path(footprint, angle, cx, cy)
            else:
                path = _box_path(float(sb["min"][0]), float(sb["min"][1]),
                                 float(sb["max"][0]), float(sb["max"][1]),
                                 angle, cx, cy)
            shapes.append(dict(type="path",
                               path=path,
                               line=dict(color="white", width=1.3),
                               fillcolor="rgba(20,20,20,0.12)", layer="above"))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    for rctx in rois:
        rx, ry = rctx["x"], rctx["y"]
        shapes.append(dict(type="path",
                           path=_box_path(float(rx.min()), float(ry.min()),
                                          float(rx.max()), float(ry.max()), angle, cx, cy),
                           line=dict(color="#ffd54f", width=1.0, dash="dot"),
                           layer="above"))

    n = len(fig.data)
    default = set(views[0][1]) | set(persistent)
    for i in range(n):
        fig.data[i].visible = i in default

    def _mask(idxs):
        keep = set(idxs) | set(persistent)
        return [i in keep for i in range(n)]

    buttons = [
        dict(label=lbl, method="update", args=[{"visible": _mask(idxs)}])
        for (lbl, idxs, _title) in views
    ]

    fig.update_layout(
        title=None,
        shapes=shapes,
        xaxis=dict(title="x [m]", constrain="domain", visible=False),
        yaxis=dict(scaleanchor="x", scaleratio=1, visible=False),
        # Contours are legend-toggled. Keep that legend outside the map on the
        # left so it never covers the active field's colorbar on the right.
        margin=dict(l=150, r=20, t=55, b=20),
        legend=dict(
            title=dict(text="Contours"),
            x=-0.02, y=0.5, xanchor="right", yanchor="middle",
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(90,90,90,0.45)", borderwidth=1,
        ),
        updatemenus=[dict(type="dropdown", direction="down", x=0.0, y=1.16,
                          xanchor="left", showactive=True, buttons=buttons)],
        annotations=[dict(x=0.0, y=1.20, xref="paper", yref="paper", showarrow=False,
                          xanchor="left", text="View:", font=dict(size=12))],
    )

    html = fig.to_html(
        include_plotlyjs='/static/plotly.min.js', full_html=True,
        config={'displayModeBar': True, 'scrollZoom': True, 'responsive': True},
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path
