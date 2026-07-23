"""Pressure-reference helpers for user-facing prediction products.

The surrogate predicts kinematic pressure in the OpenFOAM training gauge.
Public plots and reports use a deterministic relative-pressure convention:
the arithmetic mean over valid global fluid cells is zero. The raw model
output remains unchanged for NPZ/VTK exports.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


PRESSURE_REFERENCE_CONVENTION = "global_fluid_arithmetic_mean_zero"
PRESSURE_REFERENCE_LABEL = "relative pressure (global fluid-domain mean = 0)"


def global_pressure_reference_kinematic(
    pred_flow: np.ndarray,
    is_fluid: np.ndarray,
) -> float:
    """Return the global fluid-cell mean kinematic pressure."""
    flow = np.asarray(pred_flow)
    mask = (np.asarray(is_fluid) > 0.5) & np.isfinite(flow[..., 3])
    if not bool(np.any(mask)):
        return 0.0
    return float(np.mean(flow[..., 3][mask], dtype=np.float64))


def referenced_flow_copy(pred_flow: np.ndarray, reference_kinematic: float) -> np.ndarray:
    """Copy a flow array and subtract one pressure reference from finite cells."""
    out = np.array(pred_flow, copy=True)
    finite = np.isfinite(out[..., 3])
    out[..., 3][finite] -= float(reference_kinematic)
    return out


def presentation_prediction(
    predict_out: Mapping,
    reference_kinematic: float,
) -> dict:
    """Return a shallow prediction copy with pressure referenced for display.

    Global and ROI pressure use the same constant. Bundles and all arrays in
    ``predict_out`` remain untouched.
    """
    out = dict(predict_out)
    out["pred_flow"] = referenced_flow_copy(
        np.asarray(predict_out["pred_flow"]),
        reference_kinematic,
    )
    out["roi_preds"] = {
        label: referenced_flow_copy(np.asarray(flow), reference_kinematic)
        for label, flow in (predict_out.get("roi_preds") or {}).items()
    }
    out["pressure_reference_kinematic"] = float(reference_kinematic)
    out["pressure_reference_convention"] = PRESSURE_REFERENCE_CONVENTION
    return out
