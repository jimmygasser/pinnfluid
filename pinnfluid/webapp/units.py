"""Display-side unit helpers for the predict_web app.

The trained models output kinematic pressure (p/rho, m^2/s^2) — the OpenFOAM
simpleFoam convention. User-facing products first subtract one pressure
reference shared by the global and ROI fields, then multiply by air density.
The underlying model output, scalers, and stored .npz/.vtk exports remain raw
and in kinematic units so engineering post-processing can choose its own gauge
and density.
"""

from __future__ import annotations

# Standard atmosphere at sea level, 15 deg C, ~1013.25 hPa.
RHO_AIR_SEA_LEVEL = 1.225  # kg/m^3

# Display density used by stats/plots/PDF. Defaults to sea level; predict runs
# call `set_air_density_for_elevation(site_elev)` so alpine sites don't get
# ~15-20% over-stated forces. Module-level because the conversion happens in
# several modules that all do `from units import RHO_AIR` at call time.
RHO_AIR = RHO_AIR_SEA_LEVEL


def air_density(elevation_m: float) -> float:
    """ISA air density [kg/m^3] at a given elevation [m a.s.l.] (15 degC sea level).

    rho(h) = rho0 * (1 - 2.25577e-5 h)^4.2559, valid to ~11 km.
    """
    h = min(max(float(elevation_m), 0.0), 10_000.0)
    return RHO_AIR_SEA_LEVEL * (1.0 - 2.25577e-5 * h) ** 4.2559


def set_air_density(rho: float) -> float:
    """Set the display density used for kinematic->Pa conversions. Returns it."""
    global RHO_AIR
    RHO_AIR = float(rho)
    return RHO_AIR


def set_air_density_for_elevation(elevation_m: float) -> float:
    """Convenience: set display density from site elevation. Returns the rho used."""
    return set_air_density(air_density(elevation_m))


def to_pa(p_kin):
    """Convert kinematic pressure (m^2/s^2) to Pa. Accepts numpy arrays or scalars."""
    return RHO_AIR * p_kin


def cp_from_kinematic(p_kin, uref_mps: float, *, p_ref_kin: float = 0.0):
    """Pressure coefficient Cp = (p - p_ref) / (0.5 * U_ref^2).

    Because p is stored kinematic (m^2/s^2), rho cancels and we don't need it.
    Returns a value/array dimensionless.
    """
    u = float(max(uref_mps, 1e-6))
    return (p_kin - float(p_ref_kin)) / (0.5 * u * u)
