"""Small JAX-compatible EOS adapters; table preparation runs on the host."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PchipInterpolator

from src.common import MEV_FM3_TO_KM2


class LinearEOS(NamedTuple):
    """Self-bound test EOS: epsilon=epsilon_surface+P/cs2, all in km^-2."""

    epsilon_surface: float
    cs2: float = 1 / 3

    def __call__(self, pressure):
        return self.epsilon_surface + pressure / self.cs2


class TabulatedEOS(NamedTuple):
    """PCHIP epsilon(P), with a specified power law in the first vacuum cell."""

    pressure: jnp.ndarray
    coefficients: jnp.ndarray
    surface_power: float
    first_epsilon: float

    @property
    def pressure_bounds(self):
        return self.pressure[0], self.pressure[-1]

    def __call__(self, pressure):
        i = jnp.clip(
            jnp.searchsorted(self.pressure, pressure, side="right") - 1, 0, self.pressure.size - 2
        )
        dx = pressure - self.pressure[i]
        c = self.coefficients[:, i]
        epsilon = ((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3]
        # An explicit low-P exponent avoids imposing epsilon~P at a vacuum endpoint.
        power_value = (
            self.first_epsilon * (jnp.maximum(pressure, 0) / self.pressure[1]) ** self.surface_power
        )
        epsilon = jnp.where(
            (self.surface_power > 0) & (pressure <= self.pressure[1]), power_value, epsilon
        )
        valid = (pressure >= self.pressure[0]) & (pressure <= self.pressure[-1])
        return jnp.where(valid, epsilon, jnp.nan)


def tabulated_eos(pressure, epsilon, *, units="MeV/fm3", surface_power=None):
    """Prepare a monotone EOS without sorting, deleting rows or extrapolating.

    For a (0,0) endpoint, explicitly specify 0<surface_power<1 such that
    epsilon(P)=epsilon[1]*(P/P[1])**surface_power in the first cell. Choose it
    from the physical low-density EOS, not merely to force a finite radius.
    Positive minimum pressure is allowed, but solve_star reports EOS_BOUNDARY.
    First-order density jumps require a separate interface treatment.
    """
    p, e = np.asarray(pressure, dtype=float), np.asarray(epsilon, dtype=float)
    if p.ndim != 1 or e.shape != p.shape or p.size < 2:
        raise ValueError("pressure and epsilon must be equal-length 1D arrays, length >= 2")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(e))):
        raise ValueError("EOS values must be finite")
    if p[0] < 0 or e[0] < 0 or np.any(np.diff(p) <= 0) or np.any(np.diff(e) <= 0):
        raise ValueError(
            "EOS must be nonnegative and strictly increasing; select a physical branch"
        )
    vacuum = p[0] == 0 and e[0] == 0
    if vacuum and (surface_power is None or not 0 < surface_power < 1):
        raise ValueError("vacuum endpoint requires an explicit 0 < surface_power < 1")
    if not vacuum and surface_power is not None:
        raise ValueError("surface_power is only valid for a (P,epsilon)=(0,0) endpoint")
    if units not in ("MeV/fm3", "km^-2"):
        raise ValueError("units must be 'MeV/fm3' or 'km^-2'")
    factor = MEV_FM3_TO_KM2 if units == "MeV/fm3" else 1.0
    p, e = p * factor, e * factor
    return TabulatedEOS(
        jnp.asarray(p),
        jnp.asarray(PchipInterpolator(p, e).c),
        0.0 if surface_power is None else float(surface_power),
        float(e[1]),
    )
