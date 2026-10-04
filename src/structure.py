"""JIT RK4 integration of GR / regularized 4D EGB TOV backgrounds.

See docs/equations_egb.md. State is [pressure_km^-2, mass_km].
EOS callables must be JAX pytrees (e.g. src.eos classes or jax.tree_util.Partial).
"""

from enum import IntEnum
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from src.common import SOLAR_MASS_KM
from src.numerics import rk4_step


class Status(IntEnum):
    SURFACE = 0
    RADIUS_LIMIT = 1
    STEP_LIMIT = 2
    INVALID = 3
    EOS_BOUNDARY = 4


class StarResult(NamedTuple):
    radius_km: jax.Array  # NaN unless a zero-pressure surface was reached.
    mass_km: jax.Array
    redshift: jax.Array
    status: jax.Array
    steps: jax.Array  # Attempted steps, including rejected surface steps.
    last_radius_km: jax.Array
    last_state: jax.Array  # Actual [P,m]; surface pressure is not rounded to zero.

    @property
    def mass_msun(self):
        return self.mass_km / SOLAR_MASS_KM


def metric_f(radius, mass, alpha=0.0):
    """GR-connected metric branch; alpha in km², mass and radius in km."""
    gamma = jnp.sqrt(1 + 8 * alpha * mass / radius**3)
    return 1 - 4 * mass / (radius * (1 + gamma))


def tov_rhs(radius, state, epsilon, alpha=0.0):
    """Unified EGB/GR RHS, given epsilon at state[0]; radius must be positive."""
    pressure, mass = state
    q = mass / radius**3
    gamma = jnp.sqrt(1 + 8 * alpha * q)
    a = 4 * q / (1 + gamma)
    f = 1 - radius**2 * a
    dp = -(epsilon + pressure) * radius * (a - q + 4 * jnp.pi * pressure) / (gamma * f)
    return jnp.stack((dp, 4 * jnp.pi * radius**2 * epsilon))


def gr_tov_rhs(radius, state, epsilon):
    """GR is exactly the alpha=0 specialization of the shared kernel."""
    return tov_rhs(radius, state, epsilon, 0.0)


def _guarded_rhs(radius, state, args):
    """Internal state [P,q=m/r³] removes the leading center-stage mass error."""
    eos, alpha, p_min, p_max = args
    pressure, q = state
    mass = q * radius**3
    valid = (
        jnp.all(jnp.isfinite(state))
        & (pressure >= p_min)
        & (pressure <= p_max)
        & (mass >= 0)
        & (1 + 8 * alpha * mass / radius**3 > 0)
        & (metric_f(radius, mass, alpha) > 0)
    )

    def evaluate(_):
        epsilon = eos(pressure)
        dp = tov_rhs(radius, jnp.stack((pressure, mass)), epsilon, alpha)[0]
        rhs = jnp.stack((dp, (4 * jnp.pi * epsilon - 3 * q) / radius))
        return jnp.where((epsilon >= 0) & jnp.isfinite(epsilon), rhs, jnp.nan)

    # Invalid pressures never reach the EOS on a scalar solve.
    return lax.cond(valid, evaluate, lambda _: jnp.full(2, jnp.nan), operand=None)


@partial(jax.jit, static_argnames=("max_steps",))
def solve_star(
    eos,
    pc,
    *,
    alpha=0.0,
    dr=0.02,
    rmax=50.0,
    surface_tol_km=1e-8,
    max_steps=100_000,
    pressure_bounds=None,
):
    """Return M,R,z with status, using geometric units throughout.

    pc is CENTRAL PRESSURE (km^-2), alpha is km². Interior steps use fixed dr;
    center and pressure-scale caps resolve the endpoints. Invalid stages are
    rejected and retried with h/2. This is classical RK4,
    not an adaptive RK4/5 solver. Check global accuracy by halving dr.

    Tables supply their pressure_bounds automatically. Analytic EOS defaults
    to [0,+inf); pass explicit bounds for any restricted callable. Reaching a
    positive lower EOS bound is EOS_BOUNDARY, never a full-star radius/mass.
    Dynamic termination is not a reverse-mode differentiable surface solver.
    """
    if pressure_bounds is None:
        pressure_bounds = getattr(eos, "pressure_bounds", (0.0, jnp.inf))
    p_min, p_max = pressure_bounds
    pc = jnp.asarray(pc, dtype=jnp.float64)
    input_ok = (
        jnp.isfinite(pc)
        & (pc > p_min)
        & (pc <= p_max)
        & (p_min >= 0)
        & (p_max > p_min)
        & jnp.isfinite(alpha)
        & jnp.isfinite(dr)
        & (dr > 0)
        & jnp.isfinite(rmax)
        & (rmax > 0)
        & (surface_tol_km > 0)
        & (surface_tol_km < dr)
        & (max_steps > 0)
    )
    ec = lax.cond(input_ok, lambda _: eos(pc), lambda _: jnp.array(jnp.nan), None)
    qc = 4 * jnp.pi * ec / 3
    gamma_c = jnp.sqrt(1 + 8 * alpha * qc)
    ac = 4 * qc / (1 + gamma_c)
    p2 = -(ec + pc) * (ac - qc + 4 * jnp.pi * pc) / (2 * gamma_c)
    # Use q=m/r³ internally: RK stages for m alone lose accuracy near r=0.
    # The r² correction to q follows from epsilon(P)=ec+epsilon'_c*p2*r².
    dec = lax.cond(
        input_ok,
        lambda _: jax.jvp(eos, (pc,), (jnp.ones_like(pc),))[1],
        lambda _: jnp.array(jnp.nan),
        None,
    )
    r0 = jnp.minimum(jnp.minimum(1e-3, dr / 4), rmax * 1e-3)
    r0 = jnp.minimum(r0, jnp.sqrt((pc - p_min) / jnp.maximum(-p2, 1e-300)) * 1e-3)
    y0 = jnp.stack((pc + p2 * r0**2, qc + 4 * jnp.pi * dec * p2 * r0**2 / 5))
    valid = (
        input_ok
        & (ec > 0)
        & jnp.isfinite(ec)
        & jnp.isfinite(p2)
        & (p2 < 0)
        & (r0 > 0)
        & (y0[0] > p_min)
        & (y0[1] > 0)
        & (metric_f(r0, y0[1] * r0**3, alpha) > 0)
    )
    running = -1
    initial = (
        r0,
        y0,
        jnp.asarray(dr, dtype=jnp.float64),
        jnp.array(0),
        jnp.where(valid, running, int(Status.INVALID)),
    )
    args = (eos, alpha, p_min, p_max)

    def condition(carry):
        r, _, _, count, status = carry
        return (status == running) & (r < rmax) & (count < max_steps)

    def step(carry):
        r, y, h, count, _ = carry
        # q' has a removable center singularity; grow steps out of the center.
        h = jnp.minimum(jnp.minimum(h, r / 2), rmax - r)
        # Resolve the vanishing-pressure scale before attempting the surface.
        # Otherwise epsilon~sqrt(P) can produce large radius errors in a legal step.
        dp = _guarded_rhs(r, y, args)[0]
        pressure_scale = (y[0] - p_min) / jnp.maximum(-dp, 1e-300)
        h = jnp.minimum(h, jnp.maximum(8 * surface_tol_km, 0.25 * pressure_scale))
        candidate, stages = rk4_step(_guarded_rhs, r, y, h, args, return_stages=True)
        radii = r + h * jnp.array([0, 0.5, 0.5, 1, 1])
        pressures, masses = stages[:, 0], stages[:, 1] * radii**3
        stage_ok = (
            jnp.all(jnp.isfinite(stages))
            & jnp.all(pressures >= p_min)
            & jnp.all(pressures <= p_max)
            & jnp.all(masses >= 0)
            & jnp.all(1 + 8 * alpha * masses / radii**3 > 0)
            & jnp.all(metric_f(radii, masses, alpha) > 0)
            & (candidate[0] < y[0])
            & (candidate[1] * (r + h) ** 3 >= (1 - 1e-14) * y[1] * r**3)
        )
        boundary_crossed = jnp.any(jnp.isfinite(pressures) & (pressures < p_min))
        small = (h <= surface_tol_km) | (r + h == r)
        boundary_status = jnp.where(p_min == 0, int(Status.SURFACE), int(Status.EOS_BOUNDARY))
        status = jnp.where(
            stage_ok,
            running,
            jnp.where(
                small, jnp.where(boundary_crossed, boundary_status, int(Status.INVALID)), running
            ),
        )
        status = jnp.where(stage_ok & (candidate[0] == p_min), boundary_status, status)
        return (
            jnp.where(stage_ok, r + h, r),
            jnp.where(stage_ok, candidate, y),
            jnp.where(stage_ok, dr, h / 2),
            count + 1,
            status,
        )

    r, y, _, count, status = lax.while_loop(condition, step, initial)
    y = jnp.stack((y[0], y[1] * r**3))
    status = jnp.where(
        status != running,
        status,
        jnp.where(r >= rmax, int(Status.RADIUS_LIMIT), int(Status.STEP_LIMIT)),
    )
    surface = status == int(Status.SURFACE)
    return StarResult(
        jnp.where(surface, r, jnp.nan),
        jnp.where(surface, y[1], jnp.nan),
        jnp.where(surface, 1 / jnp.sqrt(metric_f(r, y[1], alpha)) - 1, jnp.nan),
        status,
        count,
        r,
        y,
    )


@partial(jax.jit, static_argnames=("max_steps",))
def mass_radius_sequence(
    eos,
    central_pressures,
    *,
    alpha=0.0,
    dr=0.02,
    rmax=50.0,
    surface_tol_km=1e-8,
    max_steps=100_000,
    pressure_bounds=None,
):
    """Compiled sequential map, retaining scalar EOS guards and early termination."""
    return lax.map(
        lambda pc: solve_star(
            eos,
            pc,
            alpha=alpha,
            dr=dr,
            rmax=rmax,
            surface_tol_km=surface_tol_km,
            max_steps=max_steps,
            pressure_bounds=pressure_bounds,
        ),
        jnp.asarray(central_pressures, dtype=jnp.float64),
    )
