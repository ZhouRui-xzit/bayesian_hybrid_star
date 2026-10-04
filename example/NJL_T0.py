"""Cold, neutral u/d/s/e/mu NJL matter with a flavor-diagonal vector interaction.

Based on ZhouRui-xzit/PNJLs, src/axion_red_vec/{pnjl_vec,constants}.jl,
commit 08a32a6660ee1b2e0d9a4bf04236071bbc9734cd. No axion or Polyakov fields.
Internal units are powers of Lambda; public units are MeV and fm.
sigma = -<qbar q> is positive, as in the reference implementation.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

jax.config.update("jax_enable_x64", True)

REFERENCE_COMMIT = "08a32a6660ee1b2e0d9a4bf04236071bbc9734cd"


@dataclass(frozen=True)
class Parameters:
    Rv: float = 0.0
    B_eff: float = 0.0  # MeV/fm^3; epsilon += B_eff, P -= B_eff
    cutoff: float = 630.0  # MeV
    Gs_Lambda2: float = 1.781
    K_Lambda5: float = 9.29
    m_u: float = 5.5  # MeV
    m_d: float = 5.5
    m_s: float = 135.7
    m_e: float = 0.511
    m_mu: float = 105.658
    hbar_c: float = 197.33  # MeV fm, deliberately matches the reference
    rho0: float = 0.16  # baryons/fm^3

    def __post_init__(self):
        if not all(np.isfinite(v) for v in self.__dict__.values()):
            raise ValueError("All model parameters must be finite")
        if (
            self.Rv < 0
            or min(
                self.cutoff,
                self.Gs_Lambda2,
                self.K_Lambda5,
                self.m_u,
                self.m_d,
                self.m_s,
                self.m_e,
                self.m_mu,
                self.hbar_c,
                self.rho0,
            )
            <= 0
        ):
            raise ValueError("Require Rv >= 0 and positive masses/scales/couplings")

    @property
    def energy_unit(self):
        return self.cutoff**4 / self.hbar_c**3  # MeV/fm^3 per Lambda^4

    @property
    def density_unit(self):
        return (self.cutoff / self.hbar_c) ** 3  # fm^-3 per Lambda^3

    def array(self):
        return jnp.array(
            [
                self.Gs_Lambda2,
                self.K_Lambda5,
                self.Rv * self.Gs_Lambda2,
                self.m_u / self.cutoff,
                self.m_d / self.cutoff,
                self.m_s / self.cutoff,
                self.m_e / self.cutoff,
                self.m_mu / self.cutoff,
            ]
        )


def scalar_integral(p, m):
    """Integral_0^p k^2/sqrt(k^2+m^2) dk; stable also at tiny p/m."""
    t = p / m
    exact = (p * jnp.hypot(p, m) - m**2 * jnp.arcsinh(t)) / 2
    series = p**3 / m * (1 / 3 - t**2 / 10 + 3 * t**4 / 56 - 5 * t**6 / 144 + 35 * t**8 / 1408)
    return jnp.where(t < 0.03, series, exact)


def energy_integral(p, m):
    """Integral_0^p k^2 sqrt(k^2+m^2) dk, with a nonrelativistic series."""
    t = p / m
    exact = (p * jnp.hypot(p, m) * (2 * p**2 + m**2) - m**4 * jnp.arcsinh(t)) / 8
    series = m * p**3 * (1 / 3 + t**2 / 10 - t**4 / 56 + t**6 / 144 - 5 * t**8 / 1408)
    return jnp.where(t < 0.03, series, exact)


def pressure_integral(p, m):
    """Integral_0^p k^4/sqrt(k^2+m^2) dk; avoids epsilon-mu*n cancellation."""
    t = p / m
    exact = (p * jnp.hypot(p, m) * (2 * p**2 - 3 * m**2) + 3 * m**4 * jnp.arcsinh(t)) / 8
    series = p**5 / m * (1 / 5 - t**2 / 14 + t**4 / 24 - 5 * t**6 / 176 + 35 * t**8 / 1664)
    return jnp.where(t < 0.03, series, exact)


def lepton_gas(mu, masses):
    """T=0 particles (mu >= 0), spin degeneracy 2; exact zero below threshold."""
    p2 = jnp.maximum((mu - masses) * (mu + masses), 0.0)
    # A floor only in sqrt avoids NaN AD at threshold; where restores exact zero.
    p = jnp.where(p2 > 0, jnp.sqrt(jnp.maximum(p2, 1e-100)), 0.0)
    n = p2**1.5 / (3 * jnp.pi**2)
    e = energy_integral(p, masses) / jnp.pi**2
    pressure = pressure_integral(p, masses) / (3 * jnp.pi**2)
    return n, e, pressure


def masses_from_sigma(sigma, a):
    return (
        a[3:6]
        + 4 * a[0] * sigma
        + 2 * a[1] * sigma[jnp.array([1, 2, 0])] * sigma[jnp.array([2, 0, 1])]
    )


def condensate(masses, momenta):
    return (
        3 * masses / jnp.pi**2 * (scalar_integral(1.0, masses) - scalar_integral(momenta, masses))
    )


@jax.jit
def vacuum_residual(x, a):
    sigma = x / 10
    return (sigma - condensate(masses_from_sigma(sigma, a), jnp.zeros(3))) * 10


@jax.jit
def unpack(x, n, a):
    """n is baryon density in Lambda^3; momenta scaled to the density."""
    sigma = x[:3] / 10
    masses = masses_from_sigma(sigma, a)
    scale = (jnp.pi**2 * n) ** (1 / 3)
    y = x[3:6]
    momenta = scale * y
    densities = n * y**3
    mu = jnp.hypot(momenta, masses) + 4 * a[2] * densities
    mu_e = x[-1] * scale
    return sigma, masses, momenta, densities, mu, mu_e


@jax.jit
def residual(x, args):
    n, a, strange = args
    sigma, masses, momenta, densities, mu, mu_e = unpack(x, n, a)
    leptons, _, _ = lepton_gas(mu_e, a[6:8])
    scale = (jnp.pi**2 * n) ** (1 / 3)
    equations = [
        *(10 * (sigma - condensate(masses, momenta))),
        jnp.sum(densities) / n - 3,
        (2 * densities[0] - densities[1] - densities[2]) / (3 * n) - jnp.sum(leptons) / n,
        (mu[1] - mu[0] - mu_e) / scale,
        jnp.where(strange, (mu[2] - mu[1]) / scale, x[5]),
    ]
    return jnp.array(equations)


@jax.jit
def newton_solve(x, n, a, strange):
    sol = optx.root_find(
        residual,
        optx.Newton(rtol=1e-10, atol=1e-11),
        x,
        args=(n, a, strange),
        max_steps=100,
        throw=False,
        options={"lower": jnp.zeros_like(x), "upper": jnp.full_like(x, 3.0)},
    )
    return sol.value, sol.stats["num_steps"], sol.result == optx.RESULTS.successful


@jax.jit
def continuation_step(carry, n, a):
    previous, strange = carry
    x, steps, success = newton_solve(previous, n, a, strange)
    _, masses, _, _, mu, _ = unpack(x, n, a)
    close = strange & (mu[1] < masses[2]) & (x[5] < 1e-7)

    def close_strange_sea(_):
        updated, extra_steps, ok = newton_solve(x.at[5].set(0.0), n, a, jnp.array(False))
        return updated, steps + extra_steps, ok

    x, steps, success = jax.lax.cond(close, close_strange_sea, lambda _: (x, steps, success), None)
    strange = strange & ~close
    _, masses, _, _, mu, _ = unpack(x, n, a)
    onset = (~strange) & (mu[1] > masses[2] + 1e-12)

    def open_strange_sea(_):
        scale = (jnp.pi**2 * n) ** (1 / 3)
        ys = jnp.sqrt(jnp.maximum(mu[1] ** 2 - masses[2] ** 2, 1e-12)) / scale
        seed = x.at[5].set(jnp.clip(ys, 1e-5, 1.0))
        seed = seed.at[4].set(jnp.cbrt(jnp.maximum(3 - seed[3] ** 3 - seed[5] ** 3, 0.1)))
        updated, extra_steps, ok = newton_solve(seed, n, a, jnp.array(True))
        return updated, steps + extra_steps, ok

    x, steps, success = jax.lax.cond(onset, open_strange_sea, lambda _: (x, steps, success), None)
    strange = strange | onset
    error = jnp.max(jnp.abs(residual(x, (n, a, strange))))
    _, _, momenta, _, _, _ = unpack(x, n, a)
    success = (
        success & jnp.isfinite(error) & (error < 2e-8) & jnp.all(x[:3] > 0) & jnp.all(momenta < 1)
    )
    # Stop advancing the carry after a failure; the Python boundary reports it.
    next_x = jnp.where(success, x, previous)
    return (next_x, strange), (x, strange, error, steps, success)


@jax.jit
def scan_states(densities, a, initial):
    """One compiled density loop; each converged point seeds the next Newton solve."""
    return jax.lax.scan(lambda carry, n: continuation_step(carry, n, a), initial, densities)[1]


# Fixed quadrature only for a stable *difference* of vacuum energies; Fermi seas
# and gap equations use the closed T=0 integrals above.
_nodes, _weights = np.polynomial.legendre.leggauss(96)
_p = jnp.array((_nodes + 1) / 2)
_w = jnp.array(_weights / 2)


@jax.jit
def thermodynamics(x, n, a, vacuum_sigma):
    sigma, masses, momenta, densities, mu, mu_e = unpack(x, n, a)
    m_vac = masses_from_sigma(vacuum_sigma, a)
    ds = sigma - vacuum_sigma
    j, k = jnp.array([1, 2, 0]), jnp.array([2, 0, 1])
    dm = 4 * a[0] * ds + 2 * a[1] * (ds[j] * sigma[k] + vacuum_sigma[j] * ds[k])
    # Cancel the linear vacuum terms algebraically BEFORE evaluating them.
    # E-Ev-(Mv/Ev)*dM is quadratic in dM. This matters at nB/rho0=1e-8.
    energy = jnp.hypot(_p, masses[:, None])
    ev = jnp.hypot(_p, m_vac[:, None])
    quadratic = (
        (dm**2 * (masses + m_vac))[:, None]
        * _p**2
        / (ev * (energy + ev) * (masses[:, None] * ev + m_vac[:, None] * energy))
    )
    vacuum_gap = condensate(m_vac, jnp.zeros(3)) - vacuum_sigma
    background = (
        2 * a[0] * jnp.sum(ds**2)
        + 2 * a[1] * jnp.sum(vacuum_sigma * ds[j] * ds[k])
        + 4 * a[1] * jnp.prod(ds)
        - jnp.sum(vacuum_gap * dm)
        - 3 / jnp.pi**2 * jnp.sum(_w * _p**2 * quadratic)
    )
    vector = 2 * a[2] * jnp.sum(densities**2)
    nl, el, pl = lepton_gas(mu_e, a[6:8])
    eq = 3 / jnp.pi**2 * energy_integral(momenta, masses)
    pq = pressure_integral(momenta, masses) / jnp.pi**2
    epsilon = background + jnp.sum(eq) + vector + jnp.sum(el)
    pressure = -background + jnp.sum(pq) + vector + jnp.sum(pl)
    return epsilon, pressure, nl, el, pl, masses, mu, mu_e


@jax.jit
def equilibrium_response(x, n, a, vacuum_sigma, strange):
    """Implicit differentiation on one homogeneous branch, never across a jump."""

    def fn(y, density):
        return residual(y, (density, a, strange))

    jac_x, jac_n = jax.jacfwd(fn, argnums=(0, 1))(x, n)
    dx_dn = jnp.linalg.solve(jac_x, -jac_n)

    def energy_pressure(y, density):
        return jnp.array(thermodynamics(y, density, a, vacuum_sigma)[:2])

    _, derivative = jax.jvp(energy_pressure, (x, n), (dx_dn, jnp.ones_like(n)))
    return derivative  # d(epsilon, P)/dnB, in units of Lambda


def density_grid(max_ratio=6.0, step=0.01, min_ratio=0.01):
    """Ascending continuation: 1e-8, 0.01, 0.02, ..., 6 by default."""
    if not (
        np.isfinite([max_ratio, step, min_ratio]).all()
        and max_ratio >= min_ratio > 1e-8
        and step > 0
    ):
        raise ValueError("Require max_ratio >= min_ratio > 1e-8 and step > 0")
    count = int(np.floor((max_ratio - min_ratio) / step + 1e-9))
    grid = min_ratio + step * np.arange(count + 1)
    if not np.isclose(grid[-1], max_ratio, rtol=0, atol=1e-12):
        grid = np.append(grid, max_ratio)
    return np.r_[1e-8, grid]


@jax.jit
def vacuum_solve(a):
    # A single chirally broken vacuum seed, not a multi-root search.
    sol = optx.root_find(
        vacuum_residual,
        optx.Newton(rtol=1e-12, atol=1e-13),
        jnp.array([0.5, 0.5, 0.7]),
        args=a,
        max_steps=100,
        throw=False,
        options={"lower": jnp.full(3, 1e-12), "upper": jnp.full(3, 3.0)},
    )
    return sol.value / 10, sol.result == optx.RESULTS.successful


@jax.jit
def evaluate_states(xs, ns, a, vacuum_sigma, strange):
    values = jax.vmap(thermodynamics, in_axes=(0, 0, None, None))(xs, ns, a, vacuum_sigma)
    derivatives = jax.vmap(equilibrium_response, in_axes=(0, 0, None, None, 0))(
        xs, ns, a, vacuum_sigma, strange
    )
    return values, derivatives


class NJLModel:
    """A single T=0 branch followed from dilute matter with JIT + lax.scan.

    There is no multi-start search or implicit SciPy fallback. Every density
    inherits the previous solution. A failed root/cutoff check aborts the scan.
    Stability diagnostics do not imply a search over all possible phases.
    """

    def __init__(self, parameters=Parameters()):
        self.parameters = parameters
        self.a = parameters.array()
        sigma, success = vacuum_solve(self.a)
        if (
            not bool(success)
            or float(jnp.max(jnp.abs(vacuum_residual(10 * sigma, self.a)))) > 1e-10
        ):
            raise RuntimeError("Chirally broken vacuum gap solve failed")
        self.vacuum_sigma = sigma
        self.vacuum_masses = np.asarray(masses_from_sigma(sigma, self.a))
        n = 1e-8 * parameters.rho0 / parameters.density_unit
        scale = (np.pi**2 * n) ** (1 / 3)
        momenta = np.array([1.0, np.cbrt(2), 0.0])
        mu_e = np.hypot(scale * momenta[1], self.vacuum_masses[1]) - np.hypot(
            scale, self.vacuum_masses[0]
        )
        self.initial = (jnp.r_[10 * sigma, momenta, max(mu_e / scale, 1e-10)], jnp.array(False))

    def _rows(self, ratios, results):
        par = self.parameters
        xs, strange, errors, steps, success = jax.device_get(results)
        if not np.all(success):
            i = int(np.flatnonzero(~success)[0])
            raise RuntimeError(
                f"Continuation failed at rho/rho0={ratios[i]:g}, Rv={par.Rv:g}: "
                f"residual={errors[i]:.3g}. Reduce density step; no root was silently replaced."
            )
        ns = jnp.asarray(ratios * par.rho0 / par.density_unit)
        values, derivatives = jax.device_get(
            evaluate_states(jnp.asarray(xs), ns, self.a, self.vacuum_sigma, jnp.asarray(strange))
        )
        eps, press, nl, el, pl, masses, mus, mu_es = values
        nq_all = ratios[:, None] * par.rho0 * xs[:, 3:6] ** 3
        nl = nl * par.density_unit
        mus = mus * par.cutoff
        masses = masses * par.cutoff
        rows = []
        for i, ratio in enumerate(ratios):
            nq = nq_all[i]
            nb = float(nq.sum() / 3)
            charge = float((2 * nq[0] - nq[1] - nq[2]) / 3 - nl[i].sum())
            mu_b = float(mus[i, 0] + 2 * mus[i, 1])
            mu_e = float(mu_es[i]) * par.cutoff
            energy = float(eps[i]) * par.energy_unit + par.B_eff
            pressure = float(press[i]) * par.energy_unit - par.B_eff
            de, dp = derivatives[i] * par.cutoff
            cs2 = float(dp / de)
            row = dict(
                rho_over_rho0=float(ratio),
                nB_fm3=nb,
                P_MeV_fm3=pressure,
                epsilon_MeV_fm3=energy,
                muB_MeV=mu_b,
                muQ_MeV=-mu_e,
                mu_e_MeV=mu_e,
                nQ_fm3=charge,
                residual=float(errors[i]),
                newton_steps=int(steps[i]),
                solver="optimistix_newton",
                branch="uds" if strange[i] else "ud",
                branch_id="dilute_upward_continuation",
                status="converged",
                thermo_error_MeV_fm3=energy + pressure - mu_b * nb + mu_e * charge,
                Rv=par.Rv,
                B_eff_MeV_fm3=par.B_eff,
                cs2=cs2,
                mechanically_stable=bool(cs2 > 0),
                d_epsilon_dnB_MeV=float(de),
                dP_dnB_MeV=float(dp),
            )
            for j, flavor in enumerate(("u", "d", "s")):
                row[f"M_{flavor}_MeV"] = float(masses[i, j])
                row[f"n_{flavor}_fm3"] = float(nq[j])
                # Physical mu_s=mu_d, also when the strange Fermi sea is absent.
                row[f"mu_{flavor}_MeV"] = float(mus[i, j if j < 2 else 1])
            for j, flavor in enumerate(("e", "mu")):
                row[f"n_{flavor}_fm3"] = float(nl[i, j])
                row[f"epsilon_{flavor}_MeV_fm3"] = float(el[i, j]) * par.energy_unit
                row[f"P_{flavor}_MeV_fm3"] = float(pl[i, j]) * par.energy_unit
            rows.append(row)
        return rows

    def scan(self, ratios=None):
        ratios = density_grid() if ratios is None else np.asarray(ratios, dtype=float)
        if (
            ratios.ndim != 1
            or len(ratios) < 1
            or not np.all(np.isfinite(ratios))
            or np.any(ratios <= 0)
            or np.any(np.diff(ratios) <= 0)
            or ratios[0] != 1e-8
        ):
            raise ValueError("Grid must be finite, strictly ascending, and start at 1e-8")
        ns = jnp.asarray(ratios * self.parameters.rho0 / self.parameters.density_unit)
        results = scan_states(ns, self.a, self.initial)
        rows = self._rows(ratios, results)
        self.last_scan_states = {
            "ratios": ratios,
            "x": np.asarray(results[0]),
            "strange": np.asarray(results[1]),
        }
        return rows

    def solve_density(self, ratio, initial=None):
        """One local continuation step, or a dilute-to-target scan if no seed is given."""
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("rho/rho0 must be finite and positive")
        if initial is None:
            grid = density_grid(max_ratio=ratio) if ratio >= 0.01 else np.unique([1e-8, ratio])
            if ratio < 1e-8:
                raise ValueError("The documented dilute starting density is 1e-8")
            row = self.scan(grid)[-1]
            return row, {
                "x": self.last_scan_states["x"][-1],
                "strange": bool(self.last_scan_states["strange"][-1]),
            }
        n = ratio * self.parameters.rho0 / self.parameters.density_unit
        _, result = continuation_step(
            (jnp.asarray(initial["x"]), jnp.asarray(initial["strange"])), jnp.asarray(n), self.a
        )
        results = jax.tree.map(lambda x: x[None], result)
        row = self._rows(np.array([ratio]), results)[0]
        return row, {"x": np.asarray(result[0]), "strange": bool(result[1])}


class TOVEOS:
    """A bounded, strictly monotone *homogeneous* P(epsilon) segment.

    Does not manufacture a crust, smooth a spinodal, or bridge a Maxwell jump.
    A segment with Pmin>0 can be used as a core EOS but not as a full star EOS.
    """

    def __init__(self, rows, surface_available=False):
        self.rows = rows
        p = np.array([r["P_MeV_fm3"] for r in rows])
        e = np.array([r["epsilon_MeV_fm3"] for r in rows])
        if len(p) < 2 or not np.all(np.isfinite([p, e])):
            raise ValueError("Need at least two finite EOS points")
        if np.any(np.diff(p) <= 0) or np.any(np.diff(e) <= 0) or p[0] < 0 or e[0] < 0:
            raise ValueError("TOV segment requires increasing P>=0 and epsilon")
        self.pressure_bounds = (float(p[0]), float(p[-1]))
        self.surface_available = surface_available
        self._epsilon = PchipInterpolator(p, e, extrapolate=False)

    def epsilon_of_pressure(self, pressure):
        p = np.asarray(pressure, dtype=float)
        lo, hi = self.pressure_bounds
        if np.any(~np.isfinite(p)) or np.any(p < lo) or np.any(p > hi):
            raise ValueError(f"Pressure outside EOS interval [{lo:g}, {hi:g}] MeV/fm^3")
        return self._epsilon(p)

    def sound_speed_sq(self, pressure):
        self.epsilon_of_pressure(pressure)
        result = 1 / self._epsilon.derivative()(pressure)
        if self.surface_available and self._epsilon(0.0) == 0:
            result = np.where(np.asarray(pressure) == 0, 0.0, result)
        return result


def _vacuum_row(model, template):
    """The analytic nB=0 endpoint for B_eff=0, not an extrapolated EOS point."""
    row = template.copy()
    for key in row:
        if key.startswith(("n_", "P_", "epsilon_")):
            row[key] = 0.0
    mu = model.vacuum_masses * model.parameters.cutoff
    row.update(
        rho_over_rho0=0.0,
        nB_fm3=0.0,
        nQ_fm3=0.0,
        mu_e_MeV=0.0,
        muQ_MeV=0.0,
        muB_MeV=float(mu[0] + 2 * mu[1]),
        residual=0.0,
        newton_steps=0,
        cs2=0.0,
        dP_dnB_MeV=0.0,
        d_epsilon_dnB_MeV=float(mu[0] + 2 * mu[1]),
        thermo_error_MeV_fm3=0.0,
        status="analytic_vacuum",
        mechanically_stable=True,
    )
    for j, flavor in enumerate(("u", "d", "s")):
        row[f"M_{flavor}_MeV"] = float(mu[j])
        row[f"mu_{flavor}_MeV"] = float(mu[j if j < 2 else 1])
    return row


def tov_segment(model, rows):
    """Extract high-density mechanical stability; report limitations explicitly.

    Internal first-order transitions require a Maxwell construction and are not
    replaced by interpolation here. The raw table always retains every point.
    """
    ordered = sorted(rows, key=lambda r: r["nB_fm3"])
    p = np.array([r["P_MeV_fm3"] for r in ordered])
    e = np.array([r["epsilon_MeV_fm3"] for r in ordered])
    cs2 = np.array([r["cs2"] for r in ordered])
    valid = (np.diff(p) > 0) & (np.diff(e) > 0) & (cs2[:-1] > 0) & (cs2[1:] > 0)
    start = len(ordered) - 1
    while start > 0 and valid[start - 1]:
        start -= 1
    stable = ordered[start:]
    status = {
        "status": "core_only",
        "surface_available": False,
        "selection": "highest-density monotone homogeneous branch",
        "global_phase_equilibrium": "not_searched_single_continuation_branch",
        "excluded_low_density_points": start,
    }
    if len(stable) < 2 or stable[-1]["P_MeV_fm3"] <= 0:
        return None, dict(status, status="no_positive_pressure_branch")
    positive = [r for r in stable if r["P_MeV_fm3"] > 0]
    if len(positive) < 2:
        return None, dict(status, status="insufficient_positive_pressure_points")
    # A genuine root on this continuous branch, not interpolation to an invented
    # zero-density point. The outer low-density phase may still be preferred.
    if stable[0]["P_MeV_fm3"] < 0:
        j = next(i for i, r in enumerate(stable) if r["P_MeV_fm3"] > 0)
        saved = model.last_scan_states

        def surface_point(ratio):
            nearest = int(np.argmin(abs(saved["ratios"] - ratio)))
            initial = {"x": saved["x"][nearest], "strange": saved["strange"][nearest]}
            return model.solve_density(ratio, initial)[0]

        root = brentq(
            lambda ratio: surface_point(ratio)["P_MeV_fm3"],
            stable[j - 1]["rho_over_rho0"],
            stable[j]["rho_over_rho0"],
            xtol=1e-11,
        )
        surface = surface_point(root)
        if abs(surface["P_MeV_fm3"]) > 1e-6:
            raise RuntimeError("Zero pressure is not on a continuous homogeneous branch")
        surface["P_MeV_fm3"] = 0.0
        positive.insert(0, surface)
        status.update(
            status="homogeneous_surface", surface_available=True, surface_rho_over_rho0=root
        )
    elif start == 0 and model.parameters.B_eff == 0:
        # Resolve the gap between 1e-8 and 0.01 for interpolation, without changing
        # the requested raw scan. The exact vacuum endpoint is known analytically.
        low_ratios = np.geomspace(1e-8, positive[1]["rho_over_rho0"], 49)
        ns = jnp.asarray(low_ratios * model.parameters.rho0 / model.parameters.density_unit)
        low = model._rows(low_ratios, scan_states(ns, model.a, model.initial))
        positive = [_vacuum_row(model, low[0]), *low[:-1], *positive[1:]]
        status.update(
            status="homogeneous_vacuum_surface",
            surface_available=True,
            surface_rho_over_rho0=0.0,
            extra_dilute_points=47,
        )
    eos = TOVEOS(positive, status["surface_available"])
    status["pressure_bounds_MeV_fm3"] = list(eos.pressure_bounds)
    status["excluded_low_density_points"] = len(rows) - sum(r["P_MeV_fm3"] > 0 for r in stable)
    cs = eos.sound_speed_sq(np.linspace(*eos.pressure_bounds, 1000))
    status.update(cs2_min=float(np.min(cs)), cs2_max=float(np.max(cs)))
    if not np.all((cs >= 0) & (cs <= 1 + 1e-8)) or np.any(cs[1:] <= 0):
        return None, dict(status, status="noncausal_or_unstable_interpolant")
    return eos, status
