"""Independent analytic/ODE checks for the RK4 stellar background solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.common import MEV_FM3_TO_KM2, SOLAR_MASS_KM
from src.eos import LinearEOS, tabulated_eos
from src.numerics import rk4_step
from src.structure import Status, gr_tov_rhs, mass_radius_sequence, metric_f, solve_star, tov_rhs


def test_rk4_order_for_vector_ivp():
    def integrate(n):
        def step(y, x):
            return rk4_step(lambda t, state, _: jnp.array([state[1], -state[0]]), x, y, 1 / n), None

        return jax.lax.scan(step, jnp.array([1.0, 0.0]), jnp.arange(n) / n)[0]

    exact = np.array([np.cos(1), -np.sin(1)])
    errors = [np.linalg.norm(jax.jit(integrate, static_argnums=0)(n) - exact) for n in [10, 20, 40]]
    assert 15 < errors[0] / errors[1] < 17
    assert 15 < errors[1] / errors[2] < 17


def test_rhs_gr_limit_and_reference_egb():
    r, m, p, e = 7.0, 1.2, 1e-4, 6e-4
    state = jnp.array([p, m])
    gr = np.array(
        [-(e + p) * (m + 4 * np.pi * r**3 * p) / (r * (r - 2 * m)), 4 * np.pi * r * r * e]
    )
    np.testing.assert_allclose(tov_rhs(r, state, e, 0.0), gr, rtol=3e-15)
    np.testing.assert_array_equal(tov_rhs(r, state, e, 0.0), gr_tov_rhs(r, state, e))
    for alpha in [-3.0, 6.0]:
        gamma = np.sqrt(1 + 8 * alpha * m / r**3)
        dp = (e + p) * (r**3 * (gamma + 8 * np.pi * alpha * p - 1) - 2 * alpha * m)
        dp /= r * r * gamma * (r * r * (gamma - 1) - 2 * alpha)
        np.testing.assert_allclose(tov_rhs(r, state, e, alpha)[0], dp, rtol=1e-13)
    for alpha in [-1e-12, 1e-12]:
        np.testing.assert_allclose(tov_rhs(r, state, e, alpha), gr, rtol=1e-12)
        np.testing.assert_allclose(metric_f(r, m, alpha), 1 - 2 * m / r, rtol=1e-12)


def test_uniform_density_star_analytic_and_fourth_order():
    e, pc = 3e-4, 1e-4
    factor = (e + pc) / (e + 3 * pc)
    radius = np.sqrt(3 * (1 - factor**2) / (8 * np.pi * e))
    mass = 4 * np.pi * e * radius**3 / 3
    errors = []
    for dr in [0.2, 0.1, 0.05]:
        star = solve_star(LinearEOS(e, np.inf), pc, dr=dr, surface_tol_km=1e-12)
        assert star.status == Status.SURFACE
        errors.append(abs(float(star.radius_km) - radius))
        assert star.last_state[0] >= 0
    assert 12 < errors[0] / errors[1] < 20
    assert 12 < errors[1] / errors[2] < 20
    np.testing.assert_allclose(star.radius_km, radius, atol=3e-9, rtol=0)
    np.testing.assert_allclose(star.mass_km, mass, atol=3e-9, rtol=0)
    np.testing.assert_allclose(star.mass_msun, mass / SOLAR_MASS_KM, atol=3e-9, rtol=0)
    np.testing.assert_allclose(star.redshift, 1 / factor - 1, atol=1e-9, rtol=0)


def reference_star(alpha):
    """Independent DOP853 integrating [P,m], using the original Julia formula."""
    pc, e0, cs2, r0 = 1e-4, 3e-4, 1 / 3, 1e-4
    ec = e0 + pc / cs2
    qc = 4 * np.pi * ec / 3
    gamma = np.sqrt(1 + 8 * alpha * qc)
    p2 = -(ec + pc) * (4 * qc / (1 + gamma) - qc + 4 * np.pi * pc) / (2 * gamma)
    initial = [pc + p2 * r0 * r0, qc * r0**3 + 4 * np.pi * p2 / cs2 * r0**5 / 5]

    def rhs(r, state):
        p, m = state
        # This linear analytic EOS admits continuation for event bracketing.
        e = e0 + p / cs2
        if alpha == 0:
            dp = -(e + p) * (m + 4 * np.pi * r**3 * p) / (r * (r - 2 * m))
        else:
            gamma = np.sqrt(1 + 8 * alpha * m / r**3)
            dp = (e + p) * (r**3 * (gamma + 8 * np.pi * alpha * p - 1) - 2 * alpha * m)
            dp /= r * r * gamma * (r * r * (gamma - 1) - 2 * alpha)
        return [dp, 4 * np.pi * r * r * e]

    def surface(r, state):
        return state[0]

    surface.terminal = True
    surface.direction = -1
    sol = solve_ivp(
        rhs,
        [r0, 30.0],
        initial,
        method="DOP853",
        events=surface,
        rtol=2e-12,
        atol=[1e-16, 1e-15],
        max_step=0.02,
    )
    return sol.t_events[0][0], sol.y_events[0][0, 1]


@pytest.mark.parametrize("alpha", [-2.0, 0.0, 6.0])
def test_compressible_star_against_independent_integrator(alpha):
    expected = reference_star(alpha)
    errors = []
    for dr in [0.4, 0.2, 0.1]:
        star = solve_star(LinearEOS(3e-4), 1e-4, alpha=alpha, dr=dr, surface_tol_km=1e-12)
        assert star.status == Status.SURFACE
        errors.append(np.linalg.norm(np.array([star.radius_km, star.mass_km]) - expected))
    assert 10 < errors[0] / errors[1] < 22
    assert 10 < errors[1] / errors[2] < 22
    np.testing.assert_allclose([star.radius_km, star.mass_km], expected, atol=1e-8, rtol=0)


def test_small_alpha_full_star_and_compiled_sequence():
    eos = LinearEOS(3e-4)
    base = solve_star(eos, 1e-4)
    for alpha in [0.0, -1e-12, 1e-12]:
        star = solve_star(eos, 1e-4, alpha=alpha)
        np.testing.assert_allclose(
            [star.radius_km, star.mass_km, star.redshift],
            [base.radius_km, base.mass_km, base.redshift],
            atol=1e-11,
        )
    pcs = jnp.array([5e-5, 1e-4, 2e-4])
    sequence = mass_radius_sequence(eos, pcs, alpha=6.0)
    assert np.all(sequence.status == Status.SURFACE)
    for i, pc in enumerate(pcs):
        star = solve_star(eos, pc, alpha=6.0)
        np.testing.assert_allclose(sequence.radius_km[i], star.radius_km, atol=1e-12)


def test_failure_statuses_are_not_stars():
    eos = LinearEOS(3e-4)
    cases = [
        (dict(rmax=1.0), Status.RADIUS_LIMIT),
        (dict(max_steps=2), Status.STEP_LIMIT),
        (dict(alpha=-1e4), Status.INVALID),
        (dict(dr=-0.1), Status.INVALID),
        (dict(pressure_bounds=(1e-5, 2e-4)), Status.EOS_BOUNDARY),
    ]
    for kwargs, expected in cases:
        star = solve_star(eos, 1e-4, **kwargs)
        assert star.status == expected
        assert np.isnan(star.radius_km) and np.isnan(star.mass_msun)


def test_eos_bounds_units_and_surface_choice():
    p = np.linspace(0, 400, 101)
    table = tabulated_eos(p, 240 + 3 * p)
    np.testing.assert_allclose(
        jax.jit(lambda eos, p: eos(p))(table, 100 * MEV_FM3_TO_KM2),
        540 * MEV_FM3_TO_KM2,
        rtol=1e-14,
    )
    assert np.isnan(table(-1e-10)) and np.isnan(table(500 * MEV_FM3_TO_KM2))
    pc = 100 * MEV_FM3_TO_KM2
    star = solve_star(table, pc)
    analytic = solve_star(LinearEOS(240 * MEV_FM3_TO_KM2), pc)
    np.testing.assert_allclose(star.radius_km, analytic.radius_km, atol=1e-9)
    assert solve_star(table, 500 * MEV_FM3_TO_KM2).status == Status.INVALID
    with pytest.raises(ValueError, match="strictly increasing"):
        tabulated_eos([0, 2, 1], [100, 102, 101])
    with pytest.raises(ValueError, match="surface_power"):
        tabulated_eos([0, 1, 2], [0, 2, 3])
    vacuum = tabulated_eos([0, 1, 2], [0, 2, 3], surface_power=0.6, units="km^-2")
    np.testing.assert_allclose(vacuum(0.1), 2 * 0.1**0.6)
    assert vacuum(0.0) == 0.0


def test_zero_density_surface_and_no_negative_eos_calls():
    observed = []

    def polytrope(p):
        jax.debug.callback(lambda x: observed.append(float(x)), p)
        return jnp.sqrt(p / 100.0) + p  # Relativistic Gamma=2 polytrope.

    star = solve_star(jax.tree_util.Partial(polytrope), 1e-4, dr=0.1)
    star.radius_km.block_until_ready()
    assert star.status == Status.SURFACE
    assert np.isfinite(star.mass_msun)
    assert min(observed) >= 0


def test_integer_step_and_vacuum_table_star():
    p = np.r_[0.0, np.geomspace(1e-14, 1e-3, 400)]
    e = np.sqrt(p / 100.0) + p
    table = tabulated_eos(p, e, units="km^-2", surface_power=0.5)
    star = solve_star(table, 1e-4, dr=1)
    fine = solve_star(table, 1e-4, dr=0.05)
    assert star.status == fine.status == Status.SURFACE
    np.testing.assert_allclose(star.radius_km, fine.radius_km, atol=2e-4, rtol=0)
