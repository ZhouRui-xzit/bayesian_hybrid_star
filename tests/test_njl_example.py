"""Independent quadrature, thermodynamics, thresholds, and continuation checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad as scipy_quad

from example.check_njl_eos import MEV_FM3_TO_CGS, MEV_FM3_TO_KM2, validate
from example.NJL_T0 import (
    NJLModel,
    Parameters,
    density_grid,
    energy_integral,
    lepton_gas,
    pressure_integral,
    scalar_integral,
    tov_segment,
)


def quad(fn, lower, upper, **kwargs):
    kwargs.setdefault("epsabs", 1e-13)
    kwargs.setdefault("epsrel", 1e-12)
    return scipy_quad(fn, lower, upper, **kwargs)


@pytest.fixture(scope="module")
def scans():
    result = {}
    for rv in (0.0, 0.25, 0.5, 1.0):
        model = NJLModel(Parameters(Rv=rv))
        result[rv] = model, model.scan()
    return result


@pytest.mark.parametrize("rv", [0.0, 0.25, 0.5, 1.0])
def test_complete_continuation_and_physics(scans, rv):
    model, rows = scans[rv]
    report = validate(rows, model.parameters)
    assert report["points"] == 601
    assert rows[0]["rho_over_rho0"] == 1e-8
    assert rows[-1]["rho_over_rho0"] == 6.0
    assert rows[0]["n_s_fm3"] == 0
    assert rows[-1]["n_s_fm3"] > 0
    assert report["max_scaled_residual"] < 1e-9
    if rv == 1:
        assert report["muon_points"] > 0
    # P/n^(5/3) has a finite dilute limit; no cancellation floor from the vacuum.
    assert 1.04e-12 < rows[0]["P_MeV_fm3"] < 1.06e-12


@pytest.mark.parametrize("p", [1e-6, 0.005, 0.08, 1.0])
def test_fermi_integrals_against_independent_quadrature(p):
    m = 0.5
    for fn, integrand in (
        (scalar_integral, lambda q: q**2 / np.hypot(q, m)),
        (energy_integral, lambda q: q**2 * np.hypot(q, m)),
        (pressure_integral, lambda q: q**4 / np.hypot(q, m)),
    ):
        expected = quad(integrand, 0, p, epsabs=1e-30, epsrel=1e-12)[0]
        assert float(fn(p, m)) == pytest.approx(expected, rel=2e-10, abs=1e-35)


def test_lepton_threshold_and_derivative():
    mass = jnp.array([0.511, 105.658])
    n, e, p = lepton_gas(0.5, mass)
    assert np.all(np.asarray([n, e, p]) == 0)
    mu = 120.0
    n, e, p = lepton_gas(mu, mass)
    dp = jax.jacfwd(lambda value: lepton_gas(value, mass)[2])(mu)
    np.testing.assert_allclose(e + p, mu * n, rtol=2e-12)
    np.testing.assert_allclose(dp, n, rtol=2e-12)
    threshold_derivative = jax.jacfwd(lambda value: lepton_gas(value, mass)[0])(mass[1])
    assert np.all(np.isfinite(threshold_derivative))
    assert float(lepton_gas(mass[1], mass)[0][1]) == 0


def test_eos_against_direct_reference_integrals(scans):
    """Rebuild the reference theta=eta=T=0 grand potential with scipy.quad."""
    model, rows = scans[0.5]
    row = rows[400]  # nB/n0=4, both three-flavor and lepton contributions
    par = model.parameters
    gs, kt, gv = np.asarray(model.a)[:3]
    nq = np.array([row[f"n_{f}_fm3"] for f in ("u", "d", "s")]) / par.density_unit
    kf = np.cbrt(np.pi**2 * nq)
    masses = np.array([row[f"M_{f}_MeV"] for f in ("u", "d", "s")]) / par.cutoff
    sigma = np.array(
        [
            3 * m / np.pi**2 * quad(lambda p: p**2 / np.hypot(p, m), k, 1, epsabs=1e-13)[0]
            for m, k in zip(masses, kf, strict=True)
        ]
    )
    vacuum = model.vacuum_masses
    sv = np.array(
        [3 * m / np.pi**2 * quad(lambda p: p**2 / np.hypot(p, m), 0, 1)[0] for m in vacuum]
    )

    def background(s, m):
        return (
            2 * gs * np.sum(s**2)
            + 4 * kt * np.prod(s)
            - 3 / np.pi**2 * sum(quad(lambda p: p**2 * np.hypot(p, mass), 0, 1)[0] for mass in m)
        )

    shift = background(sigma, masses) - background(sv, vacuum)
    vector = 2 * gv * np.sum(nq**2)
    energy, pressure = shift + vector, -shift + vector
    for mass, k in zip(masses, kf, strict=True):
        energy += 3 / np.pi**2 * quad(lambda p: p**2 * np.hypot(p, mass), 0, k)[0]
        pressure += (
            3 / np.pi**2 * quad(lambda p: p**2 * (np.hypot(k, mass) - np.hypot(p, mass)), 0, k)[0]
        )
    for m in (par.m_e, par.m_mu):
        mass, mu = m / par.cutoff, row["mu_e_MeV"] / par.cutoff
        k = np.sqrt(max(mu**2 - mass**2, 0))
        energy += quad(lambda p: p**2 * np.hypot(p, mass), 0, k)[0] / np.pi**2
        pressure += quad(lambda p: p**2 * (mu - np.hypot(p, mass)), 0, k)[0] / np.pi**2
    assert row["P_MeV_fm3"] == pytest.approx(pressure * par.energy_unit, abs=1e-8)
    assert row["epsilon_MeV_fm3"] == pytest.approx(energy * par.energy_unit, abs=1e-8)


def test_first_law_and_sound_speed_with_finite_differences(scans):
    model, rows = scans[0.5]
    state = {
        "x": model.last_scan_states["x"][400],
        "strange": model.last_scan_states["strange"][400],
    }
    delta = 1e-4
    low, _ = model.solve_density(4 - delta, state)
    high, _ = model.solve_density(4 + delta, state)
    de = high["epsilon_MeV_fm3"] - low["epsilon_MeV_fm3"]
    dp = high["P_MeV_fm3"] - low["P_MeV_fm3"]
    assert de / (2 * delta * model.parameters.rho0) == pytest.approx(rows[400]["muB_MeV"], rel=1e-7)
    assert dp / de == pytest.approx(rows[400]["cs2"], rel=1e-6)


def test_bag_shift_and_unit_conversions(scans):
    _, baseline = scans[0.5]
    bag = NJLModel(Parameters(Rv=0.5, B_eff=60.0))
    shifted = bag.scan()
    for index in (0, 100, 400, 600):
        assert shifted[index]["P_MeV_fm3"] == pytest.approx(baseline[index]["P_MeV_fm3"] - 60)
        assert shifted[index]["epsilon_MeV_fm3"] == pytest.approx(
            baseline[index]["epsilon_MeV_fm3"] + 60
        )
        for key in ("n_s_fm3", "muB_MeV", "cs2"):
            assert shifted[index][key] == pytest.approx(baseline[index][key], rel=1e-12)
    assert MEV_FM3_TO_CGS == pytest.approx(1.602176634e33)
    assert MEV_FM3_TO_KM2 == pytest.approx(1.323833e-6, rel=1e-6)
    eos, info = tov_segment(bag, shifted)
    assert info["surface_available"]
    assert eos.pressure_bounds[0] == 0
    assert eos.epsilon_of_pressure(0) > 0  # resolved finite-density surface


def test_tov_bounds_and_unstable_branch_are_explicit(scans):
    model, rows = scans[0.0]
    eos, status = tov_segment(model, rows)
    assert status["status"] == "core_only"
    assert status["excluded_low_density_points"] > 0
    assert any(r["cs2"] < 0 for r in rows)  # raw points have not disappeared
    assert all(r["cs2"] > 0 for r in eos.rows)
    with pytest.raises(ValueError, match="outside EOS interval"):
        eos.epsilon_of_pressure(0)
    with pytest.raises(ValueError, match="outside EOS interval"):
        eos.epsilon_of_pressure(eos.pressure_bounds[1] + 1)
    model, rows = scans[0.5]
    eos, status = tov_segment(model, rows)
    assert status["surface_available"]
    assert eos.epsilon_of_pressure(0) == 0
    assert eos.sound_speed_sq(0) == 0
    test_pressures = np.geomspace(1e-12, eos.pressure_bounds[1], 2000)
    cs2 = eos.sound_speed_sq(test_pressures)
    assert np.all((cs2 > 0) & (cs2 < 1))


def test_density_step_convergence(scans):
    model = NJLModel(Parameters(Rv=0.5))
    fine = model.scan(density_grid(step=0.005))
    _, coarse = scans[0.5]
    for index in (25, 100, 300, 400, 600):
        for key in ("P_MeV_fm3", "epsilon_MeV_fm3", "M_s_MeV", "n_mu_fm3", "cs2"):
            match = min(
                fine, key=lambda row: abs(row["rho_over_rho0"] - coarse[index]["rho_over_rho0"])
            )
            assert match["rho_over_rho0"] == pytest.approx(
                coarse[index]["rho_over_rho0"], abs=1e-12
            )
            assert match[key] == pytest.approx(coarse[index][key], rel=1e-8, abs=1e-10)


def test_critical_rv_and_collapsing_coexistence():
    from example.find_critical_rv import find_critical_rv, minimum_stiffness

    result = find_critical_rv()
    assert result["Rv_critical"] == pytest.approx(0.06705849, abs=2e-7)
    refinements = result["refinements"]
    assert abs(refinements[0]["Rv"] - refinements[1]["Rv"]) < 1e-7
    assert abs(refinements[-1]["min_dmuB_dratio_MeV"]) < 1e-6
    assert 0.81 < refinements[-1]["rho_over_rho0"] < 0.82
    assert result["above_critical"]["min_dmuB_dratio_MeV"] > 0
    farther, closer = result["coexistence_below"]
    assert farther["density_jump_fm3"] > closer["density_jump_fm3"] > 0
    for endpoints in (farther, closer):
        assert abs(endpoints["delta_P_MeV_fm3"]) < 1e-7
        assert abs(endpoints["delta_muB_MeV"]) < 1e-7
    # A density-independent bag constant cannot change the internal critical Rv.
    shifted = minimum_stiffness(result["Rv_critical"], parameters=Parameters(B_eff=60))
    assert abs(shifted["min_dmuB_dratio_MeV"]) < 1e-6


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_two_file_output_and_csv_modes(tmp_path, mode):
    import csv

    from example.check_njl_eos import COMPACT_COLUMNS, main

    main(["--rv", "0.25", "0.5", "--max-rho", "0.1", "--csv-mode", mode, "--output", str(tmp_path)])
    assert {p.name for p in tmp_path.iterdir()} == {
        f"eos_Rv{rv}_Beff0.{extension}" for rv in (0.25, 0.5) for extension in ("csv", "svg")
    }
    for rv in (0.25, 0.5):
        with (tmp_path / f"eos_Rv{rv}_Beff0.csv").open() as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        assert {float(row["Rv"]) for row in rows} == {rv}
        assert len(rows) == 11
        if mode == "compact":
            assert fields == COMPACT_COLUMNS
        else:
            assert {"n_mu_fm3", "P_km_minus2", "residual"} <= set(fields)
