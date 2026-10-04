"""Locate the zero-temperature end of the spinodal on the continued neutral branch.

This is a coupling search, not a multi-start gap search. Every EOS is obtained
with the same dilute-to-dense JAX continuation. No data files are written here.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar

from example.NJL_T0 import NJLModel, Parameters, density_grid


def _branch(rv, step, max_ratio, parameters):
    model = NJLModel(replace(parameters, Rv=float(rv)))
    rows = model.scan(density_grid(max_ratio=max_ratio, step=step))
    states = model.last_scan_states

    def point(ratio):
        i = int(np.argmin(abs(states["ratios"] - ratio)))
        initial = {"x": states["x"][i], "strange": states["strange"][i]}
        return model.solve_density(float(ratio), initial)[0]

    return rows, point


def minimum_stiffness(rv, step=0.01, max_ratio=6.0, parameters=Parameters()):
    """Minimize d(muB)/d(nB/n0) over the declared density interval.

    Use (dP/dnB)/(nB/n0), not min(cs2): cs2 tends to zero in dilute matter
    even when there is no critical point. Refine each resolved interior minimum
    with local, single-seed field solves so a fixed grid cannot hide a spinodal.
    """
    rows, point = _branch(rv, step, max_ratio, parameters)
    slopes = np.array([r["dP_dnB_MeV"] / r["rho_over_rho0"] for r in rows])
    ratios = np.array([r["rho_over_rho0"] for r in rows])
    candidates = [rows[0], rows[-1]]

    def slope(ratio):
        row = point(ratio)
        return row["dP_dnB_MeV"] / ratio

    for i in range(1, len(rows) - 1):
        if slopes[i] <= slopes[i - 1] and slopes[i] <= slopes[i + 1]:
            fit = minimize_scalar(
                slope,
                bounds=(ratios[i - 1], ratios[i + 1]),
                method="bounded",
                options={"xatol": 1e-9},
            )
            if not fit.success:
                raise RuntimeError("Density refinement of the stiffness minimum failed")
            candidates.append(point(fit.x))
    best = min(candidates, key=lambda r: r["dP_dnB_MeV"] / r["rho_over_rho0"])
    return {
        "Rv": float(rv),
        "rho_over_rho0": best["rho_over_rho0"],
        "min_dmuB_dratio_MeV": best["dP_dnB_MeV"] / best["rho_over_rho0"],
        "muB_MeV": best["muB_MeV"],
        "P_MeV_fm3": best["P_MeV_fm3"],
        "cs2": best["cs2"],
    }


def coexistence_check(rv, step=0.005, max_ratio=6.0, parameters=Parameters()):
    """Check a common tangent below the critical Rv, without changing the raw EOS.

    Endpoints are locally neutral phases on the traced branch. This is a
    diagnostic Maxwell check, not a globally neutral Gibbs construction.
    """
    rows, point = _branch(rv, step, max_ratio, parameters)
    n = np.array([r["nB_fm3"] for r in rows])
    energy = np.array([r["epsilon_MeV_fm3"] for r in rows])
    hull = []
    for i in range(len(rows)):
        while len(hull) >= 2:
            a, b = hull[-2:]
            left = (energy[b] - energy[a]) / (n[b] - n[a])
            right = (energy[i] - energy[b]) / (n[i] - n[b])
            if right > left:
                break
            hull.pop()
        hull.append(i)
    gaps = [(i, j) for i, j in zip(hull[:-1], hull[1:], strict=True) if j > i + 1]
    if len(gaps) != 1:
        raise RuntimeError(f"Expected one resolved coexistence interval, found {len(gaps)}")
    i, j = gaps[0]
    low, high = rows[i]["rho_over_rho0"], rows[j]["rho_over_rho0"]
    middle = (low + high) / 2

    def equations(ratios):
        a, b = point(ratios[0]), point(ratios[1])
        return np.array(
            [(a["muB_MeV"] - b["muB_MeV"]) / 100, (a["P_MeV_fm3"] - b["P_MeV_fm3"]) / 10]
        )

    def jacobian(ratios):
        a, b = point(ratios[0]), point(ratios[1])
        return np.array(
            [
                [a["dP_dnB_MeV"] / ratios[0] / 100, -b["dP_dnB_MeV"] / ratios[1] / 100],
                [a["dP_dnB_MeV"] * parameters.rho0 / 10, -b["dP_dnB_MeV"] * parameters.rho0 / 10],
            ]
        )

    fit = least_squares(
        equations,
        [low, high],
        jac=jacobian,
        bounds=([1e-8, middle], [middle, max_ratio]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    a, b = point(fit.x[0]), point(fit.x[1])
    if (
        not fit.success
        or np.max(np.abs(fit.fun)) > 1e-9
        or b["rho_over_rho0"] - a["rho_over_rho0"] < step
        or min(a["cs2"], b["cs2"]) <= 0
    ):
        raise RuntimeError("Nontrivial, mechanically stable Maxwell endpoints not resolved")
    interior = [r for r in rows if a["nB_fm3"] < r["nB_fm3"] < b["nB_fm3"]]
    if any(
        r["epsilon_MeV_fm3"]
        < a["epsilon_MeV_fm3"] + a["muB_MeV"] * (r["nB_fm3"] - a["nB_fm3"]) - 1e-7
        for r in interior
    ):
        raise RuntimeError("Common tangent does not bound the continued energy from below")
    return {
        "Rv": float(rv),
        "rho_low_over_rho0": a["rho_over_rho0"],
        "rho_high_over_rho0": b["rho_over_rho0"],
        "density_jump_fm3": b["nB_fm3"] - a["nB_fm3"],
        "muB_MeV": a["muB_MeV"],
        "P_MeV_fm3": a["P_MeV_fm3"],
        "delta_muB_MeV": a["muB_MeV"] - b["muB_MeV"],
        "delta_P_MeV_fm3": a["P_MeV_fm3"] - b["P_MeV_fm3"],
    }


def find_critical_rv(parameters=Parameters(), max_ratio=6.0, steps=(0.01, 0.005)):
    """Bracket Rv with negative/positive minimum curvature, then refine its zero."""
    results = []
    for step in steps:
        cache = {}

        def minimum(rv):
            if rv not in cache:
                cache[rv] = minimum_stiffness(rv, step, max_ratio, parameters)
            return cache[rv]["min_dmuB_dratio_MeV"]

        lo, hi = 0.0, 0.25
        if minimum(lo) >= 0:
            raise ValueError("No first-order spinodal found at Rv=0 in the searched interval")
        while minimum(hi) <= 0:
            hi *= 2
            if hi > 4:
                raise RuntimeError("No upper critical-coupling bracket found up to Rv=4")
        critical = brentq(minimum, lo, hi, xtol=2e-10)
        refined = minimum_stiffness(critical, step, max_ratio, parameters)
        if not 1e-8 < refined["rho_over_rho0"] < max_ratio:
            raise RuntimeError("Minimum is at the density boundary, not an interior critical point")
        results.append(dict(refined, density_step=step))
    critical = results[-1]["Rv"]
    if len(results) > 1 and abs(results[-1]["Rv"] - results[-2]["Rv"]) > 2e-7:
        raise RuntimeError("Critical coupling has not converged under density-step refinement")
    recommended = round(float(np.ceil((critical + 1e-6) / 0.025) * 0.025), 12)
    above = minimum_stiffness(recommended, min(steps), max_ratio, parameters)
    if above["min_dmuB_dratio_MeV"] <= 0:
        raise RuntimeError("Suggested coupling still has a spinodal")
    probes = [
        coexistence_check(critical - offset, min(steps), max_ratio, parameters)
        for offset in (0.01, 0.001)
        if critical > offset
    ]
    return {
        "Rv_critical": critical,
        "Rv_recommended": recommended,
        "Gv_MeV_minus2": critical * parameters.Gs_Lambda2 / parameters.cutoff**2,
        "Gv_fm2": critical * parameters.Gs_Lambda2 * (parameters.hbar_c / parameters.cutoff) ** 2,
        "refinements": results,
        "above_critical": above,
        "coexistence_below": probes,
        "scope": "T=0, homogeneous, locally neutral continued NJL branch; 1e-8 <= nB/n0 <= max_ratio",
    }


if __name__ == "__main__":
    from pprint import pprint

    pprint(find_critical_rv())
