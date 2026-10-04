"""Acceptance run: uv run python -m example.check_njl_eos --benchmark."""

import argparse
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from example.find_critical_rv import find_critical_rv
from example.NJL_T0 import NJLModel, Parameters, density_grid, scan_states

COMPACT_COLUMNS = [
    "Rv",
    "B_eff_MeV_fm3",
    "rho_over_rho0",
    "nB_fm3",
    "P_MeV_fm3",
    "epsilon_MeV_fm3",
    "muB_MeV",
    "cs2",
]
DEFAULT_OUTPUT = Path("data/example__NJL0T")

MEV_FM3_TO_CGS = 1.602176634e33  # erg/cm^3 = dyn/cm^2
C_CGS = 2.99792458e10
G_CGS = 6.67430e-8
MEV_FM3_TO_KM2 = MEV_FM3_TO_CGS * G_CGS / C_CGS**4 * 1e10


def validate(rows, params):
    """Physics checks, including a derivative identity independent of P+epsilon."""
    nb = np.array([r["nB_fm3"] for r in rows])
    target = np.array([r["rho_over_rho0"] * params.rho0 for r in rows])
    residual = max(r["residual"] for r in rows)
    density_error = float(np.max(abs(nb - target) / target))
    charge_error = float(np.max(np.abs([r["nQ_fm3"] for r in rows]) / nb))
    thermo_error = max(abs(r["thermo_error_MeV_fm3"]) for r in rows)
    derivative_error = max(abs(r["d_epsilon_dnB_MeV"] - r["muB_MeV"]) for r in rows)
    numeric = [[v for v in row.values() if isinstance(v, (int, float))] for row in rows]
    if not all(np.all(np.isfinite(values)) for values in numeric):
        raise AssertionError("Nonfinite EOS or response")
    if residual > 2e-8 or max(density_error, charge_error) > 2e-8:
        raise AssertionError("Gap/density/neutrality tolerance failed")
    if thermo_error > 2e-6 or derivative_error > 1e-4:
        raise AssertionError("Thermodynamic identity failed")
    for row in rows:
        for name, mass in (("e", params.m_e), ("mu", params.m_mu)):
            if row["mu_e_MeV"] <= mass and row[f"n_{name}_fm3"] != 0:
                raise AssertionError(f"Nonzero {name} density below threshold")
        if abs(row["mu_d_MeV"] - row["mu_u_MeV"] - row["mu_e_MeV"]) > 1e-5:
            raise AssertionError("Beta equilibrium failed")
    return dict(
        points=len(rows),
        max_scaled_residual=residual,
        max_relative_density_error=density_error,
        max_charge_per_baryon=charge_error,
        max_thermo_error_MeV_fm3=thermo_error,
        max_first_law_error_MeV=derivative_error,
        mechanically_unstable_points=sum(r["cs2"] <= 0 for r in rows),
        muon_points=sum(r["n_mu_fm3"] > 0 for r in rows),
        strange_points=sum(r["n_s_fm3"] > 0 for r in rows),
        max_newton_steps=max(r["newton_steps"] for r in rows),
    )


def write_csv(path, rows, mode="compact"):
    """Write one parameter combination, with no sidecar files."""
    if mode not in ("compact", "full"):
        raise ValueError("CSV mode must be compact or full")
    exported = []
    for row in rows:
        item = row.copy()
        p, e = item["P_MeV_fm3"], item["epsilon_MeV_fm3"]
        item.update(
            P_dyn_cm2=p * MEV_FM3_TO_CGS,
            epsilon_erg_cm3=e * MEV_FM3_TO_CGS,
            mass_density_g_cm3=e * MEV_FM3_TO_CGS / C_CGS**2,
            P_km_minus2=p * MEV_FM3_TO_KM2,
            epsilon_km_minus2=e * MEV_FM3_TO_KM2,
        )
        exported.append(item)
    with path.open("w", newline="") as stream:
        fields = COMPACT_COLUMNS if mode == "compact" else list(exported[0])
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(exported)


def eos_stem(rv, bag):
    """Stable readable parameter names without rounding distinct floats together."""
    rv_text = np.format_float_positional(float(rv) if rv else 0.0, unique=True, trim="-")
    bag_text = np.format_float_positional(float(bag) if bag else 0.0, unique=True, trim="-")
    return f"eos_Rv{rv_text}_Beff{bag_text}"


def plot_eos(scans, output_path, bag, critical=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(scans)))
    for color, (rv, rows) in zip(colors, sorted(scans.items()), strict=True):
        n = np.array([r["rho_over_rho0"] for r in rows])
        pressure = np.array([r["P_MeV_fm3"] for r in rows])
        energy = np.array([r["epsilon_MeV_fm3"] for r in rows])
        cs2 = np.array([r["cs2"] for r in rows])
        label = rf"$R_v={rv:.6g}$"
        axes[0].plot(energy, pressure, color=color, label=label, lw=1.7)
        axes[1].plot(n, cs2, color=color, lw=1.7)
    axes[0].set(xlabel=r"$\epsilon$ [MeV fm$^{-3}$]", ylabel=r"$P$ [MeV fm$^{-3}$]")
    axes[1].set(xlabel=r"$n_B/n_0$", ylabel=r"$c_s^2=dP/d\epsilon$")
    axes[1].axhline(0, color="#bb4455", lw=0.8, ls="--")
    axes[0].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    title = rf"Cold NJL EOS: $B_{{\rm eff}}={bag:g}$ MeV fm$^{{-3}}$"
    if critical:
        title += rf"; $R_{{v,\rm crit}}\simeq {critical['Rv_critical']:.7f}$"
    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rv", nargs="+", type=float, default=[0, 0.25, 0.5, 1])
    parser.add_argument("--b-eff", "--B-eff", type=float, default=0.0, help="MeV/fm^3")
    parser.add_argument("--max-rho", type=float, default=6.0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-mode", choices=("compact", "full"), default="compact")
    parser.add_argument("--benchmark", action="store_true", help="Print warm scan timing")
    parser.add_argument(
        "--find-critical",
        action="store_true",
        help="Find the disappearance of the first-order spinodal and add its EOS",
    )
    args = parser.parse_args(argv)
    if len(set(args.rv)) != len(args.rv):
        parser.error("Rv values must be unique")
    ratios = density_grid(args.max_rho, args.step)
    critical = None
    rvs = list(args.rv)
    if args.find_critical:
        print(
            "Locating critical Rv, with density-step refinement and Maxwell checks ...", flush=True
        )
        critical = find_critical_rv(Parameters(B_eff=args.b_eff), max_ratio=args.max_rho)
        print(
            f"Rv_critical = {critical['Rv_critical']:.10f}; "
            f"Gv = {critical['Gv_fm2']:.10g} fm^2 "
            f"= {critical['Gv_MeV_minus2']:.10g} MeV^-2",
            flush=True,
        )
        for item in critical["refinements"]:
            print(
                f"  step={item['density_step']:g}: Rv={item['Rv']:.10f}, "
                f"nB/n0={item['rho_over_rho0']:.8f}, "
                f"min dmuB/d(nB/n0)={item['min_dmuB_dratio_MeV']:.3e} MeV"
            )
        for item in critical["coexistence_below"]:
            print(
                f"  Maxwell at Rv={item['Rv']:.8f}: "
                f"nB/n0={item['rho_low_over_rho0']:.8f} -> {item['rho_high_over_rho0']:.8f}, "
                f"density jump={item['density_jump_fm3']:.8g} fm^-3"
            )
        print(
            f"  Suggested Rv={critical['Rv_recommended']:g}: "
            f"minimum stiffness={critical['above_critical']['min_dmuB_dratio_MeV']:.8g} MeV"
        )
        print(
            "  Scope: this homogeneous, locally neutral continuation branch at T=0, up to "
            f"nB/n0={args.max_rho:g}; no search over disconnected phases."
        )
        rvs = sorted(set([*rvs, critical["Rv_critical"], critical["Rv_recommended"]]))
    scans = {}
    for rv in rvs:
        par = Parameters(Rv=rv, B_eff=args.b_eff)
        start = time.perf_counter()
        model = NJLModel(par)
        rows = model.scan(ratios)
        elapsed = time.perf_counter() - start
        report = validate(rows, par)
        scans[rv] = rows
        print(
            f"Rv={rv:.8g}: {len(rows)} points, {elapsed:.3f}s, "
            f"residual={report['max_scaled_residual']:.3e}, "
            f"unstable points={report['mechanically_unstable_points']}",
            flush=True,
        )
        if args.benchmark:
            ns = jnp.asarray(ratios * par.rho0 / par.density_unit)
            start = time.perf_counter()
            jax.block_until_ready(scan_states(ns, model.a, model.initial))
            print(f"  warm scan kernel: {(time.perf_counter() - start) * 1000:.3f} ms")
    # One CSV/SVG pair per parameter combination, in both CSV modes.
    args.output.mkdir(parents=True, exist_ok=True)
    for rv, rows in scans.items():
        stem = eos_stem(rv, args.b_eff)
        csv_path, svg_path = args.output / f"{stem}.csv", args.output / f"{stem}.svg"
        write_csv(csv_path, rows, args.csv_mode)
        plt.close(plot_eos({rv: rows}, svg_path, args.b_eff, critical))
        print(f"Output ({args.csv_mode}): {csv_path}, {svg_path}")


if __name__ == "__main__":
    main()
