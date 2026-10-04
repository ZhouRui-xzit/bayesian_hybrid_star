"""NJL (u,d,s,e,mu) -> physical zero-pressure EOS -> GR/EGB stars.

Run from the project root: uv run python -m example.NJL_TOV
Writes a continuous central-pressure sequence to one MR CSV/SVG pair.
"""

import argparse
import csv
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from example.NJL_T0 import NJLModel, Parameters, density_grid, tov_segment
from src.common import MEV_FM3_TO_KM2
from src.eos import tabulated_eos
from src.structure import Status, mass_radius_sequence

# Calibrated for the fixed Parameters defaults; see example/README.md.
# A constant B_eff shifts P and epsilon, not this internal critical coupling.
RV_CRITICAL = 0.0670584907704
DEFAULT_OUTPUT = Path("data/example__NJL0T")


def build_njl_eos(rv=0.5, b_eff=10.0, *, max_rho=12.0, density_step=0.01):
    """Return the JAX EOS and surface metadata; pressures in the EOS are km^-2."""
    if not np.isfinite(rv) or rv <= RV_CRITICAL:
        raise ValueError(f"This example requires Rv > {RV_CRITICAL:.10f}")
    if not np.isfinite(b_eff) or b_eff < 0:
        raise ValueError("B_eff must be finite and >= 0 MeV/fm^3")
    model = NJLModel(Parameters(Rv=rv, B_eff=b_eff))
    rows = model.scan(density_grid(max_ratio=max_rho, step=density_step))
    if not all(np.isfinite(row["cs2"]) and row["cs2"] > 0 for row in rows):
        raise RuntimeError("The scanned NJL branch has nonpositive/nonfinite compressibility")
    branch, info = tov_segment(model, rows)
    if branch is None or not info["surface_available"]:
        raise RuntimeError(f"No usable zero-pressure NJL surface: {info['status']}")
    pressure = np.array([row["P_MeV_fm3"] for row in branch.rows])
    epsilon = np.array([row["epsilon_MeV_fm3"] for row in branch.rows])
    # At B_eff=0 only, the dilute massive Fermi gas has epsilon ~ P^(3/5).
    # B_eff=10 instead has a finite-density surface, found by solving P=0.
    eos = tabulated_eos(
        pressure,
        epsilon,
        units="MeV/fm3",
        surface_power=3 / 5 if epsilon[0] == 0 else None,
    )
    return eos, dict(info, surface_epsilon_MeV_fm3=float(epsilon[0]))


def central_pressure_grid(eos_max, pc_min=1e-3, pc_max=None, points=600):
    """Ascending log grid, covering low-mass stars and the high-pressure branch."""
    upper = eos_max if pc_max is None else pc_max
    if not np.isfinite([pc_min, upper]).all() or not 0 < pc_min < upper <= eos_max:
        raise ValueError(f"Require 0 < pc_min < pc_max <= {eos_max:.8g} MeV/fm^3")
    if points < 3:
        raise ValueError("At least three central-pressure points are required")
    return np.geomspace(pc_min, upper, points)


def mr_stem(rv, b_eff):
    rv_text = np.format_float_positional(float(rv), unique=True, trim="-")
    bag_text = np.format_float_positional(float(b_eff) if b_eff else 0.0, unique=True, trim="-")
    return f"mr_Rv{rv_text}_Beff{bag_text}"


def sequence_summary(pc, stars):
    """Report a sampled peak only when it has valid lower-mass neighbours."""
    valid = np.asarray(stars.status) == Status.SURFACE
    if not valid.any():
        return dict(valid=0, peak=None, peak_bracketed=False)
    mass = np.asarray(stars.mass_msun)
    i = int(np.argmax(np.where(valid, mass, -np.inf)))
    bracketed = (
        0 < i < len(pc) - 1
        and valid[i - 1]
        and valid[i + 1]
        and mass[i] > mass[i - 1]
        and mass[i] > mass[i + 1]
    )
    return dict(valid=int(valid.sum()), peak=i, peak_bracketed=bool(bracketed))


def write_mr(output, rv, b_eff, pc, epsilon_c, sequences):
    """One table and one figure; failures remain as status rows / gaps in the curves."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = mr_stem(rv, b_eff)
    csv_path, svg_path = output / f"{stem}.csv", output / f"{stem}.svg"
    columns = [
        "Rv",
        "B_eff_MeV_fm3",
        "alpha_km2",
        "Pc_MeV_fm3",
        "epsilon_c_MeV_fm3",
        "M_Msun",
        "R_km",
        "redshift",
        "status",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for alpha, stars in sequences.items():
            for i, pressure in enumerate(pc):
                writer.writerow(
                    [
                        rv,
                        b_eff,
                        alpha,
                        pressure,
                        epsilon_c[i],
                        stars.mass_msun[i],
                        stars.radius_km[i],
                        stars.redshift[i],
                        Status(int(stars.status[i])).name,
                    ]
                )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.8, len(sequences)))
    for color, (alpha, stars) in zip(colors, sequences.items(), strict=True):
        label = "GR" if alpha == 0 else rf"EGB, $\alpha={alpha:g}\,\mathrm{{km}}^2$"
        summary = sequence_summary(pc, stars)
        if summary["valid"] != len(pc):
            label += f" ({summary['valid']}/{len(pc)} converged)"
        axes[0].plot(stars.radius_km, stars.mass_msun, color=color, lw=1.9, label=label)
        axes[1].plot(pc, stars.mass_msun, color=color, lw=1.9)
        i = summary["peak"]
        if i is not None:
            marker = "*" if summary["peak_bracketed"] else "x"
            axes[0].plot(stars.radius_km[i], stars.mass_msun[i], marker, color=color, ms=10)
            axes[1].plot(pc[i], stars.mass_msun[i], marker, color=color, ms=10)
    axes[0].set(
        xlabel=r"Radius $R$ [km]",
        ylabel=r"Mass $M$ [$M_\odot$]",
        title="Mass–radius sequence",
        xlim=(0, None),
        ylim=(0, None),
    )
    axes[0].legend(frameon=False)
    axes[1].set(
        xlabel=r"Central pressure $P_c$ [MeV/fm$^3$]",
        ylabel=r"Mass $M$ [$M_\odot$]",
        title="Central-pressure coverage",
        xscale="log",
        ylim=(0, None),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(rf"NJL, $T=0$, $R_v={rv:g}$, $B_{{\rm eff}}={b_eff:g}$ MeV/fm$^3$")
    fig.savefig(svg_path)
    plt.close(fig)
    return csv_path, svg_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rv", type=float, default=0.5)
    parser.add_argument("--b-eff", type=float, default=10.0, help="MeV/fm^3")
    parser.add_argument("--points", type=int, default=600, help="central-pressure grid size")
    parser.add_argument("--pc-min", type=float, default=1e-3, help="MeV/fm^3")
    parser.add_argument("--pc-max", type=float, default=None, help="default: EOS upper pressure")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.0, 6.0], help="km^2")
    parser.add_argument("--max-rho", type=float, default=12.0, help="EOS maximum nB/n0")
    parser.add_argument("--density-step", type=float, default=0.01)
    parser.add_argument("--dr", type=float, default=0.02, help="RK4 radial step in km")
    parser.add_argument("--rmax", type=float, default=50.0, help="radius limit in km")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not all(np.isfinite(args.alpha)):
        parser.error("--alpha must contain finite couplings")
    if not np.isfinite(args.dr) or args.dr <= 1e-8:
        parser.error("--dr must be finite and > 1e-8 km")
    if not np.isfinite(args.rmax) or args.rmax <= 0:
        parser.error("--rmax must be finite and positive")
    if args.points < 3:
        parser.error("--points must be >= 3")
    print(
        f"NJL: Rv={args.rv:g}, B_eff={args.b_eff:g} MeV/fm^3 (Rv_crit={RV_CRITICAL:.8f})",
        flush=True,
    )
    try:
        eos, info = build_njl_eos(
            args.rv, args.b_eff, max_rho=args.max_rho, density_step=args.density_step
        )
        pc = central_pressure_grid(
            info["pressure_bounds_MeV_fm3"][1], args.pc_min, args.pc_max, args.points
        )
        # Clamp only unit-conversion roundoff at the already validated EOS bound.
        pc_geo = np.minimum(pc * MEV_FM3_TO_KM2, float(eos.pressure_bounds[1]))
        epsilon_c = np.asarray(eos(pc_geo)) / MEV_FM3_TO_KM2
        print(
            f"Surface nB/n0={info['surface_rho_over_rho0']:.8g}; "
            f"scanning {len(pc)} central pressures {pc[0]:.8g} -> {pc[-1]:.8g} MeV/fm^3",
            flush=True,
        )
        sequences = {}
        failed = False
        for alpha in dict.fromkeys(args.alpha):
            stars = jax.device_get(
                mass_radius_sequence(eos, pc_geo, alpha=alpha, dr=args.dr, rmax=args.rmax)
            )
            sequences[alpha] = stars
            summary = sequence_summary(pc, stars)
            print(f"alpha={alpha:g} km^2: {summary['valid']}/{len(pc)} reached surface", flush=True)
            i = summary["peak"]
            if i is not None:
                label = (
                    "sampled peak"
                    if summary["peak_bracketed"]
                    else "largest sampled mass; peak NOT bracketed"
                )
                print(
                    f"  {label}: M={stars.mass_msun[i]:.8f} Msun, "
                    f"R={stars.radius_km[i]:.8f} km, Pc={pc[i]:.8g} MeV/fm^3",
                    flush=True,
                )
            if summary["valid"] != len(pc):
                failed = True
                statuses, counts = np.unique(stars.status, return_counts=True)
                print(
                    "  " + ", ".join(f"{Status(int(s)).name}={n}" for s, n in zip(statuses, counts))
                )
        paths = write_mr(args.output, args.rv, args.b_eff, pc, epsilon_c, sequences)
        for path in paths:
            print(path)
        if failed:
            parser.exit(1, "Some stars failed; CSV retains their status and SVG leaves gaps.\n")
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"NJL TOV failed: {exc}\n")


if __name__ == "__main__":
    main()
