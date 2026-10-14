
#!/usr/bin/env python3
"""Three PRB-style 2x2 figures for four pure CT-HYB Bethe-DMFT runs.

Rows: U/t = 8, 10.
Columns: target <n> = 0.7, 0.9.

Figures:
    1. Im Sigma(iwn)
    2. Re and Im G(iwn)
    3. Measured G(tau)

Matsubara panels use all 512 signed frequencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import AutoMinorLocator, MultipleLocator


HERE = Path(__file__).resolve().parent

CASES = [
    (
        8, 0.7,
        "U8_n0p7_beta25_adaptive_v2",
        "U8p000_n0p700_beta25p00"
    ),
    (
        8, 0.9,
        "U8_n0p9_beta25_adaptive_v2",
        "U8p000_n0p900_beta25p00"
    ),
    (
        10, 0.7,
        "U10_n0p7_beta25_adaptive_v2",
        "U10p000_n0p700_beta25p00"
    ),
    (
        10, 0.9,
        "U10_n0p9_beta25_adaptive_v2",
        "U10p000_n0p900_beta25p00"
    ),
]

BLUE = "#0878B9"
RED = "#C7252E"
BLACK = "#171717"

SPINE_LW = 1.2
FIGSIZE = (8.6, 7.1)


def configure_matplotlib(use_tex: bool) -> None:

    mpl.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "text.latex.preamble": (
            r"\usepackage{amsmath}" if use_tex else ""
        ),

        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        "axes.linewidth": SPINE_LW,

        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,

        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,

        "lines.linewidth": 1.6,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })


def load_cases(root: Path) -> list[dict]:

    loaded = []

    for U, n, run_name, case_name in CASES:

        folder = root / run_name / case_name

        npz_path = folder / "dmft_result.npz"
        metadata_path = folder / "metadata.json"

        if not npz_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing dmft_result.npz or metadata.json in {folder}"
            )

        with np.load(npz_path, allow_pickle=False) as data:

            omega = np.imag(
                np.asarray(data["iw_over_t"], dtype=complex)
            )

            G_iw = np.asarray(
                data["G_iw"], dtype=complex
            ).reshape(-1)

            Sigma_iw = np.asarray(
                data["Sigma_iw"], dtype=complex
            ).reshape(-1)

            tau = np.asarray(
                data["tau_times_t"], dtype=float
            ).reshape(-1)

            G_tau = np.asarray(
                data["G_tau"], dtype=complex
            ).reshape(-1)

        with metadata_path.open("r", encoding="utf-8") as stream:
            meta = json.load(stream)

        if (
            len(omega) != 512
            or len(G_iw) != 512
            or len(Sigma_iw) != 512
        ):
            raise ValueError(
                f"Expected 512 signed Matsubara points in {npz_path}"
            )

        if len(tau) != len(G_tau):
            raise ValueError(
                f"Imaginary-time grid mismatch in {npz_path}"
            )

        if (
            int(meta["U_over_t"]) != U
            or abs(meta["target_filling"] - n) > 1e-9
        ):
            raise ValueError(
                f"Case metadata does not match directory: {folder}"
            )

        if abs(float(meta["beta_t"]) - 25.0) > 1e-9:
            raise ValueError(
                f"Expected beta*t=25 in {folder}"
            )

        if abs(float(meta.get("t_eV", 0.25)) - 0.25) > 1e-9:
            raise ValueError(
                f"Expected t=0.25 eV in {folder}"
            )

        if not meta.get("final_quality_passed", False):
            print(
                f"WARNING: final_quality_passed is false "
                f"for U/t={U}, n={n}"
            )

        order = np.argsort(omega)

        loaded.append({
            "U": U,
            "n": n,
            "omega": omega[order],
            "G_iw": G_iw[order],
            "Sigma_iw": Sigma_iw[order],
            "tau": tau / float(meta["beta_t"]),
            "G_tau": G_tau.real,
            "mu": float(meta["mu_over_t"]),
            "n_final": float(meta["final_filling_mean"]),
        })

    return loaded


def style_axis(ax: plt.Axes, *, is_tau: bool) -> None:

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(SPINE_LW)

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True
    )

    ax.xaxis.set_major_locator(
        MultipleLocator(0.2 if is_tau else 20)
    )

    ax.xaxis.set_minor_locator(
        MultipleLocator(0.05 if is_tau else 5)
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator(2)
    )


def save_fig(
    fig: plt.Figure,
    out: Path,
    filename: str,
    dpi: int
) -> None:

    out.mkdir(parents=True, exist_ok=True)

    for ext in ("pdf", "png"):

        path = out / f"{filename}.{ext}"

        fig.savefig(
            path,
            dpi=dpi,
            facecolor="white"
        )

        print("Saved:", path)

    plt.close(fig)


def plot_grid(
    cases: list[dict],
    which: str,
    out: Path,
    dpi: int
) -> None:

    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True
    )

    flat = axes.ravel()
    letters = "abcd"

    for idx, (ax, case) in enumerate(zip(flat, cases)):

        U = case["U"]
        n = case["n"]

        negative = case["omega"] < 0
        positive = case["omega"] > 0

        if which == "sigma":

            for mask in (negative, positive):

                ax.plot(
                    case["omega"][mask],
                    case["Sigma_iw"].imag[mask],
                    color=BLUE,
                    lw=1.55
                )

            ax.axhline(
                0, color="0.45", lw=0.8, ls=":"
            )

            ax.axvline(
                0, color="0.6", lw=0.7, ls=":"
            )

        elif which == "Giw":

            for mask in (negative, positive):

                ax.plot(
                    case["omega"][mask],
                    case["G_iw"].real[mask],
                    color=BLUE,
                    lw=1.55,
                    label=(
                        r"$\mathrm{Re}\,G$"
                        if mask is negative else None
                    )
                )

                ax.plot(
                    case["omega"][mask],
                    case["G_iw"].imag[mask],
                    color=RED,
                    lw=1.55,
                    ls="--",
                    label=(
                        r"$\mathrm{Im}\,G$"
                        if mask is negative else None
                    )
                )

            ax.axhline(
                0, color="0.45", lw=0.8, ls=":"
            )

            ax.axvline(
                0, color="0.6", lw=0.7, ls=":"
            )

        elif which == "Gtau":

            ax.plot(
                case["tau"],
                case["G_tau"],
                color=BLACK,
                lw=1.65
            )

        else:
            raise ValueError(which)

        style_axis(
            ax,
            is_tau=(which == "Gtau")
        )

        ax.set_title(
            rf"({letters[idx]})  "
            rf"$U/t={U},\;\langle n\rangle={n:.1f}$",
            loc="left",
            fontsize=12,
            pad=5
        )

    for ax in axes[0]:
        ax.tick_params(labelbottom=False)

    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)

    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        bottom=0.13,
        top=0.82,
        wspace=0.08,
        hspace=0.18
    )

    if which == "sigma":

        title = (
            r"Imaginary self-energy, "
            r"full signed Matsubara grid"
        )

        ylabel = (
            r"$\mathrm{Im}\,\Sigma(i\omega_n)/t$"
        )

        xlabel = r"$\omega_n/t$"

        basename = "01_ImSigma_full_512"

    elif which == "Giw":

        title = (
            r"Green's function, "
            r"full signed Matsubara grid"
        )

        ylabel = r"$t\,G(i\omega_n)$"

        xlabel = r"$\omega_n/t$"

        basename = "02_Giw_full_512"

        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=2,
            frameon=False,
            columnspacing=2.5
        )

    else:

        title = (
            r"Imaginary-time Green's function, "
            r"complete $\tau$ grid"
        )

        ylabel = r"$G(\tau)$"

        xlabel = r"$\tau/\beta$"

        basename = "03_Gtau_full"

    fig.suptitle(
        title,
        x=0.53,
        y=0.987,
        fontsize=16
    )

    fig.text(
        0.53,
        0.947,
        r"$\beta t=25,\quad t=0.25\,\mathrm{eV},"
        r"\quad T\simeq116.05\,\mathrm{K}$",
        ha="center",
        va="center",
        fontsize=12
    )

    fig.supxlabel(
        xlabel,
        x=0.53,
        y=0.035,
        fontsize=16
    )

    fig.supylabel(
        ylabel,
        x=0.025,
        fontsize=16
    )

    save_fig(
        fig,
        out,
        basename,
        dpi
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=HERE / "cthyb_test_results",
        help=(
            "Parent folder containing four adaptive_v2 "
            "run folders"
        )
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Where to write three figures; "
            "default ROOT/steve_three_figures"
        )
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=400
    )

    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Use external LaTeX; requires latex on PATH"
    )

    args = parser.parse_args()

    configure_matplotlib(args.usetex)

    out = (
        args.out_dir
        or args.root / "steve_three_figures"
    )

    cases = load_cases(args.root)

    for kind in ("sigma", "Giw", "Gtau"):
        plot_grid(
            cases,
            kind,
            out,
            args.dpi
        )

    print("All three 2x2 figures completed.")


if __name__ == "__main__":
    main()

