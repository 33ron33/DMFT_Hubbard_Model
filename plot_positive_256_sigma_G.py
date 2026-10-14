#!/usr/bin/env python3
"""
Make two PRB-style 2x2 figures from four CT-HYB DMFT runs:

1) Im Sigma(iwn)/t vs omega_n/t using the 256 positive Matsubara frequencies
2) t G(iwn) vs omega_n/t using the 256 positive Matsubara frequencies
   (plots both Re G and Im G)

Panels:
    (a) U/t = 8,  <n> = 0.7
    (b) U/t = 8,  <n> = 0.9
    (c) U/t = 10, <n> = 0.7
    (d) U/t = 10, <n> = 0.9
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


# -----------------------------
# Cases
# -----------------------------
CASES = [
    (8, 0.7, "U8_n0p7_beta25_adaptive_v2",  "U8p000_n0p700_beta25p00"),
    (8, 0.9, "U8_n0p9_beta25_adaptive_v2",  "U8p000_n0p900_beta25p00"),
    (10, 0.7, "U10_n0p7_beta25_adaptive_v2", "U10p000_n0p700_beta25p00"),
    (10, 0.9, "U10_n0p9_beta25_adaptive_v2", "U10p000_n0p900_beta25p00"),
]

# -----------------------------
# Styling
# -----------------------------
C_SIGMA = "#C7252E"
C_GRE   = "#1674B7"
C_GIM   = "#C7252E"

SPINE_LW = 1.2
FIGSIZE = (8.7, 7.1)


def configure_matplotlib(use_tex: bool) -> None:
    mpl.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.latex.preamble": r"\usepackage{amsmath}" if use_tex else "",
            "font.size": 13,
            "axes.labelsize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.linewidth": 1.2,
            "lines.linewidth": 1.4,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def load_cases(root: Path) -> list[dict]:
    loaded = []

    for U, n_target, run_name, case_name in CASES:
        folder = root / run_name / case_name
        npz_path = folder / "dmft_result.npz"
        metadata_path = folder / "metadata.json"

        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing file: {npz_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing file: {metadata_path}")

        with np.load(npz_path, allow_pickle=False) as data:
            iw_over_t = np.asarray(data["iw_over_t"]).ravel()
            G_iw = np.asarray(data["G_iw"]).ravel()
            Sigma_iw = np.asarray(data["Sigma_iw"]).ravel()

        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        omega = np.imag(iw_over_t)

        order = np.argsort(omega)
        omega = omega[order]
        G_iw = G_iw[order]
        Sigma_iw = Sigma_iw[order]

        pos = omega > 0
        omega_pos = omega[pos]
        G_pos = G_iw[pos]
        Sigma_pos = Sigma_iw[pos]

        if len(omega_pos) != 256:
            raise ValueError(
                f"Expected 256 positive Matsubara frequencies in {npz_path}, "
                f"found {len(omega_pos)}"
            )

        # optional consistency checks
        if int(meta["U_over_t"]) != U:
            raise ValueError(f"U/t mismatch in {folder}")
        if abs(float(meta["target_filling"]) - n_target) > 1e-8:
            raise ValueError(f"target filling mismatch in {folder}")

        loaded.append(
            {
                "U": U,
                "n_target": n_target,
                "n_final": float(meta["final_filling_mean"]),
                "mu_over_t": float(meta["mu_over_t"]),
                "omega": omega_pos,
                "G_iw": G_pos,
                "Sigma_iw": Sigma_pos,
            }
        )

    return loaded


def style_ax(ax: plt.Axes) -> None:
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LW)
        spine.set_color("black")

    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def add_panel_label(ax: plt.Axes, idx: int, U: int, n_final: float) -> None:
    label = "abcd"[idx]
    ax.set_title(
        rf"({label})  $U/t={U},\ \langle n\rangle={n_final:.4f}$",
        loc="left",
        fontsize=13,
        pad=6,
    )


def save_figure(fig: plt.Figure, out_dir: Path, basename: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / f"{basename}.pdf"
    png_path = out_dir / f"{basename}.png"

    fig.savefig(pdf_path, dpi=dpi, facecolor="white")
    fig.savefig(png_path, dpi=dpi, facecolor="white")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def plot_sigma(cases: list[dict], out_dir: Path, dpi: int) -> None:

    fig, axes = plt.subplots(
        2, 2,
        figsize=(9.6, 7.5),
        sharex=True,
        sharey=True
    )

    axes = axes.ravel()

    # Include BOTH real and imaginary Sigma
    # when determining the Y-axis limits.
    all_re = np.concatenate([
        case["Sigma_iw"].real for case in cases
    ])

    all_im = np.concatenate([
        case["Sigma_iw"].imag for case in cases
    ])

    all_y = np.concatenate([all_re, all_im])

    dy = all_y.max() - all_y.min()
    pad = 0.07 * dy

    ylo = all_y.min() - pad
    yhi = all_y.max() + pad

    for i, (ax, case) in enumerate(zip(axes, cases)):

        omega = case["omega"]
        sigma_re = case["Sigma_iw"].real
        sigma_im = case["Sigma_iw"].imag

        # Real part of self-energy
        ax.plot(
            omega,
            sigma_re,
            "-",
            color="#1674B7",
            lw=1.7,
            label=r"$\mathrm{Re}\,\Sigma(i\omega_n)$"
        )

        # Imaginary part of self-energy
        ax.plot(
            omega,
            sigma_im,
            "--",
            color="#C7252E",
            lw=1.7,
            label=r"$\mathrm{Im}\,\Sigma(i\omega_n)$"
        )

        ax.axhline(
            0,
            color="0.6",
            lw=0.8,
            ls=":"
        )

        ax.set_xlim(0, 65)
        ax.set_ylim(ylo, yhi)

        style_ax(ax)
        add_panel_label(
            ax, i, case["U"], case["n_final"]
        )

        # Show Y-axis values on every panel.
        ax.tick_params(
            labelleft=True,
            labelbottom=True
        )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=2,
        frameon=False,
        columnspacing=2.0,
        handlelength=2.5
    )

    fig.suptitle(
        r"Self-energy $\Sigma(i\omega_n)$",
        fontsize=17,
        y=0.985
    )

    fig.text(
        0.5,
        0.945,
        r"$\beta t=25,\quad "
        r"t=0.25\,\mathrm{eV},\quad "
        r"T\approx116.05\,\mathrm{K}$",
        ha="center",
        va="center",
        fontsize=12
    )

    fig.supxlabel(
        r"$\omega_n/t$",
        y=0.035,
        fontsize=16
    )

    fig.supylabel(
        r"$\mathrm{Re}\,\Sigma/t,\quad"
        r"\mathrm{Im}\,\Sigma/t$",
        x=0.018,
        fontsize=15
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        bottom=0.115,
        top=0.82,
        wspace=0.23,
        hspace=0.26
    )

    save_figure(
        fig,
        out_dir,
        "01_Sigma_Re_Im_positive_256",
        dpi
    )

    plt.close(fig)

def plot_G(cases: list[dict], out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    axes = axes.ravel()

    all_re = np.concatenate([case["G_iw"].real for case in cases])
    all_im = np.concatenate([case["G_iw"].imag for case in cases])
    all_y = np.concatenate([all_re, all_im])

    dy = all_y.max() - all_y.min()
    pad = 0.08 * dy if dy > 0 else 0.2
    ylo = all_y.min() - pad
    yhi = all_y.max() + pad

    for i, (ax, case) in enumerate(zip(axes, cases)):
        omega = case["omega"]
        Gre = case["G_iw"].real
        Gim = case["G_iw"].imag

        ax.plot(
            omega,
            Gre,
            "-o",
            color=C_GRE,
            lw=1.4,
            ms=1.8,
            label=r"$\mathrm{Re}[\,tG(i\omega_n)\,]$",
        )
        ax.plot(
            omega,
            Gim,
            "--o",
            color=C_GIM,
            lw=1.4,
            ms=1.8,
            label=r"$\mathrm{Im}[\,tG(i\omega_n)\,]$",
        )

        ax.axhline(0.0, color="0.55", lw=0.8, ls=":")
        ax.set_ylim(ylo, yhi)
        style_ax(ax)
        add_panel_label(ax, i, case["U"], case["n_final"])

    for ax in axes[:2]:
        ax.tick_params(labelbottom=False)
    for ax in (axes[1], axes[3]):
        ax.tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=2,
        frameon=False,
        columnspacing=2.0,
        handlelength=2.4,
    )

    fig.suptitle(
        r"$G(i\omega_n)$ using the 256 positive Matsubara frequencies",
        fontsize=17,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        r"$\beta t=25,\quad t=0.25\,\mathrm{eV},\quad T\approx 116.05\,\mathrm{K}$",
        ha="center",
        va="center",
        fontsize=12,
    )

    fig.supxlabel(r"$\omega_n/t$", y=0.035, fontsize=16)
    fig.supylabel(r"$t\,G(i\omega_n)$", x=0.03, fontsize=16)

    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        bottom=0.12,
        top=0.86,
        wspace=0.10,
        hspace=0.18,
    )

    save_figure(fig, out_dir, "02_Giw_positive_256", dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make 2x2 figures for Im Sigma and G using 256 positive Matsubara frequencies."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("cthyb_test_results"),
        help="Directory containing the adaptive_v2 result folders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cthyb_test_results/steve_positive_256"),
        help="Directory to save figures",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output resolution for PNG",
    )
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Use external LaTeX for text rendering",
    )

    args = parser.parse_args()

    configure_matplotlib(args.usetex)
    cases = load_cases(args.root)

    plot_sigma(cases, args.out_dir, args.dpi)
    plot_G(cases, args.out_dir, args.dpi)

    print("Done. Both 2x2 figures were created successfully.")


if __name__ == "__main__":
    main()
