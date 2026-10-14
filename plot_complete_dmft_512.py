#!/usr/bin/env python3
"""Complete four-case Bethe DMFT plots: complex Sigma and G over all 512 iw.

2x2 layout:
   (a) U/t=8,  n=0.7  | (b) U/t=8,  n=0.9
   (c) U/t=10, n=0.7  | (d) U/t=10, n=0.9

Unlike the previous plot, shows the FULL complex Sigma (both Re and Im),
shows Y tick labels on ALL panels, and saves an additional low-frequency zoom.
Reads full signed 512-point iw mesh from dmft_result.npz; does not interpolate
or omit any sampled frequency.
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
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

CASES = [
    (8, 0.7, "U8_n0p7_beta25_adaptive_v2", "U8p000_n0p700_beta25p00"),
    (8, 0.9, "U8_n0p9_beta25_adaptive_v2", "U8p000_n0p900_beta25p00"),
    (10, 0.7, "U10_n0p7_beta25_adaptive_v2", "U10p000_n0p700_beta25p00"),
    (10, 0.9, "U10_n0p9_beta25_adaptive_v2", "U10p000_n0p900_beta25p00"),
]

C_RE = "#1674B7"
C_IM = "#C72E35"


def configure(use_tex: bool) -> None:
    mpl.rcParams.update({
        "text.usetex": use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}" if use_tex else "",
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 13,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 1.2,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "lines.linewidth": 1.65,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def load_case(root: Path, U: int, n: float, folder: str, subfolder: str) -> dict:
    path = root / folder / subfolder
    fn = path / "dmft_result.npz"
    md = path / "metadata.json"
    if not fn.is_file() or not md.is_file():
        raise FileNotFoundError(f"Missing dmft_result.npz or metadata.json in {path}")
    with np.load(fn, allow_pickle=False) as z:
        omega = np.asarray(z["iw_over_t"]).imag.ravel()
        G = np.asarray(z["G_iw"]).ravel()
        Sigma = np.asarray(z["Sigma_iw"]).ravel()
        tau = np.asarray(z["tau_times_t"]).ravel()
        Gtau = np.asarray(z["G_tau"]).real.ravel()
    with open(md, encoding="utf-8") as f:
        meta = json.load(f)
    if len(omega) != 512 or len(G) != 512 or len(Sigma) != 512:
        raise ValueError(f"Expected 512 Matsubara frequencies: {fn}")
    if (int(meta["U_over_t"]) != U or
            abs(float(meta["target_filling"]) - n) > 1.e-9):
        raise ValueError(f"Case metadata does not match filename: {md}")
    if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(Sigma)):
        raise ValueError(f"Nonfinite values in {fn}")
    idx = np.argsort(omega)
    print(f"U/t={U:2d}, target={n:.1f}: "+
          f"{len(omega)} iw points, range [{omega.min():.3f},{omega.max():.3f}], "+
          f"Re Sigma=[{Sigma.real.min():.3f},{Sigma.real.max():.3f}], "+
          f"Im Sigma=[{Sigma.imag.min():.3f},{Sigma.imag.max():.3f}]")
    return dict(U=U, n=n, w=omega[idx], G=G[idx], Sigma=Sigma[idx],
                tau=tau / float(meta["beta_t"]), Gtau=Gtau,
                mu=float(meta["mu_over_t"]),
                nfinal=float(meta["final_filling_mean"]), meta=meta)


def padded_limits(arrays, *, include_zero=True, padding=.075):
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        raise ValueError("No finite plotting values")
    low, high = float(vals.min()), float(vals.max())
    if include_zero:
        low, high = min(low, 0.), max(high, 0.)
    spread = high - low or 1.
    return low - spread * padding, high + spread * padding


def make_figure(cases: list[dict], *, what: str, zoom: float | None,
                output: Path, dpi: int) -> None:
    if what not in {"Sigma", "G"}:
        raise ValueError(what)
    key = "Sigma" if what == "Sigma" else "G"
    if zoom is None:
        cutoff = max(float(np.max(np.abs(c["w"]))) for c in cases)
    else:
        cutoff = float(zoom)

    # Consistent physical y scale among all panels, encompassing BOTH components.
    # The small zoom uses its own scale so near-zero physics is actually visible.
    arrays = []
    for case in cases:
        mask = np.abs(case["w"]) <= cutoff + 1.e-12
        val = case[key][mask]
        arrays.extend([val.real, val.imag])
    ylo, yhi = padded_limits(arrays)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0),
                             sharex=True, sharey=True)
    letters = "abcd"
    for i, (ax, case) in enumerate(zip(axes.flat, cases)):
        w = case["w"]
        z = case[key]
        window = np.abs(w) <= cutoff + 1.e-12
        for side in [w < 0, w > 0]:
            m = side & window
            ax.plot(w[m], z.real[m], color=C_RE, ls="-", lw=1.7)
            ax.plot(w[m], z.imag[m], color=C_IM, ls="--", lw=1.65)

        ax.axhline(0., color="0.6", ls=":", lw=.75, zorder=0)
        ax.axvline(0., color="0.6", ls=":", lw=.75, zorder=0)
        ax.set_xlim(-cutoff*1.02, cutoff*1.02)
        ax.set_ylim(ylo, yhi)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="both", direction="in", top=True, right=True)
        # IMPORTANT: Do NOT hide right-column tick labels (old code did this).
        ax.tick_params(axis="y", which="major", labelleft=True)
        ax.tick_params(axis="x", which="major", labelbottom=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.set_title(
            rf"({letters[i]})  $U/t={case['U']},\;\langle n\rangle={case['nfinal']:.4f}$",
            loc="left", pad=7, fontsize=13)

    symbol = r"\Sigma" if what == "Sigma" else "G"
    unit = r"/t" if what == "Sigma" else r"\,t"
    ylabel = (r"$\mathrm{Re},\,\mathrm{Im}\,[\Sigma(i\omega_n)/t]$" if what == "Sigma"
              else r"$\mathrm{Re},\,\mathrm{Im}\,[tG(i\omega_n)]$")
    subtitle = ("Full 512-point signed Matsubara mesh" if zoom is None
                else rf"Low-frequency detail: $|\omega_n|/t\leq{cutoff:g}$")
    fig.suptitle(rf"${symbol}(i\omega_n)$ — {subtitle}", fontsize=17, y=.99)
    fig.text(.5, .945,
             r"$\beta t=25,\quad t=0.25\,\mathrm{eV},\quad T\simeq116.05\,\mathrm{K}$",
             ha="center", fontsize=12)
    fig.supxlabel(r"$\omega_n/t$", fontsize=16, y=.04)
    fig.supylabel(ylabel, fontsize=16, x=.02)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0], color=C_RE, lw=1.9, label=rf"$\mathrm{{Re}}\,{symbol}$"),
                        Line2D([0],[0], color=C_IM, lw=1.9, ls="--", label=rf"$\mathrm{{Im}}\,{symbol}$")],
               loc="upper center", bbox_to_anchor=(.5,.92), ncol=2, frameon=False)
    fig.subplots_adjust(left=.13, right=.98, bottom=.12, top=.84,
                        wspace=.22, hspace=.31)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        dest = output.with_suffix("." + ext)
        fig.savefig(dest, dpi=dpi, facecolor="white")
        print("Saved:", dest)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).parent / "cthyb_test_results")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--zoom", type=float, default=12.,
                   help="Frequency cutoff for additional zoom figures; 0 disables")
    p.add_argument("--usetex", action="store_true", help="Requires working TeX installation")
    args = p.parse_args()
    configure(args.usetex)
    cases = [load_case(args.root, *case) for case in CASES]
    out = args.out_dir or args.root / "steve_complete_complex"
    for quantity in ("Sigma", "G"):
        make_figure(cases, what=quantity, zoom=None,
                    output=out / f"{quantity}_complete_512", dpi=args.dpi)
        if args.zoom > 0:
            make_figure(cases, what=quantity, zoom=args.zoom,
                        output=out / f"{quantity}_zoom_{args.zoom:g}", dpi=args.dpi)
    print("Done. All 512 points are present in full plots; low-frequency detail saved separately.")


if __name__ == "__main__":
    main()

