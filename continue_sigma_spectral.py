
#!/usr/bin/env python3
"""Causality-constrained spectral continuation of CT-HYB self-energy.

Sigma(z) = Sigma_inf + Sigma_1 * sum_j p_j / (z - e_j),
where p_j >= 0 and sum_j p_j ~= 1.

Uses NNLS with smoothness regularization and a moment-sum constraint.

Outputs are model-dependent, broadened estimates of real-frequency
self-energy, NOT exact analytical continuation.

Requirements:
    numpy
    scipy
    matplotlib
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ================================================================
# 1. FOUR ADAPTIVE V2 CASES
# ================================================================

CASES = [
    (
        "U8_n0p7_beta25_adaptive_v2",
        "U8p000_n0p700_beta25p00",
    ),
    (
        "U8_n0p9_beta25_adaptive_v2",
        "U8p000_n0p900_beta25p00",
    ),
    (
        "U10_n0p7_beta25_adaptive_v2",
        "U10p000_n0p700_beta25p00",
    ),
    (
        "U10_n0p9_beta25_adaptive_v2",
        "U10p000_n0p900_beta25p00",
    ),
]


# ================================================================
# 2. LOAD MATSUBARA SELF-ENERGY
# ================================================================

def load_case(folder: Path):

    metadata_path = folder / "metadata.json"
    npz_path = folder / "dmft_result.npz"

    with metadata_path.open("r") as f:
        meta = json.load(f)

    with np.load(npz_path, allow_pickle=False) as f:

        iw = np.asarray(
            f["iw_over_t"]
        ).reshape(-1)

        sigma_iw = np.asarray(
            f["Sigma_iw"]
        ).reshape(-1)

    # iw_over_t contains i*omega_n/t.
    omega_n = np.imag(iw)

    # Keep positive fermionic Matsubara frequencies.
    positive = omega_n > 0

    omega_n = omega_n[positive]
    sigma_iw = sigma_iw[positive]

    # Sort from lowest to highest frequency.
    order = np.argsort(omega_n)

    omega_n = omega_n[order]
    sigma_iw = sigma_iw[order]

    return omega_n, sigma_iw, meta


# ================================================================
# 3. FIT CAUSAL SPECTRAL REPRESENTATION
# ================================================================

def fit_one(
    omega_n,
    sigma_iw,
    sigma_infinity,
    sigma_1,
    energy,
    nfit,
    regularization,
    weight_scale,
    moment_weight,
):

    omega_n = omega_n[:nfit]
    sigma_iw = sigma_iw[:nfit]

    # Spectral representation kernel:
    #
    # Sigma(iwn) - Sigma_inf
    #     = Sigma_1 * sum_j p_j / (iwn - energy_j)

    kernel = 1.0 / (
        1j * omega_n[:, None]
        - energy[None, :]
    )

    # Remove known Hartree term and divide by first moment.
    target = (
        sigma_iw - sigma_infinity
    ) / sigma_1

    # Give more importance to low Matsubara frequencies.
    weights = 1.0 / np.sqrt(
        1.0 + (omega_n / weight_scale) ** 2
    )

    number_of_basis_points = len(energy)

    # Second-derivative smoothness regularization.
    D2 = np.diff(
        np.eye(number_of_basis_points),
        n=2,
        axis=0,
    )

    # Stack real and imaginary constraints.
    A = np.vstack(
        (
            weights[:, None] * kernel.real,
            weights[:, None] * kernel.imag,
            np.sqrt(regularization) * D2,
            moment_weight
            * np.ones((1, number_of_basis_points)),
        )
    )

    b = np.concatenate(
        (
            weights * target.real,
            weights * target.imag,
            np.zeros(number_of_basis_points - 2),
            [moment_weight],
        )
    )

    # Non-negative least squares.
    spectral_weights, residual_norm = nnls(
        A,
        b,
        maxiter=10 * number_of_basis_points,
    )

    # Reconstruct Matsubara self-energy.
    sigma_backfit = (
        sigma_infinity
        + sigma_1 * (kernel @ spectral_weights)
    )

    # Complex Matsubara fitting error.
    rmse = np.sqrt(
        np.mean(
            np.abs(
                sigma_backfit - sigma_iw
            ) ** 2
        )
    )

    return (
        spectral_weights,
        sigma_backfit,
        rmse,
    )


# ================================================================
# 4. CONTINUE ONE DMFT CASE
# ================================================================

def process_case(folder, output_dir, args):

    omega_n, sigma_iw, meta = load_case(folder)

    sigma_infinity = float(
        meta["sigma_infinity_mean_over_t"]
    )

    sigma_1 = float(
        meta["sigma_first_moment_mean_over_t2"]
    )

    if sigma_1 <= 0:
        raise ValueError(
            "The first self-energy moment must be positive."
        )

    nfit = min(args.nfit, len(omega_n))

    # Spectral representation energies in units of t.
    energy = np.linspace(
        -args.energy_max,
        args.energy_max,
        args.nbasis,
    )

    # Real-frequency output mesh in units of t.
    omega = np.linspace(
        -args.real_max,
        args.real_max,
        args.nomega,
    )

    eta = args.eta

    # Test three regularization strengths:
    # lambda/2, lambda, 2*lambda.
    variants = []

    for factor in (0.5, 1.0, 2.0):

        (
            spectral_weights,
            sigma_backfit,
            rmse,
        ) = fit_one(
            omega_n,
            sigma_iw,
            sigma_infinity,
            sigma_1,
            energy,
            nfit,
            args.regularization * factor,
            args.weight_scale,
            args.moment_weight,
        )

        # Evaluate at z = omega + i*eta.
        #
        # This is an eta-broadened retarded estimate.
        real_axis_kernel = 1.0 / (
            omega[:, None]
            + 1j * eta
            - energy[None, :]
        )

        sigma_retarded = (
            sigma_infinity
            + sigma_1
            * (real_axis_kernel @ spectral_weights)
        )

        variants.append(
            (
                factor,
                spectral_weights,
                sigma_backfit,
                sigma_retarded,
                rmse,
            )
        )

    # Baseline regularization result.
    (
        factor,
        spectral_weights,
        sigma_backfit,
        sigma_retarded,
        rmse,
    ) = variants[1]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # 5. SAVE ACTUAL NUMERICAL VALUES
    # ============================================================

    csv_path = (
        output_dir / "Sigma_retarded_estimate.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "omega_over_t",
                "omega_eV",
                "Re_Sigma_over_t",
                "Im_Sigma_over_t",
                "Re_Sigma_eV",
                "Im_Sigma_eV",
                "spectral_B_over_t",
                "ImSigma_lambda_half_over_t",
                "ImSigma_lambda_double_over_t",
            ]
        )

        for j, w in enumerate(omega):

            writer.writerow(
                [
                    w,
                    w * meta["t_eV"],
                    sigma_retarded[j].real,
                    sigma_retarded[j].imag,
                    sigma_retarded[j].real
                    * meta["t_eV"],
                    sigma_retarded[j].imag
                    * meta["t_eV"],
                    -sigma_retarded[j].imag / np.pi,
                    variants[0][3][j].imag,
                    variants[2][3][j].imag,
                ]
            )

    # Save complex arrays and spectral parameters.
    np.savez_compressed(
        output_dir / "Sigma_retarded_estimate.npz",

        omega_over_t=omega,

        Sigma_retarded_over_t=sigma_retarded,

        spectral_support_over_t=energy,

        spectral_weight_mass=spectral_weights,

        sigma_matsubara_input_over_t=sigma_iw,

        sigma_matsubara_backfit_over_t=sigma_backfit,

        omega_matsubara_over_t=omega_n,

        eta_over_t=eta,

        sigma_infinity_over_t=sigma_infinity,

        sigma_first_moment_over_t2=sigma_1,

        sigma_retarded_lambda_half_over_t=variants[0][3],

        sigma_retarded_lambda_double_over_t=variants[2][3],
    )

    # ============================================================
    # 6. DIAGNOSTIC INFORMATION
    # ============================================================

    quality = {

        "source": str(folder),

        "method": (
            "Nonnegative spectral density NNLS "
            "with second-difference "
            "Tikhonov regularization"
        ),

        "warning": (
            "Ill-posed inverse problem. "
            "These are model-dependent, "
            "eta-broadened estimates, "
            "NOT exact Sigma(omega)."
        ),

        "U_over_t": meta["U_over_t"],

        "target_filling": meta["target_filling"],

        "achieved_filling": meta["final_filling_mean"],

        "mu_over_t": meta["mu_over_t"],

        "beta_t": meta["beta_t"],

        "n_matsubara_used": nfit,

        "eta_over_t": eta,

        "spectral_support_over_t": [
            -args.energy_max,
            args.energy_max,
        ],

        "moment_input_over_t2": sigma_1,

        "spectral_mass_fraction": float(
            np.sum(spectral_weights)
        ),

        "backfit_complex_rmse_over_t": float(rmse),

        "regularization": args.regularization,

        "regularization_sensitivity_imag_max_abs_over_t": float(
            max(
                np.max(
                    np.abs(
                        sigma_retarded.imag
                        - variants[0][3].imag
                    )
                ),
                np.max(
                    np.abs(
                        sigma_retarded.imag
                        - variants[2][3].imag
                    )
                ),
            )
        ),
    }

    diagnostics_path = (
        output_dir / "continuation_diagnostics.json"
    )

    diagnostics_path.write_text(
        json.dumps(
            quality,
            indent=2,
        ) + "\n"
    )

    # ============================================================
    # 7. PLOT FOUR DIAGNOSTIC PANELS
    # ============================================================

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7.8),
    )

    # Panel A: real and imaginary retarded Sigma.
    axes[0, 0].plot(
        omega,
        sigma_retarded.real,
        label=r"Re $\Sigma^R$",
        color="#1769aa",
    )

    axes[0, 0].plot(
        omega,
        sigma_retarded.imag,
        label=r"Im $\Sigma^R$",
        color="#bf3c3c",
    )

    axes[0, 0].axhline(
        0,
        color="0.65",
        lw=0.7,
    )

    axes[0, 0].set_ylabel(
        r"$\Sigma^R(\omega)/t$"
    )

    axes[0, 0].set_xlabel(
        r"$\omega/t$"
    )

    axes[0, 0].legend(
        frameon=False
    )

    # Panel B: eta-broadened spectral density.
    axes[0, 1].plot(
        omega,
        -sigma_retarded.imag / np.pi,
        color="#bf3c3c",
    )

    axes[0, 1].set_ylabel(
        r"$-\mathrm{Im}\Sigma^R/(\pi t)$"
    )

    axes[0, 1].set_xlabel(
        r"$\omega/t$"
    )

    # Panel C: comparison with input Matsubara data.
    axes[1, 0].plot(
        omega_n[:nfit],
        sigma_iw[:nfit].real,
        "o",
        ms=2.5,
        label="Input Re",
        color="#1769aa",
    )

    axes[1, 0].plot(
        omega_n[:nfit],
        sigma_backfit.real,
        "-",
        lw=1.5,
        label="Backfit Re",
        color="#1769aa",
    )

    axes[1, 0].plot(
        omega_n[:nfit],
        sigma_iw[:nfit].imag,
        "o",
        ms=2.5,
        label="Input Im",
        color="#bf3c3c",
    )

    axes[1, 0].plot(
        omega_n[:nfit],
        sigma_backfit.imag,
        "-",
        lw=1.5,
        label="Backfit Im",
        color="#bf3c3c",
    )

    axes[1, 0].set_xlabel(
        r"$\omega_n/t$"
    )

    axes[1, 0].set_ylabel(
        r"$\Sigma(i\omega_n)/t$"
    )

    axes[1, 0].legend(
        frameon=False,
        ncol=2,
        fontsize=9,
    )

    # Panel D: regularization sensitivity.
    axes[1, 1].plot(
        omega,
        sigma_retarded.imag,
        label="Baseline",
        color="#bf3c3c",
    )

    axes[1, 1].plot(
        omega,
        variants[0][3].imag,
        "--",
        label=r"$\lambda/2$",
        color="#1769aa",
    )

    axes[1, 1].plot(
        omega,
        variants[2][3].imag,
        ":",
        label=r"$2\lambda$",
        color="#252525",
    )

    axes[1, 1].set_ylabel(
        r"$\mathrm{Im}\Sigma^R(\omega)/t$"
    )

    axes[1, 1].set_xlabel(
        r"$\omega/t$"
    )

    axes[1, 1].legend(
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        rf"$U/t={meta['U_over_t']:g}$, "
        rf"$\langle n\rangle={meta['final_filling_mean']:.5f}$, "
        rf"$\beta t={meta['beta_t']:g}$; "
        "spectral continuation (estimated)",
        fontsize=15,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    fig.savefig(
        output_dir / "Sigma_continuation_diagnostics.pdf"
    )

    fig.savefig(
        output_dir / "Sigma_continuation_diagnostics.png",
        dpi=args.dpi,
    )

    plt.close(fig)

    print(
        f"{folder.name}: "
        f"U={meta['U_over_t']} "
        f"n={meta['final_filling_mean']:.6f} "
        f"RMSE={rmse:.4g} "
        f"mass={sum(spectral_weights):.6f} "
        f"-> {output_dir}"
    )

    return quality, omega, sigma_retarded


# ================================================================
# 8. MAIN FUNCTION
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("cthyb_test_results"),
    )

    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Process one case instead of all four.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--nfit",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--energy-max",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--real-max",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--nbasis",
        type=int,
        default=161,
    )

    parser.add_argument(
        "--nomega",
        type=int,
        default=601,
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=0.003,
    )

    parser.add_argument(
        "--weight-scale",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--moment-weight",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    if (
        args.nfit < 6
        or args.nbasis < 20
        or args.eta <= 0
        or args.regularization < 0
    ):
        parser.error("Invalid fitting parameters.")

    root = args.root.expanduser().resolve()

    output = (
        args.out_dir
        or root / "sigma_real_frequency_continuation"
    ).resolve()

    if args.case_dir:

        cases = [
            (
                "single",
                args.case_dir.expanduser().resolve(),
            )
        ]

    else:

        cases = [
            (
                run,
                root / run / sub,
            )
            for run, sub in CASES
        ]

    all_spectra = []

    for run, folder in cases:

        quality, omega, sigma_retarded = process_case(
            folder,
            output / run,
            args,
        )

        all_spectra.append(
            (
                quality,
                omega,
                sigma_retarded,
            )
        )

    # ============================================================
    # 9. FOUR-CASE COMPARISON FIGURE
    # ============================================================

    if len(all_spectra) > 1:

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 7.2),
            sharex=True,
            sharey=True,
        )

        for ax, (quality, omega, sigma) in zip(
            axes.flat,
            all_spectra,
        ):

            ax.plot(
                omega,
                sigma.real,
                color="#1769aa",
                label="Re",
            )

            ax.plot(
                omega,
                sigma.imag,
                color="#bf3c3c",
                label="Im",
            )

            ax.set_title(
                rf"$U/t={quality['U_over_t']:g},\ "
                rf"\langle n\rangle="
                rf"{quality['achieved_filling']:.4f}$"
            )

            ax.axhline(
                0,
                lw=0.7,
                color="0.65",
            )

            ax.set_xlabel(
                r"$\omega/t$"
            )

            ax.set_ylabel(
                r"$\Sigma^R(\omega)/t$"
            )

            ax.legend(
                frameon=False
            )

        fig.suptitle(
            "Real-frequency self-energy: "
            "regularized estimates (not unique)",
            fontsize=14,
        )

        fig.tight_layout(
            rect=(0, 0, 1, 0.95)
        )

        fig.savefig(
            output
            / "ALL_FOUR_Sigma_real_frequency_estimates.pdf"
        )

        fig.savefig(
            output
            / "ALL_FOUR_Sigma_real_frequency_estimates.png",
            dpi=args.dpi,
        )

        plt.close(fig)

    print(
        "Finished. Review Matsubara backfit "
        "and regularization sensitivity "
        "before interpreting peaks."
    )


if __name__ == "__main__":
    main()

