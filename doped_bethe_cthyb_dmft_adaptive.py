#!/usr/bin/env python3
"""Paramagnetic doped single-band Hubbard DMFT using only TRIQS/CTHYB.
The lattice is the infinite-coordination Bethe lattice with hopping t=1,
semicircular DOS of half-bandwidth 2t, and self-consistency
    Delta(iw) = t^2 G_imp(iw).
The impurity Hamiltonian uses the physical (unshifted) convention
    H_int = U n_up n_down,
    G0^{-1}(iw) = iw + mu - Delta(iw),
so the saved Sigma contains the Hartree constant.  Target filling is obtained
by a two-stage (coarse then refined) safeguarded chemical-potential search.
Coarse data are never used to certify the final target density.  No neural
network code or data are used.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
from triqs.atom_diag import trace_rho_op
from triqs.gf import Fourier, Gf, MeshImFreq, MeshImTime
from triqs.gf.tools import make_zero_tail
from triqs.operators import n
from triqs_cthyb import Solver
try:
    import triqs.utility.mpi as mpi
except ImportError:  # pragma: no cover - only for unusual TRIQS builds
    class _SerialMPI:
        @staticmethod
        def is_master_node():
            return True
        rank = 0
    mpi = _SerialMPI()
KB_EV_PER_K = 8.617333262145e-5
T_HOP = 1.0  # all solver energies are expressed in units of t
def master_print(*args, **kwargs):
    if mpi.is_master_node():
        print(*args, **kwargs, flush=True)
def clone_gf(source: Gf) -> Gf:
    result = Gf(mesh=source.mesh, target_shape=source.target_shape)
    result.data[:] = source.data[:]
    return result
def scalar_gf(mesh, values) -> Gf:
    result = Gf(mesh=mesh, target_shape=[1, 1])
    result.data[:, 0, 0] = np.asarray(values, dtype=np.complex128)
    return result
def mesh_iw_values(mesh) -> np.ndarray:
    return np.asarray([complex(w) for w in mesh], dtype=np.complex128)
def mesh_tau_values(mesh) -> np.ndarray:
    return np.asarray([float(x) for x in mesh], dtype=float)
def positive_indices(iw_values: np.ndarray) -> np.ndarray:
    return np.flatnonzero(iw_values.imag > 0.0)
def noninteracting_bethe_green(z: np.ndarray, t_hop: float = 1.0) -> np.ndarray:
    """Physical root of t^2 G^2-zG+1=0, selected by G~1/z."""
    disc = np.sqrt(z * z - 4.0 * t_hop * t_hop)
    root_a = (z - disc) / (2.0 * t_hop * t_hop)
    root_b = (z + disc) / (2.0 * t_hop * t_hop)
    return np.where(np.abs(root_a) <= np.abs(root_b), root_a, root_b)
def initial_delta(iw_values: np.ndarray, mu: float) -> np.ndarray:
    g_nonint = noninteracting_bethe_green(iw_values + float(mu), T_HOP)
    return T_HOP**2 * g_nonint
def safe_real(value) -> float:
    return float(np.real_if_close(value).real)
def gf_to_tau(g_iw: Gf, tau_mesh: MeshImTime, known_moments=None) -> Gf:
    result = Gf(mesh=tau_mesh, target_shape=[1, 1])
    if known_moments is None:
        result << Fourier(g_iw)
    else:
        result << Fourier(g_iw, known_moments=known_moments)
    return result
@dataclass
class ImpurityResult:
    G_iw: Gf
    G_iw_raw: Gf
    Sigma_iw: Gf
    Sigma_iw_raw: Gf
    G_tau_measured: Gf
    filling: float
    double_occupancy: float
    sigma_infinity: float
    sigma_first_moment: float
    average_sign: float
    average_order: float
    auto_corr_time: float
    wall_seconds: float
    seed: int
    solve_status: int
@dataclass
class FixedMuResult:
    mu: float
    filling: float
    converged: bool
    iterations: int
    delta_iw: Gf
    G0_iw: Gf
    impurity: ImpurityResult
    residual_history: list[float]
    filling_history: list[float]
    sign_history: list[float]
    auto_corr_history: list[float]
    wall_seconds: float
    stage: str = "refined"
def solve_impurity(
    *, U: float, beta: float, mu: float, delta_iw: Gf,
    n_iw: int, n_tau: int, n_l: int,
    n_cycles: int, n_warmup_cycles: int, length_cycle: int,
    seed: int, perform_tail_fit: bool,
    fit_min_w: float, fit_max_w: float, fit_max_moment: int,
    verbosity: int, max_time: int,
) -> ImpurityResult:
    iw_values = mesh_iw_values(delta_iw.mesh)
    g0_values = 1.0 / (iw_values + float(mu) - delta_iw.data[:, 0, 0])
    solver = Solver(
        beta=float(beta),
        gf_struct=[("up", 1), ("down", 1)],
        n_iw=int(n_iw),
        n_tau=int(n_tau),
        n_l=int(n_l),
    )
    for _, g0_spin in solver.G0_iw:
        g0_spin.data[:, 0, 0] = g0_values
    n_up = n("up", 0)
    n_down = n("down", 0)
    h_int = float(U) * n_up * n_down
    started = time.time()
    rank = int(getattr(mpi, "rank", 0))
    rank_seed = int((int(seed) + 104729 * rank) % (2**31 - 2) + 1)
    solve_parameters = dict(
        h_int=h_int,
        n_cycles=int(n_cycles),
        n_warmup_cycles=int(n_warmup_cycles),
        length_cycle=int(length_cycle),
        random_seed=rank_seed,
        max_time=int(max_time),
        measure_G_tau=True,
        measure_G_l=True,
        measure_density_matrix=True,
        use_norm_as_weight=True,
        perform_tail_fit=bool(perform_tail_fit),
        verbosity=int(verbosity),
    )
    # These are Python-wrapper post-processing options.  When tail fitting is
    # disabled they must not be forwarded to the C++ solve-parameter object.
    if perform_tail_fit:
        solve_parameters.update(
            fit_max_moment=int(fit_max_moment),
            fit_min_w=float(fit_min_w),
            fit_max_w=float(fit_max_w),
        )
    returned_status = solver.solve(**solve_parameters)
    # TRIQS/CTHYB releases differ: some return the status from solve(), while
    # others expose it only through the SolverCore property.
    solve_status = int(
        solver.solve_status if returned_status is None else returned_status
    )
    wall = time.time() - started
    if solve_status != 0:
        raise RuntimeError(
            f"CT-HYB terminated with solve_status={solve_status}. "
            "A max-time stop or interrupted Monte Carlo run is not valid data."
        )
    G_iw = 0.5 * (solver.G_iw["up"] + solver.G_iw["down"])
    Sigma_iw = 0.5 * (solver.Sigma_iw["up"] + solver.Sigma_iw["down"])
    if getattr(solver, "G_iw_raw", None) is None:
        G_iw_raw = G_iw
    else:
        G_iw_raw = 0.5 * (solver.G_iw_raw["up"] + solver.G_iw_raw["down"])
    if getattr(solver, "Sigma_iw_raw", None) is None:
        Sigma_iw_raw = Sigma_iw
    else:
        Sigma_iw_raw = 0.5 * (
            solver.Sigma_iw_raw["up"] + solver.Sigma_iw_raw["down"]
        )
    G_tau = 0.5 * (solver.G_tau["up"] + solver.G_tau["down"])
    rho = solver.density_matrix
    atom = solver.h_loc_diagonalization
    n_up_value = safe_real(trace_rho_op(rho, n_up, atom))
    n_down_value = safe_real(trace_rho_op(rho, n_down, atom))
    double_occ = safe_real(trace_rho_op(rho, n_up * n_down, atom))
    sigma_inf = float(U) * 0.5 * (n_up_value + n_down_value)
    sigma_first_moment = float(U) ** 2 * 0.5 * (n_up_value + n_down_value) * (
        1.0 - 0.5 * (n_up_value + n_down_value)
    )
    try:
        sigma_inf = 0.5 * (
            safe_real(solver.Sigma_moments["up"][0, 0, 0])
            + safe_real(solver.Sigma_moments["down"][0, 0, 0])
        )
        sigma_first_moment = 0.5 * (
            safe_real(solver.Sigma_moments["up"][1, 0, 0])
            + safe_real(solver.Sigma_moments["down"][1, 0, 0])
        )
    except Exception:
        pass
    average_order = getattr(solver, "average_order", np.nan)
    auto_corr = getattr(solver, "auto_corr_time", np.nan)
    return ImpurityResult(
        G_iw=clone_gf(G_iw),
        G_iw_raw=clone_gf(G_iw_raw),
        Sigma_iw=clone_gf(Sigma_iw),
        Sigma_iw_raw=clone_gf(Sigma_iw_raw),
        G_tau_measured=clone_gf(G_tau),
        filling=float(n_up_value + n_down_value),
        double_occupancy=float(double_occ),
        sigma_infinity=float(sigma_inf),
        sigma_first_moment=float(sigma_first_moment),
        average_sign=float(getattr(solver, "average_sign", np.nan)),
        average_order=float(average_order) if np.isscalar(average_order) else float("nan"),
        auto_corr_time=float(auto_corr) if np.isscalar(auto_corr) else float("nan"),
        wall_seconds=float(wall),
        seed=rank_seed,
        solve_status=solve_status,
    )
def run_fixed_mu_dmft(
    *, U: float, beta: float, mu: float,
    n_iw: int, n_tau: int, n_l: int,
    n_cycles: int, n_warmup_cycles: int, length_cycle: int,
    max_iter: int, min_iter: int, mix: float,
    dmft_tol: float, density_stability_tol: float, stable_iterations: int,
    conv_n_iw: int, seed_base: int, vary_seed_by_iteration: bool,
    perform_tail_fit: bool, fit_min_w: float, fit_max_w: float,
    fit_max_moment: int, verbosity: int, max_time: int,
    delta_initial: Optional[Gf] = None,
    stage: str = "refined",
) -> FixedMuResult:
    iw_mesh = MeshImFreq(beta=float(beta), S="Fermion", n_iw=int(n_iw))
    iw_values = mesh_iw_values(iw_mesh)
    pos = positive_indices(iw_values)[: int(conv_n_iw)]
    if len(pos) == 0:
        raise RuntimeError("No positive Matsubara frequencies found.")
    if delta_initial is None:
        delta_iw = scalar_gf(iw_mesh, initial_delta(iw_values, mu))
    else:
        if len(delta_initial.mesh) != len(iw_mesh):
            raise ValueError("Continuation bath has a different Matsubara mesh.")
        delta_iw = clone_gf(delta_initial)
    residual_history: list[float] = []
    filling_history: list[float] = []
    sign_history: list[float] = []
    auto_corr_history: list[float] = []
    stable = 0
    previous_filling = None
    converged = False
    last_impurity = None
    started = time.time()
    for iteration in range(int(max_iter)):
        iter_seed = int(seed_base + (7919 * iteration if vary_seed_by_iteration else 0))
        last_impurity = solve_impurity(
            U=U, beta=beta, mu=mu, delta_iw=delta_iw,
            n_iw=n_iw, n_tau=n_tau, n_l=n_l,
            n_cycles=n_cycles, n_warmup_cycles=n_warmup_cycles,
            length_cycle=length_cycle, seed=iter_seed,
            perform_tail_fit=perform_tail_fit,
            fit_min_w=fit_min_w, fit_max_w=fit_max_w,
            fit_max_moment=fit_max_moment, verbosity=verbosity,
            max_time=max_time,
        )
        delta_target = T_HOP**2 * last_impurity.G_iw.data[:, 0, 0]
        residual_values = delta_target[pos] - delta_iw.data[pos, 0, 0]
        residual = float(np.sqrt(np.mean(np.abs(residual_values) ** 2)) / T_HOP)
        dn = (
            float("inf") if previous_filling is None
            else abs(last_impurity.filling - previous_filling)
        )
        residual_history.append(residual)
        filling_history.append(last_impurity.filling)
        sign_history.append(last_impurity.average_sign)
        auto_corr_history.append(last_impurity.auto_corr_time)
        master_print(
            f"    [{stage}] DMFT {iteration + 1:3d}: mu/t={mu:+.8f} "
            f"n={last_impurity.filling:.7f} residual={residual:.3e} "
            f"dn={dn:.3e} sign={last_impurity.average_sign:.6f} "
            f"tau_auto={last_impurity.auto_corr_time:.3g} cycles"
        )
        passes = (
            iteration + 1 >= int(min_iter)
            and residual < float(dmft_tol)
            and dn < float(density_stability_tol)
        )
        stable = stable + 1 if passes else 0
        previous_filling = last_impurity.filling
        if stable >= int(stable_iterations):
            converged = True
            break
        delta_mixed = (
            (1.0 - float(mix)) * delta_iw.data[:, 0, 0]
            + float(mix) * delta_target
        )
        delta_iw.data[:, 0, 0] = delta_mixed
    if last_impurity is None:
        raise RuntimeError("DMFT completed no impurity solve.")
    g0_iw = scalar_gf(
        iw_mesh,
        1.0 / (iw_values + float(mu) - delta_iw.data[:, 0, 0]),
    )
    return FixedMuResult(
        mu=float(mu), filling=float(last_impurity.filling),
        converged=bool(converged), iterations=len(residual_history),
        delta_iw=clone_gf(delta_iw), G0_iw=g0_iw,
        impurity=last_impurity,
        residual_history=residual_history,
        filling_history=filling_history,
        sign_history=sign_history,
        auto_corr_history=auto_corr_history,
        wall_seconds=float(time.time() - started),
        stage=stage,
    )
def safeguarded_trial(low: FixedMuResult, high: FixedMuResult,
                      target: float) -> float:
    """Regula falsi with a bisection safeguard. Endpoints must be ordered."""
    width = high.mu - low.mu
    if width <= 0.0:
        raise RuntimeError("Invalid or reversed chemical-potential bracket.")
    dn = high.filling - low.filling
    if dn > 1.0e-10:
        trial = low.mu + (target - low.filling) * width / dn
    else:
        trial = 0.5 * (low.mu + high.mu)
    return float(np.clip(trial, low.mu + 0.15 * width,
                        high.mu - 0.15 * width))


def tune_mu_for_filling(args, U: float, target: float,
                        run_seed: int) -> tuple[FixedMuResult, list[dict]]:
    """Two-stage safeguarded search for a fixed-density *DMFT* solution.

    Stage 1: cheap, relaxed-convergence DMFT is allowed to SELECT a bracket.
    Stage 2: rerun BOTH sides of that bracket with full statistics and
    full convergence tolerances. All later root updates use ONLY stage-2
    measurements.  Coarse measurements cannot satisfy --density-tol.

    If refined measurements contradict a coarse bracket, the code expands
    along previously sampled coarse points, RE-EVALUATING each with refined
    statistics before it can participate in the refined bracket.
    """
    coarse: list[FixedMuResult] = []
    refined: list[FixedMuResult] = []
    evaluations: list[FixedMuResult] = []

    def nearest_bath(mu_value: float) -> Optional[Gf]:
        if not evaluations:
            return None
        # On the hole-doped side, prefer the lower-density branch for
        # continuation, as in the original implementation.
        lower_density = [r for r in evaluations if r.filling <= target]
        pool = lower_density if lower_density else evaluations
        return min(pool, key=lambda r: abs(r.mu - mu_value)).delta_iw

    def evaluate(mu_value: float, stage: str,
                 delta_initial: Optional[Gf] = None,
                 fresh_bath: bool = False) -> FixedMuResult:
        cache = coarse if stage == "coarse" else refined
        for previous in cache:
            if abs(previous.mu - mu_value) < 1.0e-11:
                return previous
        is_coarse = stage == "coarse"
        n_cycles = args.coarse_n_cycles if is_coarse else args.n_cycles
        n_warmup = (args.coarse_n_warmup_cycles if is_coarse
                    else args.n_warmup_cycles)
        length_cycle = (args.coarse_length_cycle if is_coarse
                        else args.length_cycle)
        dmft_tol = args.coarse_dmft_tol if is_coarse else args.dmft_tol
        dn_tol = (args.coarse_density_stability_tol if is_coarse
                  else args.density_stability_tol)
        max_iter = (args.coarse_max_dmft_iter if is_coarse
                    else args.max_dmft_iter)
        initial = (None if fresh_bath else
                   delta_initial if delta_initial is not None
                   else nearest_bath(mu_value))
        master_print(
            f"\n  [{stage.upper()}] Evaluating mu/t={mu_value:+.8f} "
            f"(target n={target:.6f}, cycles={n_cycles}, "
            f"length_cycle={length_cycle})"
        )
        result = run_fixed_mu_dmft(
            U=U, beta=args.beta_t, mu=mu_value,
            n_iw=args.n_iw, n_tau=args.n_tau, n_l=args.n_l,
            n_cycles=n_cycles, n_warmup_cycles=n_warmup,
            length_cycle=length_cycle,
            max_iter=max_iter, min_iter=args.min_dmft_iter,
            mix=args.mix, dmft_tol=dmft_tol,
            density_stability_tol=dn_tol,
            stable_iterations=args.stable_iterations,
            conv_n_iw=args.conv_n_iw,
            seed_base=run_seed + 100003 * len(evaluations),
            vary_seed_by_iteration=args.vary_seed_by_iteration,
            perform_tail_fit=not args.no_tail_fit,
            fit_min_w=args.fit_min_w_over_t,
            fit_max_w=args.fit_max_w_over_t,
            fit_max_moment=args.fit_max_moment,
            verbosity=args.cthyb_verbosity, max_time=args.cthyb_max_time,
            delta_initial=initial, stage=stage,
        )
        if not result.converged:
            raise RuntimeError(
                f"{stage.capitalize()} DMFT failed to converge at mu/t="
                f"{mu_value:+.8f} after {result.iterations} iterations "
                f"(last residual={result.residual_history[-1]:.3e}). "
                "Increase statistics or the applicable maximum DMFT "
                "iterations; do not use an unconverged density to bracket."
            )
        cache.append(result)
        evaluations.append(result)
        master_print(f"  [{stage.upper()}] Converged mu/t={mu_value:+.8f}: "
                     f"n={result.filling:.7f}, "
                     f"n-target={result.filling-target:+.4e}")
        return result

    mu_low = float(args.mu_min_over_t) if args.mu_min_over_t is not None else -4.0
    mu_high = (float(args.mu_max_over_t) if args.mu_max_over_t is not None
               else 0.5 * U)
    if mu_low >= mu_high:
        raise ValueError("--mu-min-over-t must be smaller than --mu-max-over-t")

    # Initial coarse bracket; save full baths so later high-stat runs can
    # warm-start without starting from the non-interacting bath.
    low = evaluate(mu_low, "coarse", fresh_bath=True)
    high = evaluate(mu_high, "coarse", fresh_bath=True)

    expansions = 0
    while (low.filling > target and args.mu_min_over_t is None
           and expansions < args.max_bracket_expansions):
        mu_low -= max(2.0, 0.25 * U)
        low = evaluate(mu_low, "coarse")
        expansions += 1
    expansions = 0
    while (high.filling < target and args.mu_max_over_t is None
           and expansions < args.max_bracket_expansions):
        mu_high += max(2.0, 0.25 * U)
        high = evaluate(mu_high, "coarse")
        expansions += 1
    if low.filling > target or high.filling < target:
        raise RuntimeError(
            "Could not establish a COARSE bracket: "
            f"n({low.mu:.5f})={low.filling:.6f}, "
            f"n({high.mu:.5f})={high.filling:.6f}, target={target:.6f}. "
            "Widen the initial --mu-min-over-t/--mu-max-over-t bounds."
        )

    # Optional known approximate mu/t, e.g. 0.549606 from a previous run.
    # It is a trial, never a value imposed on the final solution.
    if (args.mu_guess_over_t is not None
            and low.mu + 1e-10 < args.mu_guess_over_t < high.mu - 1e-10):
        guessed = evaluate(args.mu_guess_over_t, "coarse")
        if guessed.filling <= target:
            low = guessed
        else:
            high = guessed

    # Continue cheap fixed-mu DMFT until the TWO sides of the bracket are
    # both within 0.1 electrons/site (or the user-specified switch distance).
    # A single near-target coarse point is not sufficient to guarantee that
    # the refined calculation is started with a useful local bracket.
    for k in range(args.coarse_max_mu_iter + 1):
        if (abs(low.filling - target) <= args.switch_density_distance
                and abs(high.filling - target) <= args.switch_density_distance):
            break
        if k == args.coarse_max_mu_iter:
            raise RuntimeError(
                "Coarse search could not locate two near-target bracket "
                f"points within {args.coarse_max_mu_iter} steps. "
                "Increase --coarse-max-mu-iter, widen the switch distance, "
                "or inspect n(mu) for discontinuities / competing DMFT branches."
            )
        if high.mu - low.mu <= args.mu_tol_over_t:
            raise RuntimeError(
                "Coarse bracket became very narrow without locating both "
                "densities near target; check for a density discontinuity "
                "or overly noisy coarse QMC."
            )
        trial_mu = safeguarded_trial(low, high, target)
        trial = evaluate(trial_mu, "coarse")
        if trial.filling < target:
            low = trial
        else:
            high = trial

    master_print(
        f"\n  SWITCH TO REFINED QMC: coarse bracket "
        f"mu/t=[{low.mu:.7f}, {high.mu:.7f}], "
        f"n=[{low.filling:.6f}, {high.filling:.6f}]. "
        "These densities will now be RE-MEASURED with full statistics."
    )

    # Revalidate BOTH endpoint densities at full statistics, explicitly
    # starting from the corresponding converged coarse baths.
    low_f = evaluate(low.mu, "refined", delta_initial=low.delta_iw)
    high_f = evaluate(high.mu, "refined", delta_initial=high.delta_iw)

    # A loose/noisy coarse endpoint may move to the wrong side.  If that
    # happens, fall back on successively farther coarse mu values, but rerun
    # each at refined statistics before accepting its sign in the bracket.
    while low_f.filling > target:
        left_candidates = sorted(
            (r for r in coarse if r.mu < low_f.mu - 1e-10),
            key=lambda r: r.mu, reverse=True,
        )
        if not left_candidates:
            raise RuntimeError(
                "Refined lower endpoint is above target filling and no "
                "previous coarse point lies to its left. Widen initial "
                "mu bounds or improve coarse statistics."
            )
        low_f = evaluate(left_candidates[0].mu, "refined",
                         delta_initial=left_candidates[0].delta_iw)
    while high_f.filling < target:
        right_candidates = sorted(
            (r for r in coarse if r.mu > high_f.mu + 1e-10),
            key=lambda r: r.mu,
        )
        if not right_candidates:
            raise RuntimeError(
                "Refined upper endpoint is below target filling and no "
                "previous coarse point lies to its right. Widen initial "
                "mu bounds or improve coarse statistics."
            )
        high_f = evaluate(right_candidates[0].mu, "refined",
                          delta_initial=right_candidates[0].delta_iw)

    if low_f.mu >= high_f.mu:
        raise RuntimeError("Refined bracket not ordered; inspect n(mu) branch.")
    master_print(
        f"  REFINED bracket confirmed: mu/t=[{low_f.mu:.7f}, "
        f"{high_f.mu:.7f}], densities=[{low_f.filling:.7f}, "
        f"{high_f.filling:.7f}]"
    )
    for k in range(args.max_mu_iter):
        best = min(refined, key=lambda r: abs(r.filling - target))
        if abs(best.filling - target) <= args.density_tol:
            break
        if high_f.mu - low_f.mu <= args.mu_tol_over_t:
            raise RuntimeError(
                "Refined chemical-potential bracket is narrower than "
                f"--mu-tol-over-t, but best |n-target|="
                f"{abs(best.filling-target):.3e} still exceeds "
                f"--density-tol={args.density_tol:.3e}. Increase "
                "statistics or decrease --mu-tol-over-t; don't claim convergence."
            )
        trial_mu = safeguarded_trial(low_f, high_f, target)
        trial = evaluate(trial_mu, "refined")
        if trial.filling < target:
            low_f = trial
        else:
            high_f = trial

    best = min(refined, key=lambda r: abs(r.filling - target))
    if abs(best.filling - target) > args.density_tol:
        raise RuntimeError(
            "Refined search exhausted without reaching density tolerance: "
            f"best mu/t={best.mu:+.8f}, n={best.filling:.7f}, "
            f"|n-target|={abs(best.filling-target):.3e} > "
            f"{args.density_tol:.3e}. Increase --max-mu-iter and/or "
            "refined QMC statistics."
        )
    summary = [
        {
            "stage": r.stage,
            "mu_over_t": r.mu,
            "filling": r.filling,
            "density_error": r.filling - target,
            "dmft_converged": r.converged,
            "dmft_iterations": r.iterations,
            "last_residual": r.residual_history[-1],
            "average_sign": r.impurity.average_sign,
            "auto_corr_time_cycles": r.impurity.auto_corr_time,
            "n_cycles": (args.coarse_n_cycles if r.stage == "coarse"
                         else args.n_cycles),
            "length_cycle": (args.coarse_length_cycle if r.stage == "coarse"
                             else args.length_cycle),
            "wall_seconds": r.wall_seconds,
        }
        for r in evaluations
    ]
    return best, summary


def run_final_replicas(args, result: FixedMuResult, U: float, seed_base: int):
    replicas: list[ImpurityResult] = []
    for replica in range(int(args.final_replicas)):
        master_print(f"  Final fixed-bath replica {replica + 1}/{args.final_replicas}")
        replicas.append(solve_impurity(
            U=U, beta=args.beta_t, mu=result.mu, delta_iw=result.delta_iw,
            n_iw=args.n_iw, n_tau=args.n_tau, n_l=args.n_l,
            n_cycles=args.final_n_cycles,
            n_warmup_cycles=args.final_n_warmup_cycles,
            length_cycle=args.final_length_cycle,
            seed=seed_base + 10000019 * replica,
            perform_tail_fit=not args.no_tail_fit,
            fit_min_w=args.fit_min_w_over_t,
            fit_max_w=args.fit_max_w_over_t,
            fit_max_moment=args.fit_max_moment,
            verbosity=args.cthyb_verbosity,
            max_time=args.cthyb_max_time,
        ))
    return replicas
def mean_and_sem(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(arrays, axis=0)
    mean = np.mean(stack, axis=0)
    if len(arrays) < 2:
        return mean, np.full(mean.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    # Encode the independent real/imaginary standard errors in the real and
    # imaginary components of one array for compact output.
    sem = (
        np.std(stack.real, axis=0, ddof=1)
        + 1j * np.std(stack.imag, axis=0, ddof=1)
    ) / math.sqrt(len(arrays))
    return mean, sem
def tag_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace("-", "m").replace(".", "p")
def save_case(args, U: float, target: float, result: FixedMuResult,
              mu_search: list[dict], replicas: list[ImpurityResult]) -> tuple[Path, float]:
    run_root = Path(args.output_dir) / args.tag if args.tag else Path(args.output_dir)
    case_dir = run_root / (
        f"U{tag_float(U)}_n{tag_float(target)}_beta{tag_float(args.beta_t, 2)}"
    )
    if mpi.is_master_node():
        case_dir.mkdir(parents=True, exist_ok=True)
    iw = mesh_iw_values(result.delta_iw.mesh)
    tau_mesh = MeshImTime(beta=args.beta_t, S="Fermion", n_tau=args.n_tau)
    tau = mesh_tau_values(tau_mesh)
    G_mean, G_sem = mean_and_sem([r.G_iw.data[:, 0, 0] for r in replicas])
    G_raw_mean, G_raw_sem = mean_and_sem([r.G_iw_raw.data[:, 0, 0] for r in replicas])
    Sigma_mean, Sigma_sem = mean_and_sem([r.Sigma_iw.data[:, 0, 0] for r in replicas])
    Sigma_raw_mean, Sigma_raw_sem = mean_and_sem(
        [r.Sigma_iw_raw.data[:, 0, 0] for r in replicas]
    )
    Gtau_mean, Gtau_sem = mean_and_sem([r.G_tau_measured.data[:, 0, 0] for r in replicas])
    fillings = np.asarray([r.filling for r in replicas], dtype=float)
    doubles = np.asarray([r.double_occupancy for r in replicas], dtype=float)
    signs = np.asarray([r.average_sign for r in replicas], dtype=float)
    average_orders = np.asarray([r.average_order for r in replicas], dtype=float)
    auto_corr_times = np.asarray([r.auto_corr_time for r in replicas], dtype=float)
    sigma_inf_values = np.asarray([r.sigma_infinity for r in replicas], dtype=float)
    sigma_inf = float(np.mean(sigma_inf_values))
    sigma_first_values = np.asarray([r.sigma_first_moment for r in replicas], dtype=float)
    sigma_first = float(np.mean(sigma_first_values))
    sigma_dynamic_iw = scalar_gf(result.delta_iw.mesh, Sigma_mean - sigma_inf)
    sigma_known_moments = make_zero_tail(sigma_dynamic_iw, n_moments=2)
    sigma_known_moments[0, 0, 0] = 0.0
    sigma_known_moments[1, 0, 0] = sigma_first
    sigma_dynamic_tau = gf_to_tau(
        sigma_dynamic_iw, tau_mesh, known_moments=sigma_known_moments
    ).data[:, 0, 0]
    G0_values = result.G0_iw.data[:, 0, 0]
    delta_values = result.delta_iw.data[:, 0, 0]
    conv_pos = positive_indices(iw)[: int(args.conv_n_iw)]
    final_bethe_residual = float(np.sqrt(np.mean(
        np.abs(T_HOP**2 * G_mean[conv_pos] - delta_values[conv_pos]) ** 2
    )) / T_HOP)
    final_bethe_residual_raw = float(np.sqrt(np.mean(
        np.abs(T_HOP**2 * G_raw_mean[conv_pos] - delta_values[conv_pos]) ** 2
    )) / T_HOP)
    final_density_error = float(np.mean(fillings) - target)
    G0_tau = gf_to_tau(result.G0_iw, tau_mesh).data[:, 0, 0]
    delta_tau = gf_to_tau(result.delta_iw, tau_mesh).data[:, 0, 0]
    metadata = {
        "model": "paramagnetic single-band Hubbard model, Bethe lattice",
        "convention": "H_int=U*n_up*n_down; G0^-1=iw+mu-Delta",
        "t_internal": 1.0,
        "t_eV": args.t_ev,
        "U_over_t": U,
        "U_eV": None if args.t_ev is None else U * args.t_ev,
        "target_filling": target,
        "target_hole_doping": 1.0 - target,
        "beta_t": args.beta_t,
        "temperature_over_t": 1.0 / args.beta_t,
        "temperature_K": args.temperature_k_resolved,
        "mu_over_t": result.mu,
        "mu_eV": None if args.t_ev is None else result.mu * args.t_ev,
        "final_filling_mean": float(np.mean(fillings)),
        "final_filling_sem": float(np.std(fillings, ddof=1) / np.sqrt(len(fillings))) if len(fillings) > 1 else None,
        "double_occupancy_mean": float(np.mean(doubles)),
        "average_sign_mean": float(np.mean(signs)),
        "average_order_mean": float(np.nanmean(average_orders)) if np.any(np.isfinite(average_orders)) else None,
        "auto_corr_time_mean": float(np.nanmean(auto_corr_times)) if np.any(np.isfinite(auto_corr_times)) else None,
        "sigma_infinity_mean_over_t": sigma_inf,
        "sigma_first_moment_mean_over_t2": sigma_first,
        "dmft_converged": result.converged,
        "dmft_iterations": result.iterations,
        "last_dmft_residual": result.residual_history[-1],
        "final_replica_bethe_residual": final_bethe_residual,
        "final_replica_bethe_residual_raw": final_bethe_residual_raw,
        "final_density_error": final_density_error,
        "final_quality_passed": bool(
            final_bethe_residual <= args.dmft_tol
            and abs(final_density_error) <= args.density_tol
        ),
        "settings": vars(args),
        "mu_search": mu_search,
        "coarse_mu_evaluations": sum(r["stage"] == "coarse" for r in mu_search),
        "refined_mu_evaluations": sum(r["stage"] == "refined" for r in mu_search),
        "density_from_Gtau_beta_endpoint": float(-2.0 * Gtau_mean[-1].real),
        "density_matrix_minus_Gtau_endpoint": float(np.mean(fillings) + 2.0 * Gtau_mean[-1].real),
        "note_density_crosscheck": "Endpoint estimator is a diagnostic, not substituted for measured density-matrix filling.",
        "note_sigma_tau": "Sigma_dynamic_tau is Fourier[Sigma(iw)-Sigma_infinity]; the Hartree constant is a delta function at tau=0.",
    }
    if not mpi.is_master_node():
        return case_dir, final_bethe_residual
    np.savez_compressed(
        case_dir / "dmft_result.npz",
        iw_over_t=iw,
        tau_times_t=tau,
        G_iw=G_mean,
        G_iw_sem=G_sem,
        G_iw_raw=G_raw_mean,
        G_iw_raw_sem=G_raw_sem,
        Sigma_iw=Sigma_mean,
        Sigma_iw_sem=Sigma_sem,
        Sigma_iw_raw=Sigma_raw_mean,
        Sigma_iw_raw_sem=Sigma_raw_sem,
        Sigma_infinity_over_t=sigma_inf,
        Sigma_first_moment_over_t2=sigma_first,
        Sigma_dynamic_tau=sigma_dynamic_tau,
        G_tau=Gtau_mean,
        G_tau_sem=Gtau_sem,
        G0_iw=G0_values,
        G0_tau=G0_tau,
        Delta_iw=delta_values,
        Delta_tau=delta_tau,
        replica_fillings=fillings,
        replica_double_occupancies=doubles,
        replica_average_signs=signs,
        replica_average_orders=average_orders,
        replica_auto_corr_times=auto_corr_times,
        replica_solve_statuses=np.asarray([r.solve_status for r in replicas], dtype=int),
        replica_Gtau_beta_endpoints=np.asarray([r.G_tau_measured.data[-1, 0, 0] for r in replicas]),
        final_replica_bethe_residual=final_bethe_residual,
        final_replica_bethe_residual_raw=final_bethe_residual_raw,
        dmft_residual_history=np.asarray(result.residual_history),
        dmft_filling_history=np.asarray(result.filling_history),
        dmft_sign_history=np.asarray(result.sign_history),
        dmft_auto_corr_time_history=np.asarray(result.auto_corr_history),
    )
    with (case_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    pos = positive_indices(iw)
    with (case_dir / "matsubara_positive.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "n", "omega_over_t", "Re_G_times_t", "Im_G_times_t",
            "SEM_Re_G_times_t", "SEM_Im_G_times_t",
            "Re_Sigma_over_t", "Im_Sigma_over_t",
            "SEM_Re_Sigma_over_t", "SEM_Im_Sigma_over_t",
            "Re_Sigma_raw_over_t", "Im_Sigma_raw_over_t",
            "Re_G0_times_t", "Im_G0_times_t", "Re_Delta_over_t", "Im_Delta_over_t",
        ])
        for n_index, j in enumerate(pos):
            writer.writerow([
                n_index, iw[j].imag, G_mean[j].real, G_mean[j].imag,
                np.real(G_sem[j]), np.imag(G_sem[j]),
                Sigma_mean[j].real, Sigma_mean[j].imag,
                np.real(Sigma_sem[j]), np.imag(Sigma_sem[j]),
                Sigma_raw_mean[j].real, Sigma_raw_mean[j].imag,
                G0_values[j].real, G0_values[j].imag,
                delta_values[j].real, delta_values[j].imag,
            ])
    with (case_dir / "imaginary_time.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "tau_times_t", "Re_G_tau", "Im_G_tau", "SEM_Re_G_tau",
            "Re_Sigma_dynamic_times_t2", "Im_Sigma_dynamic_times_t2",
            "Re_G0_tau", "Im_G0_tau", "Re_Delta_times_t2", "Im_Delta_times_t2",
        ])
        for j in range(len(tau)):
            writer.writerow([
                tau[j], Gtau_mean[j].real, Gtau_mean[j].imag, np.real(Gtau_sem[j]),
                sigma_dynamic_tau[j].real, sigma_dynamic_tau[j].imag,
                G0_tau[j].real, G0_tau[j].imag,
                delta_tau[j].real, delta_tau[j].imag,
            ])
    omega = iw[pos].imag
    plot_n = min(int(args.plot_n_iw), len(pos))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    axes[0, 0].plot(omega[:plot_n], Sigma_mean[pos[:plot_n]].real, "o-", ms=3, label=r"Re $\Sigma$")
    axes[0, 0].plot(omega[:plot_n], Sigma_mean[pos[:plot_n]].imag, "o-", ms=3, label=r"Im $\Sigma$")
    axes[0, 0].axhline(sigma_inf, color="0.5", ls=":", label=r"$\Sigma_\infty$")
    axes[0, 0].set(xlabel=r"$\omega_n/t$", ylabel=r"$\Sigma(i\omega_n)/t$")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(omega[:plot_n], G_mean[pos[:plot_n]].real, "o-", ms=3, label=r"Re $G$")
    axes[0, 1].plot(omega[:plot_n], G_mean[pos[:plot_n]].imag, "o-", ms=3, label=r"Im $G$")
    axes[0, 1].set(xlabel=r"$\omega_n/t$", ylabel=r"$tG(i\omega_n)$")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(tau / args.beta_t, Gtau_mean.real, lw=1.4)
    axes[1, 0].set(xlabel=r"$\tau/\beta$", ylabel=r"$G(\tau)$")
    axes[1, 1].plot(tau / args.beta_t, sigma_dynamic_tau.real, lw=1.4)
    axes[1, 1].set(xlabel=r"$\tau/\beta$", ylabel=r"$\Sigma_{\rm dyn}(\tau)/t^2$")
    for ax in axes.flat:
        ax.grid(alpha=0.25, ls=":")
    fig.suptitle(
        rf"CT-HYB DMFT: $U/t={U:g}$, target $n={target:g}$, "
        rf"$\beta t={args.beta_t:g}$; achieved $n={np.mean(fillings):.5f}$"
    )
    fig.tight_layout()
    fig.savefig(case_dir / "greens_and_self_energy.png", dpi=220)
    fig.savefig(case_dir / "greens_and_self_energy.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].semilogy(np.arange(1, len(result.residual_history) + 1), result.residual_history, "o-")
    axes[0].axhline(args.dmft_tol, color="k", ls=":")
    axes[0].set(xlabel="DMFT iteration", ylabel="bath residual")
    mus = np.asarray([x["mu_over_t"] for x in mu_search])
    ns = np.asarray([x["filling"] for x in mu_search])
    for stage, marker in (("coarse", "o"), ("refined", "s")):
        which = np.asarray([x.get("stage") == stage for x in mu_search])
        if np.any(which):
            ordered_stage = np.flatnonzero(which)
            ordered_stage = ordered_stage[np.argsort(mus[ordered_stage])]
            axes[1].plot(mus[ordered_stage], ns[ordered_stage],
                         marker=marker, linestyle="--" if stage == "coarse" else "-",
                         ms=4, label=stage)
    axes[1].legend(frameon=False)
    axes[1].axhline(target, color="k", ls=":")
    axes[1].set(xlabel=r"$\mu/t$", ylabel=r"$\langle n\rangle$")
    for ax in axes:
        ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(case_dir / "convergence.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    take = pos[:min(len(pos), int(args.plot_n_iw))]
    w = iw[take].imag
    axes[0].plot(w, Sigma_raw_mean[take].real, "o", ms=3, alpha=0.6, label="raw")
    axes[0].plot(w, Sigma_mean[take].real, "-", lw=1.5, label="tail fitted")
    axes[0].axvline(args.fit_min_w_over_t, ls=":", color="0.5")
    axes[0].set(xlabel=r"$\omega_n/t$", ylabel=r"Re $\Sigma/t$")
    axes[1].plot(w, Sigma_raw_mean[take].imag, "o", ms=3, alpha=0.6, label="raw")
    axes[1].plot(w, Sigma_mean[take].imag, "-", lw=1.5, label="tail fitted")
    axes[1].axvline(args.fit_min_w_over_t, ls=":", color="0.5")
    axes[1].set(xlabel=r"$\omega_n/t$", ylabel=r"Im $\Sigma/t$")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(case_dir / "sigma_raw_vs_fitted.png", dpi=220)
    plt.close(fig)
    return case_dir, final_bethe_residual
def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure CT-HYB DMFT for doped single-band Hubbard model on the Bethe lattice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--u-over-t", type=float, nargs="+", default=[8.0, 9.0])
    parser.add_argument("--fillings", type=float, nargs="+", default=None,
                        help="Total filling n=n_up+n_down; default: 0.9 0.7.")
    parser.add_argument("--hole-dopings", type=float, nargs="+", default=None,
                        help="Hole dopings delta=1-n; mutually exclusive with --fillings.")
    parser.add_argument("--beta-t", type=float, default=None,
                        help="Dimensionless inverse temperature beta*t.")
    parser.add_argument("--temperature-k", type=float, default=None,
                        help="Temperature in kelvin; requires --t-ev.")
    parser.add_argument("--t-ev", "--t", dest="t_ev", type=float, default=None,
                        help="Physical hopping in eV (e.g. 0.25 for 250 meV); conversion/reporting ONLY, solver t stays 1.0.")
    parser.add_argument("--n-iw", type=int, default=512)
    parser.add_argument("--n-tau", type=int, default=4097)
    parser.add_argument("--n-l", type=int, default=80)
    parser.add_argument("--coarse-n-cycles", type=int, default=25000,
                        help="QMC measurement cycles per coarse fixed-mu iteration.")
    parser.add_argument("--coarse-n-warmup-cycles", type=int, default=4000)
    parser.add_argument("--coarse-length-cycle", type=int, default=350)
    parser.add_argument("--coarse-dmft-tol", type=float, default=1.0e-2)
    parser.add_argument("--coarse-density-stability-tol", type=float, default=1.0e-2)
    parser.add_argument("--coarse-max-dmft-iter", type=int, default=40)
    parser.add_argument("--coarse-max-mu-iter", type=int, default=16,
                        help="Maximum new coarse mu evaluations after the endpoints.")
    parser.add_argument("--switch-density-distance", type=float, default=0.10,
                        help="Both coarse bracket densities must lie within this distance of target before refinement.")
    parser.add_argument("--n-cycles", type=int, default=20000)
    parser.add_argument("--n-warmup-cycles", type=int, default=5000)
    parser.add_argument("--length-cycle", type=int, default=100)
    parser.add_argument("--final-replicas", type=int, default=2)
    parser.add_argument("--final-n-cycles", type=int, default=50000)
    parser.add_argument("--final-n-warmup-cycles", type=int, default=10000)
    parser.add_argument("--final-length-cycle", type=int, default=200)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument(
        "--vary-seed-by-iteration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use independent QMC streams across DMFT iterations.",
    )
    parser.add_argument("--cthyb-verbosity", type=int, default=0)
    parser.add_argument("--cthyb-max-time", type=int, default=-1,
                        help="Maximum seconds per impurity solve; -1 disables the limit.")
    parser.add_argument("--no-tail-fit", action="store_true")
    parser.add_argument("--fit-min-w-over-t", type=float, default=12.0)
    parser.add_argument("--fit-max-w-over-t", type=float, default=20.0)
    parser.add_argument("--fit-max-moment", type=int, default=3)
    parser.add_argument("--mix", type=float, default=0.3)
    parser.add_argument("--dmft-tol", type=float, default=1.0e-3)
    parser.add_argument("--density-tol", type=float, default=1.0e-3)
    parser.add_argument("--density-stability-tol", type=float, default=5.0e-4)
    parser.add_argument("--mu-tol-over-t", type=float, default=2.0e-3)
    parser.add_argument("--stable-iterations", type=int, default=2)
    parser.add_argument("--min-dmft-iter", type=int, default=4)
    parser.add_argument("--max-dmft-iter", type=int, default=60)
    parser.add_argument("--max-mu-iter", type=int, default=12)
    parser.add_argument("--conv-n-iw", type=int, default=64)
    parser.add_argument("--mu-min-over-t", type=float, default=None)
    parser.add_argument("--mu-max-over-t", type=float, default=None)
    parser.add_argument("--mu-guess-over-t", type=float, default=None,
                        help="Optional known guess, tested in coarse stage inside initial bracket.")
    parser.add_argument("--max-bracket-expansions", type=int, default=4)
    parser.add_argument("--plot-n-iw", type=int, default=100)
    parser.add_argument("--output-dir", default="doped_cthyb_dmft_results")
    parser.add_argument("--tag", default="",
                        help="Optional subdirectory label, e.g. pilot or production.")
    args = parser.parse_args()
    if args.fillings is not None and args.hole_dopings is not None:
        parser.error("Use either --fillings or --hole-dopings, not both.")
    if args.hole_dopings is not None:
        args.fillings = [1.0 - float(delta) for delta in args.hole_dopings]
    elif args.fillings is None:
        args.fillings = [0.9, 0.7]
    if args.t_ev is not None and args.t_ev <= 0.0:
        parser.error("--t-ev/--t must be positive.")
    if args.temperature_k is not None:
        if args.t_ev is None:
            parser.error("--temperature-k requires --t-ev.")
        if args.temperature_k <= 0.0:
            parser.error("--temperature-k must be positive.")
        if args.beta_t is not None:
            parser.error("Use either --beta-t or --temperature-k, not both.")
        args.beta_t = float(args.t_ev) / (KB_EV_PER_K * float(args.temperature_k))
        args.temperature_k_resolved = float(args.temperature_k)
    else:
        if args.beta_t is None:
            args.beta_t = 25.0
        if args.beta_t <= 0.0:
            parser.error("--beta-t must be positive.")
        args.temperature_k_resolved = (
            None if args.t_ev is None
            else float(args.t_ev) / (KB_EV_PER_K * float(args.beta_t))
        )
    if args.beta_t <= 0.0:
        parser.error("--beta-t must be positive.")
    if any(U <= 0.0 for U in args.u_over_t):
        parser.error("Every --u-over-t value must be positive.")
    if not 0.0 < args.mix <= 1.0:
        parser.error("--mix must satisfy 0 < mix <= 1.")
    for name in (
        "n_iw", "n_tau", "n_l", "n_cycles", "n_warmup_cycles",
        "length_cycle", "coarse_n_cycles", "coarse_n_warmup_cycles",
        "coarse_length_cycle", "coarse_max_dmft_iter", "coarse_max_mu_iter",
        "final_n_cycles", "final_n_warmup_cycles",
        "final_length_cycle", "stable_iterations", "min_dmft_iter",
        "max_dmft_iter", "max_mu_iter", "conv_n_iw",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1.")
    if (args.mu_min_over_t is not None and args.mu_max_over_t is not None
            and args.mu_min_over_t >= args.mu_max_over_t):
        parser.error("--mu-min-over-t must be smaller than --mu-max-over-t")
    if args.min_dmft_iter > args.coarse_max_dmft_iter:
        parser.error("--min-dmft-iter cannot exceed --coarse-max-dmft-iter")
    if args.min_dmft_iter > args.max_dmft_iter:
        parser.error("--min-dmft-iter cannot exceed --max-dmft-iter.")
    if args.conv_n_iw > args.n_iw:
        parser.error("--conv-n-iw cannot exceed --n-iw.")
    if args.cthyb_max_time == 0 or args.cthyb_max_time < -1:
        parser.error("--cthyb-max-time must be -1 (disabled) or positive.")
    for name in (
        "dmft_tol", "density_tol", "density_stability_tol", "mu_tol_over_t",
        "coarse_dmft_tol", "coarse_density_stability_tol", "switch_density_distance",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.fit_min_w_over_t >= args.fit_max_w_over_t:
        parser.error("--fit-min-w-over-t must be below --fit-max-w-over-t.")
    if args.final_replicas < 1:
        parser.error("--final-replicas must be at least 1.")
    for filling in args.fillings:
        if not 0.0 < filling < 2.0:
            parser.error("Every filling must satisfy 0 < n < 2.")
    omega_max = (2 * args.n_iw - 1) * math.pi / args.beta_t
    if not args.no_tail_fit and omega_max <= args.fit_max_w_over_t:
        parser.error(
            f"n_iw={args.n_iw} gives omega_max/t~{omega_max:.2f}, below the "
            f"tail-fit maximum {args.fit_max_w_over_t:.2f}. Increase --n-iw."
        )
    return args
def main():
    args = parse_args()
    master_print("\nPure CTHYB Bethe-DMFT configuration")
    master_print(f"  internal t = {T_HOP:g}; beta*t = {args.beta_t:.8f}; kB*T/t = {1.0 / args.beta_t:.8g}")
    if args.temperature_k_resolved is not None:
        master_print(f"  physical mapping: t={args.t_ev:g} eV, T={args.temperature_k_resolved:.6g} K")
    master_print(f"  lowest positive Matsubara frequency omega0/t = {math.pi / args.beta_t:.8g}")
    master_print(f"  U/t values = {args.u_over_t}; target fillings = {args.fillings}")
    master_print(f"  DMFT tolerance = {args.dmft_tol:g}; density tolerance = {args.density_tol:g}")
    master_print(
        f"  COARSE QMC: cycles={args.coarse_n_cycles}, warmup={args.coarse_n_warmup_cycles}, "
        f"length_cycle={args.coarse_length_cycle}, tol={args.coarse_dmft_tol:g}"
    )
    master_print(
        f"  REFINED QMC: cycles={args.n_cycles}, warmup={args.n_warmup_cycles}, "
        f"length_cycle={args.length_cycle}, tol={args.dmft_tol:g}"
    )
    master_print(
        f"  switch when both bracket density errors <= {args.switch_density_distance:g}; "
        "coarse densities are never used to accept final filling."
    )
    completed = []
    for u_index, U in enumerate(args.u_over_t):
        for n_index, target in enumerate(args.fillings):
            master_print("\n" + "=" * 78)
            master_print(f"Starting U/t={U:g}, target n={target:g} (hole doping={1.0-target:g})")
            seed = int(args.seed + 100000003 * u_index + 1000003 * n_index)
            result, mu_search = tune_mu_for_filling(args, float(U), float(target), seed)
            master_print(
                f"\n  Selected mu/t={result.mu:+.9f}, search filling={result.filling:.7f}, "
                f"error={result.filling-target:+.3e}"
            )
            replicas = run_final_replicas(args, result, float(U), seed + 700000001)
            case_dir, final_residual = save_case(
                args, float(U), float(target), result, mu_search, replicas
            )
            final_n = float(np.mean([r.filling for r in replicas]))
            master_print(
                f"  Final replica filling = {final_n:.7f}; "
                f"Bethe residual={final_residual:.3e}; saved to {case_dir}"
            )
            if final_residual > args.dmft_tol:
                master_print(
                    "  WARNING: production-replica Bethe residual exceeds --dmft-tol. "
                    "Increase search statistics and rerun before treating this as final data."
                )
            if abs(final_n - target) > args.density_tol:
                master_print(
                    "  WARNING: production-replica filling misses --density-tol. "
                    "Increase search statistics and rerun before treating this as final data."
                )
            completed.append({
                "U_over_t": float(U), "target_filling": float(target),
                "mu_over_t": result.mu, "final_filling": final_n,
                "final_bethe_residual": final_residual,
                "quality_passed": bool(
                    final_residual <= args.dmft_tol
                    and abs(final_n - target) <= args.density_tol
                ),
                "output_directory": str(case_dir),
            })
            # Update a partial run summary immediately: subsequent cases can be costly
            # or fail to converge, and completed results should remain discoverable.
            if mpi.is_master_node():
                root = Path(args.output_dir) / args.tag if args.tag else Path(args.output_dir)
                root.mkdir(parents=True, exist_ok=True)
                with (root / "run_summary.json").open("w", encoding="utf-8") as handle:
                    json.dump(completed, handle, indent=2)
    if mpi.is_master_node():
        root = Path(args.output_dir) / args.tag if args.tag else Path(args.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        with (root / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(completed, handle, indent=2)
    if all(case["quality_passed"] for case in completed):
        master_print("\nAll requested cases completed and passed the final quality checks.")
    else:
        master_print(
            "\nAll requested cases completed, but at least one case failed a final "
            "residual or density check; inspect run_summary.json before using the data."
        )
        # With `set -o pipefail` in the shell, the run is reported as failed.
        raise SystemExit(2)
if __name__ == "__main__":
    main()

