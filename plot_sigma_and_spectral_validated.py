#!/usr/bin/env python3
"""Validate preliminary NNLS continued Sigma and derive Bethe DMFT A(omega).

Inputs: four adaptive_v2 metadata.json files; continuation CSVs, optional NPZ
and diagnostics. Uses NumPy 1.x-compatible np.trapz. All frequency/energy
quantities are in units t, and A is per spin (dimensionless t*A).

IMPORTANT: passing numerical checks does not establish that analytical
continuation is unique or that spectral peaks are physical.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CASES = [
    ('U8_n0p7_beta25_adaptive_v2', 'U8p000_n0p700_beta25p00', 8, .7),
    ('U8_n0p9_beta25_adaptive_v2', 'U8p000_n0p900_beta25p00', 8, .9),
    ('U10_n0p7_beta25_adaptive_v2', 'U10p000_n0p700_beta25p00', 10, .7),
    ('U10_n0p9_beta25_adaptive_v2', 'U10p000_n0p900_beta25p00', 10, .9),
]


def bethe_g(z):
    """G*t = 2 / (z/t + sqrt((z/t)^2-4)), retarded branch, t=1."""
    z = np.asarray(z, dtype=np.complex128)
    # Product of principal roots gives sqrt~z at complex infinity, with
    # the correct analytic branch in the upper half-plane.
    root = np.sqrt(z - 2 + 0j) * np.sqrt(z + 2 + 0j)
    return 2.0 / (z + root)


def fermi(beta, w):
    x = np.clip(beta * w, -700, 700)
    return 1.0 / (1.0 + np.exp(x))


def evaluate(w, sigma, mu, eta):
    # Sigma has already been evaluated at w+i*eta; no second shift is
    # applied to Sigma itself. The bare inverse propagator includes +i eta.
    z = w + 1j*eta + mu - sigma
    G = bethe_g(z)
    A = -G.imag / np.pi
    return G, A


def checks_for_case(root, cont, case, sub, U, n_target, out):
    mp = root / case / sub / 'metadata.json'
    cp = cont / case / 'Sigma_retarded_estimate.csv'
    dp = cont / case / 'continuation_diagnostics.json'
    npz = cont / case / 'Sigma_retarded_estimate.npz'
    for p in (mp, cp):
        if not p.is_file():
            raise FileNotFoundError('Missing required file: ' + str(p))
    meta = json.loads(mp.read_text())
    diag = json.loads(dp.read_text()) if dp.is_file() else {}
    a = np.genfromtxt(cp, delimiter=',', names=True, encoding=None)
    w = np.atleast_1d(np.asarray(a['omega_over_t'], float))
    S = np.asarray(a['Re_Sigma_over_t'] + 1j*a['Im_Sigma_over_t'])
    if len(w) < 5 or not np.all(np.isfinite(S)) or not np.all(np.isfinite(w)):
        raise ValueError(f'Invalid real-frequency input for {case}')
    if not np.all(np.diff(w) > 0):
        raise ValueError(f'Frequency mesh not strictly increasing: {case}')
    if not (w[0] < 0 < w[-1]):
        raise ValueError(f'Frequency grid must contain positive and negative: {case}')
    eta = float(diag.get('eta_over_t', 0.15))
    mu = float(meta['mu_over_t'])
    beta = float(meta['beta_t'])
    n_dmft = float(meta['final_filling_mean'])
    t_ev = float(meta.get('t_eV', 0.25))
    if eta <= 0:
        raise ValueError('eta must be positive')
    G, A = evaluate(w, S, mu, eta)
    # Do not clamp negative spectral values: report failures explicitly.
    integral = float(np.trapz(A, w))
    n_from_A = float(2*np.trapz(A*fermi(beta, w), w))
    min_A = float(A.min())
    max_im_sigma = float(S.imag.max())
    a0 = float(np.interp(0, w, A))
    # A(omega) integration covers only [-12,12] by default; the true
    # spectral sum rule is an integral over the entire real axis.
    sensitivity = {}
    alt_curves = {}
    if npz.is_file():
        with np.load(npz, allow_pickle=False) as d:
            for label, key in [('lambda_half','sigma_retarded_lambda_half_over_t'),
                               ('lambda_double','sigma_retarded_lambda_double_over_t')]:
                if key in d:
                    alt = np.asarray(d[key])
                    if len(alt) == len(w):
                        _, A_alt = evaluate(w, alt, mu, eta)
                        alt_curves[label] = A_alt
                        sensitivity[label] = float(np.trapz(np.abs(A_alt-A), w))
    q = {
        'case':case, 'U_over_t':U, 'target_filling':n_target,
        'dmft_filling':n_dmft, 'mu_over_t':mu, 'beta_t':beta,
        'eta_over_t':eta, 'min_A_times_t':min_A,
        'maximum_Im_Sigma_over_t':max_im_sigma,
        'A_at_fermi_times_t':a0,
        'integrated_A_on_saved_window':integral,
        'window_omega_over_t':[float(w[0]),float(w[-1])],
        'n_from_truncated_broadened_A':n_from_A,
        'n_from_A_minus_DMFT':n_from_A-n_dmft,
        'continuation_backfit_rmse_over_t':diag.get('backfit_complex_rmse_over_t'),
        'regularization_max_delta_ImSigma_over_t':diag.get('regularization_sensitivity_imag_max_abs_over_t'),
        'A_integrated_L1_change_under_regularization':sensitivity,
        'note':'Area and filling are window-limited and eta-broadened; deviations are not conclusive. Positive and normalized A alone do not validate continuation.'
    }
    (out/f'{case}_checks.json').write_text(json.dumps(q,indent=2)+'\n')
    with (out/f'{case}_spectral.csv').open('w',newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['omega_over_t','omega_eV','Re_Sigma_over_t','Im_Sigma_over_t',
                     'Re_tG','Im_tG','tA','tA_lambda_half','tA_lambda_double'])
        for k in range(len(w)):
            cw.writerow([w[k],w[k]*t_ev,S[k].real,S[k].imag,G[k].real,G[k].imag,A[k],
                       alt_curves.get('lambda_half',np.full_like(w,np.nan))[k],
                       alt_curves.get('lambda_double',np.full_like(w,np.nan))[k]])
    return {'case':case,'U':U,'n':n_dmft,'w':w,'S':S,'A':A,'alt':alt_curves,'quality':q}


def figure(rows, out, kind, limit, dpi):
    fig, axs = plt.subplots(2,2,figsize=(10.0,7.1),sharex=True,sharey=True)
    xlo,xhi = (-limit,limit) if limit is not None else (min(r['w'][0] for r in rows),max(r['w'][-1] for r in rows))
    for idx,(ax,r) in enumerate(zip(axs.flat,rows)):
        w = r['w']
        m = (w>=xlo)&(w<=xhi)
        if kind=='sigma':
            ax.plot(w[m],r['S'].real[m],color='#1769aa',lw=1.65,label=r'$\mathrm{Re}\,\Sigma^R/t$')
            ax.plot(w[m],r['S'].imag[m],color='#c33b3b',lw=1.65,ls='--',label=r'$\mathrm{Im}\,\Sigma^R/t$')
            ylabel=r'$\Sigma^R(\omega)/t$'
        else:
            ax.plot(w[m],r['A'][m],color='#176e56',lw=1.8,label='Baseline')
            if kind=='sensitivity':
                for key,ls,cl in [('lambda_half','--','#bc4236'),('lambda_double',':','#60538c')]:
                    if key in r['alt']:
                        ax.plot(w[m],r['alt'][key][m],ls=ls,color=cl,lw=1.3,label=key.replace('_',' '))
            ylabel=r'$t A(\omega)$'
            ax.set_ylim(bottom=0)
        ax.axvline(0,ls=':',lw=.8,color='0.45')
        if kind=='sigma': ax.axhline(0,ls=':',lw=.7,color='0.5')
        ax.set_xlim(xlo,xhi)
        ax.set_title(rf"({'abcd'[idx]}) $U/t={r['U']},\ \langle n\rangle={r['n']:.4f}$",loc='left',fontsize=12)
        ax.tick_params(which='both',direction='in',top=True,right=True,labelleft=True)
        ax.minorticks_on()
        if idx>=2: ax.set_xlabel(r'$\omega/t$')
        if idx%2==0: ax.set_ylabel(ylabel)
    axs.flat[0].legend(frameon=False,fontsize=9)
    titles={'sigma':'Real-frequency self-energy (preliminary continuation)',
            'spectral':'Bethe-lattice local spectral function (preliminary)',
            'sensitivity':'Spectral function: sensitivity to continuation regularization'}
    fig.suptitle(titles[kind],fontsize=15,y=.985)
    fig.text(.5,.948,r'$\beta t=25$, per-spin $tA(\omega)$; finite broadening $\eta/t=0.15$',ha='center',fontsize=10)
    fig.subplots_adjust(top=.87,bottom=.1,left=.11,right=.98,hspace=.29,wspace=.18)
    basename=f'{kind}_four_cases_'+('zoom' if limit is not None else 'full')
    for ext in ('pdf','png'):
        fig.savefig(out/(basename+'.'+ext),dpi=dpi,facecolor='white')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('cthyb_test_results'))
    p.add_argument('--continuation-dir',type=Path)
    p.add_argument('--out-dir',type=Path)
    p.add_argument('--zoom',type=float,default=5)
    p.add_argument('--dpi',type=int,default=400)
    args=p.parse_args()
    if args.zoom<=0:p.error('--zoom must be > 0')
    root=args.root.expanduser().resolve()
    cont=(args.continuation_dir or root/'sigma_real_frequency_continuation').resolve()
    out=(args.out_dir or cont/'figures_Spectral_validated').resolve()
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':11,'axes.linewidth':1.1,'pdf.fonttype':42})
    rows=[checks_for_case(root,cont,*c,out) for c in CASES]
    for kind in ('sigma','spectral','sensitivity'):
        figure(rows,out,kind,None,args.dpi)
        figure(rows,out,kind,args.zoom,args.dpi)
    with (out/'summary_checks.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['case','A_min','ImSigma_max','integral_A_window','n_from_A_window','n_DMFT','delta_n','A_fermi','RMSE_Matsubara'])
        for r in rows:
            q=r['quality']
            writer.writerow([q['case'],q['min_A_times_t'],q['maximum_Im_Sigma_over_t'],q['integrated_A_on_saved_window'],q['n_from_truncated_broadened_A'],q['dmft_filling'],q['n_from_A_minus_DMFT'],q['A_at_fermi_times_t'],q['continuation_backfit_rmse_over_t']])
            print(f"{q['case']}: min(tA)={q['min_A_times_t']:.3g}; area(window)={q['integrated_A_on_saved_window']:.5f}; "
                  f"n(A; window,eta)={q['n_from_truncated_broadened_A']:.5f} vs n(DMFT)={q['dmft_filling']:.5f}; "
                  f"max ImSigma={q['maximum_Im_Sigma_over_t']:.3g}")
    print('Saved results to',out)
    print('CAUTION: Passes of causality/positivity do not validate real-frequency peak positions.')

if __name__=='__main__':
    main()

