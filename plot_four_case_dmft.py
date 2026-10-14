#!/usr/bin/env python3
"""Compare 4 saved TRIQS CT-HYB Bethe-DMFT adaptive_v2 calculations.

Reads ONLY numerical CSV files and metadata.json, does not import TRIQS,
and does not rerun Monte Carlo. Outputs high-resolution PNG, vector PDF,
combined multipage PDF, and a summary CSV.

Expected directory layout:
  ROOT/U8_n0p7_beta25_adaptive_v2/U8p000_n0p700_beta25p00/{metadata.json,matsubara_positive.csv,imaginary_time.csv}
  ROOT/U8_n0p9_beta25_adaptive_v2/U8p000_n0p900_beta25p00/...
  ROOT/U10_n0p7_beta25_adaptive_v2/U10p000_n0p700_beta25p00/...
  ROOT/U10_n0p9_beta25_adaptive_v2/U10p000_n0p900_beta25p00/...

All sigma curves labelled Sigma/t, all frequency axes omega_n/t; tG is
used for Matsubara Green function, and G(tau) for imaginary-time data.

Science notes:
 * Main comparison subtracts Sigma_infinity = U*n/2 to remove Hartree shift.
 * Full positive Matsubara axis has 256 independent-frequency positions.
 * --signed reconstructs the negative-frequency half by complex conjugation;
   those are NOT an additional independent QMC measurement.
 * G(tau) is DIRECTLY MEASURED data, not corrected to force its endpoints
   to agree with the density matrix. The discrepancy is made explicit.
 * Shading uses reported +/- 1 SEM, not total systematic error.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np

FILLINGS = (0.7, 0.9)
INTERACTIONS = (8, 10)
COL_N = {0.7: '#1764AA', 0.9: '#C65A25'}
COL_U = {8: '#147D7B', 10: '#793FA3'}


def arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('cthyb_test_results'),
                   help='Parent folder containing all four adaptive_v2 run directories')
    p.add_argument('--out-dir', type=Path, default=None,
                   help='Figure output location (default: ROOT/four_case_comparison)')
    p.add_argument('--dpi', type=int, default=350, help='PNG resolution')
    p.add_argument('--main-omega-max', type=float, default=12.0,
                   help='Upper omega/t for primary low/intermediate self-energy plot')
    p.add_argument('--low-omega-max', type=float, default=4.0,
                   help='Upper omega/t for low-frequency zoom plots')
    p.add_argument('--signed', action='store_true',
                   help='Extra self-energy plot with reconstructed 512 signed Matsubara points')
    p.add_argument('--usetex', action='store_true',
                   help='Use full external LaTeX rendering; requires latex and dvipng installed')
    p.add_argument('--no-errorbands', action='store_true',
                   help='Omit replica-SEM envelopes on low-frequency plots')
    return p.parse_args()


@dataclass
class Case:
    U: int
    target: float
    meta: dict
    mats: np.ndarray
    tau: np.ndarray
    case_dir: Path

    @property
    def mu(self): return float(self.meta['mu_over_t'])

    @property
    def n(self): return float(self.meta['final_filling_mean'])

    @property
    def sigma_infty(self):
        return float(self.meta.get('sigma_infinity_mean_over_t', self.U*self.n/2))

    def curve(self, key):
        return np.asarray(self.mats[key], dtype=float)

    def time_curve(self, key):
        return np.asarray(self.tau[key], dtype=float)


def resolve_case(root: Path, U: int, n: float) -> Path:
    run_dir = root / f'U{U}_n{str(n).replace(".","p")}_beta25_adaptive_v2'
    if not run_dir.is_dir():
        raise FileNotFoundError(f'Missing run folder: {run_dir}')
    found = sorted(run_dir.glob('*/metadata.json'))
    if (run_dir/'metadata.json').is_file():
        found = [run_dir/'metadata.json']
    if len(found) != 1:
        raise FileNotFoundError(f'Expected exactly one metadata.json below {run_dir}; found {len(found)}')
    return found[0].parent


def read_case(root, U, n):
    path = resolve_case(root, U, n)
    needed = ['metadata.json', 'matsubara_positive.csv', 'imaginary_time.csv']
    for fname in needed:
        if not (path/fname).is_file():
            raise FileNotFoundError(f'Missing {path/fname}')
    meta = json.loads((path/'metadata.json').read_text())
    mats = np.genfromtxt(path/'matsubara_positive.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
    tau = np.genfromtxt(path/'imaginary_time.csv', delimiter=',', names=True, dtype=None, encoding='utf-8')
    if abs(float(meta['U_over_t'])-U)>1e-8 or abs(float(meta['target_filling'])-n)>1e-8:
        raise ValueError(f'Incorrect physics metadata in {path}: U/t={meta["U_over_t"]}, target n={meta["target_filling"]}')
    if not meta.get('dmft_converged'):
        raise ValueError(f'DMFT NOT converged: {path}')
    if not meta.get('final_quality_passed'):
        raise ValueError(f'Final quality check NOT passed: {path}')
    for name in ['omega_over_t','Re_G_times_t','Im_G_times_t','Re_Sigma_over_t','Im_Sigma_over_t']:
        if name not in mats.dtype.names: raise KeyError(f'{name} missing in {path}')
    for name in ['tau_times_t','Re_G_tau']:
        if name not in tau.dtype.names: raise KeyError(f'{name} missing in {path}')
    if not np.all(np.diff(mats['omega_over_t'])>0): raise ValueError(f'Matsubara frequencies not sorted in {path}')
    if not np.all(np.diff(tau['tau_times_t'])>0): raise ValueError(f'Imaginary-time mesh not sorted in {path}')
    if len(mats)!=256 or len(tau)!=2049:
        print(f'NOTE: U={U}, n={n}, Matsubara points={len(mats)}, tau points={len(tau)}')
    return Case(U,n,meta,mats,tau,path)


def set_style(use_tex):
    if use_tex:
        missing = [v for v in ('latex','dvipng') if not shutil.which(v)]
        if missing: raise SystemExit(f'--usetex needs {missing}; omit --usetex for built-in STIX LaTeX-style math')
    plt.rcParams.update({
        'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif','serif'],
        'mathtext.fontset':'stix', 'text.usetex':use_tex,
        'font.size':11.5,'axes.titlesize':13,'axes.labelsize':12,
        'xtick.labelsize':10.5,'ytick.labelsize':10.5,
        'legend.fontsize':10.5,'figure.titlesize':17,
        'axes.linewidth':0.9,'lines.linewidth':2.0,
        'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,
        'savefig.facecolor':'white'
    })


def decorate(ax):
    ax.grid(True, which='major', lw=0.65, color='#999999', alpha=0.16)
    ax.tick_params(which='both', direction='in', width=0.8)
    ax.tick_params(which='major', length=4.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='minor', length=2.4)


def heading(fig, title, cases, note='', top=0.84):
    beta=float(cases[(8,0.7)].meta['beta_t'])
    tev=float(cases[(8,0.7)].meta['t_eV'])
    T=float(cases[(8,0.7)].meta['temperature_K'])
    fig.suptitle(title, y=0.985, fontsize=17.5, fontweight='bold')
    fig.text(0.5,0.947,
             rf'Paramagnetic Bethe lattice  $\bullet$  $\beta t={beta:g}$  $\bullet$  $t={tev:g}\,\mathrm{{eV}}$'
             rf'  $\bullet$  $T={T:.2f}\,\mathrm{{K}}$  $\bullet$  CT-HYB / DMFT',
             ha='center', va='center', fontsize=11.5)
    if note:
        fig.text(0.5,0.021,note,ha='center',va='center',fontsize=9.2,color='#4d5966')
    return fig


def save(fig, out, name, pdfbook, dpi):
    fig.savefig(out/(name+'.png'), dpi=dpi, bbox_inches='tight', pad_inches=0.12)
    fig.savefig(out/(name+'.pdf'), bbox_inches='tight', pad_inches=0.12)
    pdfbook.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print('  saved',name+'.png / .pdf',flush=True)


def color_lines(ax, cases, U, re_or_im, omega_limit=None, hartree=True, bands=False):
    for n in FILLINGS:
        c=cases[(U,n)]
        x=c.curve('omega_over_t')
        if re_or_im=='real':
            y=c.curve('Re_Sigma_over_t') - (c.sigma_infty if hartree else 0)
            ekey='SEM_Re_Sigma_over_t'
        else:
            y=-c.curve('Im_Sigma_over_t')
            ekey='SEM_Im_Sigma_over_t'
        kw=dict(lw=1.9, c=COL_N[n], label=rf'$\langle n\rangle={c.n:.4f}$')
        ax.plot(x,y,**kw)
        if bands and ekey in c.mats.dtype.names:
            sem=c.curve(ekey)
            ax.fill_between(x,y-sem,y+sem,color=COL_N[n],alpha=0.16,lw=0)
    ax.set_xlim(0,omega_limit or cases[(U,0.7)].curve('omega_over_t')[-1])
    decorate(ax)


def fig_sigma_fixed_U(cases, out, pdf, dpi, maxomega, bands, name, full=False):
    fig,axs=plt.subplots(2,2,figsize=(14.4,9.2),sharex='col')
    for i,U in enumerate(INTERACTIONS):
        color_lines(axs[i,0],cases,U,'real',maxomega,hartree=True,bands=bands)
        color_lines(axs[i,1],cases,U,'imag',maxomega,hartree=True,bands=bands)
        axs[i,0].set_ylabel(r'${\rm Re}[\Sigma(i\omega_n)-\Sigma_\infty]/t$')
        axs[i,1].set_ylabel(r'$-{\rm Im}\,\Sigma(i\omega_n)/t$')
        for j in (0,1):
            axs[i,j].set_title(rf'$U/t={U}$'+('   (30% vs 10% hole doping)' if j==0 else ''))
            if i==1: axs[i,j].set_xlabel(r'$\omega_n/t$')
    h,l=axs[0,1].get_legend_handles_labels()
    fig.legend(h,l,loc='upper center',bbox_to_anchor=(0.5,0.900),ncol=2,frameon=False)
    heading(fig,'Self-energy: effect of doping at fixed interaction',cases,
            note=(r'Full positive Matsubara grid; tail fitted above $\omega_n/t\approx12$.' if full else
                  r'No smoothing; Hartree term $\Sigma_\infty=Un/2$ removed from real part. Shading = $\pm1$ replica SEM.' if bands else
                  r'No smoothing; Hartree term $\Sigma_\infty=Un/2$ removed from real part.'))
    fig.subplots_adjust(left=0.105,right=0.96,top=0.835,bottom=0.10,hspace=0.32,wspace=0.24)
    save(fig,out,name,pdf,dpi)


def fig_sigma_full_not_subtracted(cases,out,pdf,dpi,maxomega):
    fig,axs=plt.subplots(2,2,figsize=(14.4,9.1),sharex=True)
    for i,U in enumerate(INTERACTIONS):
        color_lines(axs[i,0],cases,U,'real',maxomega,hartree=False)
        color_lines(axs[i,1],cases,U,'imag',maxomega,hartree=False)
        axs[i,0].set(ylabel=r'${\rm Re}\,\Sigma(i\omega_n)/t$',title=rf'$U/t={U}$: full real self-energy')
        axs[i,1].set(ylabel=r'$-{\rm Im}\,\Sigma(i\omega_n)/t$',title=rf'$U/t={U}$: imaginary self-energy')
        for n in FILLINGS: axs[i,0].axhline(cases[(U,n)].sigma_infty,color=COL_N[n],ls=':',lw=1,alpha=.6)
        if i==1:
            for ax in axs[i]:ax.set_xlabel(r'$\omega_n/t$')
    h,l=axs[0,1].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.90),ncol=2,frameon=False)
    heading(fig,'Complete self-energy (Hartree contribution retained)',cases,
            note=r'Dotted levels: $\Sigma_\infty=Un/2$; all 256 positive Matsubara frequencies; no smoothing.')
    fig.subplots_adjust(left=.105,right=.96,top=.835,bottom=.10,hspace=.32,wspace=.24)
    save(fig,out,'04_sigma_full_hartree_retained',pdf,dpi)


def fig_sigma_fixed_n(cases,out,pdf,dpi,maxomega):
    fig,axs=plt.subplots(2,2,figsize=(14.4,9.2),sharex=True)
    for i,n in enumerate(FILLINGS):
        for j,part in enumerate(('real','imag')):
            ax=axs[i,j]
            for U in INTERACTIONS:
                c=cases[(U,n)];x=c.curve('omega_over_t')
                y=(c.curve('Re_Sigma_over_t')-c.sigma_infty) if part=='real' else -c.curve('Im_Sigma_over_t')
                ax.plot(x,y,c=COL_U[U],lw=2,label=rf'$U/t={U}$')
            ax.set(xlim=(0,maxomega), ylabel=(r'${\rm Re}[\Sigma-\Sigma_\infty]/t$' if part=='real' else r'$-{\rm Im}\,\Sigma/t$'))
            if i==1:ax.set_xlabel(r'$\omega_n/t$')
            ax.set_title(rf'$\langle n\rangle\approx{n}$'+('  (real)' if j==0 else '  (imaginary)'))
            decorate(ax)
    h,l=axs[0,1].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.90),ncol=2,frameon=False)
    heading(fig,'Self-energy: effect of interaction at fixed filling',cases,
            note=r'For a fixed filling, compare $U/t=8$ and 10 directly; Hartree shift removed from real part.')
    fig.subplots_adjust(left=.11,right=.96,top=.835,bottom=.10,hspace=.32,wspace=.24)
    save(fig,out,'03_sigma_fixed_filling',pdf,dpi)


def fig_gtau(cases,out,pdf,dpi):
    fig,axs=plt.subplots(1,2,figsize=(14.4,6.0),sharex=True,sharey=True)
    for ax,U in zip(axs,INTERACTIONS):
        for n in FILLINGS:
            c=cases[(U,n)]
            x=c.time_curve('tau_times_t')/float(c.meta['beta_t'])
            y=c.time_curve('Re_G_tau')
            ax.plot(x,y,color=COL_N[n],lw=2.0,label=rf'$\langle n\rangle={c.n:.4f}$')
        ax.set(xlim=(0,1),xlabel=r'$\tau/\beta$',title=rf'$U/t={U}$')
        decorate(ax)
    axs[0].set_ylabel(r'$G(\tau)$ (spin averaged)')
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.885),ncol=2,frameon=False)
    heading(fig,'Interacting Green function on the complete imaginary-time grid',cases,
            note=r'Direct CT-HYB $G(\tau)$ (2049 grid points); measured endpoint density differs from the density-matrix estimate.')
    fig.subplots_adjust(left=.09,right=.97,top=.79,bottom=.17,wspace=.12)
    save(fig,out,'02_G_tau_full',pdf,dpi)


def fig_Giw(cases,out,pdf,dpi,maxomega,name):
    fig,axs=plt.subplots(2,2,figsize=(14.4,9.2),sharex=True)
    for i,U in enumerate(INTERACTIONS):
        for j,(key,label) in enumerate((('Re_G_times_t',r'${\rm Re}[tG(i\omega_n)]$'),('Im_G_times_t',r'${\rm Im}[tG(i\omega_n)]$'))):
            ax=axs[i,j]
            for n in FILLINGS:
                c=cases[(U,n)]
                ax.plot(c.curve('omega_over_t'),c.curve(key),color=COL_N[n],lw=1.9,label=rf'$\langle n\rangle={c.n:.4f}$')
            ax.set(xlim=(0,maxomega),ylabel=label,title=rf'$U/t={U}$'+('   real component' if j==0 else '   imaginary component'))
            if i==1:ax.set_xlabel(r'$\omega_n/t$')
            decorate(ax)
    h,l=axs[0,1].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.90),ncol=2,frameon=False)
    heading(fig,'Local interacting Green function: comparison of four cases',cases,
            note=r'$tG(i\omega_n)$ plotted without fitting beyond the data stored by TRIQS; no numerical smoothing.')
    fig.subplots_adjust(left=.10,right=.96,top=.835,bottom=.10,hspace=.32,wspace=.24)
    save(fig,out,name,pdf,dpi)


def fig_sigma_raw(cases,out,pdf,dpi):
    fig,axs=plt.subplots(2,2,figsize=(14.4,9.0),sharex=True)
    for i,U in enumerate(INTERACTIONS):
        for j,n in enumerate(FILLINGS):
            c=cases[(U,n)];ax=axs[i,j]
            if 'Re_Sigma_raw_over_t' not in c.mats.dtype.names:
                ax.text(.5,.5,'Raw self-energy not saved',transform=ax.transAxes,ha='center');continue
            x=c.curve('omega_over_t')
            raw=c.curve('Im_Sigma_raw_over_t');fit=c.curve('Im_Sigma_over_t')
            ax.plot(x,raw,'.',ms=3,color='#777777',alpha=.60,label='Raw Dyson')
            ax.plot(x,fit,'-',color=COL_N[n],lw=1.8,label='Tail fitted')
            ax.axvspan(12,20,color='gray',alpha=.07,label='Fit window' if i==0 and j==0 else None)
            ax.axvline(12,color='gray',ls=':',lw=1)
            ax.axvline(20,color='gray',ls=':',lw=1)
            ax.set(xlim=(0,24),title=rf'$U/t={U}$, $\langle n\rangle\approx{n}$',ylabel=r'${\rm Im}\,\Sigma(i\omega_n)/t$')
            if i==1:ax.set_xlabel(r'$\omega_n/t$')
            decorate(ax)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.90),ncol=3,frameon=False)
    heading(fig,'Tail-fit diagnostic: raw versus processed self-energy',cases,
            note=r'Raw QMC/Dyson estimates (points) and tail-fitted curves; do not interpret high-frequency raw noise as physics.')
    fig.subplots_adjust(left=.10,right=.96,top=.835,bottom=.10,hspace=.32,wspace=.25)
    save(fig,out,'06_sigma_raw_vs_fit',pdf,dpi)


def fig_diagnostics(cases,out,pdf,dpi):
    fig,axs=plt.subplots(1,3,figsize=(15,5.9))
    xs=np.arange(4);ordered=[cases[(U,n)] for U in INTERACTIONS for n in FILLINGS]
    labels=[rf'$U={c.U},\ n={c.target}$' for c in ordered]
    cols=[COL_N[c.target] for c in ordered]
    err=np.array([abs(c.n-c.target) for c in ordered]);res=np.array([float(c.meta['final_replica_bethe_residual']) for c in ordered]);doubl=np.array([float(c.meta['double_occupancy_mean']) for c in ordered]);sems=np.array([float(c.meta.get('final_filling_sem',0)) for c in ordered]);
    axs[0].bar(xs,err,color=cols,alpha=.82);axs[0].axhline(1e-3,color='black',ls='--',lw=1.2,label='Density tolerance');axs[0].set(ylabel=r'$|n_{\rm final}-n_{\rm target}|$',title='Target filling',xticks=xs,xticklabels=labels);axs[0].legend(frameon=False,fontsize=9)
    axs[1].bar(xs,res,color=cols,alpha=.82);axs[1].axhline(2e-3,color='black',ls='--',lw=1.2,label='DMFT tolerance');axs[1].set(ylabel=r'$R_\Delta$',title='Final Bethe residual',xticks=xs,xticklabels=labels);axs[1].legend(frameon=False,fontsize=9)
    axs[2].bar(xs,doubl,color=cols,alpha=.82);axs[2].set(ylabel=r'$D=\langle n_\uparrow n_\downarrow\rangle$',title='Double occupancy',xticks=xs,xticklabels=labels)
    for ax in axs:
        ax.tick_params(axis='x',labelrotation=27)
        decorate(ax)
    heading(fig,'Numerical quality and correlation diagnostics',cases,
            note=r'All four cases pass programmed tolerances; replica SEM excludes systematic bath and chemical-potential uncertainty.')
    fig.subplots_adjust(left=.075,right=.985,top=.79,bottom=.22,wspace=.34)
    save(fig,out,'07_quality_summary',pdf,dpi)


def fig_density_consistency(cases,out,pdf,dpi):
    fig,ax=plt.subplots(figsize=(12.6,5.8))
    cases_l=[cases[(U,n)] for U in INTERACTIONS for n in FILLINGS]
    x=np.arange(len(cases_l))
    dm=np.array([c.n for c in cases_l]);gp=np.array([float(c.meta.get('density_from_Gtau_beta_endpoint',np.nan)) for c in cases_l]);err=gp-dm
    ax.bar(x,err,color=[COL_N[c.target] for c in cases_l],alpha=.86)
    ax.axhline(0,color='black',lw=.9)
    ax.set(xticks=x,xticklabels=[rf'$U/t={c.U}$, $n={c.target}$' for c in cases_l],
           ylabel=r'$n_{G(\beta^-)} - n_{\mathrm{density\ matrix}}$',title='Endpoint density estimator discrepancy')
    decorate(ax)
    heading(fig,'Diagnostic: direct G(tau) endpoint density cross-check',cases,
            note=r'This is an unresolved estimator/reconstruction discrepancy, not evidence that the Matsubara density is incorrect.')
    fig.subplots_adjust(left=.12,right=.96,top=.81,bottom=.18)
    save(fig,out,'08_density_endpoint_crosscheck',pdf,dpi)


def fig_signed(cases,out,pdf,dpi):
    fig,axs=plt.subplots(2,2,figsize=(14.4,9.2),sharex=True)
    for i,U in enumerate(INTERACTIONS):
        for j,comp in enumerate(('real','imag')):
            ax=axs[i,j]
            for n in FILLINGS:
                c=cases[(U,n)];x=c.curve('omega_over_t');rr=c.curve('Re_Sigma_over_t')-c.sigma_infty;ii=c.curve('Im_Sigma_over_t')
                xx=np.r_[-x[::-1],x]; yy=np.r_[rr[::-1],rr] if comp=='real' else np.r_[-ii[::-1],ii]
                ax.plot(xx,yy,color=COL_N[n],lw=1.5,label=rf'$n\approx{n}$')
            ax.axvline(0,ls=':',color='#999999',lw=1)
            ax.set(ylabel=(r'${\rm Re}[\Sigma-\Sigma_\infty]/t$' if comp=='real' else r'${\rm Im}\,\Sigma/t$'),
                   title=rf'$U/t={U}$, '+comp)
            if i==1:ax.set_xlabel(r'$\omega_n/t$ (signed fermionic frequencies)')
            decorate(ax)
    h,l=axs[0,1].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.90),ncol=2,frameon=False)
    heading(fig,'Self-energy over the full signed 512-point Matsubara mesh',cases,
            note=r'Negative-frequency data reconstructed via $\Sigma(-i\omega_n)=\Sigma(i\omega_n)^*$; not additional independent QMC samples.')
    fig.subplots_adjust(left=.105,right=.96,top=.835,bottom=.11,hspace=.32,wspace=.24)
    save(fig,out,'09_signed_512_sigma',pdf,dpi)


def save_metrics(cases,out):
    cols=['U_over_t','target_n','final_n','density_error','final_n_sem','mu_over_t','mu_eV',
          'double_occupancy','bethe_residual','avg_sign','autocorr_cycles','sigma_infty_over_t',
          'Im_Sigma_at_first_iw_over_t','Re_Sigma_at_first_iw_over_t','n_from_Gtau_endpoint','Gtau_endpoint_difference']
    with (out/'four_case_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for U in INTERACTIONS:
            for n in FILLINGS:
                c=cases[(U,n)]
                row={'U_over_t':U,'target_n':n,'final_n':c.n,'density_error':c.n-n,
                  'final_n_sem':c.meta.get('final_filling_sem'),'mu_over_t':c.mu,
                  'mu_eV':c.meta.get('mu_eV'),'double_occupancy':c.meta.get('double_occupancy_mean'),
                  'bethe_residual':c.meta.get('final_replica_bethe_residual'),
                  'avg_sign':c.meta.get('average_sign_mean'),'autocorr_cycles':c.meta.get('auto_corr_time_mean'),
                  'sigma_infty_over_t':c.sigma_infty,
                  'Im_Sigma_at_first_iw_over_t':c.curve('Im_Sigma_over_t')[0],
                  'Re_Sigma_at_first_iw_over_t':c.curve('Re_Sigma_over_t')[0],
                  'n_from_Gtau_endpoint':c.meta.get('density_from_Gtau_beta_endpoint'),
                  'Gtau_endpoint_difference':c.meta.get('density_matrix_minus_Gtau_endpoint')}
                w.writerow(row)


def main():
    a=arg_parser()
    root=a.root.expanduser().resolve()
    out=(a.out_dir or root/'four_case_comparison').expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    set_style(a.usetex)
    cases={(U,n):read_case(root,U,n) for U in INTERACTIONS for n in FILLINGS}
    for U in INTERACTIONS:
        for n in FILLINGS:
            c=cases[(U,n)]
            print(f'Loaded U/t={U:2d}, target={n}, measured={c.n:.7f}, mu/t={c.mu:.7f}, '
                  f'Residual={c.meta["final_replica_bethe_residual"]:.5g} from {c.case_dir}',flush=True)
    beta={float(c.meta['beta_t']) for c in cases.values()}
    tevs={float(c.meta['t_eV']) for c in cases.values()}
    if len(beta)!=1 or len(tevs)!=1:
        raise ValueError('The four cases must have identical beta*t and t_eV for direct comparisons')
    save_metrics(cases,out)
    pdfname=out/'all_four_case_figures.pdf'
    with PdfPages(pdfname) as pdf:
        fig_sigma_fixed_U(cases,out,pdf,a.dpi,a.main_omega_max,not a.no_errorbands,'01_sigma_doping_comparison')
        fig_gtau(cases,out,pdf,a.dpi)
        fig_sigma_fixed_n(cases,out,pdf,a.dpi,a.main_omega_max)
        xfull=min(c.curve('omega_over_t')[-1] for c in cases.values())
        fig_sigma_full_not_subtracted(cases,out,pdf,a.dpi,xfull)
        fig_sigma_fixed_U(cases,out,pdf,a.dpi,xfull,False,'05_sigma_full_positive',full=True)
        fig_Giw(cases,out,pdf,a.dpi,xfull,'05b_Giw_full_positive')
        fig_Giw(cases,out,pdf,a.dpi,a.low_omega_max,'05c_Giw_low_omega')
        fig_sigma_raw(cases,out,pdf,a.dpi)
        fig_diagnostics(cases,out,pdf,a.dpi)
        fig_density_consistency(cases,out,pdf,a.dpi)
        if a.signed:fig_signed(cases,out,pdf,a.dpi)
    print('\nAll finished. Key files:')
    print(' ',out/'01_sigma_doping_comparison.pdf')
    print(' ',out/'02_G_tau_full.pdf')
    print(' ',out/'03_sigma_fixed_filling.pdf')
    print(' ',pdfname)
    print(' ',out/'four_case_metrics.csv')


if __name__=='__main__':
    main()

