#!/usr/bin/env python3
"""Publication-style post-processing of existing Bethe-DMFT/CT-HYB NPZ results.

Only requires numpy + matplotlib. No TRIQS imports and no Monte Carlo reruns.

Use --case-dir for the case directory containing dmft_result.npz and metadata.json,
or for its parent run directory if exactly one case is present.

By default plots ALL positive Matsubara frequencies (usually n_iw=256), not merely
--plot-n-iw used by the original DMFT calculation. --include-negative adds the
complete saved positive/negative Matsubara mesh in a separate figure.

Units: the underlying simulation has t_internal = 1. G(iwn) is plotted as tG,
Sigma(iwn) as Sigma/t, Delta(iwn) as Delta/t; imaginary-time G(tau) is
unitless, Delta(tau) and Sigma_dynamic(tau) are plotted in units of t^2.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--case-dir', type=Path, default=Path('.'), help='Directory with NPZ and metadata, or parent containing one case')
    p.add_argument('--npz', type=Path, default=None, help='Explicit dmft_result.npz path')
    p.add_argument('--metadata', type=Path, default=None, help='Explicit metadata.json path')
    p.add_argument('--out-dir', type=Path, default=None, help='Defaults to <case-dir>/publication_plots')
    p.add_argument('--dpi', type=int, default=300, help='Raster PNG resolution; PDF is vector')
    p.add_argument('--zoom-n-iw', type=int, default=24, help='Number of lowest positive Matsubara points in close-up')
    p.add_argument('--include-negative', action='store_true', help='Also plot ALL negative and positive Matsubara points')
    p.add_argument('--usetex', action='store_true', help='Use external LaTeX if installed (off by default for cluster portability)')
    return p.parse_args()


def locate(a):
    if a.npz is not None:
        npz = a.npz.expanduser().resolve()
        case = npz.parent
    else:
        case = a.case_dir.expanduser().resolve()
        npz = case / 'dmft_result.npz'
        if not npz.exists():
            found = sorted(case.glob('*/dmft_result.npz'))
            if len(found) != 1:
                raise FileNotFoundError(
                    f'Expected dmft_result.npz in {case}, or exactly one immediate child case; '
                    f'found {len(found)}. Pass --case-dir <case-directory>.'
                )
            npz = found[0]
            case = npz.parent
    metadata = a.metadata.expanduser().resolve() if a.metadata is not None else case / 'metadata.json'
    if not npz.is_file() or not metadata.is_file():
        raise FileNotFoundError(f'Required inputs missing: {npz}, {metadata}')
    out = a.out_dir.expanduser().resolve() if a.out_dir is not None else case / 'publication_plots'
    return npz, metadata, out


def theme(usetex):
    if usetex:
        required = ('latex', 'dvipng')
        missing = [x for x in required if not shutil.which(x)]
        if missing:
            raise SystemExit(f'--usetex requested, missing external tools: {missing}. Omit --usetex to use TeX-style STIX math fonts.')
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix', 'text.usetex': usetex,
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9.3, 'figure.titlesize': 15,
        'axes.linewidth': 0.9, 'lines.linewidth': 1.7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
        'axes.grid': False,
    })


def fmt(value, precision=5, default='n/a'):
    try:
        f = float(value)
        return f'{f:.{precision}f}' if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def header(meta, arrays):
    u = fmt(meta.get('U_over_t'), 1)
    b = fmt(meta.get('beta_t'), 1)
    t_ev = meta.get('t_eV')
    n = meta.get('final_filling_mean')
    mu = meta.get('mu_over_t')
    T = meta.get('temperature_K')
    t_label = f'$t={float(t_ev):g}\\,\\mathrm{{eV}}$' if t_ev is not None else r'$t=1$ (internal units)'
    line1 = (rf'$U/t={u}$  $\cdot$  $\beta t={b}$  $\cdot$  {t_label}  $\cdot$ '
             + (rf'$T={float(T):.2f}\,\mathrm{{K}}$' if T is not None else r'$T/t=1/(\beta t)$'))
    line2 = (rf'$\langle n\rangle={fmt(n,6)}$ (target {fmt(meta.get("target_filling"),4)})'
             rf'  $\cdot$  $\mu/t={fmt(mu,6)}$'
             rf'  $\cdot$  $D={fmt(meta.get("double_occupancy_mean"),5)}$'
             rf'  $\cdot$  $R_\Delta={fmt(meta.get("final_replica_bethe_residual"),6)}$')
    line3 = (rf'$\langle\mathrm{{sign}}\rangle={fmt(meta.get("average_sign_mean"),3)}$'
             rf'  $\cdot$  $\tau_{{\mathrm{{auto}}}}={fmt(meta.get("auto_corr_time_mean"),2)}$ cycles'
             rf'  $\cdot$  {len(arrays.get("replica_fillings", []))} final replicas'
             rf'  $\cdot$  Quality: {"PASS" if meta.get("final_quality_passed") else "CHECK"}')
    return line1, line2, line3


def decorate(fig, name, meta, arrays, bottom_note=None):
    line1, line2, line3 = header(meta, arrays)
    fig.suptitle(name, x=0.5, y=0.993, fontsize=16, fontweight='semibold')
    fig.text(0.5, 0.956, line1, ha='center', va='center', fontsize=11.5)
    fig.text(0.5, 0.926, line2, ha='center', va='center', fontsize=10.7)
    fig.text(0.5, 0.897, line3, ha='center', va='center', fontsize=9.7)
    if bottom_note:
        fig.text(0.5, 0.017, bottom_note, ha='center', va='center', fontsize=8.5, alpha=0.78)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.82, bottom=0.11 if bottom_note else 0.085,
                        hspace=0.33, wspace=0.24)


def axis_style(ax, x_minor=True):
    ax.grid(which='major', alpha=0.16, linewidth=0.65)
    ax.tick_params(direction='in', length=4, width=0.8)
    if x_minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='minor', direction='in', length=2.2)


def plot_sem(ax, x, y, sem, c=None):
    if sem is not None:
        s = np.asarray(sem, dtype=float)
        if s.shape == y.shape and np.any(np.isfinite(s)):
            ax.fill_between(x, y-s, y+s, alpha=0.12, color=c, linewidth=0)


def save_figure(fig, folder, stem, pdf_pages, dpi):
    png = folder / f'{stem}.png'
    pdf = folder / f'{stem}.pdf'
    fig.savefig(png, dpi=dpi, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    pdf_pages.savefig(fig)
    plt.close(fig)
    print(f'  {png.name}  +  {pdf.name}', flush=True)


def positive_index(data):
    if 'iw_over_t' not in data:
        raise KeyError('NPZ missing iw_over_t')
    iw = np.asarray(data['iw_over_t'])
    omega = iw.imag if np.iscomplexobj(iw) else np.asarray(iw, dtype=float)
    pos = np.where(omega > 0)[0]
    if not len(pos):
        raise ValueError('Matsubara mesh contains no positive frequencies')
    order = pos[np.argsort(omega[pos])]
    return omega, order


def plot_matsubara_full(d, meta, indices, omega, out, pdf, dpi):
    x = omega[indices]
    sig = d['Sigma_iw'][indices]
    g = d['G_iw'][indices]
    sig_sem = d.get('Sigma_iw_sem')
    g_sem = d.get('G_iw_sem')
    fig, axs = plt.subplots(2, 2, figsize=(14.8, 9.5))
    sig_inf = float(np.real(d.get('Sigma_infinity_over_t', meta.get('sigma_infinity_mean_over_t', np.nan))))
    specs = [
        (axs[0,0], sig.real, sig_sem[indices].real if sig_sem is not None else None,
         r'$\mathrm{Re}\,\Sigma(i\omega_n)/t$', 'Real self-energy'),
        (axs[0,1], sig.imag, sig_sem[indices].imag if sig_sem is not None else None,
         r'$\mathrm{Im}\,\Sigma(i\omega_n)/t$', 'Imaginary self-energy'),
        (axs[1,0], g.real, g_sem[indices].real if g_sem is not None else None,
         r'$\mathrm{Re}\,[tG(i\omega_n)]$', 'Real Green function'),
        (axs[1,1], g.imag, g_sem[indices].imag if g_sem is not None else None,
         r'$\mathrm{Im}\,[tG(i\omega_n)]$', 'Imaginary Green function'),
    ]
    for ax, y, e, ylabel, title in specs:
        line, = ax.plot(x, y, lw=1.8)
        plot_sem(ax, x, y, e, line.get_color())
        ax.set(xlabel=r'$\omega_n/t$', ylabel=ylabel, title=title)
        axis_style(ax)
    if np.isfinite(sig_inf):
        axs[0,0].axhline(sig_inf, ls='--', lw=1.1, alpha=0.72, label=r'$\Sigma_\infty/t$')
        axs[0,0].legend(frameon=False, loc='best')
    decorate(fig, f'Full positive Matsubara grid  |  {len(x)} frequencies', meta, d,
             bottom_note=r'Solid curves: final-replica mean. Shaded envelopes: $\pm 1$ replica SEM (when available).')
    save_figure(fig, out, '01_matsubara_full', pdf, dpi)


def plot_tau_full(d, meta, out, pdf, dpi):
    tau = np.asarray(d['tau_times_t'], dtype=float)
    beta = float(meta['beta_t'])
    x = tau/beta
    vals = [
        ('G_tau', r'$G(\tau)$', 'Measured interacting Green function'),
        ('G0_tau', r'$\mathcal{G}_0(\tau)$', 'Weiss Green function'),
        ('Delta_tau', r'$\Delta(\tau)/t^2$', 'Hybridization function'),
        ('Sigma_dynamic_tau', r'$\Sigma_{\mathrm{dyn}}(\tau)/t^2$', 'Dynamic self-energy (Hartree removed)'),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(14.8, 9.5))
    for ax, (key, label, title) in zip(axs.flat, vals):
        if key not in d:
            ax.text(0.5, 0.5, 'Not saved in NPZ', transform=ax.transAxes, ha='center')
            ax.set_axis_off()
            continue
        y = np.asarray(d[key]).real
        line, = ax.plot(x, y, lw=1.75)
        if key=='G_tau' and 'G_tau_sem' in d:
            plot_sem(ax, x, y, np.asarray(d['G_tau_sem']).real, line.get_color())
        if key=='Sigma_dynamic_tau':
            # Same data, symlog view displays near-zero ringing and boundary peaks
            # without destroying information through smoothing or cropping.
            ax.set_yscale('symlog', linthresh=0.06, linscale=1.0)
            ax.axhline(0, ls=':', lw=0.9, alpha=0.5)
            ax.text(0.98, 0.06, 'symmetric-log y axis', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8, alpha=0.7)
        ax.set(xlabel=r'$\tau/\beta$', ylabel=label, title=title, xlim=(0,1))
        axis_style(ax)
    decorate(fig, f'Full imaginary-time grid  |  {len(x)} time points', meta, d,
             bottom_note=r'$\Sigma_{\mathrm{dyn}}(\tau)$ excludes the instantaneous Hartree term; it is not the full self-energy.')
    save_figure(fig, out, '02_imaginary_time_full', pdf, dpi)


def plot_sigma_fit(d, meta, indices, omega, out, pdf, dpi):
    if 'Sigma_iw_raw' not in d:
        return
    x = omega[indices]
    fit = d['Sigma_iw'][indices]
    raw = d['Sigma_iw_raw'][indices]
    settings = meta.get('settings') or {}
    lo = settings.get('fit_min_w_over_t', 12)
    hi = settings.get('fit_max_w_over_t', 20)
    fig, axs = plt.subplots(2, 1, figsize=(14.8, 9.5), sharex=True)
    for ax, value, lab in zip(axs, ('real', 'imag'),
                               (r'$\mathrm{Re}\,\Sigma(i\omega_n)/t$', r'$\mathrm{Im}\,\Sigma(i\omega_n)/t$')):
        r = getattr(raw, value)
        f = getattr(fit, value)
        # Plot all raw points (small, semi-transparent) and fitted result as line.
        ax.plot(x, r, '.', ms=3.7, alpha=0.50, label='Raw Dyson result')
        ax.plot(x, f, '-', lw=1.85, label='Moment-constrained tail-fitted')
        ax.axvline(lo, ls=':', lw=1.0, alpha=0.6)
        ax.axvline(hi, ls=':', lw=1.0, alpha=0.6)
        ax.axvspan(lo, hi, alpha=0.045)
        ax.set(ylabel=lab)
        axis_style(ax)
    axs[0].legend(frameon=False, ncol=2)
    axs[0].set_title('Real self-energy: all positive Matsubara frequencies')
    axs[1].set_title('Imaginary self-energy: all positive Matsubara frequencies')
    axs[1].set_xlabel(r'$\omega_n/t$')
    decorate(fig, 'Raw and fitted self-energy  |  full frequency range', meta, d,
             bottom_note='Dotted vertical lines mark the user-selected tail-fit window; raw points are shown without smoothing.')
    save_figure(fig, out, '03_sigma_raw_vs_fitted_full', pdf, dpi)


def plot_sigma_fit_window(d, meta, indices, omega, out, pdf, dpi):
    if 'Sigma_iw_raw' not in d:
        return
    settings = meta.get('settings') or {}
    fit_lo = float(settings.get('fit_min_w_over_t', 12))
    fit_hi = float(settings.get('fit_max_w_over_t', 20))
    window_max = min(float(omega[indices[-1]]), fit_hi + 4.)
    inds = indices[omega[indices] <= window_max]
    x = omega[inds]
    raw = d['Sigma_iw_raw'][inds]
    fitted = d['Sigma_iw'][inds]
    fig,axs=plt.subplots(1,2,figsize=(14.8,6.6))
    for ax,component,lab in zip(axs,('real','imag'),
            (r'$\mathrm{Re}\,\Sigma(i\omega_n)/t$',r'$\mathrm{Im}\,\Sigma(i\omega_n)/t$')):
        ax.plot(x,getattr(raw,component),'.',ms=5.4,alpha=0.7,label='Raw QMC/Dyson')
        ax.plot(x,getattr(fitted,component),'-',lw=1.8,label='Tail-fitted')
        ax.axvline(fit_lo,ls=':',alpha=0.65)
        ax.axvline(fit_hi,ls=':',alpha=0.65)
        ax.axvspan(fit_lo,fit_hi,alpha=0.05)
        ax.set(xlabel=r'$\omega_n/t$',ylabel=lab)
        axis_style(ax)
    axs[0].legend(frameon=False)
    axs[0].set_title('Real self-energy near fitting boundary')
    axs[1].set_title('Imaginary self-energy near fitting boundary')
    decorate(fig, rf'Self-energy fitting region  |  $0 < \omega_n/t < {window_max:g}$',meta,d,
             bottom_note='No smoothing of raw data; dotted lines mark the user-chosen tail-fit window.')
    fig.subplots_adjust(left=0.08,right=0.97,top=0.75,bottom=0.18,wspace=0.22)
    save_figure(fig,out,'04_sigma_fit_window_zoom',pdf,dpi)


def plot_zoom(d, meta, indices, omega, nzoom, out, pdf, dpi):
    inds=indices[:min(nzoom,len(indices))]
    x=omega[inds]
    sig=d['Sigma_iw'][inds];g=d['G_iw'][inds]
    sig_sem=d.get('Sigma_iw_sem'); g_sem=d.get('G_iw_sem')
    fig, axs=plt.subplots(2,2,figsize=(14.8,9.5))
    specs=[(axs[0,0],sig.real,sig_sem[inds].real if sig_sem is not None else None,r'$\mathrm{Re}\,\Sigma(i\omega_n)/t$'),
           (axs[0,1],sig.imag,sig_sem[inds].imag if sig_sem is not None else None,r'$\mathrm{Im}\,\Sigma(i\omega_n)/t$'),
           (axs[1,0],g.real,g_sem[inds].real if g_sem is not None else None,r'$\mathrm{Re}\,[tG(i\omega_n)]$'),
           (axs[1,1],g.imag,g_sem[inds].imag if g_sem is not None else None,r'$\mathrm{Im}\,[tG(i\omega_n)]$')]
    for ax,y,sem,ylabel in specs:
        line,=ax.plot(x,y,'o-',ms=3.7,lw=1.5)
        plot_sem(ax,x,y,sem,line.get_color())
        ax.set(xlabel=r'$\omega_n/t$',ylabel=ylabel)
        axis_style(ax)
    decorate(fig, f'Low-frequency detail  |  first {len(x)} positive points', meta, d,
             bottom_note='Low-frequency quantities are especially relevant for metallicity and quasiparticle diagnostics.')
    save_figure(fig,out,'05_low_frequency_zoom',pdf,dpi)


def plot_diagnostics(d, meta, out, pdf, dpi):
    h=np.asarray(d.get('dmft_residual_history',[]),dtype=float)
    dens=np.asarray(d.get('dmft_filling_history',[]),dtype=float)
    replicas=np.asarray(d.get('replica_fillings',[]),dtype=float)
    mus=meta.get('mu_search',[])
    fig,axs=plt.subplots(2,2,figsize=(14.8,9.5))
    ax=axs[0,0]
    if len(h):
        ax.semilogy(np.arange(1,len(h)+1),h,'o-',markersize=4,label='Bath residual')
        tol=(meta.get('settings') or {}).get('dmft_tol')
        if tol:
            ax.axhline(tol,ls='--',lw=1.1,label=f'DMFT tol = {tol:.1e}')
        ax.legend(frameon=False)
    ax.set(xlabel=r'DMFT iteration (selected $\mu$)',ylabel=r'$R_\Delta$',title=r'Fixed-$\mu$ Bethe convergence')
    axis_style(ax)
    ax=axs[0,1]
    if len(dens):
        ax.plot(np.arange(1,len(dens)+1),dens,'o-',ms=4,label='Measured density')
    target=meta.get('target_filling')
    if target is not None:
        ax.axhline(target,ls='--',lw=1.1,label='Target filling')
    ax.set(xlabel=r'DMFT iteration (selected $\mu$)',ylabel=r'$\langle n\rangle$',title=r'Fixed-$\mu$ density evolution')
    ax.legend(frameon=False)
    axis_style(ax)
    ax=axs[1,0]
    if mus:
        stages=sorted({x.get('stage','refined') for x in mus})
        markers={'coarse':'o','refined':'s'}
        for stage in stages:
            points=sorted([x for x in mus if x.get('stage','refined')==stage],key=lambda x:x['mu_over_t'])
            ax.plot([x['mu_over_t'] for x in points],[x['filling'] for x in points],
                    marker=markers.get(stage,'D'),linestyle='--' if stage=='coarse' else '-',
                    lw=1.45,ms=5,label=stage.title())
    if target is not None:
        ax.axhline(target,ls=':',lw=1.2,label='Target')
    ax.set(xlabel=r'$\mu/t$',ylabel=r'$\langle n\rangle$',title=r'Chemical-potential search (sorted by $\mu$)')
    ax.legend(frameon=False)
    axis_style(ax)
    ax=axs[1,1]
    if len(replicas):
        xs=np.arange(1,len(replicas)+1)
        ax.plot(xs,replicas,'o',ms=7,label='Independent replica filling')
        ax.axhline(np.mean(replicas),lw=1.25,label='Replica mean')
        if target is not None:
            ax.axhline(target,ls='--',lw=1.0,label='Target')
            tol=(meta.get('settings') or {}).get('density_tol')
            if tol:
                ax.axhspan(target-tol,target+tol,alpha=0.08,label='Density tolerance')
        ax.set_xticks(xs)
    ax.set(xlabel='Final replica',ylabel=r'$\langle n\rangle$',title='Final density reproducibility')
    ax.legend(frameon=False,fontsize=8.3)
    axis_style(ax,False)
    decorate(fig,'DMFT convergence and sampling diagnostics',meta,d)
    save_figure(fig,out,'06_dmft_diagnostics',pdf,dpi)


def plot_highfreq(d,meta,indices,omega,out,pdf,dpi):
    x=omega[indices]
    G=d['G_iw'][indices]
    S=d['Sigma_iw'][indices]
    first=float(np.real(d.get('Sigma_first_moment_over_t2',meta.get('sigma_first_moment_mean_over_t2',np.nan))))
    inf=float(np.real(d.get('Sigma_infinity_over_t',meta.get('sigma_infinity_mean_over_t',np.nan))))
    fig,axs=plt.subplots(1,3,figsize=(15.6,6.2))
    axs[0].plot(x,-x*G.imag,lw=1.7,label=r'$-\omega_n\mathrm{Im}\,G$')
    axs[0].axhline(1,ls='--',lw=1.15,label='Expected limit = 1')
    axs[0].set(ylabel=r'$-\omega_n\mathrm{Im}\,G(i\omega_n)$',title='Green-function moment')
    axs[1].plot(x,-x*S.imag,lw=1.7,label=r'$-\omega_n\mathrm{Im}\,\Sigma$')
    if np.isfinite(first): axs[1].axhline(first,ls='--',lw=1.15,label=rf'$\Sigma_1/t^2={first:.4f}$')
    axs[1].set(ylabel=r'$-\omega_n\mathrm{Im}\,\Sigma(i\omega_n)/t$',title='Self-energy first moment')
    axs[2].plot(x,S.real,lw=1.7,label=r'$\mathrm{Re}\,\Sigma$')
    if np.isfinite(inf): axs[2].axhline(inf,ls='--',lw=1.15,label=rf'$\Sigma_\infty/t={inf:.4f}$')
    axs[2].set(ylabel=r'$\mathrm{Re}\,\Sigma(i\omega_n)/t$',title='Hartree asymptote')
    for ax in axs:
        ax.set_xlabel(r'$\omega_n/t$')
        ax.legend(frameon=False,fontsize=9)
        axis_style(ax)
    decorate(fig,'High-frequency consistency checks',meta,d,
             bottom_note='Moment-constrained tail fitting uses known asymptotes; agreement is not an independent measurement of those moments.')
    fig.subplots_adjust(left=0.07,right=0.97,top=0.77,bottom=0.16,wspace=0.31)
    save_figure(fig,out,'07_high_frequency_checks',pdf,dpi)


def plot_signed(d,meta,omega,out,pdf,dpi):
    inds=np.argsort(omega)
    x=omega[inds]
    G=d['G_iw'][inds];S=d['Sigma_iw'][inds]
    fig,axs=plt.subplots(2,2,figsize=(14.8,9.5))
    items=[(axs[0,0],S.real,r'$\mathrm{Re}\,\Sigma/t$'),(axs[0,1],S.imag,r'$\mathrm{Im}\,\Sigma/t$'),
           (axs[1,0],G.real,r'$\mathrm{Re}\,(tG)$'),(axs[1,1],G.imag,r'$\mathrm{Im}\,(tG)$')]
    for ax,y,lab in items:
        ax.plot(x,y,lw=1.4)
        ax.axvline(0,ls=':',lw=0.8,alpha=0.6)
        ax.set(xlabel=r'$\omega_n/t$',ylabel=lab)
        axis_style(ax)
    decorate(fig,f'Entire signed Matsubara mesh  |  {len(inds)} frequencies',meta,d,
             bottom_note='Shows both negative and positive fermionic Matsubara frequencies saved in the NPZ.')
    save_figure(fig,out,'08_signed_matsubara_full',pdf,dpi)


def main():
    a=args_parser()
    if a.dpi<75 or a.zoom_n_iw<1:
        raise SystemExit('--dpi must be >= 75 and --zoom-n-iw >= 1')
    npz_path,metadata_path,out=locate(a)
    with metadata_path.open(encoding='utf-8') as fh:
        meta=json.load(fh)
    with np.load(npz_path,allow_pickle=False) as npz:
        d={name:npz[name] for name in npz.files}
    for name in ('iw_over_t','Sigma_iw','G_iw','tau_times_t','G_tau'):
        if name not in d:
            raise SystemExit(f'Missing required NPZ field: {name}')
    omega, pos=positive_index(d)
    out.mkdir(parents=True,exist_ok=True)
    theme(a.usetex)
    print(f'Input: {npz_path}\nMetadata: {metadata_path}\nPositive Matsubara points: {len(pos)}; full mesh: {len(omega)}; tau points: {len(d["tau_times_t"])}\nOutput: {out}')
    with PdfPages(out / 'DMFT_all_plots.pdf') as pages:
        plot_matsubara_full(d,meta,pos,omega,out,pages,a.dpi)
        plot_tau_full(d,meta,out,pages,a.dpi)
        plot_sigma_fit(d,meta,pos,omega,out,pages,a.dpi)
        plot_sigma_fit_window(d,meta,pos,omega,out,pages,a.dpi)
        plot_zoom(d,meta,pos,omega,a.zoom_n_iw,out,pages,a.dpi)
        plot_diagnostics(d,meta,out,pages,a.dpi)
        plot_highfreq(d,meta,pos,omega,out,pages,a.dpi)
        if a.include_negative:
            plot_signed(d,meta,omega,out,pages,a.dpi)
    print(f'  DMFT_all_plots.pdf  ({7 + int(a.include_negative)} pages)')
    print('All figures generated without running TRIQS or CT-HYB.')


if __name__ == '__main__':
    main()

