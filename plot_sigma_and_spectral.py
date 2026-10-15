#!/usr/bin/env python3
"""Plot 4-case Bethe-DMFT continued self energies and derived local spectral functions.
Requires saved NNLS continuation CSV and original case metadata JSON.
Outputs 2x2 vector PDFs, PNGs, and per-case spectral CSVs.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CASES = [
  ('U8_n0p7_beta25_adaptive_v2','U8p000_n0p700_beta25p00',8,.7),
  ('U8_n0p9_beta25_adaptive_v2','U8p000_n0p900_beta25p00',8,.9),
  ('U10_n0p7_beta25_adaptive_v2','U10p000_n0p700_beta25p00',10,.7),
  ('U10_n0p9_beta25_adaptive_v2','U10p000_n0p900_beta25p00',10,.9),
]

def bethe_g_retarded(z):
    """t=1 semicircular DOS; choose analytic square-root branch for Im(z)>0.
    g=(z-sqrt(z^2-4))/2, numerically use 2/(z+sqrt) away from zero.
    """
    root=np.sqrt(z*z-4+0j)
    # Retarded branch: sqrt(z^2-4) has positive imaginary part for Im z >0.
    root=np.where(root.imag < 0, -root, root)
    # for very large positive real z, root should have positive real part
    near=(np.abs(root.imag)<1e-13)&(z.real>0)&(root.real<0)
    root=np.where(near,-root,root)
    return 2.0/(z+root)

def read_numeric_csv(path):
    a=np.genfromtxt(path,delimiter=',',names=True,encoding=None)
    return a

def plot_grid(rows, field, out, zoom, dpi):
    fig,axarr=plt.subplots(2,2,figsize=(9.8,7.0),sharex=True,sharey=(field!='sigma'))
    axs=axarr.ravel()
    for i,(ax,row) in enumerate(zip(axs,rows)):
        om=row['omega']; title=rf"({'abcd'[i]}) $U/t={row['U']},\ \langle n\rangle={row['density']:.4f}$"
        if field=='sigma':
            ax.plot(om,row['sigma'].real,c='#1769aa',lw=1.6,label=r'$\mathrm{Re}\,\Sigma^R/t$')
            ax.plot(om,row['sigma'].imag,c='#c0392b',lw=1.6,ls='--',label=r'$\mathrm{Im}\,\Sigma^R/t$')
            ax.axhline(0,color='0.6',lw=.7,ls=':')
        else:
            ax.plot(om,row['A'],c='#2e7057',lw=1.8,label=r'$A(\omega)$')
            ax.axhline(0,color='0.7',lw=.6)
            ax.set_ylim(bottom=0)
        ax.axvline(0,color='0.65',lw=.7,ls=':')
        ax.set_title(title,fontsize=12,loc='left',pad=7)
        ax.set_xlim(-zoom,zoom)
        ax.tick_params(direction='in',top=True,right=True,labelleft=True)
        ax.minorticks_on()
        ax.grid(False)
        if i>=2: ax.set_xlabel(r'$\omega/t$')
        if i%2==0:
            ax.set_ylabel(r'$\Sigma^R(\omega)/t$' if field=='sigma' else r'$t\,A(\omega)$')
    if field=='sigma':
        axs[0].legend(frameon=False,fontsize=10,loc='best')
        supertitle=r'Retarded self-energy: regularized analytical-continuation estimates'
    else:
        supertitle=r'Local spectral function from Bethe DMFT (continuation estimates)'
    fig.suptitle(supertitle,fontsize=14,y=.985)
    fig.text(.5,.94,r'$\beta t=25,\quad t=0.25\ \mathrm{eV};\quad \eta/t=0.15$',ha='center',fontsize=11)
    fig.subplots_adjust(top=.87,bottom=.1,left=.12,right=.98,wspace=.21,hspace=.28)
    for ext in ('pdf','png'):
        f=out/f'{field}_four_cases_{"zoom" if zoom<10 else "full"}.{ext}'
        fig.savefig(f,dpi=dpi,facecolor='white')
        print('Saved',f)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('cthyb_test_results'))
    p.add_argument('--continuation-dir',type=Path,default=None)
    p.add_argument('--out-dir',type=Path,default=None)
    p.add_argument('--zoom',type=float,default=5)
    p.add_argument('--dpi',type=int,default=400)
    args=p.parse_args()
    root=args.root.resolve()
    cont=(args.continuation_dir or root/'sigma_real_frequency_continuation').resolve()
    out=(args.out_dir or cont/'figures_for_steve').resolve();out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':11,'axes.linewidth':1.1,'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in'})
    rows=[]
    for case,sub,U,n in CASES:
        csvpath=cont/case/'Sigma_retarded_estimate.csv'
        metapath=root/case/sub/'metadata.json'
        if not csvpath.is_file():raise FileNotFoundError(csvpath)
        if not metapath.is_file():raise FileNotFoundError(metapath)
        arr=read_numeric_csv(csvpath)
        meta=json.loads(metapath.read_text())
        w=np.asarray(arr['omega_over_t'],dtype=float)
        S=np.asarray(arr['Re_Sigma_over_t']+1j*arr['Im_Sigma_over_t'])
        mu=float(meta['mu_over_t'])
        eta=.15 # must match CSV generator's broadening
        diag=cont/case/'continuation_diagnostics.json'
        if diag.is_file():eta=float(json.loads(diag.read_text())['eta_over_t'])
        if np.any(S.imag>1e-7):print('WARNING noncausal Sigma:',case)
        z=w+mu-S+1j*eta
        G=bethe_g_retarded(z)
        A=-G.imag/np.pi
        if np.any(A< -1e-8):print('WARNING negative A:',case)
        A=np.maximum(A,0)
        rows.append(dict(omega=w,sigma=S,G=G,A=A,U=U,density=float(meta['final_filling_mean'])))
        outcsv=out/f'{case}_A_local_estimate.csv'
        with outcsv.open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['omega_over_t','Re_Sigma_over_t','Im_Sigma_over_t','Re_G_times_t','Im_G_times_t','A_times_t'])
            writer.writerows(zip(w,S.real,S.imag,G.real,G.imag,A))
        area=np.trapezoid(A,w)
        print(f'{case}: mu/t={mu:.7f}; int A domega over plotted range={area:.5f}; A(0)~{np.interp(0,w,A):.5f}')
    plot_grid(rows,'sigma',out,12,args.dpi)
    plot_grid(rows,'sigma',out,args.zoom,args.dpi)
    plot_grid(rows,'A',out,12,args.dpi)
    plot_grid(rows,'A',out,args.zoom,args.dpi)
    print('Note: A is evaluated consistently at omega+i*eta; finite eta broadens the result.')

if __name__=='__main__':main()

