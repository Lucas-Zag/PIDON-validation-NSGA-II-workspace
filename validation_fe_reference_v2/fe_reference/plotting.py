from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_case(key,case_id,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    g=key[key.case_id==case_id]
    mid=g[g.position=='part_mid']
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(mid.t_min,mid.Ta_C,label='Autoclave air $T_a$',linestyle='--')
    ax.plot(mid.t_min,mid.T_FE_C,label='FE part midpoint $T_c$')
    ax.set_xlabel('Time [min]');ax.set_ylabel('Temperature [°C]');ax.legend();ax.grid(alpha=.25)
    ax.set_title(f'{case_id}: complete cure cycle and exotherm')
    fig.tight_layout();fig.savefig(outdir/f'{case_id}_part_mid_temperature.png',dpi=220);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(mid.t_min,mid.alpha_FE,label='FE part midpoint $\\alpha$')
    ax.set_xlabel('Time [min]');ax.set_ylabel('Degree of cure');ax.set_ylim(0,1);ax.grid(alpha=.25);ax.legend()
    ax.set_title(f'{case_id}: degree of cure')
    fig.tight_layout();fig.savefig(outdir/f'{case_id}_part_mid_alpha.png',dpi=220);plt.close(fig)


def plot_convergence(df,xcol,title,ycol,outpath):
    fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(df[xcol],df[ycol],marker='o')
    ax.set_xlabel(xcol);ax.set_ylabel(ycol);ax.set_title(title);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(outpath,dpi=220);plt.close(fig)
