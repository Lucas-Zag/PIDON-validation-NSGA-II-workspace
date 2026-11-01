"""Plot held-out SAC evaluation history after training."""
import argparse,csv
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path('rl_results_v3/eval_history.csv'))
    p.add_argument('--output',type=Path,default=Path('rl_results_v3/training_diagnostics.png'))
    a=p.parse_args()
    with a.input.open(newline='') as f: rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No evaluation rows')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=[int(x['training_steps']) for x in rows]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(8,8),sharex=True)
    specs=[('mean_score','Mean held-out final score'),('feasible_rate','Held-out feasible rate'),('mean_min_doc','Mean final minimum DoC')]
    for ax,(key,label) in zip(axes,specs):
        ax.plot(t,[float(r[key]) for r in rows],marker='o'); ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[-1].set_xlabel('SAC training steps')
    fig.tight_layout();fig.savefig(a.output,dpi=150);plt.close(fig)
    print(a.output)
if __name__=='__main__':main()
