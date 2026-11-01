from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from fe_reference.plotting import plot_case,plot_convergence


def main():
    key=pd.read_csv(ROOT/'outputs'/'reference'/'fe_key_curves.csv');fig=ROOT/'outputs'/'figures';fig.mkdir(parents=True,exist_ok=True)
    for c in key.case_id.unique():plot_case(key,c,fig)
    mesh=pd.read_csv(ROOT/'outputs'/'verification'/'mesh_convergence.csv')
    time=pd.read_csv(ROOT/'outputs'/'verification'/'time_convergence.csv')
    plot_convergence(mesh,'n_elements_each_layer','Mesh convergence','part_mid_Tmax_C',fig/'mesh_convergence_peak.png')
    plot_convergence(time,'max_step_s','Time-step convergence','part_mid_Tmax_C',fig/'time_convergence_peak.png')
    print(f'Figures saved to {fig}')

if __name__=='__main__':main()
