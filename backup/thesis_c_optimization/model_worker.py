"""Isolated model process. Never execute existing comparison entrypoints."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time

from contracts import read_case, validate_case, scaled_input, select_tags, sampled_metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xpidon_fields(root, case, times, positions, tags, model_dir):
    import numpy as np
    import jax.numpy as jnp
    sys.path.insert(0,str(root))
    from utils.phy_params import PhyParams
    from utils.exp_params import ExpParams
    from models.pidon import XPIDON

    raw_exp = json.loads((root/'exp_params.json').read_text())
    phy = PhyParams(str(root/'phy_params.json'))
    exp = ExpParams(str(root/'exp_params.json'))
    tags = select_tags(model_dir,tags)
    start_load = time.perf_counter()
    models, hashes = [], {}
    start_fraction = 0.0
    for tag in tags:
        end_fraction = int(tag)/10000
        model = XPIDON(phy,exp,[9,50,50,50,100],[2,50,50,50,50,50,100],
                       [100,50,50,50,50,1],start_fraction*exp.t_max,end_fraction*exp.t_max)
        weights = []
        for variable in ('T','a'):
            p = model_dir/f'xpidon_class_{tag}_{variable}.pkl'
            data = p.read_bytes()
            hashes[str(p)] = hashlib.sha256(data).hexdigest()
            weights.append(pickle.loads(data))
        models.append((start_fraction,end_fraction,model,*weights))
        start_fraction = end_fraction
    load_s = time.perf_counter()-start_load
    vector = np.asarray(scaled_input(case,raw_exp),dtype=np.float32)

    def predict():
        temperature = np.empty((len(times),len(positions)))
        alpha = np.empty_like(temperature)
        global_time = times/float(exp.t_max)
        # Same right-hand boundary convention as the supplied comparison caller.
        ends = np.asarray([r[1] for r in models])
        domain_indices = np.minimum(np.searchsorted(ends,global_time,side='right'),len(models)-1)
        for index,(lo,hi,model,T_weights,a_weights) in enumerate(models):
            selected = np.where(domain_indices == index)[0]
            if not len(selected):
                continue
            local = np.clip((global_time[selected]-lo)/(hi-lo),0,1)
            query = np.column_stack([np.repeat(local,len(positions)),
                                     np.tile(positions,len(selected))])
            inputs = jnp.asarray(np.tile(vector,(len(query),1)))
            query = jnp.asarray(query)
            # np.asarray synchronizes JAX work before timing ends.
            temperature[selected,:] = np.asarray(model.pred_T(T_weights,inputs,query)).reshape(len(selected),-1)/exp.T_scaler
            alpha[selected,:] = np.asarray(model.pred_a(a_weights,inputs,query)).reshape(len(selected),-1)
        return temperature,alpha

    start = time.perf_counter()
    predict()
    first_s = time.perf_counter()-start
    start = time.perf_counter()
    temperature,alpha = predict()
    warm_s = time.perf_counter()-start
    return temperature,alpha,{
        'model_load_and_initialization_s':load_s,
        'first_grid_inference_including_compilation_s':first_s,
        'warm_grid_inference_s':warm_s,
        'checkpoint_sha256':hashes,
        'selected_tags':tags,
        'subdomain_boundary_source':'Tag-derived, same as comparison; exact original run boundaries must be confirmed',
    }


def fe_fields(root, case, times, positions):
    import copy
    import numpy as np
    sys.path.insert(0,str(root))
    from fe_reference.config import load_config
    from fe_reference.solver import solve_case
    config_path = root/'config/reference_config.yaml'
    cfg = copy.deepcopy(load_config(config_path))
    if not np.isclose(float(cfg['cure_cycle']['total_process_time_s']),case['total_process_time_s']):
        raise ValueError('FE configuration and XPIDON case horizon differ')
    # Solve with its existing verified settings, then interpolate onto common queries.
    start = time.perf_counter()
    result = solve_case(case,cfg)
    elapsed = time.perf_counter()-start
    if result.t_s[-1] < times[-1]-1e-9:
        raise ValueError('FE output does not cover requested end time')
    indices = result.mesh.part_node_indices
    x_nodes = (result.mesh.z[indices]-result.mesh.Lt_m)/result.mesh.Lc_m
    def interpolate(field):
        at_times = np.array([np.interp(times,result.t_s,node) for node in field])
        return np.array([np.interp(positions,x_nodes,at_times[:,i]) for i in range(len(times))])
    return interpolate(result.T_C[indices,:]),interpolate(result.alpha),{
        'fe_solve_s':elapsed,'fe_config_sha256':digest(config_path),
        'interpolation':'linear in time and physical composite coordinate',
    }


def main():
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    workspace = Path(request['workspace'])
    xp_root = workspace/'XPIDON-Composites-FV1'
    fe_root = workspace/'validation_fe_reference_v2'
    case = read_case(request['cases_file'],request['case_id'])
    exp = json.loads((xp_root/'exp_params.json').read_text())
    phy = json.loads((xp_root/'phy_params.json').read_text())
    timing = validate_case(case,exp,phy)
    total = float(exp['t_max'])
    # Include initial/end/hold-end times even when off the uniform time grid.
    base = np.arange(0,total,float(request['time_step_s']))
    times = np.unique(np.r_[base,total,timing['heating_hold_end_s']])
    positions = np.linspace(0,1,int(request['spatial_points']))
    if request['backend'] == 'xpidon':
        tags = request['tags']
        times = np.unique(np.r_[times,[int(tag)/10000*total for tag in tags]])
        temperature,alpha,provenance = xpidon_fields(xp_root,case,times,positions,tags,
                                                    Path(request['model_directory']))
    else:
        temperature,alpha,provenance = fe_fields(fe_root,case,times,positions)
    summary = sampled_metrics(times,positions,temperature,alpha)
    end_index = int(np.searchsorted(times,timing['heating_hold_end_s']))
    summary['sampled_alpha_min_at_heating_hold_end'] = float(alpha[end_index].min())
    summary.update(timing)
    summary['complete_cycle_min'] = total/60
    summary['backend'] = request['backend']
    summary['case'] = case
    summary['grid'] = {'times':len(times),'spatial_points':len(positions),
                       'uniform_time_step_s':request['time_step_s'],
                       'extrema_scope':'Sampled composite field; grid refinement pending'}
    summary['provenance'] = provenance
    summary['config_sha256'] = {str(xp_root/k):digest(xp_root/k)
                               for k in ('exp_params.json','phy_params.json')}
    summary['versions'] = {}
    for name in ('numpy','pandas','scipy','jax','jaxlib','optax'):
        try:
            summary['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    out = Path(request['output_directory'])
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'t_s':np.repeat(times,len(positions)),
                  'x_norm':np.tile(positions,len(times)),
                  'T_part_C':temperature.reshape(-1),
                  'alpha':alpha.reshape(-1)}).to_csv(out/'part_field.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
