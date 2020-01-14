import os
import re

import numpy as np
import pandas as pd

Plotpath = os.path.abspath('./Plots//')


def construct_df(calib_data, valid_data, col_names=['c_v_data', 'Calib/Eval', 'Algorithms', 'Objective Function']):
    gofs = list(calib_data)
    algorithms = list(calib_data[gofs[0]])

    dfs = {c_name: [ ] for c_name in col_names}
    mean_list = {g: { } for g in gofs}

    for g in gofs:
        for al in algorithms:
            calib_dt = calib_data[g][al]

            dfs['Calib/Eval'].extend(['Calibration' for i in range(len(calib_dt))])
            dfs['Algorithms'].extend([al.upper() for i in range(len(calib_dt))])

            if g == 'aic':
                dfs['c_v_data'].extend([np.nan for i in range(len(calib_dt))])
                dfs['Objective Function'].extend(['Δ'+g.upper() for i in range(len(calib_dt))])
            else:
                mean_list[g].update({al: [np.mean(calib_dt)]})
                dfs['Objective Function'].extend([g.upper() for i in range(len(calib_dt))])
                dfs['c_v_data'].extend(list(calib_dt))

            valid_dt = valid_data[g][al]
            dfs['Calib/Eval'].extend(['Evaluation' for i in range(len(valid_dt))])
            dfs['Algorithms'].extend([al.upper() for i in range(len(valid_dt))])

            if g == 'aic':
                delta_aic = valid_dt - calib_dt
                dfs['c_v_data'].extend(list(delta_aic))
                mean_list[g].update({al: np.mean(delta_aic)})
                dfs['Objective Function'].extend(['Δ'+g.upper() for i in range(len(valid_dt))])
            else:
                mean_list[g][al].append(np.mean(valid_dt))
                dfs['c_v_data'].extend(list(valid_dt))
                dfs['Objective Function'].extend([g.upper() for i in range(len(valid_dt))])

            # Create the pandas DataFrame
    return pd.DataFrame(dfs, columns=col_names), mean_list


def construct_param_df(parameter_data, calib_valid, info_polys,
                       col_names=['Parameter', 'Parameter Vals', 'Algorithm', 'Runs', 'Canal Position', 'Soil Class', 'Poly',
                                  'C/V Poly']):

    algorithms = list(parameter_data)
    if calib_valid == 'calib' or calib_valid =='valid':
        polys = info_polys[0]
        total_sc = info_polys[1]
    elif calib_valid =='all':
        c_poly, v_poly = info_polys['calib'][0], info_polys['valid'][0]
        polys =  c_poly + v_poly
        c_sc, v_sc = info_polys['calib'][1], info_polys['valid'][1]
        total_sc = {k1: v1 + v2 for k1, v1 in c_sc.items() for k2, v2 in v_sc.items() if k1==k2}


    b1_dfs = {c_name: [ ] for c_name in col_names}
    b2_dfs = {c_name: [ ] for c_name in col_names}
    dfs_socioeco = {c_name: [ ] for c_name in col_names[:4]}

    for al in algorithms:
        param_dt = parameter_data[al]

        for params, vals in param_dt.items():
            if '-' in params:
                array_val = np.take(vals, np.array([p - 1 for p in polys]), axis=1)
                len_v = len(vals)
                for i, p in enumerate(polys):

                    def bf_dfs(dfs, i):
                        dfs['Poly'].extend([p for i in range(len_v)])
                        dfs['Runs'].extend(np.arange(len_v))
                        dfs['Parameter'].extend([params[:params.find(' ')] for i in range(len_v)])
                        dfs['Parameter Vals'].extend(array_val[:, i])
                        dfs['Algorithm'].extend([al.upper() for i in range(len_v)])

                        if calib_valid in ['calib', 'valid']:
                            dfs['C/V Poly'].extend([f'{calib_valid.capitalize()} Poly'] * len_v)
                        elif calib_valid =='all':
                            dfs['C/V Poly'].extend(['Calib Poly' if p in c_poly else 'Valid Poly'] * len_v)

                        dfs['Canal Position'].extend(
                            [[sc[sc.find(',') + 2]] * len_v for sc, l_pol in total_sc.items() if p in l_pol][0])
                        dfs['Soil Class'].extend(
                            [[sc[0]] * len_v for sc, l_pol in total_sc.items() if p in l_pol][0])

                    if params.startswith("K"):
                        bf_dfs(b1_dfs, i)
                    else:
                        bf_dfs(b2_dfs, i)

            else:
                len_v = len(vals)
                prms = params[:params.find(' ')] + ','+[params[i] for i in [m.end() for m in re.finditer(' ', params)]][0]

                dfs_socioeco['Parameter'].extend([prms]*len_v)
                dfs_socioeco['Parameter Vals'].extend(vals)
                dfs_socioeco['Algorithm'].extend([al.upper() for i in range(len_v)])
                dfs_socioeco['Runs'].extend(np.arange(len_v))

    return pd.DataFrame(b1_dfs, columns=col_names), pd.DataFrame(b2_dfs, columns=col_names), pd.DataFrame(dfs_socioeco, columns=col_names[:4])