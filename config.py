import os

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

            dfs[col_names[1]].extend(['Calibration' for i in range(len(calib_dt))])
            dfs[col_names[2]].extend([al.upper() for i in range(len(calib_dt))])

            if g == 'aic':
                dfs[col_names[0]].extend([np.nan for i in range(len(calib_dt))])
                dfs[col_names[3]].extend(['Δ'+g.upper() for i in range(len(calib_dt))])
            else:
                mean_list[g].update({al: [np.mean(calib_dt)]})
                dfs[col_names[0]].extend(list(calib_dt))
                dfs[col_names[3]].extend([g.upper() for i in range(len(calib_dt))])

            valid_dt = valid_data[g][al]
            dfs[col_names[1]].extend(['Evaluation' for i in range(len(valid_dt))])
            dfs[col_names[2]].extend([al.upper() for i in range(len(valid_dt))])

            if g == 'aic':
                abs_aic = np.abs(valid_dt - calib_dt)
                dfs[col_names[0]].extend(list(abs_aic))
                mean_list[g].update({al: np.mean(abs_aic)})
                dfs[col_names[3]].extend(['Δ'+g.upper() for i in range(len(valid_dt))])
            else:
                dfs[col_names[0]].extend(list(valid_dt))
                mean_list[g][al].append(np.mean(valid_dt))
                dfs[col_names[3]].extend([g.upper() for i in range(len(valid_dt))])

            # Create the pandas DataFrame
    return pd.DataFrame(dfs, columns=col_names), mean_list
