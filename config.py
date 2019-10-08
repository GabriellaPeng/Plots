import os
import pandas as pd

Plotpath = os.path.abspath('./Plots//')


def construct_df(calib_data, valid_data, col_names=['c_v_data', 'Calib/Eval', 'Algorithms', 'Objective Function']):
    gofs = list(calib_data)
    algorithms = list(calib_data[gofs[0]])

    dfs = {c_name: [ ] for c_name in col_names}
    for g in gofs:
        for al in algorithms:
            calib_dt = calib_data[g][al]
            dfs[col_names[0]].extend(list(calib_dt))
            dfs[col_names[1]].extend(['Calibration' for i in range(len(calib_dt))])
            dfs[col_names[2]].extend([al.upper() for i in range(len(calib_dt))])
            dfs[col_names[3]].extend([g.upper() for i in range(len(calib_dt))])

            valid_dt = valid_data[g][al]
            dfs[col_names[0]].extend(list(valid_dt))
            dfs[col_names[1]].extend(['Evaluation' for i in range(len(valid_dt))])
            dfs[col_names[2]].extend([al.upper() for i in range(len(valid_dt))])
            dfs[col_names[3]].extend([g.upper() for i in range(len(valid_dt))])

            # Create the pandas DataFrame
    return pd.DataFrame(dfs, columns=col_names)
