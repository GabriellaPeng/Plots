import os

import numpy as np
from config import construct_df
from Paths import plot_path, valid_polygon, calib_polygon
from data_process import equal_len_calib_valid, _soil_canal, proc_soil_canal_mask, proc_data_to_soil_canal, ylims
from examples.load_data import load_calib_gof_data, load_valid_likes, load_valid_res, load_obs_data
from Boxplots import dist_parameters, boxplot_gof_loc

beh_gofs, num_gofs = ['aic', 'mic'], ['nse', 'rmse']
algorithms = ['fscabc', 'mle', 'demcz', 'dream']

obj_type = 'beh'
gof_location = True
calib_valid = 'valid_polys'

gofs = [beh_gofs + num_gofs if obj_type == 'all' else beh_gofs if obj_type == 'beh' else num_gofs][0]

calib_data = load_calib_gof_data(algorithms, gofs, tops=True)  # 500 likes

if not gof_location:
    plot_path = plot_path + ['gofs/' if os.name == 'posix' else 'gofs\\'][0]

    valid_data = load_valid_likes(algorithms, gofs, weighted=True)
    calib_data, valid_data = equal_len_calib_valid(calib_data, valid_data)
    dfs, mean_list = construct_df(calib_data, valid_data)
    dist_parameters(dfs, save=plot_path + f'{obj_type}', kind='box')

else:
    plot_path = plot_path + ['gofs_soil_canal/' if os.name == 'posix' else 'gofs_soil_canal\\'][0]
    soil_canal = 'soil'  # 'soil', 'canal'
    sc_name = ['' if soil_canal == 'all' else soil_canal][0]

    v_data = load_valid_res(algorithms, gofs, gof_loc=True)
    s_cnl = _soil_canal(list(load_obs_data(valid_polygon)))  # calib_polygon
    s_cnl, mask = proc_soil_canal_mask(soil_canal, s_cnl, gen_mask=True)

    for gof in gofs:
        ylims = ylims(gof, v_data, algorithms)

        for al in algorithms:
            vdata = [
                np.asarray([calib_data[gof][al][i] - v for i, v in enumerate(v_data[gof][al])]) if gof == 'aic' else
                v_data[gof][al]][0]
            vdata = proc_data_to_soil_canal(data=vdata, soil_canal=s_cnl)

            print(f"\nPlotting {gof}, {al.upper()}")
            boxplot_gof_loc({gof: calib_data[gof][al]}, {gof: vdata}, soil_canal_mask=mask,
                            save=plot_path + f'{gof}_{sc_name}_{al[:2]}', ylims=ylims)
