import numpy as np
from config import construct_df
from Paths import plot_path, valid_polygon
from data_process import equal_len_calib_valid, _soil_canal, proc_soil_canal_mask, proc_data_to_soil_canal
from examples.load_data import load_calib_gof_data, load_valid_likes, load_valid_res, load_obs_data
from Boxplots import dist_parameters, boxplot_gof_loc

beh_gofs = ['aic']  # ['aic', 'mic', 'nse', 'rmse']
num_gofs = ['nse', 'rmse']
algorithms = ['fscabc', 'mle', 'demcz', 'dream']

obj_type = 'beh'
gof_location = True
soil_canal = 'soil'  # 'soil', 'canal'

if obj_type == 'all':
    gofs = beh_gofs + num_gofs
elif obj_type == 'beh':
    gofs = beh_gofs
elif obj_type == 'num':
    gofs = num_gofs

calib_data = load_calib_gof_data(algorithms, gofs, tops=True)  # 500 likes

if not gof_location:
    valid_data = load_valid_likes(algorithms, gofs, weighted=True)
    calib_data, valid_data = equal_len_calib_valid(calib_data, valid_data)
    dfs, mean_list = construct_df(calib_data, valid_data)
    dist_parameters(dfs, save=plot_path + f'Box/gof_values/{obj_type}', kind='box')

else:
    v_data = load_valid_res(algorithms, gofs, gof_loc=True)
    s_cnl = _soil_canal(list(load_obs_data(valid_polygon)))
    s_cnl, mask = proc_soil_canal_mask(soil_canal, s_cnl, gen_mask=True)

    for gof in gofs:
        if gof == 'mic':
            ymin, ymax = 0.6, 1.0 + 0.05
        elif gof == 'aic':
            ymin, ymax = -60, 100
        else:
            ymin, ymax = min([np.min(v_data[gof][al]) for al in algorithms]), max(
                [np.max(v_data[gof][al]) for al in algorithms])

        for al in algorithms:
            if gof == 'aic':
                vdata = np.asarray([calib_data[gof][al][i] - v for i, v in enumerate(v_data[gof][al])])
            else:
                vdata = v_data[gof][al]

            vdata = proc_data_to_soil_canal(data=vdata, soil_canal=s_cnl)

            print(f"\nPlotting {gof}, {al.upper()}")
            sc_name = ['' if soil_canal == 'all' else soil_canal][0]
            boxplot_gof_loc({gof: calib_data[gof][al]}, {gof: vdata}, soil_canal_mask=mask,
                            save=plot_path + f'Box/{sc_name}_{al[:2]}_{gof}', ylims=(ymin, ymax))
