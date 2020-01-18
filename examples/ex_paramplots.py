from Paths import calib_polygon, valid_polygon, plot_path
from ParameterPlots import group_parameters_at_loc
from config import construct_param_df
from data_process import _soil_canal, retrieve_name, find_optimiza_parameter
from examples.load_data import load_calib_param_data, load_obs_data

gofs = ['aic', 'mic'] #
algorithms = ['mle','demcz', 'dream', 'fscabc'] #'mle', 'demcz', 'dream', 'fscabc'

calib_valid = 'valid'
sc_type = 'canal'

parameter_data = load_calib_param_data(algorithms, gofs, tops=False)

if calib_valid == 'all':
    c_polys = list(load_obs_data(calib_polygon))
    v_polys = list(load_obs_data(valid_polygon))
    info_polys =  {'calib': [c_polys, _soil_canal(c_polys)], 'valid': [v_polys, _soil_canal(v_polys)]}
else:
    if calib_valid == 'valid':
        polys = list(load_obs_data(valid_polygon))
    elif calib_valid == 'calib':
        polys = list(load_obs_data(calib_polygon))
    soil_canal = _soil_canal(polys)
    info_polys = [polys, soil_canal]
    bf_opt_params, sd_opt_params = find_optimiza_parameter(parameter_data, algorithms, gofs, sc_type)

type = 'hist'
sc_type1 = ['Soil Class' if sc_type == 'soil' else 'Canal Position'][0]

for gof in gofs:
    bf1_dfs, bf2_dfs, sd_dfs = construct_param_df(parameter_data[gof], calib_valid, info_polys)
    # for al in algorithms:
    #     save = plot_path + f'parameters/{gof}/{calib_valid}_{al[:2]}_{sc_type}_'
    save = plot_path + f'parameters/{gof}/{calib_valid}_{sc_type}_'
    for data in [bf1_dfs, bf2_dfs]: #bf1_dfs, bf2_dfs,
        name = retrieve_name(data)
        dict_optimize = [bf_opt_params[gof] if 'bf' in name else sd_opt_params[gof]][0]
        group_parameters_at_loc(data=data, save=save, type=type, col_type=sc_type1, dict_optimize=dict_optimize)