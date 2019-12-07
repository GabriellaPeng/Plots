from Paths import calib_polygon, valid_polygon, plot_path
from ParameterPlots import group_parameters_at_loc
from config import construct_param_df
from data_process import _soil_canal, retrieve_name
from examples.load_data import load_calib_param_data, load_obs_data

gofs = ['aic', 'mic']
algorithms = ['mle', 'demcz', 'dream', 'fscabc']

calib = True
valid  =False

parameter_data = load_calib_param_data(algorithms, gofs, tops=True)

c_polys = list(load_obs_data(calib_polygon))
c_soil_canal = _soil_canal(c_polys)

v_polys = list(load_obs_data(valid_polygon))
v_soil_canal = _soil_canal(v_polys)

type = 'hist'

for gof in gofs:
    bf1_dfs, bf2_dfs, sd_dfs = construct_param_df(parameter_data[gof], [c_polys, c_soil_canal], [v_polys, v_soil_canal])

    for al in algorithms:
        save  = plot_path + f'parameters/{gof}/{al}_'
        for i in [bf1_dfs, bf2_dfs, sd_dfs]:
            name = retrieve_name(i)
            data = i[i['Algorithm'] == al.upper()]
            group_parameters_at_loc(data=data, save=save + name[:name.find('_')], type=type)


