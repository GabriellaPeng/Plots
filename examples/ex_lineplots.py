from LinePlots import plot_top_sim_obs
from Paths import valid_polygon, valid_obs_norm, plot_path
from data_process import _soil_canal, _combine_soil_canal_data
from examples.load_data import load_obs_data, load_observe_data, load_valid_likes, load_valid_res

gofs = ['nse', 'rmse', 'aic', 'mic']
algorithms = ['fscabc', 'mle', 'demcz', 'dream']

res_data = load_valid_res(algorithms, gofs, weighted_sim=False)
sim_norm = load_valid_res(algorithms, gofs, weighted_sim=True)

npoly = list(load_obs_data(valid_polygon))
obs_norm = load_observe_data(valid_obs_norm)

soils, canals = _combine_soil_canal_data(npoly, soil_canal='soil'), _combine_soil_canal_data(npoly, soil_canal='canal')

for sc, dt, in soils.items():
    for fig in ['bounds', 'cis']:
        for al in algorithms:
            for gof in gofs:
                plot_top_sim_obs(sim_norm[gof][al], obs_norm, npoly, plot_path+f'{al}/{gof}/', res_data[gof][al],
                                 l_poly=dt, alg=al+gof+sc, figures=[fig])
