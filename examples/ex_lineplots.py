import os

from LinePlots import plot_top_sim_obs, plot_gof_convergence
from Paths import valid_polygon, plot_path, calib_polygon
from data_process import _soil_canal
from examples.load_data import load_obs_data, load_observe_data, load_valid_res, load_calib_gof_data

gofs = ['rmse', 'nse', 'aic', 'mic']
algorithms = ['fscabc', 'mle', 'demcz', 'dream']

data = load_valid_res(algorithms, gofs, top_weighted_sim=True)

plot_simVobs = True
plot_converge = False

if plot_simVobs:
    type_sim = 'top_weighted_sim'
    type_res = 'top_res'
    figure = ['bounds']
    soil_canal = 'soil'
    poly_calib_valid  = 'valid_poly'
    warmup_period = None
    plot_path = plot_path + ['simVobs/' if os.name == 'posix' else 'simVobs\\'][0]

    polygon = [calib_polygon if poly_calib_valid == 'calib_poly' else valid_polygon][0]

    npoly = list(load_obs_data(polygon))
    obs_norm = load_observe_data(polygon, warmup_period=warmup_period)
    d_polys = _soil_canal(npoly)

    soil = {s: [ ] for s in ['B', 'F', 'J', 'C']}
    canal = {c: [ ] for c in ['H', 'M', 'T']}

    l_poly = {'nrow': len(gofs)}

    for s_c, l_polys in d_polys.items():
        soil[s_c[0]].extend(l_polys)
        canal[s_c[s_c.find(' ')+1]].extend(l_polys)

    # plot by soils
    if soil_canal == 'soil':
        sc = soil
    elif soil_canal == 'canal':
        sc = canal

    for name, l_polys in sc.items():
        l_poly.update({'ncol':l_polys})

        for al in algorithms:
            if l_poly['nrow'] > 1:
                sim_norm = {gof: {type_sim: data[gof][al][type_sim]} for gof in gofs}
                res_dt = {gof: {type_res: data[gof][al][type_res]} for gof in gofs}
                plot_top_sim_obs(sim_norm, obs_norm, npoly, plot_path + f'{al}_{name}', res_dt,
                                 l_poly=l_poly, figures=figure)
            else:
                for gof in gofs:
                    sim_norm = {type_sim: data[gof][al][type_sim]}
                    res_dt = {type_res: data[gof][al][type_res]}
                    plot_top_sim_obs(sim_norm, obs_norm, npoly, plot_path+f'{al}_{gof}_{name}_', res_dt,
                                     l_poly=l_poly, figures=figure)

elif plot_converge:
    plot_path = plot_path + ['converge/' if os.name == 'posix' else 'converge\\'][0]

    for gof in gofs:
        data = load_calib_gof_data(algorithms, gofs, tops=False)[gof]
        plot_gof_convergence(gof, algorithms, data, plot_path+f'{gof}')


