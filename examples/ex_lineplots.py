import os

from LinePlots import plot_top_sim_obs, plot_gof_convergence
from Paths import valid_polygon, plot_path, calib_polygon
from data_process import load_res_sim, plot_soil_canal, run_sim_vs_obs
from examples.load_data import load_observe_data, load_valid_res, load_calib_gof_data

gofs = ['rmse', 'nse', 'aic', 'mic']  # 'rmse', 'nse', 'aic', 'mic'
algorithms = ['fscabc', 'mle', 'demcz', 'dream'] #'fscabc', 'mle', 'demcz', 'dream'

data = load_valid_res(algorithms, gofs)

plot_simVobs = True
plot_converge = False


if plot_simVobs:
    figure, warmup_period = ['bounds'], None

    type_res = 'all'
    type_res, type_sim = load_res_sim(type_res)

    soil_canal = 'soil'
    calib_valid = 'valid'
    selected_polys = 'all' #'best'
    combine_polys  = True

    plot_path = plot_path + ['simVobs/' if os.name == 'posix' else 'simVobs\\'][0]
    polygon_path = [calib_polygon if calib_valid == 'calib' else valid_polygon][0]
    obs_norm = load_observe_data(polygon_path, warmup_period=warmup_period)
    sc, npoly = plot_soil_canal(soil_canal, polys=selected_polys, calib_valid=calib_valid)

    l_poly = {'nrow': len(gofs)}

    if selected_polys == 'best':
        l_poly.update({'ncol': npoly})

        for al in algorithms:
            sim_norm, res_dt = run_sim_vs_obs(l_poly, algorithms, gofs, type_sim, type_res, data)
            plot_top_sim_obs(sim_norm, obs_norm, npoly, plot_path + f'{al[:2]}', res_dt,
                             l_poly=l_poly, figures=figure, calib_valid=calib_valid, polys=selected_polys, combine_polys=combine_polys)

    else:
        for name, l_polys in sc.items():
            l_poly.update({'ncol': l_polys})

            for al in algorithms:
                sim_norm, res_dt = run_sim_vs_obs(l_poly, al, gofs, type_sim, type_res, data)
                plot_top_sim_obs(sim_norm, obs_norm, npoly, plot_path + f'{name}_{al[:2]}_', res_dt,
                             l_poly=l_poly, figures=figure, calib_valid=calib_valid, polys=selected_polys, combine_polys=combine_polys)


elif plot_converge:
    plot_path = plot_path + ['converge/' if os.name == 'posix' else 'converge\\'][0]

    for gof in gofs:
        data = load_calib_gof_data(algorithms, gofs, tops=False)[gof]
        plot_gof_convergence(gof, algorithms, data, plot_path + f'{gof}')


