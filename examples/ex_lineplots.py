from LinePlots import plot_top_sim_obs, plot_gof_convergence
from Paths import valid_polygon, plot_path
from examples.load_data import load_obs_data, load_observe_data, load_valid_res, load_calib_gof_data

gofs = ['aic', 'mic'] #'aic', 'mic'
algorithms = ['fscabc', 'mle', 'demcz', 'dream']

res_data = load_valid_res(algorithms, gofs, weighted_sim=False)
sim_norm = load_valid_res(algorithms, gofs, weighted_sim=True)

npoly = list(load_obs_data(valid_polygon))
obs_norm = load_observe_data(valid_polygon, warmup_period=2)


plot_sim = True
plot_converge = False

if plot_sim:
    for al in algorithms:
        dt = {'ncol': [17, 168, 175] , 'nrow': 4}
        for fig in ['bounds']: #, 'cis']:
            #sim_norms = {gof: sim_norm[gof][al] for gof in gofs}
            # res_dt = {gof: res_data[gof][al] for gof in gofs}
            for gof in gofs:
                sim_norms = {type_sim: dt for type_sim, dt in sim_norm[gof][al].items() if
                             type_sim == 'top_weighted_sim'}

                res_dt = {type_sim: dt for type_sim, dt in res_data[gof][al].items() if
                 type_sim == 'weighted_res'}

                plot_top_sim_obs(sim_norms, obs_norm, npoly, plot_path+f'{al}/{gof}/', res_dt,
                                 l_poly=None, alg=f'{al}', figures=[fig])

elif plot_converge:
    for gof in gofs:
        data = load_calib_gof_data(algorithms, gofs, tops=False)[gof]
        plot_gof_convergence(gof, algorithms, data, plot_path + f'converge/')


