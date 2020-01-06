from Paths import plot_path
from Histograms import plot_theil
from Paths import valid_polygon, calib_polygon
from data_process import _soil_canal, proc_soil_canal_mask
from examples.load_data import load_obs_data, load_theil_data

gofs =['aic', 'mic']
algorithms = [ 'demcz','dream', 'fscabc', 'mle'] #'fscabc', 'mle', , 'dream'

soil_canal = 'canal'
calib_valid = 'calib'
type_sim = ['weighted_sim','top_weighted_sim']

npoly = [list(load_obs_data(calib_polygon)) if calib_valid == 'calib' else list(load_obs_data(valid_polygon))][0]
s_cnl = _soil_canal(npoly)
s_cnl = proc_soil_canal_mask(soil_canal, s_cnl)

for type in type_sim:
    theil_data = load_theil_data(algorithms, gofs, type_sim=type, calib_valid=calib_valid)

    for gof in gofs:
        plot_theil(methods=algorithms, data=theil_data, polys=npoly, gof=gof, soil_canal_mask=s_cnl, soil_canal=soil_canal,
                   save=plot_path + f'Theil/{type}/{calib_valid}_{gof}_')
