from Histograms import plot_theil
from Paths import valid_polygon
from data_process import _soil_canal, proc_soil_canal_mask
from examples.load_data import load_obs_data, load_theil_data

gofs =['aic', 'mic']
algorithms = ['fscabc', 'mle', 'demcz', 'dream']
soil_canal = 'soil'

theil_data = load_theil_data(algorithms, gofs)

npoly = list(load_obs_data(valid_polygon))
s_cnl = _soil_canal(npoly)
s_cnl = proc_soil_canal_mask(soil_canal, s_cnl)

from Paths import plot_path

for gof in gofs:
    plot_theil(methods=algorithms, data=theil_data, polys=npoly, gof=gof, soil_canal_mask=s_cnl, soil_canal=soil_canal,
               save=plot_path + f'Theil/{gof}')
