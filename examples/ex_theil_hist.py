from Histograms import plot_theil
from Paths import valid_polygon
from data_process import _soil_canal
from examples.load_data import load_obs_data, load_theil_data

gofs =['aic', 'mic']
algorithms = ['fscabc', 'mle', 'demcz', 'dream']

theil_data = load_theil_data(algorithms, gofs)

npoly = list(load_obs_data(valid_polygon))
soil_canal_mask = _soil_canal(npoly)

from Paths import plot_path

for gof in gofs:
    plot_theil(methods=algorithms, data=theil_data, polys=npoly, gof=gof, soil_canal_mask=soil_canal_mask, save=plot_path+f'Theil/N{gof}')
