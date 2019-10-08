from Boxplots import boxplot_calib_parameters
from Paths import valid_polygon
from config import construct_df
from data_process import equal_len_calib_valid
from examples.load_data import load_calib_gof_data, load_valid_gof_data, load_obs_data

gofs = ['nse', 'rmse', 'aic', 'mic']
algorithms = ['fscabc', 'mle', 'demcz', 'dream']

calib_data = load_calib_gof_data(algorithms, gofs, tops=True) #500 likes
valid_data = load_valid_gof_data(algorithms, gofs)
calib_data, valid_data = equal_len_calib_valid(calib_data, valid_data)
dfs = construct_df(calib_data, valid_data)

boxplot_calib_parameters(gofs, algorithms, dfs)
