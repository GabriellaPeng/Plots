from config import construct_df
from data_process import equal_len_calib_valid
from examples.load_data import load_calib_gof_data, load_valid_likes

gofs = ['nse', 'rmse', 'aic', 'mic']
algorithms = ['fscabc', 'mle', 'demcz', 'dream']

calib_data = load_calib_gof_data(algorithms, gofs, tops=True) #500 likes
valid_data = load_valid_likes(algorithms, gofs, weighted=True)
calib_data, valid_data = equal_len_calib_valid(calib_data, valid_data)
dfs, mean_list= construct_df(calib_data, valid_data)

from Paths import plot_path
from Boxplots import boxplot_calib_parameters

boxplot_calib_parameters(dfs, save=plot_path, kind='violin')
