from config import construct_df
from Paths import plot_path
from data_process import equal_len_calib_valid
from examples.load_data import load_calib_gof_data, load_valid_likes
from Boxplots import boxplot_calib_parameters, boxplot_gof_loc

beh_gofs =  ['aic', 'mic'] #['aic', 'mic', 'nse', 'rmse']
num_gofs = ['nse', 'rmse']
algorithms = ['fscabc', 'mle', 'demcz', 'dream']

all_obj_func = True

if all_obj_func:
    gofs = beh_gofs +num_gofs
else:
    gofs = beh_gofs

calib_data = load_calib_gof_data(algorithms, gofs, tops=True) #500 likes
valid_data = load_valid_likes(algorithms, gofs, weighted=True)
calib_data, valid_data = equal_len_calib_valid(calib_data, valid_data)
dfs, mean_list= construct_df(calib_data, valid_data)

if all_obj_func:
    boxplot_calib_parameters(dfs, save=plot_path, kind='box')

else:
    boxplot_gof_loc()