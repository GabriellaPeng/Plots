from Paths import calib_polygon, valid_polygon

from examples.load_data import load_obs_data, load_valid_res

calib_obs = load_obs_data(calib_polygon)
valid_obs = load_obs_data(valid_polygon)

gofs = ['rmse', 'nse', 'aic', 'mic']
algorithms = ['fscabc', 'mle', 'demcz', 'dream']

dict_res = load_valid_res(algorithms, gofs)
dict_res = {gof: {al: v2['all_res'] for al, v2 in v1.items()} for gof, v1 in dict_res.items()}

print()
