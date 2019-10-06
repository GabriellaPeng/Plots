from Boxplots import boxplot_calib_parameters
from Paths import valid_polygon
from examples.load_data import load_calib_gof_data, load_valid_gof_data, load_obs_data

gofs = ['nse', 'rmse', 'aic', 'mic']
algorithms = ['fscabc', 'mle', 'demcz', 'dream']

gof_data = load_calib_gof_data(algorithms, gofs) #500 likes

# valid_polygons = load_obs_data(csv_path=valid_polygon)

load_valid_gof_data(algorithms, gofs) #load 18 for each of the 16 combinations

# boxplot_calib_parameters(gofs, algorithms, calib_gofs, valid_gofs)
