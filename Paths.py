import os

variable = 'mds_Watertable depth Tinamit'
month = 'dec'
if os.name == 'posix':
    root_path = "/Users/gabriellapeng/OneDrive - Concordia University - Canada/gaby/pp2_data/calib"

    res_path = root_path + f'/npy_res/{month}/'

    plot_path = root_path + f'/plot/{month}/'

    calib_polygon = root_path + '/valid.csv'

    valid_polygon = root_path + '/calib.csv'

else:
    root_path = "C:\\Users\\umroot\\OneDrive - Concordia University - Canada\\gaby\\pp2_data\\calib"

    res_path = root_path + f'\\npy_res\\{month}\\'

    plot_path = root_path + f'\\plot\\{month}\\'

    calib_polygon = root_path + '\\valid.csv'

    valid_polygon = root_path + '\\calib.csv'

valid_obs_norm = res_path + 'valid_obs_norm.npy'
