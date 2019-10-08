import numpy as np
import pandas as pd
from Paths import res_path, variable
from data_process import process_calib_likes


def load_obs_data(csv_path):
    res = pd.read_csv(csv_path)

    obs_data = {}
    for row in res.values:
        obs_data[row[1]] = row[2:]
    return  obs_data #res.columns[2:len(res.columns)]


def load_calib_gof_data(algorithms, gofs, res_path=res_path, tops=False):
    dict_gof = {gof: {m: {} for m in algorithms} for gof in gofs}

    for gof in gofs:
        for m in algorithms:
            calib_likes = np.load(res_path + f'{m}/{m}_{gof}_PrmProb.npy', allow_pickle=True).tolist()['likes']
            dict_gof[gof][m] = calib_likes
    if tops:
        dict_gof = process_calib_likes(dict_gof, algorithms, gofs, top_percent=0.2)

    return dict_gof


def load_parameter_data(algorithms, gofs, res_path=res_path):
    dict_params = {gof: {m: {} for m in algorithms} for gof in gofs}

    for gof in gofs:
        for m in algorithms:
            a = np.load(res_path + f'{m}/{m}_{gof}_PrmProb.npy', allow_pickle=True).tolist()
            dict_params[gof][m] = {p: v for p, v in a.items() if p != 'likes'}

    return dict_params


def load_valid_gof_data(algorithms, gofs, res_path=res_path, variable=variable):
    dict_gof = {gof: {m: {} for m in algorithms} for gof in gofs}

    numerical_gofs = {'gofs': ['nse', 'rmse'], 'type': "multidim"}
    behavior_gofs = {'gofs': ['aic', 'mic'], 'type': "patrón"}

    for gof in gofs:
        if gof in numerical_gofs['gofs']:
            g, type  =gof , numerical_gofs['type']
        elif gof in behavior_gofs['gofs']:
            g, type = gof, behavior_gofs['type']

        for m in algorithms:
             a = np.load(res_path + f'{m}/valid_{g}.npy', allow_pickle=True).tolist()[variable][type][gof]['likes']['weighted_res']
             dict_gof[gof][m] = np.average(a, axis=1)

    return dict_gof


def load_observe_data(calib=True):
    pass


def load_simlated_data(calib=True):
    pass
