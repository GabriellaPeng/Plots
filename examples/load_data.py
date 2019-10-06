import numpy as np
import pandas as pd
from Paths import res_path, variable

def load_obs_data(csv_path):
    res = pd.read_csv(csv_path)

    obs_data = {}
    for row in res.values:
        obs_data[row[1]] = row[2:]
    return  obs_data #res.columns[2:len(res.columns)]


def load_calib_gof_data(algorithms, gofs, res_path=res_path):
    dict_gof = {gof: {m: {} for m in algorithms} for gof in gofs}

    for gof in gofs:
        for m in algorithms:
            dict_gof[gof][m] = np.load(res_path + f'{m}/{m}_{gof}_PrmProb.npy', allow_pickle=True).tolist()['likes']

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

    numerical_gofs = {'gofs': ['nse', 'rmse'], 'type': "multidim", 'like_name': 'likes'}
    behavior_gofs = {'gofs': ['aic', 'mic'], 'type': "patrón", 'like_name': 'aic_likes'}

    for gof in gofs:
        if gof in numerical_gofs['gofs']:
            g, type, likes =gof + '_multidim', numerical_gofs['type'], numerical_gofs['like_name']
        elif gof in behavior_gofs['gofs']:
            g, type, likes= gof, behavior_gofs['type'], behavior_gofs['like_name']

        for m in algorithms:
             a = np.load(res_path + f'{m}/valid_{g}.npy', allow_pickle=True).tolist()[variable][type][gof]

             if gof in numerical_gofs['gofs']:
                 dict_gof[gof][m] = a[likes]
             else:
                dict_gof[gof][m] = a['weighted_sim'][likes]

    return dict_gof


def load_observe_data(calib=True):
    pass


def load_simlated_data(calib=True):
    pass
