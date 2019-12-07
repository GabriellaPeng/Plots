import json
import os

# from xarray import Dataset
import numpy as np
import pandas as pd
from Paths import res_path, variable
from data_process import process_calib_likes


def load_obs_data(csv_path):
    res = pd.read_csv(csv_path)

    obs_data = {}
    for row in res.values:
        obs_data[row[1]] = row[2:]
    return obs_data  # res.columns[2:len(res.columns)]


def load_calib_gof_data(algorithms, gofs, res_path=res_path, tops=False):
    dict_gof = {gof: {m: {} for m in algorithms} for gof in gofs}

    for gof in gofs:
        for m in algorithms:
            calib_likes = np.load(res_path + f'{m}/calib_{gof}_PrmProb.npy', allow_pickle=True).tolist()['likes']
            dict_gof[gof][m] = calib_likes
    if tops:
        dict_gof = process_calib_likes(dict_gof, algorithms, gofs, top_percent=0.2)

    return dict_gof


def load_calib_param_data(algorithms, gofs, res_path=res_path, tops=False):
    dict_params = {gof: {m: {} for m in algorithms} for gof in gofs}

    for gof in gofs:
        for m in algorithms:
            mask = np.load(res_path + f'{m}/calib_{gof}.npy', allow_pickle=True).tolist()['buenas']
            calib_params = np.load(res_path + f'{m}/calib_{gof}.npy', allow_pickle=True).tolist()['parameters']

            if tops:
                calib_params = {prm: np.take(vals, mask, axis=0) for prm, vals in calib_params.items()}
            dict_params[gof][m] = calib_params

    return dict_params


def load_parameter_data(algorithms, gofs, res_path=res_path):
    dict_params = {gof: {m: {} for m in algorithms} for gof in gofs}

    for gof in gofs:
        for m in algorithms:
            a = np.load(res_path + f'{m}/calib_{gof}_PrmProb.npy', allow_pickle=True).tolist()
            dict_params[gof][m] = {p: v for p, v in a.items() if p != 'likes'}

    return dict_params


def load_valid_likes(algorithms, gofs, res_path=res_path, variable=variable, weighted=False):
    dict_gof = {gof: {m: {} for m in algorithms} for gof in gofs}

    numerical_gofs = {'gofs': ['nse', 'rmse'], 'type': "multidim"}
    behavior_gofs = {'gofs': ['aic', 'mic'], 'type': "patrón"}

    for gof in gofs:
        if gof in numerical_gofs['gofs']:
            g, type = gof, numerical_gofs['type']
        elif gof in behavior_gofs['gofs']:
            g, type = gof, behavior_gofs['type']

        for m in algorithms:
            if weighted:
                a = np.load(res_path + f'{m}/valid_{g}.npy', allow_pickle=True).tolist()[variable][type][gof]['likes'][
                    'weighted_res']
                dict_gof[gof][m] = np.average(a, axis=1)

            else:
                a = np.load(res_path + f'{m}/valid_{g}.npy', allow_pickle=True).tolist()[variable][type][gof]['likes']
                dict_gof[gof][m] = a

    return dict_gof


def load_valid_res(algorithms, gofs, res_path=res_path, variable=variable, weighted_sim=False):
    dict_gof = {gof: {m: {} for m in algorithms} for gof in gofs}

    numerical_gofs = {'gofs': ['nse', 'rmse'], 'type': "multidim"}
    behavior_gofs = {'gofs': ['aic', 'mic'], 'type': "patrón"}

    for gof in gofs:
        if gof in numerical_gofs['gofs']:
            g, type = gof, numerical_gofs['type']
        elif gof in behavior_gofs['gofs']:
            g, type = gof, behavior_gofs['type']

        for m in algorithms:
            if weighted_sim:
                a = np.load(res_path + f'{m}/valid_{g}.npy', allow_pickle=True).tolist()[variable][type][gof]
                dict_gof[gof][m]['weighted_sim'] = a['weighted_sim']
                dict_gof[gof][m].update({'top_weighted_sim': a['top_weighted_sim']})

            else:
                a = np.load(res_path + f'{m}/valid_{g}.npy', allow_pickle=True).tolist()[variable][type][gof]
                dict_gof[gof][m]['weighted_res'] = a['weighted_res']
                dict_gof[gof][m].update({'all_res': a['all_res']})

    return dict_gof


def load_theil_data(algorithms, gofs, res_path=res_path, variable=variable):
    theil_data = {al: { } for al in algorithms}

    numerical_gofs = {'gofs': ['nse', 'rmse'], 'type': "multidim"}
    behavior_gofs = {'gofs': ['aic', 'mic'], 'type': "patrón"}

    type_sim = 'top_weighted_sim'

    for gof in gofs:
        if gof in numerical_gofs['gofs']:
            type = numerical_gofs['type']
        elif gof in behavior_gofs['gofs']:
            type = behavior_gofs['type']

        for m in algorithms:
            a = np.load(res_path + f'{m}/valid_{gof}.npy', allow_pickle=True).tolist()[variable][type][gof]['Theil'][type_sim]
            theil_data[m][gof] = a

    return theil_data


def load_observe_data(csv_path, warmup_period=None):
    data = load_obs_data(csv_path)

    if warmup_period is not None:
        obs_data = np.zeros([len(data), len(list(data.values())[0])-warmup_period])
        for i, (p, vals) in enumerate(data.items()):
            obs_data[i] = vals[warmup_period:]
    else:
        obs_data = np.zeros([len(data), len(list(data.values())[0])])
        for i, (p, vals) in enumerate(data.items()):
            obs_data[i] = vals

    return obs_data.T #39*18


def load_simlated_data(algorithm, gof, simul_path, mask, variable,
                       warmup_period=True):  # mask = {obs: obs_norm, 39*19, polys:[]}
    d_val = {'dream': {'aic': (500, 1000), 'nse': (0, 500), 'rmse': (0, 500), 'mic': (0, 555)},
             'mle': {'aic': (0, 500), 'nse': (500, 969), 'rmse': (500, 1000), 'mic': (1055, 1555)},
             'fscabc': {'aic': (144, 288), 'nse': (0, 500), 'rmse': (0, 500), 'mic': (0, 500)},
             'demcz': {'aic': (0, 500), 'nse': (500, 1000), 'rmse': (1000, 1500), 'mic': (555, 1055)}}

    simul_path = simul_path + f'{algorithm}/{gof}/'
    n_sim = d_val[algorithm][gof]

    obs_data_mask = mask['obs']
    l_polys = mask['polys']
    top_ind = mask['top']  # [list of top 20 indices]

    sim_data = np.empty([len(top_ind), *obs_data_mask.values.shape])  # 100*39*19

    for i, v in enumerate(top_ind):
        if not warmup_period:
            sim_data[i, :] = np.asarray(
                [Dataset.from_dict(load_json(os.path.join(simul_path, f'{v + n_sim[0]}')))[
                     variable[0]].values[warmup_period:, j - 1] for j in l_polys]).T
        else:
            sim_data[i, :] = np.asarray(
                [Dataset.from_dict(load_json(os.path.join(simul_path, f'{v + n_sim[0]}')))[
                     variable[0]].values[:, j - 1] for j in l_polys]).T

    return sim_data


def load_json(arch, codif='UTF-8'):
    """
    Cargar un fuente json.
    Parameters
    ----------
    arch : str
        El fuente en el cual se encuentra el objeto json.
    codif : str
        La codificación del fuente.
    Returns
    -------
    dict | list
        El objeto json.
    """

    nmbr, ext = os.path.splitext(arch)
    if not len(ext):
        arch = nmbr + '.json'

    with open(arch, 'r', encoding=codif) as d:
        return json.load(d)
