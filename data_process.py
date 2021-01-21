import inspect
import re

import numpy as np

from Paths import calib_polygon, valid_polygon
from examples.load_data import load_obs_data, load_calib_gof_data, load_valid_likes

maxi = ['nse', 'mic']
mini = ['aic', 'rmse']


def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if
                 var_val is var]
        if len(names) > 0:
            return names[0]


def process_calib_likes(calib_likes, algorithms, gofs, top_percent=0.2):
    maxi = ['nse', 'mic']
    mini = ['aic', 'rmse']

    proc_likes = {g: {} for g in gofs}

    for gof in gofs:
        for m in algorithms:
            likes = calib_likes[gof][m]
            if gof == 'aic' and np.nanmean(likes) > 0:
                likes = np.negative(likes)
            elif gof == 'rmse' and np.nanmean(likes) < 0:
                likes = np.negative(likes)

            no_likes = int(len(likes) * top_percent)

            if gof in maxi:
                a = np.sort(likes)[-no_likes:]
            elif gof in mini:
                a = np.sort(likes)[:no_likes]
            proc_likes[gof][m] = a

    return proc_likes


def equal_len_calib_valid(calib_data, valid_data):
    for gof in calib_data:
        for m in calib_data[gof]:
            if len(calib_data[gof][m]) != len(valid_data[gof][m]):
                min_len = min(len(calib_data[gof][m]), len(valid_data[gof][m]))
                if gof in mini:
                    calib_data[gof][m] = np.sort(calib_data[gof][m])[:min_len]
                    valid_data[gof][m] = np.sort(valid_data[gof][m])[:min_len]

                elif gof in maxi:
                    calib_data[gof][m] = np.sort(calib_data[gof][m])[-min_len:]
                    valid_data[gof][m] = np.sort(valid_data[gof][m])[-min_len:]

    return calib_data, valid_data


def mu_sg(npoly, data):
    mu = np.array([np.nanmean(data[:, i]) for i, p in enumerate(npoly)])  # 215
    sg = np.array([np.nanstd(data[:, i]) for i, p in enumerate(npoly)])  # 215
    return mu, sg


def normalize_data(npoly, mu, sg, data):
    norm = np.array(
        [((data[:, i] - mu[i]) / sg[i]) for i, p in enumerate(npoly)])  # 38*61
    return norm


def _soil_canal(all_poly_dt, good_polys):
    cls = ['Chuharkana, Tail', 'Buchanan, Head', 'Buchanan, Middle', 'Buchanan, Tail',
           'Farida, Head', 'Farida, Middle',
           'Farida, Tail', 'Jhang, Middle', 'Jhang, Tail']

    s_p = {c: [] for c in cls}

    if good_polys:  # only good evaluation polygons
        for p in all_poly_dt:
            if p ==175:  # CT
                s_p[cls[0]].append(p)
            # elif p == 185:  # BH
            #     s_p[cls[1]].append(p)
            # elif p == 36:  # BM
            #     s_p[cls[2]].append(p)
            # elif p == 125:  # BT
            #     s_p[cls[3]].append(p)
            # elif p in (76, 71):  # FH
            #     s_p[cls[4]].append(p)
            # elif p == 168:  # FM
            #     s_p[cls[5]].append(p)
            elif p == 178:  # FT 130, 174,
                s_p[cls[6]].append(p)
            # elif p == 22:  # JM
            #     s_p[cls[7]].append(p)
            # elif p == 50:  # JT
            #     s_p[cls[8]].append(p)
    else:
        for p in all_poly_dt:
            if p in (143, 164, 175, 203):  # CT
                s_p[cls[0]].append(p)
            elif p in (17, 52, 185):  # BH
                s_p[cls[1]].append(p)
            elif p in (36, 85, 132):  # BM
                s_p[cls[2]].append(p)
            elif p in (110, 125, 215):  # BT
                s_p[cls[3]].append(p)
            elif p in (7, 13, 76, 71):  # FH
                s_p[cls[4]].append(p)
            elif p in (25, 77, 123, 168, 171):  # FM
                s_p[cls[5]].append(p)
            elif p in (54, 130, 172, 174, 178, 187, 191, 202, 205):  # FT
                s_p[cls[6]].append(p)
            elif p in (16, 22, 80, 94):  # JM
                s_p[cls[7]].append(p)
            elif p in (50, 121):  # JT
                s_p[cls[8]].append(p)
    return s_p


def _combine_soil_canal_data(polys, soil_canal='soil'):
    sc_data = _soil_canal(polys)

    def _nrow(len_cols):
        if len_cols <= 4:
            return {'nrow': 1}
        elif len_cols > 3:
            return {'nrow': len_cols}

    posi = ['Head', 'Middle', 'Tail']
    soils = list(set([sc[:sc.index(',')] for sc in sc_data]))

    soil_data = {s: {'ncol': []} for s in soils}
    canal_data = {pos: {'ncol': []} for pos in posi}

    if soil_canal == 'soil':
        for soil in soils:
            for sc, v in sc_data.items():
                if sc.startswith(soil):
                    soil_data[soil]['ncol'].extend(v)
            soil_data[soil].update(_nrow(len(soil_data[soil]['ncol'])))

        return soil_data

    elif soil_canal == 'canal':
        for pos in posi:
            for sc, v in sc_data.items():
                if sc.endswith(pos):
                    canal_data[pos]['ncol'].extend(v)
            canal_data[pos].update(_nrow(len(canal_data[pos]['ncol'])))

        return canal_data


def clr_marker(mtd_clr=False, mtd_mkr=False, obj_fc_clr=False, obj_fc_mkr=False,
               wt_mu_m=False):
    if mtd_clr:
        return {'fscabc': 'b', 'dream': 'orange', 'mle': 'r', 'demcz': 'g'}
    elif mtd_mkr:
        return {'fscabc': 'o', 'dream': 'v', 'mle': 'x', 'demcz': '*'}
    elif obj_fc_clr:
        return {'aic': 'b', 'nse': 'orange', 'rmse': 'g'}
    elif obj_fc_mkr:
        return {'aic': 'o', 'nse': 'v', 'rmse': 'x'}
    elif wt_mu_m:
        return {'weighted_sim': 'orange', 'mean_sim': 'b', 'median_sim': 'g'}


def proc_soil_canal_mask(soil_canal, soil_canal_mask, gen_mask=False):
    if soil_canal == 'soil':
        soil_canal = {s: [] for s in ['C', 'B', 'F', 'J']}
        for sc, vals in soil_canal_mask.items():
            soil_canal[sc[0]].extend(vals)
    elif soil_canal == 'canal':
        soil_canal = {s: [] for s in ['H', 'M', 'T']}
        for sc, vals in soil_canal_mask.items():
            soil_canal[sc[sc.find(',') + 2]].extend(vals)

    elif soil_canal == 'all':
        soil_canal = {sc[0] + sc[sc.find(',') + 2]: vals for sc, vals in
                      soil_canal_mask.items()}

    if gen_mask and soil_canal != 'all':
        mask = {s: [i + 0.5] for i, s in enumerate(soil_canal)}
        return soil_canal, mask
    else:
        return soil_canal


def proc_data_to_soil_canal(data, soil_canal):
    proc_data = np.zeros([len(data), len(soil_canal)])
    polys = list(np.sort([j for i in list(soil_canal.values()) for j in i]))

    for i, (sc, d_poly) in enumerate(soil_canal.items()):
        proc_data[:, i] = np.average(
            np.take(data[:, ], [polys.index(j) for j in d_poly], axis=1), axis=1)

    return proc_data


def ylims(gof, data, algorithms):
    if gof == 'mic':
        ymin, ymax = 0.6, 1.0 + 0.05
    elif gof == 'aic':
        ymin, ymax = -60, 100
    else:
        ymin, ymax = min([np.min(data[gof][al]) for al in algorithms]), max(
            [np.max(data[gof][al]) for al in algorithms])
    return ymin, ymax


def load_res_sim(type_results):
    if type_results == 'all':
        type_res = 'all_res'
        type_sim = 'weighted_sim'

    elif type_results == 'top':
        type_res = 'top_res'
        type_sim = 'top_weighted_sim'

    return type_res, type_sim


def plot_soil_canal(soil_canal, polys='all', calib_valid='valid', good_polys=False):
    soil = {s: [] for s in ['C', 'B', 'F', 'J']}
    canal = {c: [] for c in ['H', 'M', 'T']}

    if polys == 'best':
        if calib_valid == 'valid':
            if soil_canal == 'soil':
                npoly = [164, 36, 76, 50]  # TODO: this can be changed
            elif soil_canal == 'canal':
                npoly = [76, 36, 164]
        elif calib_valid == 'calib':
            print('Please add the the best calibration polygons to present')

    elif polys == 'all':
        polygon_path = [calib_polygon if calib_valid == 'calib' else valid_polygon][0]
        npoly = list(load_obs_data(polygon_path))

    d_polys = _soil_canal(npoly, good_polys)

    for s_c, l_polys in d_polys.items():
        soil[s_c[0]].extend(l_polys)
        canal[s_c[s_c.find(' ') + 1]].extend(l_polys)

    # plot by soils
    if soil_canal == 'soil':
        sc = soil
    elif soil_canal == 'canal':
        sc = canal
    return sc, npoly


def run_sim_vs_obs(l_poly, algorithm, gofs, type_sim, type_res, data):
    sim_norm = [{gof: {} for gof in gofs} if l_poly['nrow'] > 1 else {}][0]
    res_dt = [{gof: {} for gof in gofs} if l_poly['nrow'] > 1 else {}][0]

    for gof in gofs:
        if l_poly['nrow'] > 1:
            sim_norm[gof].update({type_sim: data[gof][algorithm][type_sim]})
            res_dt[gof].update({type_res: data[gof][algorithm][type_res]})

        else:
            sim_norm[type_sim] = data[gof][algorithm][type_sim]
            res_dt[type_sim] = data[gof][algorithm][type_res]

    return sim_norm, res_dt


def find_optimiza_parameter(parameter_data, algorithms, gofs, sc_type):
    maxi, mini = ['aic', 'mic', 'nse'], ['rmse']
    dict_sc, polys = plot_soil_canal(sc_type)
    valid_gofs = load_valid_likes(algorithms, gofs,
                                  top_weighted=True)  # TODO: later change to False
    bf_opt_params, sd_opt_params = {gof: {} for gof in gofs}, {
        gof: {al: {} for al in algorithms} for gof in gofs}

    if 'aic' in gofs:
        calib_gofs = load_calib_gof_data(algorithms, ['aic'], tops=True)['aic']
        for al in algorithms:
            valid_gofs['aic'][al] = valid_gofs['aic'][al] - calib_gofs[al]

    for gof in gofs:
        for al in algorithms:
            ind_opt = [np.argmax(valid_gofs[gof][al]) if gof in maxi else np.argmin(
                valid_gofs[gof][al])][0]
            for prm, val in parameter_data[gof][al].items():
                if '-' in prm:
                    prm = prm[:prm.find(' ')]
                    if prm not in bf_opt_params[gof]:
                        bf_opt_params[gof].update({prm: {}})
                    if al not in bf_opt_params[gof][prm]:
                        bf_opt_params[gof][prm].update({al: {}})

                    val = np.take(val, [i - 1 for i in polys], axis=1)  # 500*18
                    val = val[ind_opt, :]  # 18
                    for sc, sc_val in dict_sc.items():
                        bf_opt_params[gof][prm][al][sc] = np.mean(
                            np.take(val, [polys.index(i) for i in sc_val]))
                else:
                    prm = prm[:prm.find(' ')] + ',' + \
                          [prm[i] for i in [m.end() for m in re.finditer(' ', prm)]][0]
                    prm = [
                        prm[:-1] + 'Summer' if prm[-1] == 'K' else prm[:-1] + 'Winter' if
                        prm[-1] == 'r' else 'CTW'][0]
                    sd_opt_params[gof][al][prm] = val[ind_opt]

    return bf_opt_params, sd_opt_params