import numpy as np

maxi = ['nse', 'mic']
mini = ['aic', 'rmse']


def process_calib_likes(calib_likes, algorithms, gofs, top_percent=0.2):
    maxi = ['nse', 'mic']
    mini = ['aic', 'rmse']

    proc_likes = {g: {} for g in gofs}

    for gof in gofs:
        for m in algorithms:
            likes = calib_likes[gof][m]
            if gof == 'aic' and np.nanmean(likes) > 0:
                likes = np.negative(likes)
            elif gof =='rmse' and np.nanmean(likes) < 0:
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
    norm = np.array([((data[:, i] - mu[i]) / sg[i]) for i, p in enumerate(npoly)])  # 38*61
    return norm


def _soil_canal(all_poly_dt):
    cls = ['Buchanan, Head', 'Buchanan, Middle', 'Buchanan, Tail', 'Farida, Head', 'Farida, Middle', 'Farida, Tail',
           'Jhang, Middle', 'Jhang, Tail', 'Chuharkana, Tail']

    s_p = {c: [] for c in cls}

    for p in all_poly_dt:
        if p in (17, 52, 185):
            s_p[cls[0]].append(p)
        elif p in (36, 85, 132):
            s_p[cls[1]].append(p)
        elif p in (110, 125, 215):
            s_p[cls[2]].append(p)
        elif p in (7, 13, 76, 71):
            s_p[cls[3]].append(p)
        elif p in (25, 77, 123, 168, 171):
            s_p[cls[4]].append(p)
        elif p in (54, 130, 172, 174, 178, 187, 191, 202, 205):
            s_p[cls[5]].append(p)
        elif p in (16, 22, 80, 94):
            s_p[cls[6]].append(p)
        elif p in (50, 121):
            s_p[cls[7]].append(p)
        elif p in (143, 164, 175, 203):
            s_p[cls[8]].append(p)
    return s_p

def _combine_soil_canal_data(polys, soil_canal='soil'):
    sc_data = _soil_canal(polys)

    def _nrow(len_cols):
        if len_cols <= 4:
            return {'nrow':1}
        elif len_cols > 3:
            return {'nrow': len_cols}


    posi = ['Head', 'Middle', 'Tail']
    soils = list(set([sc[:sc.index(',')] for sc in sc_data]))

    soil_data = {s: { 'ncol': []} for s in soils}
    canal_data = {pos: { 'ncol': []} for pos in posi}

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

