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