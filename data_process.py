import numpy as np


def process_calib_likes(calib_likes, algorithms, gofs, top_percent=0.2):
    maxi = ['nse', 'mic']
    mini = ['aic', 'rmse']

    proc_likes = {g: { } for g in gofs}

    for gof in gofs:
        for m in algorithms:
            likes = calib_likes[gof][m]
            no_likes = len(likes) * top_percent

            if gof in maxi:
                proc_likes[gof][m] = np.sort(likes)[-no_likes:]
            elif gof in mini:
                proc_likes[gof][m] = np.sort(likes)[:no_likes]
    return proc_likes