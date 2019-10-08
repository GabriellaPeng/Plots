import numpy as np

def theil_inequal(y_predict, y_obs):
    not_nan = np.where(~np.isnan(y_obs))
    y_obs = y_obs[not_nan]
    y_predict = y_predict[not_nan]

    mu_pred = np.nanmean(y_predict)
    mu_obs = np.nanmean(y_obs)

    std_pred = np.nanstd(y_predict)
    std_obs = np.nanstd(y_obs)

    mse = np.nanmean((y_predict - y_obs) ** 2)

    r = np.nanmean(((y_obs - mu_obs) * (y_predict - mu_pred)) / (std_obs * std_pred))
    # um = np.abs((mu_pred ** 2 - mu_obs ** 2 ) / mse)
    # us = np.abs((std_pred ** 2 - std_obs ** 2) / mse)
    # uc = np.abs(2 * (1 - r) * std_pred * std_obs / mse)

    um = (mu_pred-mu_obs)**2/ mse
    us = (std_pred -std_obs) ** 2/ mse
    uc = 2 * (1 - r)*std_pred * std_obs/ mse

    return mse, um, us, uc