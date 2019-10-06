


def _set_ax_marker(ax, xlabel, ylabel, title, xlim=None, ylim=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    elif ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_title(title)


def clr_marker(mtd_clr=False, mtd_mkr=False, obj_fc_clr=False, obj_fc_mkr=False, wt_mu_m=False):
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
