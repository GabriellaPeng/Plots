import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_process import clr_marker
from examples.ex_Marks import _clr_marker, _set_ax_marker


def bounds_cis(figures, l_poly, npoly, obs_norm, sims, p_sim, save_plot, plot_name):
    types = plot_name['type']
    alg = plot_name['alg']

    l_pol = l_poly['ncol']
    nrow = l_poly['nrow']

    if isinstance(p_sim, dict):
        ncol = len(l_pol)
    else:
        ncol = [len(l_pol) if nrow == 1 else 1][0]

    def _ind_p(c, r, sims=sims, p_sim=p_sim):
        if isinstance(p_sim, dict):
            gofs = list(sims)
            ind_p = npoly.index(l_pol[c])
            p = l_pol[c]
            ind_ax = [c, r, ncol, nrow]
            sims = sims[gofs[r]][:, :, ind_p]
            p_sim = p_sim[gofs[r]][:, ind_p]
            gof = gofs[r]
        else:
            ind_p = npoly.index(l_pol[c + r])
            p = l_pol[c + r]
            ind_ax = [c, r, ncol]
            sims = sims[:, :, ind_p]
            p_sim = p_sim[:, ind_p]
            gof = ''
        return ind_p, p, ind_ax, sims, p_sim, gof

    if 'bounds' in figures:
        plt.ioff()
        fig1 = plt.figure(figsize=(4 * 2 * ncol, 4 * nrow), constrained_layout=False)
        gs1 = GridSpec(ncols=ncol, nrows=nrow, wspace=0.0, hspace=0.0)

        for c in range(ncol):
            for r in range(nrow):
                vars()[f'ax1{r}{c}'] = fig1.add_subplot(gs1[r, c])

        for c in range(ncol):
            for r in range(nrow):
                ind_p, p, ind_ax, sims, p_sim, gof= _ind_p(c, r)
                param_uncertainty_bounds(sims, obs_norm[:, ind_p], p, p_sim,
                                         ax=vars()[f'ax1{r}{c}'], ind_ax=ind_ax, ylabel=gof)

        fig1.savefig(save_plot + f'Bs.{types[:3]}.{alg}.png')
        plt.close(fig1)

    elif 'cis' in figures:
        plt.ioff()
        fig2 = plt.figure(figsize=(4 * 2 * ncol, 4 * nrow), constrained_layout=False)
        gs2 = GridSpec(ncols=ncol, nrows=nrow, wspace=0.0, hspace=0.0)

        for c in range(ncol):
            for r in range(nrow):
                vars()[f'ax2{r}{c}'] = fig2.add_subplot(gs2[r, c])

        for c in range(ncol):
            for r in range(nrow):
                ind_p, p, ind_ax, sims, p_sim, gof= _ind_p(c, r)
                plot_ci(sims, obs_norm[:, ind_p], f'Poly-{p}', ax=vars()[f'ax2{r}{c}'],
                        ind_ax=ind_ax, ylabel=gof)
        fig2.savefig(save_plot + f'CI.{types[:3]}.{alg}.png')
        plt.close(fig2)


def plot_top_sim_obs(dict_sim_norm, obs_norm, npoly, save_plot, dict_sim, l_poly=None,
                     figures=['cis'], alg=None):
    '''l_poly = {nrow: int, ncol:[list of list]}'''

    sns.set(style="darkgrid")
    if l_poly is not None:
        if 'weighted_res' not in dict_sim:
            type = 'weighted_res'
            p_sim = {gof: v['top_weighted_sim'] for gof, v in dict_sim_norm.items()}
            sims = {gof: v['weighted_res'] for gof, v in dict_sim.items()}

            bounds_cis(figures, l_poly, npoly, obs_norm, sims, p_sim, save_plot, {'type': type, 'alg': alg})

        else:
            for type, sims in dict_sim.items():
                if type == 'all_res':
                    p_sim = dict_sim_norm['weighted_sim']
                elif type == 'weighted_res':
                    p_sim = dict_sim_norm['top_weighted_sim']

                bounds_cis(figures, l_poly, npoly, obs_norm, sims, p_sim, save_plot, {'type': type, 'alg': alg})


    else:
        for type, sims in dict_sim.items():
            if type == 'all_res':
                p_sim = dict_sim_norm['weighted_sim']
            elif type == 'weighted_res':
                p_sim = dict_sim_norm['top_weighted_sim']

            for ind_p, p in enumerate(npoly):

                if 'bounds' in figures:
                    plt.ioff()
                    fig1 = plt.figure(figsize=(4 * 2 * 1, 4), constrained_layout=False)
                    gs1 = GridSpec(ncols=1, nrows=1, wspace=0.0, hspace=0.0)
                    ax1 = fig1.add_subplot(gs1[0])

                    param_uncertainty_bounds(sims[:, :, ind_p], obs_norm[:, ind_p], p, p_sim[:, ind_p],
                                             ax=ax1, ind_ax=[0])
                    fig1.savefig(save_plot + f'bounds.{type[:3]}.{p}.png', dpi=400)
                    plt.close(fig1)

                elif 'cis' in figures:
                    plt.ioff()
                    fig2 = plt.figure(figsize=(4 * 2 * 1, 4), constrained_layout=False)
                    gs2 = GridSpec(ncols=1, nrows=1, wspace=0.0, hspace=0.0)

                    ax2 = fig2.add_subplot(gs2[0])

                    plot_ci(sims[:, :, ind_p], obs_norm[:, ind_p], f'Poly-{p}', ax=ax2, ind_ax=[0])
                    fig2.savefig(save_plot + f'ci.{type[:3]}_{p}.png', dpi=400)
                    plt.close(fig2)


def param_uncertainty_bounds(sim_res, observations, poly, proc_sim, ax, ind_ax=None, ylabel=None):
    q5, q25, q75, q95 = [], [], [], []
    for t in range(len(observations)):
        q5.append(np.percentile(sim_res[:, t], 2.5))
        q95.append(np.percentile(sim_res[:, t], 97.5))

    # ax.plot(q5, color='lightblue', linestyle='solid')
    # ax.plot(q95, color='lightblue', linestyle='solid')
    ax.fill_between(np.arange(0, len(q5), 1), list(q5), list(q95), facecolor='lightblue', zorder=0,
                    linewidth=0, label='5-95% simulation bound', alpha=0.8)

    ax.plot(proc_sim, color=f"{_clr_marker(wt_mu_m=True)['weighted_sim']}", label=f'Weighted average simulation',
            linestyle='dashed')

    ax.plot(observations, 'r-', label=f'Polygon{poly} observation')
    ax.set_ylim(-6, 9)

    yrange = [str(i) for i in np.arange(-4, 9, 2)]
    yrange.insert(0, '')

    ax.set_yticks(np.arange(-6, 9, 2))
    ax.set_xticks(np.arange(0, len(observations), 2))

    xticklabel = np.arange(2, len(observations) + 2, 2)
    ind_ax_set(ind_ax, yrange, 15, xticklabel, 10, ax, ylabel=ylabel)

    ax.legend(loc='upper left', prop={'size': 10})


def plot_ci(sims, obs, poly, ax, ind_ax=None, ylabel=None):
    time = sims.shape[1]

    x_data = np.arange(0, 101, 10)
    y_data = {100 - 2 * i: [] for i in np.arange(0, 50, 5)}

    for i in np.arange(0, 50, 5):
        vars()[f'q{i}'], vars()[f'q{100 - i}'] = [], []

    for t in range(time):
        for i in np.arange(0, 50, 5):
            lower_b, upper_b = np.float(i / 2), np.float(100 - i / 2)
            lower, upper = np.percentile(sims[:, t], lower_b), np.percentile(sims[:, t], upper_b)
            vars()[f'q{i}'].append(lower)
            vars()[f'q{100 - i}'].append(upper)

            if lower <= obs[t] <= upper:
                y_data[100 - 2 * i].append(obs[t])

    y_dt = [len(v) / time for i, v in y_data.items()]
    y_dt.append(0)
    y_data = np.sort(y_dt)

    yarang = [np.round(i, 2) for i in np.arange(0, 1.1, 0.1)]
    ax.plot(x_data, yarang, 'g-.')
    ax.plot(x_data, y_data, 'r.-', label=f"{poly} CI")

    ax.set_yticks(yarang)
    ax.set_xticks(x_data)

    ind_ax_set(ind_ax, yarang, 11, [f'{i}CI' for i in x_data], 12, ax, ylabel=ylabel)

    ax.legend(loc='upper left', frameon=False, prop={'size': 13})

    # if isinstance(label_poly, list):
    #     label_poly = label_poly[0]
    #
    #
    # x_data = np.insert(x_data, 0, 0)
    # y_data = np.insert(y_data, 0, 0)
    # y_data[np.isnan(y_data)] = 1

    # ax.plot(x_data, x_data, 'g-.', label="% of simulations for evaluation over CIs")
    # ax.plot(x_data, y_data, 'r.-', label=f"{label_poly} CI")

    # ax.xaxis.set_major_formatter(plt.NullFormatter())
    # ax.set_xticks(x_data, [np.round(i, 1) for i in x_data])
    # ax.tick_params(axis="x", labelsize=8)
    # for i in x_data:
    #     ax.axvline(i, color='grey', alpha=0.4, linestyle='--')
    # ax.set_yticks(y_data, [np.round(i, 2) for i in y_data])
    # ax.legend(loc='upper left', frameon=False)

    # if ind_ax is not None and ind_ax == 0:
    #     ax.set_yticklabels([np.round(i, 2) for i in np.arange(0, 1.1, 0.1)], size=15)
    # else:
    #     ax.yaxis.set_major_formatter(plt.NullFormatter())


def ind_ax_set(ind_ax, yticklabel, yticksize, xticklabel, xticksize, ax, ylabel, ylabelsize=20):
    if len(ind_ax) > 1:
        c, r, nrow = ind_ax[0], ind_ax[1], ind_ax[3]
        if c == 0:
            ax.set_yticklabels(yticklabel, size=yticksize)
            ax.set_ylabel(ylabel.upper(), fontsize=20)
            if r == nrow-1:
                ax.set_xticklabels(xticklabel, size=xticksize)
            else:
                ax.xaxis.set_major_formatter(plt.NullFormatter())

        elif r == nrow - 1:
            ax.set_xticklabels(xticklabel, size=xticksize)
            ax.yaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())

    else:
        ax.set_yticklabels(yticklabel, size=yticksize)
        ax.set_xticklabels(xticklabel, size=xticksize)


def _ci(data, obs_data, confidence=0.95, probs=None, counts=None):
    if not isinstance(data, np.ndarray):
        a = 1.0 * np.array(data)
    else:
        a = data
        # a=100*39

    n = len(a)

    if probs is None:
        m, se = np.nanmean(a), stats.sem(a)
    else:
        weight = np.asarray([(p - np.min(probs)) / np.ptp(probs) for p in probs])
        m = np.average(a, weights=weight)
        se = np.sqrt(np.average((a - m) ** 2, weights=weight) / len(weight))

    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    intervals = [m - h, m + h]

    if counts is not None:
        if intervals[0] <= obs_data <= intervals[1]:
            counts.append(1)
    else:
        return intervals


def confidence_interval(obs_norm, proc_sim, dict_probs):  # 495*41*19, 41*19
    poly = obs_norm.shape[1]
    time = len(obs_norm)

    pass
    # cis, no_ci = np.arange(0.05, 1.05, 0.05), len(np.arange(0.05, 1.05, 0.05))
    # mean_pcentl, top_wt_pcentl, = np.zeros([no_ci, poly]), np.zeros([no_ci, poly])
    #
    # top_prob = dict_probs['weighted_res']  # 100*18
    #
    # all_res = proc_sim['all_res']
    # weighted_res = proc_sim['weighted_res']
    #
    # for p in range(poly):
    #     for i, ci in enumerate(cis):
    #         mean_count, top_wt_count = [], []
    #         for t in range(time):
    #             obs = obs_norm[t, p]
    #             _ci(all_res[:, t, p], obs, ci, probs=None, counts=mean_count)
    #             _ci(weighted_res[:, t, p], obs, ci, top_prob[:, p], counts=top_wt_count)
    #
    #         mean_pcentl[i, p], top_wt_pcentl[i, p] = len(mean_count) / time, len(top_wt_count) / time
    # pcentl = np.zeros_like(obs_norm, dtype=float)
    # for t in range(len(obs_norm)):
    #     for p in range(obs_norm.shape[1]):
    #         if np.isnan(obs_norm[t, p]):
    #             pcentl[t, p] = np.nan
    #         else:
    #             perc = estad.percentileofscore(sim_norm[:, t, p], obs_norm[t, p], kind='weak')
    #             pcentl[t, p] = abs(0.5 - perc / 100) * 2
    # return {'all_mean': mean_pcentl, 'top_wt': top_wt_pcentl}


def plot_gof_convergence(gof, methods, data, save=None):
    def scale(x, out_range=(0, 1)):
        domain = np.min(x), np.max(x)
        y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
        return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range)

    vals = [i for m, v in data.items() for i in v]
    if gof == 'rmse':
        mini, maxi = -3.0, 0.0
    elif gof == 'nse':
        mini, maxi = -15.0, 1.0
    elif gof == 'mic':
        mini, maxi = np.min(vals), np.max(vals)+0.1
    elif gof=='aic':
        mini, maxi = np.min(vals), np.max(vals)+10

    lines = []
    sns.set()
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.set_ylim(mini, maxi)

    for m, v in data.items():
        x = np.arange(0, len(v))
        lines += ax.plot(x, v, '-', color=clr_marker(mtd_clr=True)[m])
        # line = ax.plot(x, v, '-', color=clr_marker(mtd_clr=True)[m])
        # ax.legend(line, [m.upper()], loc='upper left', frameon=False)
    ax.set_xlabel('Number of Runs', fontname="Cambria", fontsize=20)
    ax.set_ylabel(f'{gof.upper()}', fontname="Cambria", fontsize=20)
    ax.legend(lines, [m.upper() for m in methods], ncol=4, loc='upper left', frameon=False)
    if save is not None:
        fig.savefig(save + f'{gof}', dpi=500)
        plt.clf()
        plt.close(fig)