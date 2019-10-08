import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_top_sim_obs(sim_norm, obs_norm, npoly, save_plot, proc_sim, percentails=None, l_poly=None):
    for pcentl in percentails:  # for pcentl, array in percentails.items()
        pcentl = np.asarray(pcentl)

        plt.ioff()
        if l_poly is not None:
            ncol = len(l_poly)

            fig1 = plt.figure(figsize=(4 * 2 * ncol, 4), constrained_layout=False)
            gs1 = GridSpec(ncols=len(l_poly), nrows=1, wspace=0.0, hspace=0.0)

            fig2 = plt.figure(figsize=(4 * 2 * ncol, 4), constrained_layout=False)
            gs2 = GridSpec(ncols=len(l_poly), nrows=1, wspace=0.0, hspace=0.0)
            for i in range(ncol):
                vars()[f'ax1' + str(i)] = fig1.add_subplot(gs1[i])
                vars()[f'ax2' + str(i)] = fig2.add_subplot(gs2[i])

            for i in range(ncol):
                ind_p = npoly.index(l_poly[i])
                p_sim = {'weighted_sim': proc_sim['weighted_sim'][:, ind_p]}
                # p_sim = {n: val[:, ind_p] for n, val in proc_sim.items() if val != 'prob'}
                param_uncertainty_bounds(sim_norm[:, :, ind_p], obs_norm[:, ind_p], l_poly[i], p_sim, save_plot,
                                         fig1, ax=vars()[f'ax1' + str(i)], ind_ax=i)

                plot_ci(np.arange(0.05, 1.05, 0.05), np.sort(pcentl[:, ind_p]), f'Poly-{l_poly[i]}', save_plot,
                        fig=fig2,
                        ax=vars()[f'ax2' + str(i)], ind_ax=i)

            fig1.savefig(save_plot + f'bounds_{l_poly}')
            fig2.savefig(save_plot + f'ci_{l_poly}')

        else:
            for ind_p, p in enumerate(npoly):
                fig1 = plt.figure(figsize=(4 * 2 * ncol, 4), constrained_layout=False)
                gs1 = GridSpec(ncols=len(l_poly), nrows=1, wspace=0.0, hspace=0.0)

                fig2 = plt.figure(figsize=(4 * 2 * ncol, 4), constrained_layout=False)
                gs2 = GridSpec(ncols=len(l_poly), nrows=1, wspace=0.0, hspace=0.0)
                for i in range(ncol):
                    vars()[f'ax1' + str(i)] = fig1.add_subplot(gs1[i])
                    vars()[f'ax2' + str(i)] = fig2.add_subplot(gs2[i])

                p_sim = {'weighted_sim': proc_sim['weighted_sim'][:, ind_p]}
                param_uncertainty_bounds(sim_norm[:, :, ind_p], obs_norm[:, ind_p], p, p_sim,
                                         ax=vars()[f'ax1' + str(i)], ind_ax=i)

                plot_ci(np.arange(0.05, 1.05, 0.05), np.sort(pcentl[:, ind_p]), f'Poly-{l_poly[i]}',
                        ax=vars()[f'ax2' + str(i)], ind_ax=i)

                fig1.savefig(save_plot + f'bounds_{l_poly}')
                fig2.savefig(save_plot + f'ci_{l_poly}')


def param_uncertainty_bounds(sim_res, observations, poly, proc_sim, ax, ind_ax=None):
    q5, q25, q75, q95 = [], [], [], []
    for t in range(len(observations)):
        q5.append(np.percentile(sim_res[:, t], 2.5))
        q95.append(np.percentile(sim_res[:, t], 97.5))
    ax.plot(q5, color='lightblue', linestyle='solid')
    ax.plot(q95, color='lightblue', linestyle='solid')
    ax.fill_between(np.arange(0, len(q5), 1), list(q5), list(q95), facecolor='lightblue', zorder=0,
                    linewidth=0, label='5th-95th percentile parameter uncertainty', alpha=0.4)
    for n, array in proc_sim.items():
        ax.plot(array, color=f'{clr_marker(wt_mu_m=True)[n]}', label=f'{n.capitalize()[:n.index("_")]} Simulation',
                linestyle='dashed')

    ax.plot(observations, 'r-', label=f'Poly{poly} observation')
    ax.set_ylim(-6, 9)
    ax.set_yticks(np.arange(-6, 9, 2))
    yrange = [str(i) for i in np.arange(-4, 9, 2)]
    yrange.insert(0, '')
    if ind_ax is not None and ind_ax == 0:
        ax.set_yticklabels(yrange, size=15)
    else:
        ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax.set_xticks(np.arange(len(observations)))
    ax.xaxis.set_major_formatter(plt.NullFormatter())


def plot_ci(x_data, y_data, label_poly, ax, ind_ax=None):
    if isinstance(label_poly, list):
        label_poly = label_poly[0]

    x_data = np.insert(x_data, 0, 0)
    y_data = np.insert(y_data, 0, 0)
    y_data[np.isnan(y_data)] = 1

    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'g-.', label="CI over Percentiles")
    ax.plot(x_data, y_data, 'r.-', label=f"{label_poly} CI")
    ax.set_xticks(x_data[1:])

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    # plt.xticks(np.arange(0, 1.1, 0.1), [np.round(i, 2) for i in np.arange(0, 1.1, 0.1)])
    # for i in x_data:
    #     ax.axvline(i, color='grey', alpha=0.4, linestyle='--')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    if ind_ax is not None and ind_ax == 0:
        ax.set_yticklabels([np.round(i, 2) for i in np.arange(0, 1.1, 0.1)], size=15)
    else:
        ax.yaxis.set_major_formatter(plt.NullFormatter())
