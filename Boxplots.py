import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def boxplot_calib_parameters(gofs, algorithms, data):
    dt = data
    sns.set(style="whitegrid") #"darkgrid"

    dt_col = list(dt.head())
    g = sns.catplot(x=dt_col[0], y=dt_col[2], col=dt_col[3], hue=dt_col[1], data=dt, palette="vlag", kind='violin',
                height=4,
                aspect=1.5, sharex=False, legend_out=True)

    ax1, ax2, ax3, ax4 = g.axes[0]

    ax1.axvline(10, ls='--')
    ax2.axvline(30, ls='--')

    g.savefig("combine calib_valid.png")
    # sns.despine(trim=True, left=True, bottom=True)

    # ncol = len(gofs)
    # fig = plt.figure(figsize=(4 * 2 * len(gofs), 4), constrained_layout=False)
    # gs = GridSpec(ncols=ncol, nrows=1, wspace=0.0, hspace=0.0)

    # for i in range(ncol):
    #     vars()[f'ax' + str(i)] = fig.add_subplot(gs[i])

        # vars()[f'ax' + str(i)].legend(ncol=2, loc="lower right", frameon=True)
        # vars()[f'ax' + str(i)].set(xlim=(0, 24), ylabel="Algorithms", xlabel=f"{gof.upper()}")
        #
        # vars()[f'ax' + str(i)].xaxis.grid(True)



def boxplot_calib_valid_gofs():
    pass


def param_uncertainty_bounds(sim_res, observations, poly, proc_sim, save_plot, fig, ax, ind_ax=None):
    q5, q25, q75, q95 = [], [], [], []
    for t in range(len(observations)):
        q5.append(np.percentile(sim_res[:, t], 2.5))
        q95.append(np.percentile(sim_res[:, t], 97.5))
    ax.plot(q5, color='dimgrey', linestyle='solid')
    ax.plot(q95, color='dimgrey', linestyle='solid')
    ax.fill_between(np.arange(0, len(q5), 1), list(q5), list(q95), facecolor='dimgrey', zorder=0,
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
    # ax.set_xticklabels([f'{i}-season' for i in np.arange(len(observations))], rotation=45, size=6)
    # ax.legend()
    if ind_ax is None:
        fig.savefig(save_plot + f'{poly}')
        plt.close('all')
        plt.clf()
