import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def boxplot_calib_parameters(gofs, algorithms, calib_gofs, valid_gofs):
    sns.set(style="ticks")


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
