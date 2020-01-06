import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data_process import clr_marker


def plot_theil(methods, data, polys, gof, soil_canal_mask, soil_canal='all', save=None):
    pos = 0
    bar_width = 0.15
    epsilon = .005
    line_width = 1
    opacity = 0.7

    def _plt_bar(pos, m, ind, l_poly=None):
        mtd = m.upper()
        if ind == 0:
            mlabel = f'{mtd} Um'
            slabel = f'{mtd} Us'
            clabel = f'{mtd} Uc'

        else:
            mlabel = f'{mtd}'
            slabel = None
            clabel = None

        if soil_canal != 'all':
            u_data = {u: np.zeros(len(soil_canal_mask)) for u in data[m][gof]}
            for i, (sc, l_poly) in enumerate(soil_canal_mask.items()):
                poly_ind = [np.where(np.asarray(polys) == p)[0][0] for p in l_poly]
                for u in data[m][gof]:
                    u_data[u][i] = np.average(np.take(data[m][gof][u], poly_ind))

        else:
            poly_ind = [np.where(np.asarray(polys) == p)[0][0] for p in l_poly]
            u_data = {u: np.take(data[m][gof][u], poly_ind) for u in data[m][gof]}

        sns.set()
        plt.ioff()
        plt.bar(pos, u_data['Um'], bar_width,
                color=clr_marker(mtd_clr=True)[m],
                label=mlabel)

        plt.bar(pos, u_data['Us'], bar_width - epsilon,
                bottom=u_data['Um'],
                alpha=opacity,
                color='white',
                edgecolor=clr_marker(mtd_clr=True)[m],
                linewidth=line_width,
                hatch='//',
                label=slabel)

        plt.bar(pos, u_data['Uc'], bar_width - epsilon,
                bottom=u_data['Um'] + u_data['Us'],
                alpha=opacity,
                color='white',
                edgecolor=clr_marker(mtd_clr=True)[m],
                linewidth=line_width,
                hatch='0',
                label=clabel)

    if soil_canal != 'all':
        xticks = [f'{sc}' for sc in soil_canal_mask]
        N = len(soil_canal_mask)
        bar_position = {m: np.arange(N) + i * bar_width for i, m in enumerate(methods)}

        for m in methods:
            _plt_bar(bar_position[m], m, ind=methods.index(m))

        _end_plot(pos, xticks, bar_position, name=soil_canal, save=save)

    else:
        for name, l_poly in soil_canal_mask.items():
            xticks = [f'{p}' for p in l_poly]
            bar_position = {m: np.arange(len(xticks)) + i * bar_width for i, m in enumerate(methods)}

            for m in methods:
                _plt_bar(bar_position[m], m, ind=methods.index(m), l_poly=l_poly)

            _end_plot(pos, xticks, bar_position, name, save)


def _end_plot(pos, xticks, bar_position,name, save, hlines=True):
    pos_tick = [np.average([v[i] for m, v in bar_position.items()]) for i in range(len(xticks))]
    plt.xticks(pos_tick, xticks, fontsize=16)
    left, right = plt.xlim()
    if hlines:
        plt.hlines(y=0.5, xmin=left, xmax=right, linestyles='dashed', linewidth=4)
    # plt.ylabel('Errors')
    sns.despine()

    pos += 1
    if pos == 2:
        plt.legend(bbox_to_anchor=(1.1, 1.1), fontsize=10)
        plt.savefig(save+f'{name}', dpi=500, bbox_inches='tight')
    else:
        plt.savefig(save + f'{name}', dpi=500)
    plt.clf()