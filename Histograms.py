import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data_process import clr_marker


def plot_theil(methods, data, polys, gof, soil_canal_mask, save=None):
    pos = 0
    for name, l_poly in soil_canal_mask.items():
        name = name[0]+name[name.index(',')+2]
        xticks = [f'{p}' for p in l_poly]

        sns.set()
        plt.ioff()
        # plot details
        bar_width = 0.15
        epsilon = .005
        line_width = 1
        opacity = 0.7

        def _plt_bar(pos, m, ind):
            mtd = m.upper()
            if ind == 0:
                mlabel = f'{mtd} Um'
                slabel = f'{mtd} Us'
                clabel = f'{mtd} Uc'

            else:
                mlabel = f'{mtd}'
                slabel = None
                clabel = None


            poly_ind = [np.where(np.asarray(polys) == p)[0][0] for p in l_poly]
            u_data = {u: np.take(data[m][gof][u], poly_ind) for u in data[m][gof]}

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

        N = len(l_poly)
        bar_position = {m: np.arange(N)+i*bar_width for i, m in enumerate(methods)}

        for m in methods:
            _plt_bar(bar_position[m], m, ind=methods.index(m))

        pos_tick = [np.average([v[i] for m, v in bar_position.items()]) for i in range(N)]
        plt.xticks(pos_tick, xticks, fontsize=16)
        left, right = plt.xlim()
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