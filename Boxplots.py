import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.patches import Polygon


def dist_parameters(data, save=None, kind='box'):
    '''Figure 4, Boxplots for all obj funcs'''

    dt = data

    plt.ioff()
    sns.set(style="darkgrid")
    dt_col = list(dt.head())

    g = sns.catplot(x=dt_col[0], y=dt_col[2], col=dt_col[3], hue=dt_col[1], data=dt, kind=kind, palette="vlag",
                    height=5, aspect=1.1, sharex=False, legend_out=True, legend=False)

    plt.subplots_adjust(wspace=.01)

    fontsize = 12
    ticksize = 10
    for i in range(len(g.axes[0])):
        vars()[f'ax{i}'] = g.axes[0][i]
        vars()[f'ax{i}'].set_xlabel("",fontsize=fontsize)
        vars()[f'ax{i}'].set_ylabel("",fontsize=fontsize)
        vars()[f'ax{i}'].tick_params(labelsize=ticksize)
    # for i, (gof, als) in enumerate(means.items()):
    #     vars()[f'ax{i}'] = g.axes[0][i]
    #     for al, m in als.items():
    #         if gof =='aic':
    #             vars()[f'ax{i}'].axvline(m, ls='--', color=colors[1])
    #         else:
    #             vars()[f'ax{i}'].axvline(m[0], ls='--', color=colors[0])
    #             vars()[f'ax{i}'].axvline(m[1], ls='--', color=colors[1])

    if save is not None:
        g.savefig(save + f"{kind}.png", dpi=500)
    plt.clf()



def boxplot_gof_loc(calib_data, valid_data, soil_canal_mask, save=None):
    '''sim_data={gof: np.array}'''

    min_gof=['nse', 'mic']
    opt_val = [np.argmax(vals) if gof in min_gof else np.argmin(vals) for gof, vals in calib_data.items()][0]

    xpol = [j for i, v in soil_canal_mask.items() for j in v]
    xlim = [0, len(xpol)]
    xpos = np.linspace(xlim[0] + 0.5, xlim[1] + 0.5, len(xpol), endpoint=False)

    l_xpol = [np.arange(xpos[xpol.index(l_p[0])], xpos[xpol.index(l_p[0])] + len(l_p) * 0.5, 0.5) for j, l_p in
              soil_canal_mask.items()]

    xlabels = [ ]
    for j in [[[i for i in soil_canal_mask][i]] * len(v) for i, v in enumerate(l_xpol)]:
        xlabels.extend(j)

    # whiskers = [ ]
    sns.set( )
    fig, ax1 = plt.subplots(figsize=(10, 6))

    for gof, v_data in valid_data.items():

        # Set the axes ranges and axes labels
        if gof == 'mic':
            ylim, step = [np.nanmin(v_data)-0.05, 1.0 + 0.1], 0.1
        else:
            ylim, step= [-300, 0], 50

        fig.canvas.set_window_title(f'{gof.upper()} Boxplot')
        fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

        bp = ax1.boxplot(v_data, notch=0, sym='.', positions=np.asarray(xpos), whis=[5, 95])
        # whiskers.append(min([i[0] for i in [item.get_ydata() for item in bp['whiskers']]]))

        # Add a horizontal grid to the plot
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax1.set_axisbelow(True)
        # _set_ax_marker(ax1, 'Canal position', f'{obj_func.upper()}', #f'{obj_func[:obj_func.index("_")].upper()}_AIC'
        #                f' {m.upper()} comparison of {obj_func.upper()} Across All Validation Canal Positions')

        #fill the boxes with desired colors
        box_colors = ['#F5B7B1','#00BCD4', '#e3931b', '#A5D6A7', '#B39DDB', '#F48FB1', '#C8E851', '#039BE5', '#3949AB']
        rbg = [mcolors.to_rgba(c) for c in box_colors]
        num_boxes = len(xpol)
        medians = np.empty(num_boxes)

        color = []
        for i, l_pos in enumerate(l_xpol):
            for j in range(len(l_pos)):
                color.append(rbg[i])

        for i, v in enumerate(xpos):
            box = bp['boxes'][i]
            boxX = []
            boxY = []
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])

            box_coords = np.column_stack([boxX, boxY])

            for scloc, l_v in enumerate(l_xpol):
                if l_v[0] == v:
                    sc = [sc for sc in list(soil_canal_mask)]
                    label = sc[scloc]
                    ax1.add_patch(Polygon(box_coords, facecolor=color[i], label=label))
                else:
                    ax1.add_patch(Polygon(box_coords, facecolor=color[i]))

            # Now draw the median lines back over what we just filled in
            med = bp['medians'][i]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                l_median = ax1.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]

            # add mean
            l_mean = ax1.plot(np.average(med.get_xdata()), np.average(v_data[:, i]), color='w', marker='*')

        for tick, label in zip(range(num_boxes), ax1.get_xticks()):
            ax1.text(label, .95, xpol[tick], transform=ax1.get_xaxis_transform(),
                     horizontalalignment='center', size='small', color=color[tick])

        # the optimal run in calibration
        l_opt = ax1.axhline(np.mean(v_data[opt_val]), ls="dashed", color='r')

        ax1.set_xlim(xlim[0], xlim[1])
        # ax1.set_xticks([np.average(i) for i in l_xpol])
        ax1.set_xticklabels([j for i in l_xpol for j in i])
        ax1.set_xticklabels(xlabels, rotation=45, fontsize=14)

        if gof != 'mic':
            ax1.set_ylim(ylim[0], ylim[1])
            ax1.set_yticks(np.arange(ylim[0], ylim[1], step))

        ax1.legend(frameon=False, ncol=len(soil_canal_mask), bbox_to_anchor=(0.95, -0.1))
        # add a basic legend
        # pos = [i for i in np.flip(np.arange(0.01, 0.7, 0.05))][:len([i[0] + i[i.index(',') + 2] for i in s_cnl_msk])] #pos=0.035 is right
        # for i, v in enumerate([i[0] + i[i.index(',') + 2] for i in s_cnl_msk]):
        #     fig.text(0.97, pos[i], f'{v}',
        #              backgroundcolor=rbg[i], color='white', weight='roman', size='small', bbox={'pad': 3.5, 'ec':rbg[i], 'fc':rbg[i]})

        # fig.legend((l_median[0], l_mean[0], l_opt), ('Median', 'Mean', 'Optimum calibrated value'),
        #            bbox_to_anchor=(0.9, 0.1), facecolor='#CACFD2', fontsize='x-small')
        if save:
            fig.savefig(save, dpi=500, bbox_inches='tight')
        plt.close(fig)



