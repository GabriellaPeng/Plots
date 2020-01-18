import seaborn as sns
import matplotlib.pyplot as plt

from data_process import clr_marker


def group_parameters_at_loc(data, hue='Algorithm', save=None, type='hist', col_type = 'Canal Position', dict_optimize=None):
    clr_pal = {i: clr_marker(mtd_clr=True)[i.lower()] for i in set(data[hue])}
    # clr_pal =clr_marker(mtd_clr=True)[list(data[hue])[0].lower()]# {list(data[hue])[0]: }
    def plot(plot_data, bf_data, type=type, prm=None, dict_opt=None):
        aspect = [0.5 if type == 'scatterplots' else 0.7][0]
        row = [hue if len(set(data[hue])) > 1 else None][0]

        if bf_data:
            sharex, col, sharey= True, col_type, False
        else:
            sharex, col, sharey = False, 'Parameter', False

        g = sns.FacetGrid(plot_data, row=row, col=col, height=4, aspect=aspect, margin_titles=True,
                          palette=clr_pal, hue_order=list(clr_pal), sharex=sharex, sharey=sharey, hue=hue, legend_out=True)

        if type == 'scatterplots':
            g = g.map(plt.scatter, "Runs", 'Parameter Vals', alpha=.4, edgecolor="w").add_legend()
        elif type =='hist':
            g = g.map(sns.distplot, 'Parameter Vals', hist=True, norm_hist=True, label=list(data[hue])[0]).add_legend()
        # g.add_legend(legend_data=data['Parameter Vals'])
        # g.set_axis_labels(x_var="Parameter Values", y_var="Parameter frequency")
        if row is not None:
            if dict_optimize is not None:
                rows = plot_data[row][~plot_data.duplicated(row, keep='first')]
                cols = plot_data[col][~plot_data.duplicated(col, keep='first')]
                for r, v1 in enumerate(rows):
                    for c, v2 in enumerate(cols):
                        v = dict_opt[v1.lower()][v2]
                        g.axes[r, c].axvline(v, color='k', label=v)

        if save is not None:
            if bf_data:
                g.savefig(save + f"{prm}.png", dpi=350)
            else:
                g.savefig(save + f".png", dpi=350)

        plt.clf()

        return g

    if 'Poly' in list(data):
        for prm in set(data['Parameter']):
            dt = data[data['Parameter'] == prm]
            dict_opt = dict_optimize[prm]
            plot(plot_data = dt, bf_data=True, prm=prm, dict_opt=dict_opt)
    else:
        plot(plot_data = data, bf_data=False, dict_opt=dict_optimize)
