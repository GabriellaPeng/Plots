import seaborn as sns
import matplotlib.pyplot as plt

from data_process import clr_marker


def group_parameters_at_loc(data, hue='Algorithm', save=None, type='hist', col_type = 'Canal Position'):
    clr_pal =clr_marker(mtd_clr=True)[list(data[hue])[0].lower()]# {list(data[hue])[0]: }

    def plot(plot_data, bf_data, type=type, prm=None):
        aspect = [0.5 if type == 'scatterplots' else 0.7][0]

        if bf_data:
            g = sns.FacetGrid(plot_data, row="Parameter", col=col_type, height=4, aspect=aspect, margin_titles=True,
                                  sharex=True, sharey=True, hue=hue, legend_out=True)

        else:
            g = sns.FacetGrid(plot_data, col="Parameter", height=4, aspect=aspect, margin_titles=True,
                              sharex=False, sharey=True, hue=hue, legend_out=True)

        if type == 'scatterplots':
            g = g.map(plt.scatter, "Runs", 'Parameter Vals', alpha=.4, edgecolor="w", color=clr_pal).add_legend()
        elif type =='hist':
            g = g.map(sns.distplot, 'Parameter Vals', color=clr_pal, hist=True, norm_hist=True, label=list(data[hue])[0]).add_legend()

        # g.add_legend(legend_data=data['Parameter Vals'])
        g.set_axis_labels(x_var="Parameter Values", y_var="Parameter frequency")

        if save is not None and prm is not None:
            g.savefig(save + f"{prm}.png", dpi=500)

        plt.clf()

        return g

    if 'Poly' in list(data):
        for prm in set(data['Parameter']):
            dt = data[data['Parameter'] == prm]
            plot(plot_data = dt, bf_data=True, prm=prm)
    else:
        plot(plot_data = data, bf_data=False)
