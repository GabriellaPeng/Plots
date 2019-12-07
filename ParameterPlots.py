import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data_process import clr_marker


def group_parameters_at_loc(data, hue='Algorithm', save=None, type='scatterplots'):
    clr_pal =clr_marker(mtd_clr=True)[list(data[hue])[0].lower()]# {list(data[hue])[0]: }

    def plot(poly=True, type=type):
        aspect = [0.5 if type == 'scatterplots' else 0.7][0]

        if poly:
            g = sns.FacetGrid(data, row="Parameter", col='Canal Pos', height=4, aspect=aspect, margin_titles=True,
                              sharex=False, sharey=True, hue=hue, legend_out=True)
        else:
            g = sns.FacetGrid(data, col="Parameter", height=4, aspect=aspect, margin_titles=True,
                              sharex=False, sharey=False, hue=hue, legend_out=True)

        if type == 'scatterplots':
            g = g.map(plt.scatter, "Runs", 'Parameter Vals', alpha=.4, edgecolor="w", color=clr_pal).add_legend()
        elif type =='hist':
            g = g.map(sns.distplot, 'Parameter Vals', color=clr_pal, hist=True, norm_hist=True, label=list(data[hue])[0]).add_legend()

        # g.add_legend(legend_data=data['Parameter Vals'])
        g.set_axis_labels(x_var="Parameter Values", y_var="Parameter frequency")
        return g

    if 'Poly' in list(data):
        g = plot(poly=True)
    else:
        g = plot(poly=False)

    if save is not None:
        g.savefig(save + f".png", dpi=500)
    plt.clf()
