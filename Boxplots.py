import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_calib_parameters(data, means=None, save=None, kind='box'):
    dt = data
    plt.ioff()
    sns.set(style="darkgrid")  # "darkgrid"
    # colors = ['#ff7878',  '#84a9cd']
    # sns.set_palette(sns.color_palette(colors))
    dt_col = list(dt.head())

    g = sns.catplot(x=dt_col[0], y=dt_col[2], col=dt_col[3], hue=dt_col[1], data=dt, kind=kind, palette="vlag",
                    height=4, aspect=1.5, sharex=False, legend_out=True)
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
        g.savefig(save + f"{kind}1.png")
    plt.clf()


def boxplot_calib_valid_gofs():
    pass
