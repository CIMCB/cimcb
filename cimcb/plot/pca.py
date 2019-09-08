import numpy as np
import pandas as pd
from sklearn import decomposition
from bokeh.plotting import output_notebook, show
from bokeh.layouts import gridplot
from .scatter import scatter
from ..utils import ci95_ellipse


def pca(X, pcx=1, pcy=2, group_label=None, sample_label=None, peak_label=None):
    """Creates a PCA scores and loadings plot using Bokeh.

    Required Parameters
    -------------------
    X : array-like, shape = [n_samples]
        Inpute data
    """

    # Set model
    model = decomposition.PCA()
    model.fit(X)
    scores_ = model.transform(X)
    explained_var_ = model.explained_variance_ratio_ * 100

    # Extract scores, explained variance, and loadings for pcx and pcy
    x_score = scores_[:, (pcx - 1)]
    y_score = scores_[:, (pcy - 1)]
    x_expvar = explained_var_[(pcx - 1)]
    y_expvar = explained_var_[(pcy - 1)]
    x_load = model.components_[(pcx - 1), :]
    y_load = model.components_[(pcy - 1), :]

    # Colour for fig_score
    if group_label is None:
        col = ["blue", "green", "red"]
    else:
        col = None

    # Ensure group_label is an np.array
    group_label = np.array(group_label)

    # Scores plot
    fig_score = scatter(x_score, y_score, group=group_label, label=sample_label, size=5, xlabel="PC {} ({:0.1f}%)".format(pcx, x_expvar), ylabel="PC {} ({:0.1f}%)".format(pcy, y_expvar), title="PCA Score Plot (PC{} vs. PC{})".format(pcx, pcy), font_size="15pt", width=490, height=430, hover_xy=False, col_palette=col)
    print(len(x_load))
    # Loadings plot
    fig_load = scatter(x_load, y_load, size=7, label=peak_label, xlabel="PC {} ({:0.1f}%)".format(pcx, x_expvar), ylabel="PC {} ({:0.1f}%)".format(pcy, y_expvar), title="PCA Loadings Plot (PC{} vs. PC{})".format(pcx, pcy), font_size="15pt", width=490, height=430, hover_xy=False, shape="triangle", legend=False, hline=True, vline=True)

    # if group is None
    if group_label is None:
        group_label = [0] * len(X)

    # Score plot extra: 95% confidence ellipse using PCA
    unique_group = np.sort(np.unique(group_label))

    # Set colour per group
    list_color = ["red", "blue", "green", "black", "orange", "yellow", "brown", "cyan"]
    while len(list_color) < len(unique_group):  # Loop over list_color if number of groups > len(list_colour)
        list_color += list_color

    # Add 95% confidence ellipse for each unique group in a loop
    for i in range(len(unique_group)):
        # Get scores for the corresponding group
        group_i_x = []
        group_i_y = []
        for j in range(len(group_label)):
            if group_label[j] == unique_group[i]:
                group_i_x.append(x_score[j])
                group_i_y.append(y_score[j])

        # Calculate ci95 ellipse for each group
        data_circ_group = pd.DataFrame({"0": group_i_x, "1": group_i_y})
        m, outside_m = ci95_ellipse(data_circ_group, type="mean")
        p, outside_p = ci95_ellipse(data_circ_group, type="pop")

        # Plot ci95 ellipse outer line
        fig_score.line(m[:, 0], m[:, 1], color=list_color[i], line_width=2, alpha=0.8, line_dash="solid")
        fig_score.line(p[:, 0], p[:, 1], color=list_color[i], alpha=0.4)

        # Plot ci95 ellipse shade
        fig_score.patch(m[:, 0], m[:, 1], color=list_color[i], alpha=0.07)
        fig_score.patch(p[:, 0], p[:, 1], color=list_color[i], alpha=0.01)

    # Output this figure with fig_score and fig_load
    output_notebook()
    fig = gridplot([[fig_score, fig_load]])
    show(fig)
