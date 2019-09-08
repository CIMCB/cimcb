import numpy as np
import pandas as pd
from scipy import stats
from bokeh.plotting import ColumnDataSource, figure
from bokeh.models import HoverTool


def boxplot(X, group, violin=False, title="", xlabel="Group", ylabel="Value", font_size="20pt", label_font_size="13pt", width=500, height=400, color="whitesmoke", color_violin="mediumturquoise", width_boxplot=1, width_violin=1, y_range=None, group_name=None, group_name_sort=None):
    """Creates a boxplot using Bokeh.

    Required Parameters
    -------------------
    X : array-like, shape = [n_samples]
        Inpute data

    group : array-like, shape = [n_samples]
        Group label for sample
    """

    if group_name_sort is None:
        group_name_sort = [str(i) for i in list(set(group))]

    # unique groups as strings
    if group_name is None:
        group_name = [str(i) for i in list(set(group))]

    # Colors
    if isinstance(color, str):
        color = [color] * len(group_name)
    if isinstance(color_violin, str):
        color_violin = [color_violin] * len(group_name)

    # Combine X and group into pandas table
    X_pd = pd.Series(X, name="X")
    group_pd = pd.Series(group, name="group")
    table = pd.concat([X_pd, group_pd], axis=1)

    # Find the quartiles and IQR for each group
    X_groupby = table.groupby("group", sort=True)
    q1 = X_groupby.quantile(q=0.25)
    q2 = X_groupby.quantile(q=0.50)
    q3 = X_groupby.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    # Find the outliers for each category
    def outliers(group):
        group_name = group.name
        return group[(group.X > upper.loc[group_name]["X"]) | (group.X < lower.loc[group_name]["X"])]["X"]

    out = X_groupby.apply(outliers).dropna()

    # prepare outlier data for plotting, we need coordinates for every outlier.
    outx = []
    outy = []
    outidx = []
    if not out.empty:
        outx = []
        outy = []
        outidx = []
        for keys in out.index:
            outx.append(str(keys[0]))
            outy.append(out.loc[keys[0]].loc[keys[1]])
            outidx.append(keys[1])

    # If no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = X_groupby.quantile(q=0.00)
    qmax = X_groupby.quantile(q=1.00)
    upper.X = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, "X"]), upper.X)]
    lower.X = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, "X"]), lower.X)]

    # Bokeh data source boxplot
    source_boxplot = ColumnDataSource(data=dict(group_name=group_name_sort, upper=upper.X, lower=lower.X, q1=q1.X, q2=q2.X, q3=q3.X, color=color))

    # Bokeh data source outlier
    source_outliers = ColumnDataSource(data=dict(outx=outx, outy=outy, outidx=outidx))

    # Set default y_range if None
    if y_range is None:
        max_val = max(abs(np.min(X)), abs(np.max(X)))
        violin_add = (np.max(X) - np.min(X)) * 0.15
        max_val_1 = 0.1 * max_val + violin_add
        y_range = (np.min(X) - max_val_1, np.max(X) + max_val_1)

    # Base figure
    fig = figure(title=title, x_range=group_name, x_axis_label=xlabel, y_axis_label=ylabel, plot_width=width, plot_height=height, y_range=y_range)

    # Plot probability density shade if violin is True
    try:
        if violin is True:
            vwidth = 0.5
            table["group"] = table.group.astype("category")
            for i in table["group"].cat.categories:
                y_data = table[table["group"] == i]["X"]
                y_min, y_max = y_data.min(), y_data.max()
                y_padding = (y_max - y_min) * 0.15
                y_grid = np.linspace(y_min - y_padding, y_max + y_padding, 60)
                pdf = stats.gaussian_kde(y_data, "scott")
                x_pdf = pdf(y_grid)
                x_pdf = x_pdf / x_pdf.max() * vwidth / (2 / width_violin)
                x_patch = np.append(x_pdf, -x_pdf[::-1])
                y_patch = np.append(y_grid, y_grid[::-1])
                for j in range(len(group_name)):
                    if group_name[j] == str(i):
                        val = j
                fig.patch((x_patch + val + 0.5), y_patch, alpha=0.3, color=color_violin[val], line_color="grey", line_width=1)
    except np.linalg.LinAlgError as err:
        pass

    ## Boxplot (Stems, Boxes, Whiskers, and Outliers) ##
    # Stems
    stem1 = fig.segment("group_name", "upper", "group_name", "q3", line_color="black", source=source_boxplot)
    stem2 = fig.segment("group_name", "lower", "group_name", "q1", line_color="black", source=source_boxplot)

    # Boxes
    box1 = fig.vbar("group_name", width_boxplot / 10, "q2", "q3", fill_color="color", line_color="black", alpha=0.8, source=source_boxplot)
    box2 = fig.vbar("group_name", width_boxplot / 10, "q1", "q2", fill_color="color", line_color="black", alpha=0.8, source=source_boxplot)

    # Whiskers (almost-0 height rects)
    whisker1 = fig.rect("group_name", "lower", 0.05, 0.001, line_color="black", source=source_boxplot)
    whisker2 = fig.rect("group_name", "upper", 0.05, 0.001, line_color="black", source=source_boxplot)

    # Outliers
    outliers = fig.circle("outx", "outy", size=4, color="red", fill_alpha=0.4, source=source_outliers)

    # Hovertool for boxplot
    fig.add_tools(HoverTool(renderers=[stem1, stem2, box1, box2], tooltips=[("Upper", "@upper{1.11}"), ("Q3", "@q3{1.11}"), ("Median", "@q2{1.11}"), ("Q1", "@q1{1.11}"), ("Lower", "@lower{1.11}")]))

    # Hovertool for outliers
    fig.add_tools(HoverTool(renderers=[outliers], tooltips=[("Index", "@outidx"), (ylabel, "@outy")]))

    # Font-sizes
    fig.title.text_font_size = font_size
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    return fig
