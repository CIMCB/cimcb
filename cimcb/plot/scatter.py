import numpy as np
from bokeh.plotting import ColumnDataSource, figure
from bokeh.models import Slope, Span, HoverTool


def scatter(x, y, label=None, group=None, title="Scatter Plot", xlabel="x", ylabel="y", width=600, height=600, legend=True, size=4, shape="circle", font_size="16pt", label_font_size="13pt", col_palette=None, hover_xy=True, gradient=False, hline=False, vline=False, xrange=None, yrange=None):
    """Creates a scatterplot using Bokeh.

    Required Parameters
    -------------------
    x : array-like, shape = [n_samples]
        Inpute data for x-axis.

    y : array-like, shape = [n_samples]
        Inpute data for y-axis.
    """

    # Error check
    if len(x) != len(y):
        raise ValueError("length of X does not match length of Y.")

    # If label is None, give an index based on input order
    if label is None:
        label_copy = {}
        label_copy["Idx"] = list(range(len(x)))
    else:
        try:
            label2 = label.copy()
            label2_dict = label2.to_dict("series")
            label_copy = label2_dict  # Ensure I don't overwrite label (when plot_groupmean=True)
        except TypeError:
            label2 = label.copy()
            label_copy = {}
            label_copy[label2.name] = label2.values.tolist()

    # If colour palette is None (default):
    if col_palette is None:
        col_palette = ["red", "blue", "green", "orange", "blueviolet", "gold", "peru", "pink", "darkblue", "olive", "teal", "slategray"]

    # Group is None or allow for multiple classes 
    if group is None:
        group_copy = [None] * len(x)
        col = []
        for i in range(len(x)):
            col.append(col_palette[2])
    else:
        group_copy = group.copy()
        group_unique = np.sort(np.unique(group_copy))
        col = []
        for i in range(len(group_copy)):
            for j in range(len(group_unique)):
                if group_copy[i] == group_unique[j]:
                    col.append(col_palette[j])

    # Bokeh data source with data labels
    data = {"x": x, "y": y, "group": group_copy, "col": col}
    data_label = {}
    for name, val in label_copy.items():
        data_label[name] = val
    data.update(data_label)
    source = ColumnDataSource(data=data)

    # Tool-tip (add everything in label_copy)
    TOOLTIPS = []
    if hover_xy is True:
        TOOLTIPS = [("x", "@x{1.111}"), ("y", "@y{1.111}")]
    for name, val in data_label.items():
        TOOLTIPS.append((str(name), "@" + str(name)))

    # Base figure
    fig = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, plot_width=width, plot_height=height, x_range=xrange, y_range=yrange)

    # Add to plot
    if shape is "circle":
        shape = fig.circle("x", "y", size=size, alpha=0.6, color="col", legend="group", source=source)
    elif shape is "triangle":
        shape = fig.triangle("x", "y", size=size, alpha=0.6, color="col", legend="group", source=source)
    else:
        raise ValueError("shape has to be either 'circle' or 'triangle'.")

    shape_hover = HoverTool(renderers=[shape], tooltips=TOOLTIPS)
    fig.add_tools(shape_hover)

    if gradient is not False:
        slope = Slope(gradient=gradient, y_intercept=0, line_color="black", line_width=2, line_alpha=0.3)
        fig.add_layout(slope)
        new_gradient = -(1 / gradient)
        slope2 = Slope(gradient=new_gradient, y_intercept=0, line_color="black", line_dash="dashed", line_width=2, line_alpha=0.10)
        fig.add_layout(slope2)

    if hline is not False:
        h = Span(location=0, dimension="width", line_color="black", line_width=3, line_alpha=0.15)
        fig.add_layout(h)

    if vline is not False:
        v = Span(location=0, dimension="height", line_color="black", line_width=3, line_alpha=0.15)
        fig.add_layout(v)

    # Font-sizes
    fig.title.text_font_size = font_size
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    # Remove legend
    if legend is False:
        fig.legend.visible = False

    return fig
