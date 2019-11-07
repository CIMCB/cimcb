import numpy as np
import pandas as pd
from bokeh.plotting import ColumnDataSource, figure
from bokeh.models import Slope, Span, HoverTool
from bokeh.models import Range1d
from ..utils import ci95_ellipse


def scatter_ellipse(x, y, x1, y1, label=None, group=None, title="Scatter Plot", xlabel="x", ylabel="y", width=600, height=600, legend=True, size=4, shape="circle", font_size="16pt", label_font_size="13pt", col_palette=None, hover_xy=True, gradient=False, gradient_alt=False, hline=False, vline=False, xrange=None, yrange=None, ci95=False, scatterplot=True, extraci95_x=False, extraci95_y=False, extraci95=False, scattershow=None, extraci95_x2=False, extraci95_y2=False, orthog_line=True, grid_line=False, mirror_range=False, legend_title=False):
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
        label = pd.Series(np.zeros(len(x),))
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
        col_palette = ["red", "blue", "green"]

    # Group is None or allow for multiple classes (can add more in the Future)

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
            if group_copy[i] == group_unique[0]:
                col.append(col_palette[0])
            elif group_copy[i] == group_unique[1]:
                col.append(col_palette[1])
            else:
                col.append(col_palette[2])

    if group is None:
        group_label = [0] * len(X)

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
    fig = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, plot_width=width, plot_height=height)

    # Add to plot
    if scattershow in [2, 4]:
        if shape is "circle":
            shape = fig.circle("x", "y", size=size * 0.75, alpha=0.6, color="col", source=source)
        elif shape is "triangle":
            shape = fig.triangle("x", "y", size=size, alpha=0.6, color="col", source=source)
        else:
            raise ValueError("shape has to be either 'circle' or 'triangle'.")

        shape_hover = HoverTool(renderers=[shape], tooltips=TOOLTIPS)
        fig.add_tools(shape_hover)

    data1 = {"x": x1, "y": y1, "group": group_copy, "col": col}
    data_label1 = {}
    for name, val in label_copy.items():
        data_label1[name] = val
    data1.update(data_label1)
    source1 = ColumnDataSource(data=data1)

    # Tool-tip (add everything in label_copy)
    TOOLTIPS1 = []
    if hover_xy is True:
        TOOLTIPS1 = [("x", "@x{1.111}"), ("y", "@y{1.111}")]
    for name, val in data_label.items():
        TOOLTIPS1.append((str(name), "@" + str(name)))

    if scattershow in [3, 4]:
        shape1 = fig.cross("x", "y", size=size * 2, alpha=0.6, color="col", source=source1)
        shape_hover1 = HoverTool(renderers=[shape1], tooltips=TOOLTIPS1)
        fig.add_tools(shape_hover1)

    if gradient is not False:
        if gradient_alt is False:
            if orthog_line == True:
                slope = Slope(gradient=gradient, y_intercept=0, line_color="black", line_width=2, line_alpha=0.3)
                fig.add_layout(slope)
                new_gradient = -(1 / gradient)
                slope2 = Slope(gradient=new_gradient, y_intercept=0, line_dash="dashed", line_width=2, line_alpha=0.3)
                fig.add_layout(slope2)
            else:
                new_gradient = -(1 / gradient)
                slope2 = Slope(gradient=new_gradient, y_intercept=0, line_dash="solid", line_width=3, line_alpha=0.3)
                fig.add_layout(slope2)
                h = Span(location=0, dimension="width", line_color="black", line_width=3, line_alpha=0.06)
                fig.add_layout(h)
                v = Span(location=0, dimension="height", line_color="black", line_width=3, line_alpha=0.06)
                fig.add_layout(v)

        else:
            c = 0.5 - gradient * 0.5
            slope = Slope(gradient=gradient, y_intercept=c, line_color="black", line_width=2, line_alpha=0.3)
            fig.add_layout(slope)
            new_gradient = -(1 / gradient)
            new_c = 0.5 - new_gradient * 0.5
            slope2 = Slope(gradient=new_gradient, y_intercept=new_c, line_color="black", line_dash="dashed", line_width=2, line_alpha=0.10)
            fig.add_layout(slope2)

    if hline is not False:
        h = Span(location=0, dimension="width", line_color="black", line_width=3, line_alpha=0.15)
        fig.add_layout(h)

    if vline is not False:
        v = Span(location=0, dimension="height", line_color="black", line_width=3, line_alpha=0.15)
        fig.add_layout(v)

    # if ci95 is true
    if ci95 is True:

        # if group is None
        if group is None:
            group_label = [0] * len(X)

        group_label = group_copy
        x_score = extraci95_x2
        y_score = extraci95_y2
        # Score plot extra: 95% confidence ellipse using PCA
        unique_group = np.sort(np.unique(group_label))
        unique_group_label = np.sort(label.unique())
        # print(unique_group_label)

        # Set colour per group
        list_color = ["red", "blue", "green", "black", "orange", "yellow", "brown", "cyan"]
        while len(list_color) < len(unique_group):  # Loop over list_color if number of groups > len(list_colour)
            list_color += list_color

        # Add 95% confidence ellipse for each unique group in a loop
        max_val = []
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
            if scattershow is 1:
                fig.line(m[:, 0], m[:, 1], color=list_color[i], line_width=2, alpha=0.8, line_dash="solid")
            elif scattershow in [0, 2, 4]:
                fig.line(m[:, 0], m[:, 1], color=list_color[i], line_width=2, alpha=0.8, line_dash="solid")
                fig.line(p[:, 0], p[:, 1], color=list_color[i], line_width=3, alpha=0.4)
            else:
                pass

            # Plot ci95 ellipse shade
            unique_group_label = np.sort(label.unique())
            if scattershow is 1:
                fig.patch(m[:, 0], m[:, 1], color=list_color[i], alpha=0.07)
                fig.x(np.median(m[:, 0]), np.median(m[:, 1]), size=size, alpha=0.6, color=list_color[i], line_width=2)
            elif scattershow in [0, 4]:
                fig.patch(m[:, 0], m[:, 1], color=list_color[i], alpha=0.15)
                fig.patch(p[:, 0], p[:, 1], color=list_color[i], alpha=0.02)
                fig.x(np.median(m[:, 0]), np.median(m[:, 1]), size=size, alpha=0.6, color=list_color[i], line_width=2)
            elif scattershow is 2:
                fig.patch(m[:, 0], m[:, 1], color=list_color[i], alpha=0.15, legend=unique_group_label[i])
                fig.patch(p[:, 0], p[:, 1], color=list_color[i], alpha=0.02)
                fig.x(np.median(m[:, 0]), np.median(m[:, 1]), size=size, alpha=0.6, color=list_color[i], line_width=2)
            else:
                pass

            if scattershow is 1:
                maxv = max(np.abs(m).flatten())
            else:
                maxv = max(np.abs(p).flatten())
            max_val.append(maxv)

        if extraci95 is True:
                # if group is None
            if group is None:
                group_label = [0] * len(X)

            group_label = group_copy
            x_score = extraci95_x
            y_score = extraci95_y
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
                if scattershow is 1:
                    fig.line(m[:, 0], m[:, 1], color=list_color[i], line_width=2, alpha=0.8, line_dash="dashed")
                elif scattershow in [0, 3, 4]:
                    fig.line(m[:, 0], m[:, 1], color=list_color[i], line_width=2, alpha=0.8, line_dash="dashed")
                    fig.line(p[:, 0], p[:, 1], color=list_color[i], line_width=3, alpha=0.4, line_dash="dashed")
                else:
                    pass

                unique_group_label = np.sort(label.unique())
                # Plot ci95 ellipse shade
                if scattershow is 1:
                    fig.patch(m[:, 0], m[:, 1], color=list_color[i], alpha=0.07, legend=unique_group_label[i])
                    fig.x(np.median(m[:, 0]), np.median(m[:, 1]), size=size, alpha=0.6, color=list_color[i], line_width=2)
                elif scattershow in [0, 3, 4]:
                    fig.patch(m[:, 0], m[:, 1], color=list_color[i], alpha=0.15, legend=unique_group_label[i])
                    fig.patch(p[:, 0], p[:, 1], color=list_color[i], alpha=0.02)
                    fig.x(np.median(m[:, 0]), np.median(m[:, 1]), size=size, alpha=0.6, color=list_color[i], line_width=2, legend=unique_group_label[i])
                else:
                    pass

                if scattershow is 1:
                    maxv = max(np.abs(m).flatten())
                else:
                    maxv = max(np.abs(p).flatten())
                max_val.append(maxv)

        if mirror_range == False:
            max_range = max(max_val)
            new_range_min = -max_range - 0.05 * max_range
            new_range_max = max_range + 0.05 * max_range
            fig.y_range = Range1d(new_range_min, new_range_max)
            fig.x_range = Range1d(new_range_min, new_range_max)
        else:
            pass

    # Add a legend
    #unique_group_label = np.sort(label.unique())
    # unique_color_label = ["red", "blue"]
    # unique_group_label[0] = str(unique_group_label[0])

    # y = 0
    # x = np.max(x) * 100 + np.max(x)
    # fig.square(x, y, color="red", legend=unique_group_label[0], alpha=0.5)
    # fig.square(x, y, color="blue", legend=unique_group_label[1], alpha=0.5)

    # Font-sizes
    fig.title.text_font_size = font_size
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size

    if grid_line == False:
        fig.xgrid.visible = False
        fig.ygrid.visible = False

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    # Remove legend
    if legend is True:
        if legend_title == False:
            fig.legend.visible = True
            fig.legend.location = "bottom_right"
        else:
            fig.legend.visible = False
            fig.title.text = "Groups: {} (Red) & {} (Blue)".format(unique_group_label[0], unique_group_label[1])
    else:
        fig.legend.visible = False

    return fig
