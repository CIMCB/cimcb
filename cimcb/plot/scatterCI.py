import numpy as np
from copy import deepcopy
from bokeh.models import Span, Whisker
from bokeh.plotting import ColumnDataSource, figure
from bokeh.models import SingleIntervalTicker, LinearAxis


def scatterCI(x, ci=None, label=None, hoverlabel=None, hline=0, sort_abs=False, col_hline=True, col_palette=None, title="Scatter CI Plot", xlabel="Peak", ylabel="Value", width=200, height=300, legend=True, font_size="20pt", label_font_size="13pt", linkrange=None, sort_ci=True, sort_ci_abs=False, grid_line=False, plot_x=False, x_axis_below=True):
    """Creates a scatterCI plot using Bokeh.

    Required Parameters
    -------------------
    X : array-like, shape = [n_samples]
        Inpute data
    """
    # If label is None, give an index based on input order
    if label is None:
        label_copy = []
        for i in range(len(x)):
            label_copy.append(str(i))
    else:
        label_copy = deepcopy(label)

    # Ensure label_copy is unique and under 60 characters until supported by Bokeh
    label_copy = [elem[:59] for elem in label_copy]  # Limit Label characters limit to 59
    label_copy = np.array(label_copy).tolist()
    label_unique = set(label_copy)
    label_indices = {value: [i for i, v in enumerate(label_copy) if v == value] for value in label_unique}
    for key, value in label_indices.items():
        if len(value) > 1:
            for i in range(len(value)):
                label_copy[value[i]] = (str(" ") * i) + label_copy[value[i]]

    # Make sure height accounts for the max length of label
    height = height + 5 * len(max(label_copy, key=len))

    # If colour palette is None (default):
    if col_palette is None:
        col_palette = ["blue", "red", "green"]

    # if col_hline is True, color depends on if error bar overlaps hline
    if col_hline is False:
        col = []
        for i in range(len(x)):
            col.append(col_palette[2])
    else:
        col = []
        if ci is None:
            for i in range(len(x)):
                if x[i] < hline < x[i]:
                    col.append(col_palette[0])
                else:
                    col.append(col_palette[1])
        else:
            for i in range(len(x)):
                if ci[:, 0][i] < hline < ci[:, 1][i]:
                    col.append(col_palette[0])
                else:
                    col.append(col_palette[1])

    # Sort data (absolute)
    if sort_abs is True:
        if ci is None:
            sorted_idx = np.argsort(abs(x))[::-1]
        else:
            if sort_ci is False:
                sorted_idx = np.argsort(abs(x))[::-1]
            else:
                if sort_ci_abs is False:
                    hline2 = hline
                else:
                    hline2 = -100000
                ciorder = ci - hline2
                ciorder_idx = list(range(len(x)))
                ciorder_min = []
                ciorder_sign = []
                for i in range(len(ciorder)):
                    minval = np.min(np.abs(ciorder[i]))
                    ciorder_min.append(minval)
                    if np.sign(ciorder[i][0]) == np.sign(ciorder[i][1]):
                        sign = True
                    else:
                        sign = False
                    ciorder_sign.append(sign)

                ciorder_min = np.array(ciorder_min)
                ciorder_sign = np.array(ciorder_sign)
                ciorder_idx = np.array(ciorder_idx)

                ciorder_min_signSame = []
                ciorder_min_signDiff = []

                ciorder_min_signSameidx = []
                ciorder_min_signDiffidx = []

                for i in range(len(ciorder_min)):
                    if ciorder_sign[i] == True:
                        ciorder_min_signSame.append(ciorder_min[i])
                        ciorder_min_signSameidx.append(ciorder_idx[i])
                    else:
                        ciorder_min_signDiff.append(ciorder_min[i])
                        ciorder_min_signDiffidx.append(ciorder_idx[i])

                sorted_idx = []
                sorted_idx2 = []
                ciorder_min_signSame_argsort = np.argsort(ciorder_min_signSame)[::-1]
                for i in ciorder_min_signSame_argsort:
                    sorted_idx.append(ciorder_min_signSameidx[i])
                    sorted_idx2.append(ciorder_min_signSameidx[i])

                ciorder_min_signDiff_argsort = np.argsort(ciorder_min_signDiff)[::-1]
                for i in ciorder_min_signDiff_argsort:
                    sorted_idx.append(ciorder_min_signDiffidx[i])

                sorted_idx = np.array(sorted_idx)

        x = x[sorted_idx]
        label_copy = np.array(label_copy)
        label_copy = label_copy[sorted_idx]
        col = np.array(col)
        col = col[sorted_idx]
        # Sort ci if it exists
        if ci is not None:
            ci_low = ci[:, 0][sorted_idx]
            ci_high = ci[:, 1][sorted_idx]
            ci = []
            for i in range(len(ci_low)):
                ci.append([ci_low[i], ci_high[i]])
            ci = np.array(ci)
        hoverlabel = hoverlabel.copy()
        # hoverlabel = hoverlabel.reset_index()
        # hoverlabel = hoverlabel.reindex(sorted_idx).drop('index', axis=1)

    elif sort_abs is False:
        pass

    if hoverlabel is None:
        hoverlabel_copy = {}
        hoverlabel_copy["Idx"] = list(range(len(x)))
    else:
        try:
            hoverlabel2 = hoverlabel.copy()
            hoverlabel2_dict = hoverlabel2.to_dict("series")
            hoverlabel_copy = hoverlabel2_dict
            # print(hoverlabel_copy)
        except TypeError:
            hoverlabel2 = label.copy()
            hoverlabel_copy = {}
            hoverlabel_copy[label2.name] = hoverlabel2.values.tolist()

        if sort_abs is True:
            hoverlabel2 = {}
            for key, value in hoverlabel_copy.items():
                hoverlabel2[key] = np.array(value)[sorted_idx]
            hoverlabel_copy = hoverlabel2

    # Linking to another plot
    if linkrange is None:
        xrange = label_copy
    else:
        xrange = linkrange.x_range

    # Bokeh data source
    if ci is None:
        data = {"x": x, "col": col, "label": label_copy}
    else:
        data = {"x": x, "lowci": ci[:, 0], "uppci": ci[:, 1], "col": col, "label": label_copy}
    data_label = {}
    for name, val in hoverlabel_copy.items():
        data_label[name] = val
    data.update(data_label)
    source = ColumnDataSource(data=data)

    # Tool-tip
    TOOLTIPS = []
    for name, val in data_label.items():
        TOOLTIPS.append((str(name), "@" + str(name)))
    TOOLTIPS.append(("Value", "@x{1.111}"))
    TOOLTIPS.append(("Upper", "@uppci{1.111}"))
    TOOLTIPS.append(("Lower", "@lowci{1.111}"))

    if ci is None:
        y_range_max = max(abs(np.min(x)), abs(np.max(x)), abs(np.min(x)), abs(np.max(x))) * 0.1
        y_range = (min(np.min(x) - y_range_max, np.min(x) - y_range_max), max(np.max(x) + y_range_max, np.max(x) + y_range_max))
    else:
        y_range_max = max(abs(np.min(ci[:, 0])), abs(np.max(ci[:, 0])), abs(np.min(ci[:, 1])), abs(np.max(ci[:, 1]))) * 0.1
        y_range = (min(np.min(ci[:, 0]) - y_range_max, np.min(ci[:, 0]) - y_range_max), max(np.max(ci[:, 1]) + y_range_max, np.max(ci[:, 0]) + y_range_max))

    # Base figure
    if x_axis_below == True:
        fig = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, x_range=xrange, y_range=y_range, plot_width=int(len(x) / 10 * width), plot_height=height, tooltips=TOOLTIPS, toolbar_location="left", toolbar_sticky=False)
    else:
        fig = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, x_range=xrange, y_range=y_range, plot_width=int(len(x) / 10 * width), plot_height=height, tooltips=TOOLTIPS, toolbar_location="left", toolbar_sticky=False, x_axis_location="above")

    # Add circles
    fig.circle("label", "x", size=10, alpha=0.6, color="col", source=source)

    #fig.x(label_copy, plot_x, size=20, alpha=0.6, color="black")

    # Add hline
    hline = Span(location=hline, dimension="width", line_color="grey", line_width=2, line_alpha=0.9)
    fig.add_layout(hline)

    # Add error bars
    if ci is not None:
        fig.add_layout(Whisker(base="label", lower="lowci", upper="uppci", line_color="col", line_width=1.5, source=source))

    # Font-sizes
    fig.title.text_font_size = font_size
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size

    # X-axis orientation
    fig.xaxis.major_label_orientation = np.pi / 2

    fig.outline_line_width = 2
    fig.outline_line_alpha = 1
    fig.outline_line_color = "black"

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    if grid_line == False:
        fig.xgrid.visible = False
        fig.ygrid.visible = False

    # y_axis_type=None
    # ticker = SingleIntervalTicker(interval=1, num_minor_ticks=5)
    # yaxis = LinearAxis(ticker=ticker)
    # fig.add_layout(yaxis, 'left')

    return fig
