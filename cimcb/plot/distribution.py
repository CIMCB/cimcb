import numpy as np
from scipy import stats
from bokeh.plotting import figure


def distribution(X, group, kde=True, title="Density Plot", group_label=None, xlabel="x", ylabel="Pr(x)", font_size="20pt", label_font_size="13pt", width=500, height=400, color_hist="green", color_kde="mediumturquoise", padding=0.5, smooth=None, sigmoid=False, legend=False, legend_location="top_right", plot_num=0, grid_line=False, legend_title=False):
    """Creates a distribution plot using Bokeh.

    Required Parameters
    -------------------
    X : array-like, shape = [n_samples]
        Inpute data

    group : array-like, shape = [n_samples]
        Group label for sample
    """
    if group_label == None:
        group_label = ['0', '1']

    # Split into groups
    group_label = np.sort(np.unique(group_label))
    group_unique = np.sort(np.unique(group))
    x1 = X[group == group_unique[0]]
    x2 = X[group == group_unique[1]]
    if len(group_unique) == 4:
        x3 = X[group == group_unique[2]]
        x4 = X[group == group_unique[3]]
        x3 = np.append(x3, [x3[0] + 0.05, x3[0] - 0.05])
        x4 = np.append(x4, [x4[0] + 0.05, x4[0] - 0.05])

    x1 = np.append(x1, [x1[0] + 0.05, x1[0] - 0.05])
    x2 = np.append(x2, [x2[0] + 0.05, x2[0] - 0.05])

    # Density curve
    x1_min, x1_max = x1.min(), x1.max()
    x1_padding = (x1_max - x1_min) * padding
    x1_grid = np.linspace(x1_min - x1_padding, x1_max + x1_padding, 500)
    x1_pdf = stats.gaussian_kde(x1, smooth)
    x1_pdf_grid = x1_pdf(x1_grid)
    x1_grid = np.insert(x1_grid, 0, x1_min - x1_padding)
    x1_grid = np.insert(x1_grid, 0, x1_max + x1_padding)
    x1_pdf_grid = np.insert(x1_pdf_grid, 0, 0)
    x1_pdf_grid = np.insert(x1_pdf_grid, 0, 0)

    # Density curve
    x2_min, x2_max = x2.min(), x2.max()
    x2_padding = (x2_max - x2_min) * padding
    x2_grid = np.linspace(x2_min - x2_padding, x2_max + x2_padding, 500)
    x2_pdf = stats.gaussian_kde(x2, smooth)
    x2_pdf_grid = x2_pdf(x2_grid)
    x2_grid = np.insert(x2_grid, 0, x2_min - x2_padding)
    x2_grid = np.insert(x2_grid, 0, x2_max + x2_padding)
    x2_pdf_grid = np.insert(x2_pdf_grid, 0, 0)
    x2_pdf_grid = np.insert(x2_pdf_grid, 0, 0)

    # Density curve x3 and x4
    if len(group_unique) == 4:
        # Density curve
        x3_min, x3_max = x3.min(), x3.max()
        x3_padding = (x3_max - x3_min) * padding
        x3_grid = np.linspace(x3_min - x3_padding, x3_max + x3_padding, 500)
        x3_pdf = stats.gaussian_kde(x3, smooth)
        x3_pdf_grid = x3_pdf(x3_grid)
        x3_grid = np.insert(x3_grid, 0, x3_min - x3_padding)
        x3_grid = np.insert(x3_grid, 0, x3_max + x3_padding)
        x3_pdf_grid = np.insert(x3_pdf_grid, 0, 0)
        x3_pdf_grid = np.insert(x3_pdf_grid, 0, 0)

        # Density curve
        x4_min, x4_max = x4.min(), x4.max()
        x4_padding = (x4_max - x4_min) * padding
        x4_grid = np.linspace(x4_min - x4_padding, x4_max + x4_padding, 500)
        x4_pdf = stats.gaussian_kde(x4, smooth)
        x4_pdf_grid = x4_pdf(x4_grid)
        x4_grid = np.insert(x4_grid, 0, x4_min - x4_padding)
        x4_grid = np.insert(x4_grid, 0, x4_max + x4_padding)
        x4_pdf_grid = np.insert(x4_pdf_grid, 0, 0)
        x4_pdf_grid = np.insert(x4_pdf_grid, 0, 0)

    max_val_a = max(abs(max(x1_grid)), abs(min(x1_grid)))
    max_val_b = max(abs(max(x2_grid)), abs(min(x2_grid)))
    max_val_final = 0.2 * max(max_val_a, max_val_b)
    x_range_min = min(min(x1_grid) - max_val_final, min(x2_grid) - max_val_final)
    x_range_max = max(max(x1_grid) + max_val_final, max(x2_grid) + max_val_final)
    new_x_range = (x_range_min, x_range_max)
    new_y_range = (0, max(max(x1_pdf_grid) * 1.1, max(x2_pdf_grid) * 1.1))
    if len(group_unique) == 4:
        max_val_a = max(abs(max(x1_grid)), abs(min(x1_grid)))
        max_val_b = max(abs(max(x2_grid)), abs(min(x2_grid)))
        max_val_c = max(abs(max(x3_grid)), abs(min(x3_grid)))
        max_val_d = max(abs(max(x4_grid)), abs(min(x4_grid)))
        max_val_final = 0.1 * max(max_val_a, max_val_b, max_val_c, max_val_d)
        x_range_min = min(min(x1_grid) - max_val_final, min(x2_grid) - max_val_final, min(x3_grid) - max_val_final, min(x4_grid) - max_val_final)
        x_range_max = max(max(x1_grid) + max_val_final, max(x2_grid) + max_val_final, max(x3_grid) + max_val_final, min(x4_grid) + max_val_final)
        new_x_range = (x_range_min, x_range_max)
        new_y_range = (0, max(max(x1_pdf_grid) * 1.1, max(x2_pdf_grid) * 1.1, max(x3_pdf_grid) * 1.1, max(x4_pdf_grid) * 1.1))

    # If sigmoid set min 0, max 1 # ignore this
    if sigmoid is 10:
        x1_idx = np.intersect1d(np.where(x1_grid <= 1)[0], np.where(x1_grid >= 0)[0])
        x2_idx = np.intersect1d(np.where(x2_grid <= 1)[0], np.where(x2_grid >= 0)[0])
        x1_grid = x1_grid[x1_idx]
        x1_pdf_grid = x1_pdf_grid[x1_idx]
        x1_pdf_grid[0] = 0
        x1_pdf_grid[1] = 0
        x1_pdf_grid[-1] = 0
        x2_grid = x2_grid[x2_idx]
        x2_pdf_grid = x2_pdf_grid[x2_idx]
        x2_pdf_grid[0] = 0
        x2_pdf_grid[1] = 0
        x2_pdf_grid[-1] = 0
        if len(group_unique) == 4:
            x3_idx = np.intersect1d(np.where(x3_grid <= 1)[0], np.where(x3_grid >= 0)[0])
            x4_idx = np.intersect1d(np.where(x4_grid <= 1)[0], np.where(x4_grid >= 0)[0])
            x3_grid = x3_grid[x3_idx]
            x3_pdf_grid = x3_pdf_grid[x3_idx]
            x3_pdf_grid[0] = 0
            x3_pdf_grid[1] = 0
            x3_pdf_grid[-1] = 0
            x4_grid = x4_grid[x4_idx]
            x4_pdf_grid = x4_pdf_grid[x4_idx]
            x3_pdf_grid[0] = 0
            x4_pdf_grid[1] = 0
            x3_pdf_grid[-1] = 0

    # Figure
    fig = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, plot_width=width, plot_height=height, x_range=new_x_range, y_range=new_y_range)
    if kde is True:
        if len(group_unique) == 4:
            if plot_num is 2:
                fig.patch(x1_grid, x1_pdf_grid, alpha=0.28, color="red", line_color="grey", line_width=1, legend=group_label[0])
                fig.patch(x2_grid, x2_pdf_grid, alpha=0.28, color="blue", line_color="grey", line_width=1, legend=group_label[1])
            elif plot_num is 3:
                fig.patch(x3_grid, x3_pdf_grid, alpha=0.28, color="red", line_color="grey", line_width=1, legend=group_label[0])
                fig.patch(x4_grid, x4_pdf_grid, alpha=0.28, color="blue", line_color="grey", line_width=1, legend=group_label[1])
            else:
                fig.patch(x1_grid, x1_pdf_grid, alpha=0.16, color="red", line_color="grey", line_width=1, legend=group_label[0])
                fig.patch(x2_grid, x2_pdf_grid, alpha=0.16, color="blue", line_color="grey", line_width=1, legend=group_label[1])
                fig.patch(x3_grid, x3_pdf_grid, alpha=0.16, color="red", line_color="grey", line_width=1)
                fig.patch(x4_grid, x4_pdf_grid, alpha=0.16, color="blue", line_color="grey", line_width=1)
        else:
            fig.patch(x1_grid, x1_pdf_grid, alpha=0.3, color="red", line_color="grey", line_width=1, legend=group_label[0])
            fig.patch(x2_grid, x2_pdf_grid, alpha=0.3, color="blue", line_color="grey", line_width=1, legend=group_label[1])

    # Remove legend
    if legend is True:
        if legend_title == False:
            fig.legend.visible = True
            fig.legend.location = "bottom_right"
        else:
            fig.legend.visible = False
            fig.title.text = "Groups: {} (Red) & {} (Blue)".format(group_label[0], group_label[1])
    else:
        fig.legend.visible = False

    fig.y_range.start = 0

    # Font-sizes
    fig.title.text_font_size = font_size
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    if grid_line == False:
        fig.xgrid.visible = False
        fig.ygrid.visible = False

    return fig
