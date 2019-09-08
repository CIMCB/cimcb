import numpy as np
import scipy
from copy import deepcopy, copy
from bokeh.layouts import gridplot
from bokeh.models import HoverTool, Slope, Span
from bokeh.plotting import ColumnDataSource, figure
from scipy.stats import ttest_1samp
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from ..utils import binary_metrics


def permutation_test(model, X, Y, nperm=100, folds=8):
    """Creates permutation test plots using Bokeh.

    Required Parameters
    -------------------

    model : object
        This object is assumed to store bootlist attributes in .model (e.g. modelPLS.model.x_scores_).

    X : array-like, shape = [n_samples, n_features]
        Predictor variables, where n_samples is the number of samples and n_features is the number of predictors.

    Y : array-like, shape = [n_samples, 1]
        Response variables, where n_samples is the number of samples.
    """

    try:
        model = deepcopy(model)  # Make a copy of the model
    except TypeError:
        model = copy(model)

    # Get train and test idx using Stratified KFold
    skf = StratifiedKFold(n_splits=folds)
    trainidx = []
    testidx = []
    for train, test in skf.split(X, Y):
        trainidx.append(train)
        testidx.append(test)

    # Calculate binary_metrics for stats_full
    y_pred_full = model.test(X)
    stats_full = binary_metrics(Y, y_pred_full)

    # Calculate binary_metrics for stats_cv
    y_pred_cv = [None] * len(Y)
    for j in range(len(trainidx)):
        X_train = X[trainidx[j], :]
        Y_train = Y[trainidx[j]]
        X_test = X[testidx[j], :]
        model.train(X_train, Y_train)
        y_pred = model.test(X_test)
        for (idx, val) in zip(testidx[j], y_pred):
            y_pred_cv[idx] = val.tolist()
    stats_cv = binary_metrics(Y, y_pred_cv)

    # Extract R2, Q2
    stats = []
    stats.append([stats_full["R²"], stats_cv["R²"], 1])

    # For each permutation, shuffle Y and calculate R2, Q2 and append to stats
    for i in tqdm(range(nperm), desc="Permutation Resample"):
        # Shuffle
        Y_shuff = Y.copy()
        np.random.shuffle(Y_shuff)

        # Model and calculate full binary_metrics
        model.train(X, Y_shuff)
        y_pred_full = model.test(X)
        stats_full = binary_metrics(Y_shuff, y_pred_full)

        # Get train and test idx using Stratified KFold for Y_shuff
        skf_nperm = StratifiedKFold(n_splits=folds)
        trainidx_nperm = []
        testidx_nperm = []
        for train, test in skf_nperm.split(X, Y_shuff):
            trainidx_nperm.append(train)
            testidx_nperm.append(test)

        # Model and calculate cv binary_metrics
        y_pred_cv = [None] * len(Y_shuff)
        for j in range(len(trainidx_nperm)):
            X_train = X[trainidx_nperm[j], :]
            Y_train = Y_shuff[trainidx_nperm[j]]
            X_test = X[testidx_nperm[j], :]
            model.train(X_train, Y_train)
            y_pred = model.test(X_test)
            for (idx, val) in zip(testidx_nperm[j], y_pred):
                y_pred_cv[idx] = val.tolist()
        stats_cv = binary_metrics(Y_shuff, y_pred_cv)

        # Calculate correlation using Pearson product-moment correlation coefficients and append permuted R2, Q2 and correlation coefficient
        corr = abs(np.corrcoef(Y_shuff, Y)[0, 1])
        stats.append([stats_full["R²"], stats_cv["R²"], corr])

    # Split data for plotting (corr, r2, q2)
    stats_r2 = []
    stats_q2 = []
    stats_corr = []
    for i in range(len(stats)):
        stats_r2.append(stats[i][0])
        stats_q2.append(stats[i][1])
        stats_corr.append(stats[i][2])

    # Calculate gradient, and y-intercept for plot 1
    r2gradient = (stats_r2[0] - np.mean(stats_r2[1:])) / (1 - np.mean(stats_corr[1:]))
    q2gradient = (stats_q2[0] - np.mean(stats_q2[1:])) / (1 - np.mean(stats_corr[1:]))
    r2yintercept = stats_r2[0] - r2gradient
    q2yintercept = stats_q2[0] - q2gradient

    # Figure 1
    data = {"corr": stats_corr, "r2": stats_r2, "q2": stats_q2}
    source = ColumnDataSource(data=data)
    fig1 = figure(plot_width=470, plot_height=410, x_range=(-0.15, 1.15), x_axis_label="Correlation", y_axis_label="R² & Q²")
    # Lines
    r2slope = Slope(gradient=r2gradient, y_intercept=r2yintercept, line_color="black", line_width=2, line_alpha=0.3)
    q2slope = Slope(gradient=q2gradient, y_intercept=q2yintercept, line_color="black", line_width=2, line_alpha=0.3)
    fig1.add_layout(r2slope)
    fig1.add_layout(q2slope)

    # Points
    r2_square = fig1.square("corr", "r2", size=6, alpha=0.5, color="red", legend="R²", source=source)
    q2_square = fig1.square("corr", "q2", size=6, alpha=0.5, color="blue", legend="Q²", source=source)

    # Add Hovertool
    fig1.add_tools(HoverTool(renderers=[r2_square], tooltips=[("R² Value", "@r2")]))
    fig1.add_tools(HoverTool(renderers=[q2_square], tooltips=[("Q² Value", "@q2")]))

    # Extra padding
    fig1.min_border_left = 20
    fig1.min_border_right = 20
    fig1.min_border_top = 20
    fig1.min_border_bottom = 20
    fig1.legend.location = "bottom_right"

    # Calculate Density cure for Figure 2
    # Density curve
    X1 = np.array(stats_r2[1:])
    x1_min, x1_max = X1.min(), X1.max()
    x1_padding = (x1_max - x1_min) * 0.6
    x1_grid = np.linspace(x1_min - x1_padding, x1_max + x1_padding, 50)
    x1_pdf = scipy.stats.gaussian_kde(X1, "scott")
    x1_pdf_grid = x1_pdf(x1_grid)

    # Density curve
    X2 = np.array(stats_q2[1:])
    x2_min, x2_max = X2.min(), X2.max()
    x2_padding = (x2_max - x2_min) * 0.6
    x2_grid = np.linspace(x2_min - x2_padding, x2_max + x2_padding, 50)
    x2_pdf = scipy.stats.gaussian_kde(X2, "scott")
    x2_pdf_grid = x2_pdf(x2_grid)
    x2_pdf_grid = [-x for x in x2_pdf_grid]

    # Figure 2
    fig2 = figure(plot_width=470, plot_height=410, x_range=(min(x2_grid) * 1.1, max(stats_r2[0], max(x1_grid)) + 0.65), y_range=((min(x2_pdf_grid) - 1) * 1.2, (max(x1_pdf_grid) + 1) * 1.1), x_axis_label="R² & Q²", y_axis_label="p.d.f.")
    slope_0 = Span(location=0, dimension="width", line_color="black", line_width=2, line_alpha=0.3)
    fig2.add_layout(slope_0)

    # Plot distribution
    fig2.patch(x1_grid, x1_pdf_grid, alpha=0.35, color="red", line_color="grey", line_width=1)
    fig2.patch(x2_grid, x2_pdf_grid, alpha=0.35, color="blue", line_color="grey", line_width=1)

    # Extra padding
    fig2.min_border_left = 60
    fig2.min_border_right = 20
    fig2.min_border_top = 20
    fig2.min_border_bottom = 20

    # Lollipops R2
    # Do a t-test
    a = ttest_1samp(stats_r2[1:], [stats_r2[0]])[1][0]
    b = a / 2
    if b > 0.005:
        data2_manu = "%0.2f" % b
    else:
        data2_manu = "%0.2e" % b

    # Plot
    data2 = {"x": [stats_r2[0]], "y": [max(x1_pdf_grid) + 1], "hover": [data2_manu]}
    source2 = ColumnDataSource(data=data2)
    data2_line = {"x": [stats_r2[0], stats_r2[0]], "y": [max(x1_pdf_grid) + 1, 0], "hover": [str(data2_manu), str(data2_manu)]}
    source2_line = ColumnDataSource(data=data2_line)
    r2fig2_line = fig2.line("x", "y", line_width=2, line_color="red", source=source2_line)
    r2fig2 = fig2.circle("x", "y", fill_color="red", size=6, legend="R²", source=source2)

    # Lollipops Q2
    # Do a t-test
    if ttest_1samp(stats_q2[1:], [stats_q2[0]])[1][0] / 2 > 0.005:
        a = ttest_1samp(stats_q2[1:], [stats_q2[0]])[1][0]
        b = a / 2
        data3_manu = "%0.2f" % b
    else:
        a = ttest_1samp(stats_q2[1:], [stats_q2[0]])[1][0]
        b = a / 2
        data3_manu = "%0.2e" % b

    # Plot
    data3 = {"x": [stats_q2[0]], "y": [min(x2_pdf_grid) - 1], "hover": [data3_manu]}
    source3 = ColumnDataSource(data=data3)
    data3_line = {"x": [stats_q2[0], stats_q2[0]], "y": [(min(x2_pdf_grid) - 1), 0], "hover": [data3_manu, data3_manu]}
    source3_line = ColumnDataSource(data=data3_line)
    q2fig2_line = fig2.line("x", "y", line_width=2, line_color="blue", source=source3_line)
    q2fig2 = fig2.circle("x", "y", fill_color="blue", size=6, legend="Q²", source=source3)

    # Add text
    textr2 = "True R²\nP-Value: {}".format(data2_manu)
    textq2 = "True Q²\nP-Value: {}".format(data3_manu)
    fig2.text(x=[stats_r2[0] + 0.05, stats_q2[0] + 0.05], y=[(max(x1_pdf_grid) + 0.5), (min(x2_pdf_grid) - 1.5)], text=[textr2, textq2], angle=0, text_font_size="8pt")

    # Font-sizes
    fig1.xaxis.axis_label_text_font_size = "13pt"
    fig1.yaxis.axis_label_text_font_size = "13pt"
    fig2.xaxis.axis_label_text_font_size = "12pt"
    fig2.yaxis.axis_label_text_font_size = "12pt"
    fig2.legend.location = "top_left"

    fig = gridplot([[fig1, fig2]])
    return fig
