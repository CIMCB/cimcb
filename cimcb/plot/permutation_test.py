import numpy as np
import scipy
import multiprocessing
from pydoc import locate
from copy import deepcopy, copy
from joblib import Parallel, delayed
from bokeh.layouts import gridplot
from statsmodels.stats.weightstats import ttest_ind
from bokeh.models import HoverTool, Slope, Span
from bokeh.plotting import ColumnDataSource, figure
from scipy.stats import ttest_1samp
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from ..utils import binary_metrics


class permutation_test():
    def __init__(self, model, params, X, Y, nperm=100, folds=5):
        self.model = locate(model.__name__)
        self.params = params
        self.skf = StratifiedKFold(n_splits=folds)
        self.folds = folds
        self.X = X
        self.Y = Y
        self.nperm = nperm
        self.n_cores = multiprocessing.cpu_count()

    def _calc_original(self):
        skf = self.skf
        X = self.X
        Y = self.Y
        model = self.model(**self.params)

        trainidx = []
        testidx = []
        for train, test in skf.split(X, Y):
            trainidx.append(train)
            testidx.append(test)

        # Calculate binary_metrics for stats_full
        y_pred_test = model.train(X, Y)
        #y_pred_full = model.test(X)
        stats_full = binary_metrics(Y, y_pred_test)

        # if seed is set, make sure it's none
        if 'seed' in self.params:
            self.params['seed'] = None
        model = self.model(**self.params)

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
        self.stats_original = [stats_full, stats_cv]

    def _calc_perm(self):
        stats = Parallel(n_jobs=self.n_cores)(delayed(self._calc_perm_loop)(i) for i in tqdm(range(self.nperm)))
        self.stats_perm = stats

    def _calc_perm_loop(self, i):
        skf = self.skf
        X = self.X
        Y = self.Y
        folds = self.folds
        model_i = self.model(**self.params)

        # Shuffle
        Y_shuff = Y.copy()
        np.random.shuffle(Y_shuff)

        # Model and calculate full binary_metrics
        model_i.train(X, Y_shuff)
        y_pred_full = model_i.test(X)
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
            model_i.train(X_train, Y_train)
            y_pred = model_i.test(X_test)
            for (idx, val) in zip(testidx_nperm[j], y_pred):
                y_pred_cv[idx] = val.tolist()
        stats_cv = binary_metrics(Y_shuff, y_pred_cv)
        corr = abs(np.corrcoef(Y_shuff, Y)[0, 1])
        stats_comb = [stats_full, stats_cv, corr]
        return stats_comb

    def run(self):
        self._calc_original()
        self._calc_perm()

    def plot(self, metric="r2q2", hide_pval=True, grid_line=False, legend=True):

        # Choose metric to plot
        metric_title = np.array(["ACCURACY", "AIC", "AUC", "BIC", "F1-SCORE", "PRECISION", "R²", "SENSITIVITY", "SPECIFICITY", "SSE"])
        metric_list = np.array(["acc", "aic", "auc", "bic", "f1score", "prec", "r2q2", "sens", "spec", "sse"])
        metric_idx = np.where(metric_list == metric)[0][0]

        mname = metric_title[metric_idx]
        stats = []
        stats.append([self.stats_original[0][mname], self.stats_original[1][mname], 1])
        for i in self.stats_perm:
            stats.append([i[0][mname], i[1][mname], i[2]])

        self.stats = stats

        if metric == "r2q2":
            full_text = "R²"
            cv_text = "Q²"
        else:
            full_text = mname + "full"
            cv_text = mname + "cv"

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

        max_vals = max(np.max(stats_r2), np.max(stats_q2))
        min_vals = min(np.min(stats_r2), np.min(stats_q2))
        y_range_share = (min_vals - abs(0.2 * min_vals), max_vals + abs(0.1 * min_vals))
        # Figure 1
        data = {"corr": stats_corr, "r2": stats_r2, "q2": stats_q2}
        source = ColumnDataSource(data=data)
        fig1 = figure(plot_width=470, plot_height=410, x_range=(-0.15, 1.15), x_axis_label="Correlation", y_range=y_range_share, y_axis_label=full_text + " & " + cv_text)
        # Lines
        r2slope = Slope(gradient=r2gradient, y_intercept=r2yintercept, line_color="black", line_width=2, line_alpha=0.3)
        q2slope = Slope(gradient=q2gradient, y_intercept=q2yintercept, line_color="black", line_width=2, line_alpha=0.3)
        fig1.add_layout(r2slope)
        fig1.add_layout(q2slope)

        # Points
        r2_square = fig1.square("corr", "r2", size=6, alpha=0.5, color="red", legend=full_text, source=source)
        q2_square = fig1.square("corr", "q2", size=6, alpha=0.5, color="blue", legend=cv_text, source=source)

        # Add Hovertool
        fig1.add_tools(HoverTool(renderers=[r2_square], tooltips=[(full_text + " Value", "@r2")]))
        fig1.add_tools(HoverTool(renderers=[q2_square], tooltips=[(cv_text + " Value", "@q2")]))

        # Extra padding
        fig1.min_border_left = 20
        fig1.min_border_right = 20
        fig1.min_border_top = 20
        fig1.min_border_bottom = 20
        #fig1.legend.location = "bottom_right"

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
        if hide_pval == True:
            y_range_share2 = (min_vals - abs(0.2 * min_vals), max_vals + abs(0.1 * max_vals))
            ymin = min(x2_pdf_grid) - 1
            xmin = max(x1_pdf_grid) + 1
            yy_range = (ymin - abs(0.1 * ymin), xmin + abs(0.1 * xmin))
        else:
            y_range_share2 = [min_vals - abs(0.2 * min_vals), max_vals + 0.8]
            ymin = min(x2_pdf_grid) - 1.2
            xmin = max(x1_pdf_grid) + 1.2
            yy_range = (ymin - 1, xmin + 1)
            if metric == "auc":
                if y_range_share2[1] > 1.5:
                    y_range_share2[1] = 1.5
            y_range_share2 = tuple(y_range_share2)

        fig2 = figure(plot_width=470, plot_height=410, x_axis_label=full_text + " & " + cv_text, y_axis_label="p.d.f.", x_range=y_range_share2, y_range=yy_range)
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
        #a = ttest_1samp(stats_r2[1:], [stats_r2[0]])[1][0]
        #b = a / 2
        b = ttest_ind(stats_r2[1:], [stats_r2[0]], alternative='smaller')[1]
        if b > 0.005:
            data2_manu = "%0.2f" % b
        else:
            data2_manu = "%0.2e" % b

        # Plot
        data2 = {"x": [stats_r2[0]], "y": [max(x1_pdf_grid) + 1], "hover": [data2_manu]}
        source2 = ColumnDataSource(data=data2)
        data2_line = {"x": [stats_r2[0], stats_r2[0]], "y": [max(x1_pdf_grid) + 1, 0], "hover": [str(data2_manu), str(data2_manu)]}
        source2_line = ColumnDataSource(data=data2_line)
        r2fig2_line = fig2.line("x", "y", line_width=2.25, line_color="red", alpha=0.5, source=source2_line)
        r2fig2 = fig2.circle("x", "y", fill_color="red", line_color="grey", alpha=0.75, size=7, legend=full_text, source=source2)

        # Lollipops Q2
        # Do a t-test
        # if ttest_1samp(stats_q2[1:], [stats_q2[0]])[1][0] / 2 > 0.005:
        #     a = ttest_1samp(stats_q2[1:], [stats_q2[0]])[1][0]
        #     b = a / 2
        #     data3_manu = "%0.2f" % b
        # else:
        #     a = ttest_1samp(stats_q2[1:], [stats_q2[0]])[1][0]
        #     b = a / 2
        #     data3_manu = "%0.2e" % b

        b = ttest_ind(stats_q2[1:], [stats_q2[0]], alternative='smaller')[1]
        if b > 0.005:
            data3_manu = "%0.2f" % b
        else:
            data3_manu = "%0.2e" % b

        # Plot
        data3 = {"x": [stats_q2[0]], "y": [min(x2_pdf_grid) - 1], "hover": [data3_manu]}
        source3 = ColumnDataSource(data=data3)
        data3_line = {"x": [stats_q2[0], stats_q2[0]], "y": [(min(x2_pdf_grid) - 1), 0], "hover": [data3_manu, data3_manu]}
        source3_line = ColumnDataSource(data=data3_line)
        q2fig2_line = fig2.line("x", "y", line_width=2.25, line_color="blue", alpha=0.5, source=source3_line)
        q2fig2 = fig2.circle("x", "y", fill_color="blue", line_color="grey", alpha=0.75, size=7, legend=cv_text, source=source3)

        if hide_pval == False:
            # Add text
            textr2 = "True " + full_text + "\nP-Value: {}".format(data2_manu)
            textq2 = "True " + cv_text + "\nP-Value: {}".format(data3_manu)
            fig2.text(x=[stats_r2[0] + 0.05, stats_q2[0] + 0.05], y=[(max(x1_pdf_grid) + 0.5), (min(x2_pdf_grid) - 1.5)], text=[textr2, textq2], angle=0, text_font_size="8pt")

        # Font-sizes
        fig1.xaxis.axis_label_text_font_size = "13pt"
        fig1.yaxis.axis_label_text_font_size = "13pt"
        fig2.xaxis.axis_label_text_font_size = "12pt"
        fig2.yaxis.axis_label_text_font_size = "12pt"
        fig1.legend.location = "bottom_right"
        fig2.legend.location = "top_left"
        fig1.legend.visible = True
        fig2.legend.visible = True

        if grid_line == False:
            fig1.xgrid.visible = False
            fig1.ygrid.visible = False
            fig2.xgrid.visible = False
            fig2.ygrid.visible = False

        if legend == False:
            fig1.legend.visible = False
            fig2.legend.visible = False

        fig = gridplot([[fig1, fig2]])
        return fig
