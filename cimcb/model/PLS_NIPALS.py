import math
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import combinations
from sklearn.cross_decomposition import PLSRegression
from bokeh.plotting import output_notebook, show
from bokeh.layouts import gridplot
from bokeh.plotting import ColumnDataSource, figure
from .BaseModel import BaseModel
from ..plot import scatter, distribution, roc_calculate, roc_plot, boxplot
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
import scipy
import collections
import math
from scipy.stats import logistic
from itertools import combinations
from copy import deepcopy, copy
from bokeh.layouts import widgetbox, gridplot, column, row, layout
from bokeh.models import HoverTool, Band
from bokeh.models.widgets import DataTable, Div, TableColumn
from bokeh.models.annotations import Title
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from scipy import interp
from sklearn import metrics
from sklearn.utils import resample
from ..bootstrap import Perc, BC, BCA
from ..plot import scatter, scatterCI, boxplot, distribution, permutation_test, roc_calculate, roc_plot, roc_calculate_boot, roc_plot_boot
from ..utils import binary_metrics


class PLS_NIPALS(BaseModel):
    """ Partial least-squares regression using the SIMPLS algorithm.

    Parameters
    ----------
    n_components : int, (default 2)
        Number of components to keep.

    Methods
    -------
    train : Fit model to data.

    test : Apply model to test data.

    evaluate : Evaluate model.

    calc_bootci : Calculate bootstrap intervals for plot_featureimportance.

    plot_featureimportance : Plot coefficient and Variable Importance in Projection (VIP).

    plot_permutation_test : Perform a permutation test and plot.
    """

    parametric = True
    bootlist = ["model.vip_", "model.coef_"]  # list of metrics to bootstrap

    def __init__(self, n_components=2):
        self.model = PLSRegression(n_components=n_components)  # Should change this to an empty model
        self.n_component = n_components
        self.k = n_components

        self.__name__ = 'cimcb.model.PLS_NIPALS'
        self.__params__ = {'n_components': n_components}

    def set_params(self, params):
        self.__init__(**params)

    def train(self, X, Y):
        """ Fit the PLS model, save additional stats (as attributes) and return Y predicted values.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Predictor variables, where n_samples is the number of samples and n_features is the number of predictors.

        Y : array-like, shape = [n_samples, 1]
            Response variables, where n_samples is the number of samples.

        Returns
        -------
        y_pred_train : array-like, shape = [n_samples, 1]
            Predicted y score for samples.
        """
        # Error check
        X, Y = self.input_check(X, Y)

        # Fit model
        self.model.fit(X, Y)

        # Calculate vip, pctvar (Explained variance in X) and flatten coef_ for future use
        self.model.pctvar_ = (
            sum(abs(self.model.x_scores_) ** 2) / sum(sum(abs(X) ** 2)) * 100
        )
        #self.model.vip_ = vip(self.model)
        #self.model.coef_ = self.model.coef_.flatten()

        T = self.model.x_scores_
        W = self.model.x_weights_
        Q = self.model.y_loadings_
        w0, w1 = W.shape
        s = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
        s_sum = np.sum(s, axis=0)
        w_norm = np.array([(W[:, i] / np.linalg.norm(W[:, i]))
                           for i in range(w1)])
        self.model.vip_ = np.sqrt(w0 * np.sum(s * w_norm.T ** 2, axis=1) / s_sum)

        # Calculate and return Y predicted value
        y_pred_train = self.model.predict(X)

        # Storing X and Y (for now?)
        # Now should I save Y and Y_train, if so... then should we be initialising
        # Pro -> X and Y are always stored with model, won't need to require it as input for evaluate, bootstrap etc.
        # Cons -> Using up storage (but we are using small matrices), and stretching the purpose of these classes
        self.Y_pred = y_pred_train  # Y_pred vs. Y_pred_train
        self.Y_true = Y
        self.X = X
        self.Y = Y  # Y vs. Y_true
        return y_pred_train

    def test(self, X):
        """Calculate and return Y predicted value.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Test variables, where n_samples is the number of samples and n_features is the number of predictors.

        Returns
        -------
        y_pred_test : array-like, shape = [n_samples, 1]
            Predicted y score for samples.
        """
        # Convert to X to numpy array if a DataFrame
        if isinstance(X, pd.DataFrame or pd.Series):
            X = np.array(X)

        # Overwrite x_scores_ from model.fit with using test X (or do model.x_scores_test_) ?
        self.model.x_scores_ = self.model.transform(X)
        self.lv_projections = (
            self.model.x_scores_
        )  # Should lv_projections be a standard?

        # Calculate and return Y predicted value
        y_pred_test = self.model.predict(X)
        return y_pred_test

    def plot_projections(self, label=None, size=12, ci95=True, scatterplot=True, weight_alt=False):
        """ Plots latent variables projections against each other in a Grid format.

        Parameters
        ----------
        label : DataFrame or None, (default None)
            hovertool for scatterplot.

        size : positive integer, (default 12)
            size specifies circle size for scatterplot.
        """

        if not hasattr(self, "bootci"):
            print("Use method calc_bootci prior to plot_projections.")
            bootx = None
        else:
            bootx = 1

        Yrem = deepcopy(self.Y)
        try:
            if len(self.Y[0]) > 1:
                dummy = pd.DataFrame(self.Y)
                stratY = dummy.idxmax(axis=1)
            else:
                stratY = self.Y
        except TypeError:
            stratY = self.Y
        self.Y = stratY

        x_scores_true = deepcopy(self.model.x_scores_)
        if weight_alt is True:
            self.model.x_scores_ = self.model.x_scores_alt

        num_x_scores = len(self.model.x_scores_.T)

        # If there is only 1 x_score, Need to plot x_score vs. peak (as opposided to x_score[i] vs. x_score[j])
        if num_x_scores == 1:
            # Violin plot
            violin_bokeh = boxplot(self.Y_pred.flatten(), self.Y, title="", xlabel="Class", ylabel="Predicted Score", violin=True, color=["#FFCCCC", "#CCE5FF"], width=320, height=315)
            # Distribution plot
            dist_bokeh = distribution(self.Y_pred, group=self.Y, kde=True, title="", xlabel="Predicted Score", ylabel="p.d.f.", width=320, height=315)
            # ROC plot
            fpr, tpr, tpr_ci = roc_calculate(self.Y, self.Y_pred, bootnum=100)
            roc_bokeh = roc_plot(fpr, tpr, tpr_ci, width=310, height=315)
            # Score plot
            y = self.model.x_scores_[:, 0].tolist()
            # get label['Idx'] if it exists
            try:
                x = label["Idx"].values.ravel()
            except:
                x = []
                for i in range(len(y)):
                    x.append(i)
            scatter_bokeh = scatter(x, y, label=label, group=self.Y, ylabel="LV {} ({:0.1f}%)".format(1, self.model.pctvar_[0]), xlabel="Idx", legend=True, title="", width=950, height=315, hline=0, size=int(size / 2), hover_xy=False)

            # Combine into one figure
            fig = gridplot([[violin_bokeh, dist_bokeh, roc_bokeh], [scatter_bokeh]])

        else:
            if bootx is None:
                comb_x_scores = list(combinations(range(num_x_scores), 2))

                # Width/height of each scoreplot
                width_height = int(950 / num_x_scores)
                circle_size_scoreplot = size / num_x_scores
                label_font = str(13 - num_x_scores) + "pt"

                # Create empty grid
                grid = np.full((num_x_scores, num_x_scores), None)

                # Append each scoreplot
                for i in range(len(comb_x_scores)):
                    # Make a copy (as it overwrites the input label/group)
                    label_copy = deepcopy(label)
                    group_copy = self.Y.copy()

                    # Scatterplot
                    x, y = comb_x_scores[i]
                    xlabel = "LV {} ({:0.1f}%)".format(x + 1, self.model.pctvar_[x])
                    ylabel = "LV {} ({:0.1f}%)".format(y + 1, self.model.pctvar_[y])
                    gradient = self.model.y_loadings_[0][y] / self.model.y_loadings_[0][x]

                    max_range = max(np.max(np.abs(self.model.x_scores_[:, x])), np.max(np.abs(self.model.x_scores_[:, y])))
                    new_range_min = -max_range - 0.05 * max_range
                    new_range_max = max_range + 0.05 * max_range
                    new_range = (new_range_min, new_range_max)

                    grid[y, x] = scatter(self.model.x_scores_[:, x].tolist(), self.model.x_scores_[:, y].tolist(), label=label_copy, group=group_copy, title="", xlabel=xlabel, ylabel=ylabel, width=width_height, height=width_height, legend=False, size=circle_size_scoreplot, label_font_size=label_font, hover_xy=False, xrange=new_range, yrange=new_range, gradient=gradient, ci95=ci95, scatterplot=scatterplot)

                # Append each distribution curve
                for i in range(num_x_scores):
                    xlabel = "LV {} ({:0.1f}%)".format(i + 1, self.model.pctvar_[i])
                    grid[i, i] = distribution(self.model.x_scores_[:, i], group=group_copy, kde=True, title="", xlabel=xlabel, ylabel="density", width=width_height, height=width_height, label_font_size=label_font)

                # Append each roc curve
                for i in range(len(comb_x_scores)):
                    x, y = comb_x_scores[i]

                    # Get the optimal combination of x_scores based on rotation of y_loadings_
                    theta = math.atan(self.model.y_loadings_[0][y] / self.model.y_loadings_[0][x])
                    x_rotate = self.model.x_scores_[:, x] * math.cos(theta) + self.model.x_scores_[:, y] * math.sin(theta)

                    # ROC Plot with x_rotate
                    fpr, tpr, tpr_ci = roc_calculate(np.abs(1 - self.Y), x_rotate, bootnum=100)
                    #print('k- inv')

                    if metrics.auc(fpr, tpr) < 0.5:
                        fpr, tpr, tpr_ci = roc_calculate(group_copy, x_rotate, bootnum=100)
                    grid[x, y] = roc_plot(fpr, tpr, tpr_ci, width=width_height, height=width_height, xlabel="1-Specificity (LV{}/LV{})".format(x + 1, y + 1), ylabel="Sensitivity (LV{}/LV{})".format(x + 1, y + 1), legend=False, label_font_size=label_font)

            else:

                if weight_alt is False:
                    a = 'model.x_scores_'
                else:
                    a = 'model.x_scores_alt'

                x_scores_boot = {}
                for i in range(len(self.boot.bootstat[a])):
                    # This is the scores for each bootstrap
                    # Do some checks and switches ect... then append
                    for j in range(len(self.boot.bootidx[i])):
                        try:
                            x_scores_boot[self.boot.bootidx[i][j]].append(self.boot.bootstat[a][i][j])
                        except KeyError:
                            x_scores_boot[self.boot.bootidx[i][j]] = [self.boot.bootstat[a][i][j]]

                boot_xscores = []
                for i in range(len(self.Y)):
                    scoresi = x_scores_boot[i]
                    scoresi = np.array(scoresi)
                    scoresmed = np.median(np.array(scoresi), axis=0)
                    boot_xscores.append(scoresmed)

                boot_xscores = np.array(boot_xscores)

                self.a = boot_xscores

                comb_x_scores = list(combinations(range(num_x_scores), 2))

                # Width/height of each scoreplot
                width_height = int(950 / num_x_scores)
                circle_size_scoreplot = size / num_x_scores
                label_font = str(13 - num_x_scores) + "pt"

                # Create empty grid
                grid = np.full((num_x_scores, num_x_scores), None)

                # Append each scoreplot
                for i in range(len(comb_x_scores)):
                    # Make a copy (as it overwrites the input label/group)
                    label_copy = deepcopy(label)
                    group_copy = self.Y.copy()

                    # Scatterplot
                    x, y = comb_x_scores[i]
                    xlabel = "LV {} ({:0.1f}%)".format(x + 1, self.model.pctvar_[x])
                    ylabel = "LV {} ({:0.1f}%)".format(y + 1, self.model.pctvar_[y])
                    gradient = self.model.y_loadings_[0][y] / self.model.y_loadings_[0][x]

                    max_range = max(np.max(np.abs(self.model.x_scores_[:, x])), np.max(np.abs(self.model.x_scores_[:, y])))
                    new_range_min = -max_range - 0.05 * max_range
                    new_range_max = max_range + 0.05 * max_range
                    new_range = (new_range_min, new_range_max)

                    grid[y, x] = scatter(self.model.x_scores_[:, x].tolist(), self.model.x_scores_[:, y].tolist(), label=label_copy, group=group_copy, title="", xlabel=xlabel, ylabel=ylabel, width=width_height, height=width_height, legend=False, size=circle_size_scoreplot, label_font_size=label_font, hover_xy=False, xrange=new_range, yrange=new_range, gradient=gradient, ci95=ci95, scatterplot=scatterplot, extraci95_x=boot_xscores[:, x].tolist(), extraci95_y=boot_xscores[:, y].tolist(), extraci95=True)

                # Append each distribution curve
                group_dist = np.concatenate((self.Y, (self.Y + 2)))

                for i in range(num_x_scores):
                    score_dist = np.concatenate((self.model.x_scores_[:, i], boot_xscores[:, i]))
                    xlabel = "LV {} ({:0.1f}%)".format(i + 1, self.model.pctvar_[i])
                    grid[i, i] = distribution(score_dist, group=group_dist, kde=True, title="", xlabel=xlabel, ylabel="density", width=width_height, height=width_height, label_font_size=label_font)

                # Append each roc curve
                for i in range(len(comb_x_scores)):
                    x, y = comb_x_scores[i]

                    # Get the optimal combination of x_scores based on rotation of y_loadings_
                    theta = math.atan(self.model.y_loadings_[0][y] / self.model.y_loadings_[0][x])
                    x_rotate = self.model.x_scores_[:, x] * math.cos(theta) + self.model.x_scores_[:, y] * math.sin(theta)
                    x_rotate_boot = boot_xscores[:, x] * math.cos(theta) + boot_xscores[:, y] * math.sin(theta)

                    # ROC Plot with x_rotate
                    fpr, tpr, tpr_ci = roc_calculate(group_copy, x_rotate, bootnum=100)
                    fpr_boot, tpr_boot, tpr_ci_boot = roc_calculate(group_copy, x_rotate_boot, bootnum=100)

                    grid[x, y] = roc_plot(fpr, tpr, tpr_ci, width=width_height, height=width_height, xlabel="1-Specificity (LV{}/LV{})".format(x + 1, y + 1), ylabel="Sensitivity (LV{}/LV{})".format(x + 1, y + 1), legend=False, label_font_size=label_font, roc2=True, fpr2=fpr_boot, tpr2=tpr_boot, tpr_ci2=tpr_ci_boot)
            # Bokeh grid
            fig = gridplot(grid.tolist())

        self.Y = Yrem
        self.model.x_scores_ = x_scores_true
        output_notebook()
        show(fig)

    def plot_featureimportance(self, PeakTable, peaklist=None, ylabel="Label", sort=True):
        """Plots feature importance metrics.

        Parameters
        ----------
        PeakTable : DataFrame
            Peak sheet with the required columns.

        peaklist : list or None, (default None)
            Peaks to include in plot (the default is to include all samples).

        ylabel : string, (default "Label")
            Name of column in PeakTable to use as the ylabel.

        sort : boolean, (default True)
            Whether to sort plots in absolute descending order.

        Returns
        -------
        Peaksheet : DataFrame
            New PeakTable with added "Coef" and "VIP" columns (+ "Coef-95CI" and  "VIP-95CI" if calc_bootci is used prior to plot_featureimportance).
        """
        if not hasattr(self, "bootci"):
            print("Use method calc_bootci prior to plot_featureimportance to add 95% confidence intervals to plots.")
            ci_coef = None
            ci_vip = None
        else:
            ci_coef1 = self.bootci["model.coef_"]
            ci_vip = self.bootci["model.vip_"]

        # Remove rows from PeakTable if not in peaklist
        if peaklist is not None:
            PeakTable = PeakTable[PeakTable["Name"].isin(peaklist)]
        peaklabel = PeakTable[ylabel]

        # Plot
        fig_1 = scatterCI(self.model.coef_[:, 0], ci=ci_coef, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=0, col_hline=True, title="Coefficient Plot: [1 0 0]", sort_abs=sort)
        fig_2 = scatterCI(self.model.coef_[:, 1], ci=ci_coef, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=0, col_hline=True, title="Coefficient Plot: [0 1 0]", sort_abs=sort)
        fig_3 = scatterCI(self.model.coef_[:, 2], ci=ci_coef, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=0, col_hline=True, title="Coefficient Plot: [0 0 1]", sort_abs=sort)
        fig_4 = scatterCI(self.model.vip_, ci=ci_vip, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=1, col_hline=False, title="Variable Importance in Projection (VIP)", sort_abs=sort)
        fig = layout([[fig_1], [fig_2], [fig_3], [fig_4]])
        output_notebook()
        show(fig)

        # Return table with: Idx, Name, Label, Coefficient, 95CI, VIP, 95CI
        Peaksheet = []
        return Peaksheet
