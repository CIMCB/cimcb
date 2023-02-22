import numpy as np
import scipy
import pandas as pd
from scipy.stats import norm
import math
import multiprocessing
from copy import deepcopy
from bokeh.layouts import gridplot, column, row, layout
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from .BaseBootstrap import BaseBootstrap
from itertools import combinations
from ..plot import scatterCI, boxplot, distribution, roc_plot_boot2, scatter, scatter_ellipse
from ..utils import nested_getattr, dict_95ci, dict_median_scores


class BC(BaseBootstrap):
    """ Returns bootstrap confidence intervals using the bias-corrected boostrap interval.

    Parameters
    ----------
    model : object
        This object is assumed to store bootlist attributes in .model (e.g. modelPLS.model.x_scores_).

    X : array-like, shape = [n_samples, n_features]
        Predictor variables, where n_samples is the number of samples and n_features is the number of predictors.

    Y : array-like, shape = [n_samples, 1]
        Response variables, where n_samples is the number of samples.

    bootlist : array-like, shape = [n_bootlist, 1]
        List of attributes to calculate and return bootstrap confidence intervals.

    bootnum : a positive integer, (default 100)
        The number of bootstrap samples used in the computation.

    seed: integer or None (default None)
        Used to seed the generator for the resample with replacement.

    Returns
    -------
    bootci : dict of arrays
        Keys correspond to attributes in bootlist.
        Each array contains 95% confidence intervals.
        To return bootci, initalise then use method run().
    """

    def __init__(self, model, bootnum=100, seed=None, n_cores=-1):
        super().__init__(model=model, bootnum=bootnum, seed=seed, n_cores=n_cores)

    def calc_bootidx(self):
        super().calc_bootidx()

    def calc_bootstat(self):
        super().calc_bootstat()

    def calc_bootci(self):
        self.bootci = {}
        for i in self.bootlist:
            self.bootci[i] = self.bootci_method(self.bootstat[i], self.stat[i])

    def run(self):
        self.calc_bootidx()
        self.calc_bootstat()
        self.calc_bootci()

    def evaluate(self, parametric=True, errorbar=False, specificity=False, cutoffscore=False, title_align="left", dist_smooth=None, bc='nonparametric', label=None, legend='roc'):

        legend_violin = False
        legend_dist = False
        legend_roc = False
        if legend in [True, 'all']:
            legend_violin = True
            legend_dist = True
            legend_roc = True
        if legend in [False, 'none', None]:
            legend_violin = False
            legend_dist = False
            legend_roc = False
        if legend is "violin":
            legend_violin = True
        if legend is "dist":
            legend_dist = True
        if legend is "roc":
            legend_roc = True

        Y = self.Y
        if label is None:
            label = ['0', '1']
        else:
            label1 = np.array(label[self.Y == 0])[0]
            label2 = np.array(label[self.Y == 1])[0]
            label = [str(label1), str(label2)]

        violin_title = ""

        # OOB
        x_loc_oob_dict = {k: [] for k in range(len(self.Y))}
        for i in range(len(self.bootidx_oob)):
            for j in range(len(self.bootidx_oob[i])):
                val = self.bootstat_oob['Y_pred'][i][j]
                idx = self.bootidx_oob[i][j]
                x_loc_oob_dict[idx].append(val)

        x_loc_oob_ci_dict = dict_95ci(x_loc_oob_dict)
        x_loc_oob_ci = []
        for key in x_loc_oob_ci_dict.keys():
            x_loc_oob_ci.append(x_loc_oob_ci_dict[key])
        ypred_oob = np.array(x_loc_oob_ci)

        # IB
        x_loc_ib_dict = {k: [] for k in range(len(self.Y))}
        for i in range(len(self.bootidx)):
            for j in range(len(self.bootidx[i])):
                val = self.bootstat['Y_pred'][i][j]
                idx = self.bootidx[i][j]
                x_loc_ib_dict[idx].append(val)

        x_loc_ib_ci_dict = dict_95ci(x_loc_ib_dict)
        x_loc_ib_ci = []
        for key in x_loc_ib_ci_dict.keys():
            x_loc_ib_ci.append(x_loc_ib_ci_dict[key])
        ypred_ib = np.array(x_loc_ib_ci)

        self.ypred_ib = ypred_ib
        self.ypred_oob = ypred_oob

        Ytrue_train = self.Y
        Ytrue_test = self.Y
        Yscore_train = ypred_ib[:, 2]
        Yscore_test = ypred_oob[:, 2]

        Yscore_combined_nan = np.concatenate([Yscore_train, Yscore_test])
        Ytrue_combined_nan = np.concatenate([Ytrue_train, Ytrue_test + 2])  # Each Ytrue per group is unique
        Yscore_combined = []
        Ytrue_combined = []
        for i in range(len(Yscore_combined_nan)):
            if np.isnan(Yscore_combined_nan[i]) == False:
                Yscore_combined.append(Yscore_combined_nan[i])
                Ytrue_combined.append(Ytrue_combined_nan[i])
        Yscore_combined = np.array(Yscore_combined)
        Ytrue_combined = np.array(Ytrue_combined)
        self.Yscore_combined = Yscore_combined
        self.Ytrue_combined = Ytrue_combined
        Ytrue_combined_name = Ytrue_combined.astype(np.str)

        Ytrue_combined_name[Ytrue_combined == 0] = "IB (0)"
        Ytrue_combined_name[Ytrue_combined == 1] = "IB (1)"
        Ytrue_combined_name[Ytrue_combined == 2] = "OOB (0)"
        Ytrue_combined_name[Ytrue_combined == 3] = "OOB (1)"
        group_name = ["IB (0)", "OOB (0)", "IB (1)", "OOB (1)"]
        group_name_sort = np.sort(group_name)
        label_violin = label + label
        violin_bokeh = boxplot(Yscore_combined, Ytrue_combined_name, xlabel="Class", ylabel="Median Predicted Score", violin=True, color=["#FFCCCC", "#CCE5FF", "#FFCCCC", "#CCE5FF"], width=320, height=315, group_name=group_name, group_name_sort=group_name_sort, title=violin_title, legend=legend_violin, label=label_violin, font_size="11pt", label_font_size="10pt")

        # Distribution plot
        dist_bokeh = distribution(Yscore_combined, group=Ytrue_combined_name, kde=True, title="", xlabel="Median Predicted Score", ylabel="p.d.f.", width=320, height=315, padding=0.7, label_font_size="10pt", smooth=dist_smooth, group_label=label, legend_location="top_left", legend=legend_dist)

        roc_bokeh = roc_plot_boot2(ypred_ib, ypred_oob, Y, self.bootstat['Y_pred'], self.bootidx, self.bootstat_oob['Y_pred'], self.bootidx_oob, self.stat['Y_pred'], width=320, height=315, label_font_size="10pt", parametric=bc, bc=True, legend=legend_roc)

        fig = layout([[violin_bokeh, dist_bokeh, roc_bokeh]])
        output_notebook()
        show(fig)

    def plot_loadings(self, PeakTable, peaklist=None, ylabel="Label", sort=False, sort_ci=True):
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

        n_loadings = len(self.bootci["model.x_loadings_"])
        ci_loadings = self.bootci["model.x_loadings_"]

        # Remove rows from PeakTable if not in peaklist
        if peaklist is not None:
            PeakTable = PeakTable[PeakTable["Name"].isin(peaklist)]
        peaklabel = PeakTable[ylabel]

        a = [None] * 2

        if self.name == 'cimcb.model.NN_SigmoidSigmoid':
            lv_name = "Neuron"
        else:
            lv_name = "LV"

        # Plot
        plots = []
        for i in range(n_loadings):
            if ci_loadings is None:
                cii = None
            else:
                cii = ci_loadings[i]
            fig = scatterCI(self.stat['model.x_loadings_'][:, i],
                            ci=cii,
                            label=peaklabel,
                            hoverlabel=PeakTable[["Idx", "Name", "Label"]],
                            hline=0,
                            col_hline=True,
                            title="Loadings Plot: {} {}".format(lv_name, i + 1),
                            sort_abs=sort,
                            sort_ci=sort_ci)
            plots.append([fig])

        fig = layout(plots)
        output_notebook()
        show(fig)

    def plot_projections(self, label=None, size=12, ci95=True, scatterplot=False, weight_alt=False, bc="nonparametric", legend='all', plot='ci', scatter_ib=True):
        bootx = 1
        num_x_scores = len(self.stat['model.x_scores_'].T)

        legend_scatter = False
        legend_dist = False
        legend_roc = False
        if legend in [True, 'all']:
            legend_scatter = True
            legend_dist = True
            legend_roc = True
        if legend in [False, 'none', None]:
            legend_scatter = False
            legend_dist = False
            legend_roc = False
        if legend is "scatter":
            legend_scatter = True
        if legend is "dist":
            legend_dist = True
        if legend is "roc":
            legend_roc = True

        if plot == "ci":
            scatter_show = 0
        elif plot == "innerci":
            scatter_show = 1
        elif plot == "ib":
            scatter_show = 2
        elif plot == "oob":
            scatter_show = 3
        elif plot == "all":
            scatter_show = 4
        else:
            raise ValueError("plot has to be either 'ci', 'innerci', 'ib', 'oob', 'all'.")

        if self.name == 'cimcb.model.NN_SigmoidSigmoid':
            lv_name = "Neuron"
        else:
            lv_name = "LV"

        # pctvar
        pctvar_all = []
        for i in range(len(self.bootidx_oob)):
            val = self.bootstat_oob['model.pctvar_'][i]
            pctvar_all.append(val)
        pctvar_ = np.median(pctvar_all, 0)

        # y_loadings_
        y_loadings_all = []
        for i in range(len(self.bootidx_oob)):
            val = self.bootstat_oob['model.y_loadings_'][i]
            y_loadings_all.append(val)
        y_loadings_ = np.median(y_loadings_all, 0)[0]

        # OOB
        x_loc_oob_dict = {k: [] for k in range(len(self.Y))}
        for i in range(len(self.bootidx_oob)):
            for j in range(len(self.bootidx_oob[i])):
                val = self.bootstat_oob['model.x_scores_'][i][j]
                idx = self.bootidx_oob[i][j]
                x_loc_oob_dict[idx].append(val)

        x_loc_oob_ci_dict = dict_median_scores(x_loc_oob_dict)
        x_loc_oob_ci = []
        for key in x_loc_oob_ci_dict.keys():
            x_loc_oob_ci.append(x_loc_oob_ci_dict[key])
        x_scores_oob = np.array(x_loc_oob_ci)

        # IB
        x_loc_ib_dict = {k: [] for k in range(len(self.Y))}
        for i in range(len(self.bootidx)):
            for j in range(len(self.bootidx[i])):
                val = self.bootstat['model.x_scores_'][i][j]
                idx = self.bootidx[i][j]
                x_loc_ib_dict[idx].append(val)

        x_loc_ib_ci_dict = dict_median_scores(x_loc_ib_dict)
        x_loc_ib_ci = []
        for key in x_loc_ib_ci_dict.keys():
            x_loc_ib_ci.append(x_loc_ib_ci_dict[key])
        x_scores_ib = np.array(x_loc_ib_ci)

        # Original Scores
        x_scores_ = self.stat['model.x_scores_']

        # BC
        x_scores_bc = []
        for i in self.bootci['model.x_scores_']:
            x_scores_bc.append(i[:, 2])
        x_scores_bc = np.array(x_scores_bc)
        x_scores_bc = x_scores_bc.T

        if num_x_scores == 1:
            pass
        else:
            if scatter_ib is True:
                x_scores_full = x_scores_ib
            else:
                x_scores_full = x_scores_
            #x_scores_full = x_scores_bc
            x_scores_cv = x_scores_oob

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
                if label is None:
                    group_copy = self.Y.copy()
                    label_copy = pd.Series(self.Y)
                else:
                    newlabel = np.array(label)
                    label_copy = deepcopy(label)
                    #group_copy = deepcopy(newlabel)
                    group_copy = self.Y.copy()

                # Scatterplot
                x, y = comb_x_scores[i]
                xlabel = "{} {} ({:0.1f}%)".format(lv_name, x + 1, pctvar_[x])
                ylabel = "{} {} ({:0.1f}%)".format(lv_name, y + 1, pctvar_[y])
                gradient = y_loadings_[y] / y_loadings_[x]

                max_range = max(np.max(np.abs(x_scores_full[:, x])), np.max(np.abs(x_scores_cv[:, y])))
                new_range_min = -max_range - 0.05 * max_range
                new_range_max = max_range + 0.05 * max_range
                new_range = (new_range_min, new_range_max)

                grid[y, x] = scatter_ellipse(x_scores_full[:, x].tolist(), x_scores_full[:, y].tolist(), x_scores_cv[:, x].tolist(), x_scores_cv[:, y].tolist(), label=label_copy, group=group_copy, title="", xlabel=xlabel, ylabel=ylabel, width=width_height, height=width_height, legend=legend_scatter, size=circle_size_scoreplot, label_font_size=label_font, hover_xy=False, xrange=new_range, yrange=new_range, gradient=gradient, ci95=True, scatterplot=scatterplot, extraci95_x=x_scores_cv[:, x].tolist(), extraci95_y=x_scores_cv[:, y].tolist(), extraci95=True, scattershow=scatter_show)

            # Append each distribution curve
            group_dist = np.concatenate((self.Y, (self.Y + 2)))

            dist_label1 = np.array(label_copy[self.Y == 0])[0]
            dist_label2 = np.array(label_copy[self.Y == 1])[0]
            dist_label = [str(dist_label1), str(dist_label2)]

            for i in range(num_x_scores):
                score_dist = np.concatenate((x_scores_full[:, i], x_scores_cv[:, i]))
                xlabel = "{} {} ({:0.1f}%)".format(lv_name, i + 1, pctvar_[i])
                grid[i, i] = distribution(score_dist, group=group_dist, kde=True, title="", xlabel=xlabel, ylabel="p.d.f.", width=width_height, height=width_height, label_font_size=label_font, legend=legend_dist, group_label=dist_label)

            # Append each roc curve
            for i in range(len(comb_x_scores)):
                x, y = comb_x_scores[i]

                # Get the optimal combination of x_scores based on rotation of y_loadings_
                gradient = y_loadings_[y] / y_loadings_[x]
                theta = math.atan(gradient)
                x_rotate_stat = self.stat['model.x_scores_'][:, x] * math.cos(theta) + self.stat['model.x_scores_'][:, y] * math.sin(theta)
                x_rotate_ib = []
                for i in self.bootstat['model.x_scores_']:
                    val = i[:, x] * math.cos(theta) + i[:, y] * math.sin(theta)
                    x_rotate_ib.append(val)

                x_rotate_oob = []
                for i in self.bootstat_oob['model.x_scores_']:
                    val = i[:, x] * math.cos(theta) + i[:, y] * math.sin(theta)
                    x_rotate_oob.append(val)

                # x_rotate = x_scores_full[:, x] * math.cos(theta) + x_scores_full[:, y] * math.sin(theta)
                # x_rotate_boot = x_scores_cv[:, x] * math.cos(theta) + x_scores_cv[:, y] * math.sin(theta)
                Y = self.Y
                grid[x, y] = roc_plot_boot2(Y, Y, Y, x_rotate_ib, self.bootidx, x_rotate_oob, self.bootidx_oob, x_rotate_stat, width=width_height, height=width_height, xlabel="1-Specificity (LV{}{}/LV{}{})".format(lv_name, x + 1, lv_name, y + 1), ylabel="Sensitivity ({}{}/{}{})".format(lv_name, x + 1, lv_name, y + 1), label_font_size=label_font, parametric=bc, bc=True, legend=legend_roc)

                # self.x_rotate = x_rotate
                # self.Y = group_copy
                # self.x_rotate_boot = []
                # for i in range(len(x_scores_cvall)):
                #     x_rot = x_scores_cvall[i][:, x] * math.cos(theta) + x_scores_cvall[i][:, y] * math.sin(theta)
                #     self.x_rotate_boot.append(x_rot)
                # self.x_rotate_boot = np.array(self.x_rotate_boot)
                # x_rotate_boot = self.x_rotate_boot
                # # ROC Plot with x_rotate
                # fpr, tpr, tpr_ci = roc_calculate(group_copy, x_rotate, bootnum=100)
                # fpr_boot, tpr_boot, tpr_ci_boot = roc_calculate(group_copy, x_rotate_boot, bootnum=100)

                # grid[x, y] = roc_plot(fpr, tpr, tpr_ci, width=width_height, height=width_height, xlabel="1-Specificity (LV{}/LV{})".format(x + 1, y + 1), ylabel="Sensitivity (LV{}/LV{})".format(x + 1, y + 1), legend=False, label_font_size=label_font, roc2=True, fpr2=fpr_boot, tpr2=tpr_boot, tpr_ci2=tpr_ci_boot)

                # grid[x, y] = roc_plot_cv(x_rotate, x_rotate_boot, group_copy, width=width_height, height=width_height, xlabel="1-Specificity (LV{}/LV{})".format(x + 1, y + 1), ylabel="Sensitivity (LV{}/LV{})".format(x + 1, y + 1), legend=False, label_font_size=label_font)

            # Bokeh grid
            fig = gridplot(grid.tolist())

        output_notebook()
        show(fig)

    def plot_featureimportance(self, PeakTable, peaklist=None, ylabel="Label", sort=True, sort_ci=True):
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

        ci_coef = self.bootci["model.coef_"]
        ci_vip = self.bootci["model.vip_"]

        if self.name == 'cimcb.model.NN_SigmoidSigmoid':
            name_coef = "Feature Importance: Connection Weight"
            name_vip = "Feature Importance: Garlson's Algorithm"
        else:
            name_coef = "Coefficient Plot"
            name_vip = "Variable Importance in Projection (VIP)"

        # Remove rows from PeakTable if not in peaklist
        if peaklist is not None:
            PeakTable = PeakTable[PeakTable["Name"].isin(peaklist)]
        peaklabel = PeakTable[ylabel]

        # Plot
        fig_1 = scatterCI(self.stat['model.coef_'], ci=ci_coef, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=0, col_hline=True, title=name_coef, sort_abs=sort, sort_ci=sort_ci)
        if name_vip == "Variable Importance in Projection (VIP)":
            fig_2 = scatterCI(self.stat['model.vip_'], ci=ci_vip, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=1, col_hline=False, title=name_vip, sort_abs=sort, sort_ci=sort_ci, sort_ci_abs=True)
        else:
            fig_2 = scatterCI(self.stat['model.vip_'], ci=ci_vip, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=np.average(self.stat['model.vip_']), col_hline=False, title=name_vip, sort_abs=sort, sort_ci_abs=True)

        ######
        fig = layout([[fig_1], [fig_2]])
        output_notebook()
        show(fig)

        # Return table with: Idx, Name, Label, Coefficient, 95CI, VIP, 95CI
        coef = pd.DataFrame([self.stat['model.coef_'], self.bootci["model.coef_"]]).T
        coef.rename(columns={0: "Coef", 1: "Coef-95CI"}, inplace=True)
        vip = pd.DataFrame([self.stat['model.vip_'], self.bootci["model.vip_"]]).T
        vip.rename(columns={0: "VIP", 1: "VIP-95CI"}, inplace=True)

        Peaksheet = PeakTable.copy()
        Peaksheet["Coef"] = coef["Coef"].values
        Peaksheet["VIP"] = vip["VIP"].values

        Peaksheet["Coef-95CI"] = coef["Coef-95CI"].values
        Peaksheet["VIP-95CI"] = vip["VIP-95CI"].values

        return Peaksheet

    @staticmethod
    def bootci_method(bootstat, stat):
        """Calculates bootstrap confidence intervals using the bias-corrected bootstrap interval."""
        if stat.ndim == 1:
            nboot = len(bootstat)
            zalpha = norm.ppf(0.05 / 2)
            obs = stat  # Observed mean
            meansum = np.zeros((1, len(obs))).flatten()
            for i in range(len(obs)):
                for j in range(len(bootstat)):
                    if bootstat[j][i] >= obs[i]:
                        meansum[i] = meansum[i] + 1
            prop = meansum / nboot  # Proportion of times boot mean > obs mean
            z0 = -norm.ppf(prop)

            # new alpha
            pct1 = 100 * norm.cdf((2 * z0 + zalpha))
            pct2 = 100 * norm.cdf((2 * z0 - zalpha))
            pct3 = 100 * norm.cdf((2 * z0))
            boot_ci = []
            for i in range(len(pct1)):
                bootstat_i = [item[i] for item in bootstat]
                append_low = np.percentile(bootstat_i, pct1[i])
                append_mid = np.percentile(bootstat_i, pct3[i])
                append_upp = np.percentile(bootstat_i, pct2[i])
                boot_ci.append([append_low, append_upp, append_mid])
            boot_ci = np.array(boot_ci)

        # Recursive component (to get ndim = 1, and append)
        else:
            ncomp = stat.shape[1]
            boot_ci = []
            for k in range(ncomp):
                bootstat_k = []
                for j in range(len(bootstat)):
                    bootstat_k.append(bootstat[j][:, k])
                boot_ci_k = BC.bootci_method(bootstat_k, stat[:, k])
                boot_ci.append(boot_ci_k)
            boot_ci = np.array(boot_ci)
        return boot_ci
