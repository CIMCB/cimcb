from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
import scipy
import collections
import math
from tqdm import tqdm
from copy import deepcopy, copy
from scipy.stats import logistic
from itertools import combinations
from copy import deepcopy, copy
from sklearn.model_selection import StratifiedKFold
from bokeh.layouts import widgetbox, gridplot, column, row, layout
from bokeh.models import HoverTool, Band
from bokeh.models.widgets import DataTable, Div, TableColumn
from bokeh.models.annotations import Title
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from scipy import interp
from sklearn import metrics
from sklearn.utils import resample
from ..bootstrap import Perc, BC, BCA
from ..plot import scatter, scatterCI, boxplot, distribution, permutation_test, roc_plot, roc_calculate_boot, roc_plot_boot
from ..utils import binary_metrics, dict_mean, dict_median


class BaseModel(ABC):
    """Base class for models: PLS_SIMPLS."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        """Trains the model."""
        pass

    @abstractmethod
    def test(self):
        """Tests the model."""
        pass

    @abstractproperty
    def bootlist(self):
        """A list of attributes for bootstrap resampling."""
        pass

    def input_check(self, X, Y):
        # Convert to numpy array if a DataFrame
        if isinstance(X, pd.DataFrame or pd.Series):
            X = np.array(X)
            Y = np.array(Y).ravel()
        # Error checks
        if np.isnan(X).any() is True:
            raise ValueError("NaNs found in X.")
        if len(np.unique(Y)) != 2:
            raise ValueError("Y needs to have 2 groups. There is {}".format(len(np.unique(Y))))
        if np.sort(np.unique(Y))[0] != 0:
            raise ValueError("Y should only contain 0s and 1s.")
        if np.sort(np.unique(Y))[1] != 1:
            raise ValueError("Y should only contain 0s and 1s.")
        if len(X) != len(Y):
            raise ValueError("length of X does not match length of Y.")
        return X, Y

    # def calc_bootci(self, bootnum=100, type="bca"):
    #     """Calculates bootstrap confidence intervals based on bootlist.

    #     Parameters
    #     ----------
    #     bootnum : a positive integer, (default 100)
    #         The number of bootstrap samples used in the computation.

    #     type : 'bc', 'bca', 'perc', (default 'bca')
    #         Methods for bootstrap confidence intervals. 'bc' is bias-corrected bootstrap confidence intervals. 'bca' is bias-corrected and accelerated bootstrap confidence intervals. 'perc' is percentile confidence intervals.
    #     """
    #     bootlist = self.bootlist
    #     if type is "bca":
    #         boot = BCA(self, self.X, self.Y, self.bootlist, bootnum=bootnum)
    #     if type is "bc":
    #         boot = BC(self, self.X, self.Y, self.bootlist, bootnum=bootnum)
    #     if type is "perc":
    #         boot = Perc(self, self.X, self.Y, self.bootlist, bootnum=bootnum)
    #     self.boot = boot
    #     self.bootci = self.boot.run()

    def permutation_test(self, metric='r2q2', nperm=100, folds=5):
        """Plots permutation test figures.

        Parameters
        ----------
        nperm : positive integer, (default 100)
            Number of permutations.
        """
        params = self.__params__
        perm = permutation_test(self, params, self.X, self.Y, nperm=nperm, folds=folds)
        perm.run()

        if type(metric) != list:
            fig = perm.plot(metric=metric)
        else:
            fig_list = []
            for i in metric:
                fig_i = perm.plot(metric=i)
                fig_list.append([fig_i])
            fig = layout([[fig_list]])

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
        if not hasattr(self, "bootci"):
            # print("Use method calc_bootci prior to plot_featureimportance to add 95% confidence intervals to plots.")
            ci_coef = None
            ci_vip = None
        else:
            ci_coef = self.bootci["model.coef_"]
            ci_vip = self.bootci["model.vip_"]

        if self.__name__ == 'cimcb.model.NN_SigmoidSigmoid':
            name_coef = "Feature Importance: Connection Weight"
            name_vip = "Feature Importance: Garlson's Algorithm"
        elif self.__name__ == 'cimcb.model.NN_LinearSigmoid':
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
        fig_1 = scatterCI(self.model.coef_, ci=ci_coef, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=0, col_hline=True, title=name_coef, sort_abs=sort, sort_ci=sort_ci)
        if name_vip == "Variable Importance in Projection (VIP)":
            fig_2 = scatterCI(self.model.vip_, ci=ci_vip, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=1, col_hline=False, title=name_vip, sort_abs=sort, sort_ci=sort_ci, sort_ci_abs=True)
        else:
            fig_2 = scatterCI(self.model.vip_, ci=ci_vip, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=0, col_hline=False, title=name_vip, sort_abs=sort, sort_ci_abs=True)
        fig = layout([[fig_1], [fig_2]])
        output_notebook()
        show(fig)

        # Return table with: Idx, Name, Label, Coefficient, 95CI, VIP, 95CI
        if not hasattr(self, "bootci"):
            coef = pd.DataFrame([self.model.coef_]).T
            coef.rename(columns={0: "Coef"}, inplace=True)
            vip = pd.DataFrame([self.model.vip_]).T
            vip.rename(columns={0: "VIP"}, inplace=True)
        else:
            coef = pd.DataFrame([self.model.coef_, self.bootci["model.coef_"]]).T
            coef.rename(columns={0: "Coef", 1: "Coef-95CI"}, inplace=True)
            vip = pd.DataFrame([self.model.vip_, self.bootci["model.vip_"]]).T
            vip.rename(columns={0: "VIP", 1: "VIP-95CI"}, inplace=True)

        Peaksheet = PeakTable.copy()
        Peaksheet["Coef"] = coef["Coef"].values
        Peaksheet["VIP"] = vip["VIP"].values
        if hasattr(self, "bootci"):
            Peaksheet["Coef-95CI"] = coef["Coef-95CI"].values
            Peaksheet["VIP-95CI"] = vip["VIP-95CI"].values

        return Peaksheet

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
        n_loadings = len(self.model.x_loadings_[0])
        if not hasattr(self, "bootci"):
            # print("Use method calc_bootci prior to plot_loadings to add 95% confidence intervals to plots.")
            ci_loadings = None
        else:
            ci_loadings = self.bootci["model.x_loadings_"]

        # Remove rows from PeakTable if not in peaklist
        if peaklist is not None:
            PeakTable = PeakTable[PeakTable["Name"].isin(peaklist)]
        peaklabel = PeakTable[ylabel]

        a = [None] * 2

        # Plot
        plots = []
        for i in range(n_loadings):
            if ci_loadings is None:
                cii = None
            else:
                cii = ci_loadings[i]
            fig = scatterCI(self.model.x_loadings_[:, i],
                            ci=cii,
                            label=peaklabel,
                            hoverlabel=PeakTable[["Idx", "Name", "Label"]],
                            hline=0,
                            col_hline=True,
                            title="Loadings Plot: LV {}".format(i + 1),
                            sort_abs=sort,
                            sort_ci=sort_ci)
            plots.append([fig])

        fig = layout(plots)
        output_notebook()
        show(fig)

        # Return table with: Idx, Name, Label, Coefficient, 95CI, VIP, 95CI
        if not hasattr(self, "bootci"):
            returnlist = []
            for i in range(n_loadings):
                df = pd.DataFrame([self.model.x_loadings_[:, i]]).T
                df.rename(columns={0: "LV {}".format(i + 1)}, inplace=True)
        # else:
        #     coef = pd.DataFrame([self.model.coef_, self.bootci["model.coef_"]]).T
        #     coef.rename(columns={0: "Coef", 1: "Coef-95CI"}, inplace=True)
        #     vip = pd.DataFrame([self.model.vip_, self.bootci["model.vip_"]]).T
        #     vip.rename(columns={0: "VIP", 1: "VIP-95CI"}, inplace=True)

        # Peaksheet = PeakTable.copy()
        # Peaksheet["Coef"] = coef["Coef"].values
        # Peaksheet["VIP"] = vip["VIP"].values
        # if hasattr(self, "bootci"):
        #     Peaksheet["Coef-95CI"] = coef["Coef-95CI"].values
        #     Peaksheet["VIP-95CI"] = vip["VIP-95CI"].values
        # return Peaksheet

    def evaluate(self, testset=None, plot_median=False, specificity=False, cutoffscore=False, bootnum=100, title_align="left", dist_smooth=None):
        """Plots a figure containing a Violin plot, Distribution plot, ROC plot and Binary Metrics statistics.

        Parameters
        ----------
        testset : array-like, shape = [n_samples, 2] or None, (default None)
            If testset is None, use train Y and train Y predicted for evaluate. Alternatively, testset is used to evaluate model in the format [Ytest, Ypred].

        specificity : number or False, (default False)
            Use the specificity to draw error bar. When False, use the cutoff score of 0.5.

        cutoffscore : number or False, (default False)
            Use the cutoff score to draw error bar. When False, use the specificity selected.

        bootnum : a positive integer, (default 1000)
            The number of bootstrap samples used in the computation.
        """
        Ytrue_train = self.Y
        Yscore_train = self.Y_pred.flatten()

        # Get Ytrue_test, Yscore_test from testset
        if testset is not None:
            Ytrue_test = np.array(testset[0])
            Yscore_test = np.array(testset[1])

            # Error checking
            if len(Ytrue_test) != len(Yscore_test):
                raise ValueError("evaluate can't be used as length of Ytrue does not match length of Yscore in test set.")
            if len(np.unique(Ytrue_test)) != 2:
                raise ValueError("Ytrue_test needs to have 2 groups. There is {}".format(len(np.unique(Y))))
            if np.sort(np.unique(Ytrue_test))[0] != 0:
                raise ValueError("Ytrue_test should only contain 0s and 1s.")
            if np.sort(np.unique(Ytrue_test))[1] != 1:
                raise ValueError("Ytrue_test should only contain 0s and 1s.")

            # Get Yscore_combined and Ytrue_combined_name (Labeled Ytrue)
            Yscore_combined = np.concatenate([Yscore_train, Yscore_test])
            Ytrue_combined = np.concatenate([Ytrue_train, Ytrue_test + 2])  # Each Ytrue per group is unique
            Ytrue_combined_name = Ytrue_combined.astype(np.str)
            Ytrue_combined_name[Ytrue_combined == 0] = "Train (0)"
            Ytrue_combined_name[Ytrue_combined == 1] = "Train (1)"
            Ytrue_combined_name[Ytrue_combined == 2] = "Test (0)"
            Ytrue_combined_name[Ytrue_combined == 3] = "Test (1)"

        # Expliclity states which metric and value is used for the error_bar
        if specificity is not False:
            metric = "specificity"
            val = specificity
        elif cutoffscore is not False:
            metric = "cutoffscore"
            val = cutoffscore
        else:
            metric = "specificity"
            val = 0.8

        # ROC plot
        tpr, fpr, tpr_ci, stats, stats_bootci = roc_calculate(Ytrue_train, Yscore_train, bootnum=bootnum, metric=metric, val=val, parametric=self.parametric)
        # roc_title = "Specificity: {}".format(np.round(stats["val_specificity"], 2))
        roc_title = "AUC: {} ({}, {})".format(np.round(stats["AUC"], 2), np.round(stats_bootci["AUC"][0], 2), np.round(stats_bootci["AUC"][1], 2))
        roc_bokeh = roc_plot(tpr, fpr, tpr_ci, median=plot_median, width=320, height=315, title=roc_title, errorbar=stats["val_specificity"])
        if testset is not None:
            fpr_test, tpr_test, threshold_test = metrics.roc_curve(Ytrue_test, Yscore_test, pos_label=1, drop_intermediate=False)
            fpr_test = np.insert(fpr_test, 0, 0)
            tpr_test = np.insert(tpr_test, 0, 0)
            roc_bokeh.line(fpr_test, tpr_test, color="red", line_width=3.5, alpha=0.6, legend="ROC Curve (Test)")  # Add ROC Curve(Test) to roc_bokeh

            # Get Ytrue_test, Yscore_test from testset
            Ytrue_test = np.array(testset[0])
            Yscore_test = np.array(testset[1])

            # Get Yscore_combined and Ytrue_combined_name (Labeled Ytrue)
            Yscore_combined = np.concatenate([Yscore_train, Yscore_test])
            Ytrue_combined = np.concatenate([Ytrue_train, Ytrue_test + 2])  # Each Ytrue per group is unique
            Ytrue_combined_name = Ytrue_combined.astype(np.str)
            self.Yscore_combined = Yscore_combined
            self.Ytrue_combined = Ytrue_combined
            Ytrue_combined_name[Ytrue_combined == 0] = "Train (0)"
            Ytrue_combined_name[Ytrue_combined == 1] = "Train (1)"
            Ytrue_combined_name[Ytrue_combined == 2] = "Test (0)"
            Ytrue_combined_name[Ytrue_combined == 3] = "Test (1)"

        # Violin plot
        self.score = Yscore_train
        self.true = Ytrue_train

        violin_title = "Cut-off: {}".format(np.round(stats["val_cutoffscore"], 2))
        if testset is None:
            violin_bokeh = boxplot(Yscore_train, Ytrue_train, xlabel="Class", ylabel="Predicted Score", violin=True, color=["#FFCCCC", "#CCE5FF"], width=320, height=315, title=violin_title, font_size="11pt")
        else:
            violin_bokeh = boxplot(Yscore_combined, Ytrue_combined_name, xlabel="Class", ylabel="Predicted Score", violin=True, color=["#fcaeae", "#aed3f9", "#FFCCCC", "#CCE5FF"], width=320, height=315, group_name=["Train (0)", "Test (0)", "Train (1)", "Test (1)"], group_name_sort=["Test (0)", "Test (1)", "Train (0)", "Train (1)"], title=violin_title, font_size="11pt")
        violin_bokeh.multi_line([[-100, 100]], [[stats["val_cutoffscore"], stats["val_cutoffscore"]]], line_color="black", line_width=2, line_alpha=1.0, line_dash="dashed")

        # Distribution plot
        if testset is None:
            dist_bokeh = distribution(Yscore_train, group=Ytrue_train, kde=True, title="", xlabel="Predicted Score", ylabel="p.d.f.", width=320, height=315, smooth=dist_smooth)
        else:
            dist_bokeh = distribution(Yscore_combined, group=Ytrue_combined_name, kde=True, title="", xlabel="Predicted Score", ylabel="p.d.f.", width=320, height=315, smooth=dist_smooth)
        dist_bokeh.multi_line([[stats["val_cutoffscore"], stats["val_cutoffscore"]]], [[-100, 100]], line_color="black", line_width=2, line_alpha=1.0, line_dash="dashed")

        # Man-Whitney U for Table (round and use scienitic notation if p-value > 0.001)
        manw_pval = scipy.stats.mannwhitneyu(Yscore_train[Ytrue_train == 0], Yscore_train[Ytrue_train == 1], alternative="two-sided")[1]
        if manw_pval > 0.001:
            manw_pval_round = "%0.2f" % manw_pval
        else:
            manw_pval_round = "%0.2e" % manw_pval
        if testset is not None:
            testmanw_pval = scipy.stats.mannwhitneyu(Yscore_test[Ytrue_test == 0], Yscore_test[Ytrue_test == 1], alternative="two-sided")[1]
            if testmanw_pval > 0.001:
                testmanw_pval_round = "%0.2f" % testmanw_pval
            else:
                testmanw_pval_round = "%0.2e" % testmanw_pval

        # Create a stats table for test
        if testset is not None:
            teststats = binary_metrics(Ytrue_test, Yscore_test, cut_off=stats["val_cutoffscore"], parametric=self.parametric)
            teststats_round = {}
            for i in teststats.keys():
                teststats_round[i] = str(np.round(teststats[i], 2))

        # Round stats, and stats_bootci for Table
        stats_round = {}
        for i in stats.keys():
            stats_round[i] = np.round(stats[i], 2)
        bootci_round = {}
        for i in stats_bootci.keys():
            bootci_round[i] = np.round(stats_bootci[i], 2)

        # Create table
        tabledata = dict(
            evaluate=[["Train"]],
            manw_pval=[["{}".format(manw_pval_round)]],
            auc=[["{} ({}, {})".format(stats_round["AUC"], bootci_round["AUC"][0], bootci_round["AUC"][1])]],
            accuracy=[["{} ({}, {})".format(stats_round["ACCURACY"], bootci_round["ACCURACY"][0], bootci_round["ACCURACY"][1])]],
            precision=[["{} ({}, {})".format(stats_round["PRECISION"], bootci_round["PRECISION"][0], bootci_round["PRECISION"][1])]],
            sensitivity=[["{} ({}, {})".format(stats_round["SENSITIVITY"], bootci_round["SENSITIVITY"][0], bootci_round["SENSITIVITY"][1])]],
            specificity=[["{}".format(stats_round["SPECIFICITY"])]],
            F1score=[["{} ({}, {})".format(stats_round["F1-SCORE"], bootci_round["F1-SCORE"][0], bootci_round["F1-SCORE"][1])]],
            R2=[["{} ({}, {})".format(stats_round["R²"], bootci_round["R²"][0], bootci_round["R²"][1])]],
        )

        # Append test data
        if testset is not None:
            tabledata["evaluate"].append(["Test"])
            tabledata["manw_pval"].append([testmanw_pval_round])
            tabledata["auc"].append([teststats_round["AUC"]])
            tabledata["accuracy"].append([teststats_round["ACCURACY"]])
            tabledata["precision"].append([teststats_round["PRECISION"]])
            tabledata["sensitivity"].append([teststats_round["SENSITIVITY"]])
            tabledata["specificity"].append([teststats_round["SPECIFICITY"]])
            tabledata["F1score"].append([teststats_round["F1-SCORE"]])
            tabledata["R2"].append([teststats_round["R²"]])

        # Save Table
        self.table = tabledata
        self.table_eval = tabledata

        # Plot table
        source = ColumnDataSource(data=tabledata)
        columns = [TableColumn(field="evaluate", title="Evaluate"), TableColumn(field="manw_pval", title="MW-U Pvalue"), TableColumn(field="R2", title="R2"), TableColumn(field="accuracy", title="Accuracy"), TableColumn(field="precision", title="Precision"), TableColumn(field="F1score", title="F1score"), TableColumn(field="sensitivity", title="Sensitivity"), TableColumn(field="specificity", title="Specificity")]
        table_bokeh = widgetbox(DataTable(source=source, columns=columns, width=950, height=90), width=950, height=80)

        # Title
        if specificity is not False:
            title = "Specificity fixed to: {}".format(np.round(val, 2))
        elif cutoffscore is not False:
            title = "Score cut-off fixed to: {}".format(np.round(val, 2))
        else:
            title = "Specificity fixed to: {}".format(np.round(val, 2))
        title_bokeh = "<h3>{}</h3>".format(title)

        if title_align == "center":
            violin_bokeh.title.align = "center"
            dist_bokeh.title.align = "center"
            roc_bokeh.title.align = "center"

        # Combine table, violin plot and roc plot into one figure
        fig = layout([[violin_bokeh, dist_bokeh, roc_bokeh], [table_bokeh]])
        output_notebook()
        show(column(Div(text=title_bokeh, width=900, height=50), fig))

    def booteval(self, X, Y, errorbar=False, specificity=False, cutoffscore=False, bootnum=100, title_align="left", dist_smooth=None):
        """Plots a figure containing a Violin plot, Distribution plot, ROC plot and Binary Metrics statistics.

            Parameters
            ----------
            testset : array-like, shape = [n_samples, 2] or None, (default None)
                If testset is None, use train Y and train Y predicted for evaluate. Alternatively, testset is used to evaluate model in the format [Ytest, Ypred].

            specificity : number or False, (default False)
                Use the specificity to draw error bar. When False, use the cutoff score of 0.5.

            cutoffscore : number or False, (default False)
                Use the cutoff score to draw error bar. When False, use the specificity selected.

            bootnum : a positive integer, (default 1000)
                The number of bootstrap samples used in the computation.
            """
        model_boot = copy(self)
        #X, Y = self.input_check(X, Y)
        Ytrue_train = Y
        Yscore_train = model_boot.train(X, Y).flatten()

        # Expliclity states which metric and value is used for the error_bar
        if specificity is not False:
            metric = "specificity"
            val = specificity
        elif cutoffscore is not False:
            metric = "cutoffscore"
            val = cutoffscore
        else:
            metric = "cutoffscore"
            val = 0.5

        if errorbar is True:
            errorbar = True

        # ROC plot
        fpr_ib, tpr_ib_ci, stat_ib, median_ib, fpr_oob, tpr_oob_ci, stat_oob, median_oob, stats, median_y_ib, median_y_oob, manw_pval = roc_calculate_boot(self, X, Ytrue_train, Yscore_train, bootnum=bootnum, metric=metric, val=val, parametric=self.parametric)

        self.fpr_oob = np.insert(fpr_ib[0], 0, 0)
        self.tpr_oob = tpr_oob_ci[0]
        self.fpr_ib = np.insert(fpr_ib[0], 0, 0)
        self.tpr_ib = tpr_ib_ci[0]
        auc_ib_median = metrics.auc(self.fpr_ib, self.tpr_ib)
        auc_oob_median = metrics.auc(self.fpr_oob, self.tpr_oob)

        if errorbar is True:
            val_bar = stats["val_specificity"]
        else:
            val_bar = False

        roc_title = "AUC: {} ({})".format(np.round(auc_ib_median, 2), np.round(auc_oob_median, 2))
        roc_bokeh = roc_plot_boot(fpr_ib, tpr_ib_ci, fpr_oob, tpr_oob_ci, width=320, height=315, title=roc_title, errorbar=val_bar, label_font_size="10pt")

        # train is ib, test is oob
        Yscore_train_dict = {}
        for key, value in median_y_ib.items():
            if len(value) == 0:
                vals = np.nan
            else:
                vals = np.mean(value)
            Yscore_train_dict[key] = vals
        Yscore_test_dict = {}
        for key, value in median_y_oob.items():
            if len(value) == 0:
                vals = np.nan
            else:
                vals = np.mean(value)
            Yscore_test_dict[key] = vals
        Yscore_train = np.zeros(len(Y))
        for key, values in Yscore_train_dict.items():
            Yscore_train[key] = values
        Yscore_test = np.zeros(len(Y))
        for key, values in Yscore_test_dict.items():
            Yscore_test[key] = values
        Ytrue_train = Y
        Ytrue_test = Y
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

        # Violin plot
        if errorbar is False:
            violin_title = ""
        else:
            violin_title = "Cut-off: {}".format(np.round(stats["val_cutoffscore"], 2))
        violin_bokeh = boxplot(Yscore_combined, Ytrue_combined_name, xlabel="Class", ylabel="Median Predicted Score", violin=True, color=["#fcaeae", "#aed3f9", "#FFCCCC", "#CCE5FF"], width=320, height=315, group_name=["IB (0)", "OOB (0)", "IB (1)", "OOB (1)"], group_name_sort=["IB (0)", "IB (1)", "OOB (0)", "OOB (1)"], title=violin_title, font_size="11pt", label_font_size="10pt")
        if errorbar is True:
            violin_bokeh.multi_line([[-100, 100]], [[stats["val_cutoffscore"], stats["val_cutoffscore"]]], line_color="black", line_width=2, line_alpha=1.0, line_dash="dashed")

        # Distribution plot
        dist_bokeh = distribution(Yscore_combined, group=Ytrue_combined_name, kde=True, title="", xlabel="Median Predicted Score", ylabel="p.d.f.", width=320, height=315, padding=0.7, label_font_size="10pt", smooth=dist_smooth)
        if errorbar is True:
            dist_bokeh.multi_line([[stats["val_cutoffscore"], stats["val_cutoffscore"]]], [[-100, 100]], line_color="black", line_width=2, line_alpha=1.0, line_dash="dashed")

        # Manu
        manw_ib = []
        manw_oob = []
        for i in range(len(manw_pval)):
            manw_ib.append(manw_pval[i][0])
            manw_oob.append(manw_pval[i][1])
        manw_ib_med = np.percentile(manw_ib, 50, axis=0)
        manw_oob_med = np.percentile(manw_oob, 50, axis=0)
        if manw_ib_med > 0.01:
            manw_ib_pval_round = "%0.2f" % manw_ib_med
        else:
            manw_ib_pval_round = "%0.2e" % manw_ib_med
        if manw_oob_med > 0.01:
            manw_oob_pval_round = "%0.2f" % manw_oob_med
        else:
            manw_oob_pval_round = "%0.2e" % manw_oob_med

        # Round stats, and stats_bootci for Table
        stats_round_ib = {}
        for i in stat_ib.keys():
            stats_round_ib[i] = np.round(stat_ib[i], 2)

        # Round stats, and stats_bootci for Table
        stats_round_oob = {}
        for i in stat_oob.keys():
            stats_round_oob[i] = np.round(stat_oob[i], 2)

        # Create table
        if errorbar is 10:
            tabledata = dict(
                evaluate=[["In Bag"]],
                manw_pval=[["{}".format(manw_ib_pval_round)]],
                auc=[["{} ({}, {})".format(np.round(auc_ib_median, 2), stats_round_ib["AUC"][1], stats_round_ib["AUC"][2])]],
                R2=[["{} ({}, {})".format(stats_round_ib["R²"][0], stats_round_ib["R²"][1], stats_round_ib["R²"][2])]],
            )

            # Append test data
            tabledata["evaluate"].append(["Out of Bag"])
            tabledata["manw_pval"].append([manw_oob_pval_round])
            tabledata["auc"].append(["{} ({}, {})".format(np.round(auc_oob_median, 2), stats_round_oob["AUC"][1], stats_round_oob["AUC"][2])])
            tabledata["R2"].append(["{} ({}, {})".format(stats_round_oob["R²"][0], stats_round_oob["R²"][1], stats_round_oob["R²"][2])])
        else:
            tabledata = dict(
                evaluate=[["In Bag"]],
                auc=[["{} ({}, {})".format(np.round(auc_ib_median, 2), stats_round_ib["AUC"][1], stats_round_ib["AUC"][2])]],
                manw_pval=[["{}".format(manw_ib_pval_round)]],
                accuracy=[["{} ({}, {})".format(stats_round_ib["ACCURACY"][0], stats_round_ib["ACCURACY"][1], stats_round_ib["ACCURACY"][2])]],
                precision=[["{} ({}, {})".format(stats_round_ib["PRECISION"][0], stats_round_ib["PRECISION"][1], stats_round_ib["PRECISION"][2])]],
                sensitivity=[["{} ({}, {})".format(stats_round_ib["SENSITIVITY"][0], stats_round_ib["SENSITIVITY"][1], stats_round_ib["SENSITIVITY"][2])]],
                specificity=[["{}".format(stats_round_ib["SPECIFICITY"][0])]],
                F1score=[["{} ({}, {})".format(stats_round_ib["F1-SCORE"][0], stats_round_ib["F1-SCORE"][1], stats_round_ib["F1-SCORE"][2])]],
                R2=[["{} ({}, {})".format(stats_round_ib["R²"][0], stats_round_ib["R²"][1], stats_round_ib["R²"][2])]],
            )

            # Append test data
            tabledata["evaluate"].append(["Out of Bag"])
            tabledata["manw_pval"].append([manw_oob_pval_round])
            tabledata["auc"].append(["{} ({}, {})".format(np.round(auc_oob_median, 2), stats_round_oob["AUC"][1], stats_round_oob["AUC"][2])])
            tabledata["accuracy"].append(["{} ({}, {})".format(stats_round_oob["ACCURACY"][0], stats_round_oob["ACCURACY"][1], stats_round_oob["ACCURACY"][2])])
            tabledata["precision"].append(["{} ({}, {})".format(stats_round_oob["PRECISION"][0], stats_round_oob["PRECISION"][1], stats_round_oob["PRECISION"][2])])
            tabledata["sensitivity"].append(["{} ({}, {})".format(stats_round_oob["SENSITIVITY"][0], stats_round_oob["SENSITIVITY"][1], stats_round_oob["SENSITIVITY"][2])])
            tabledata["specificity"].append(["{}".format(stats_round_oob["SPECIFICITY"][0])])
            tabledata["F1score"].append(["{} ({}, {})".format(stats_round_oob["F1-SCORE"][0], stats_round_oob["F1-SCORE"][1], stats_round_oob["F1-SCORE"][2])])
            tabledata["R2"].append(["{} ({}, {})".format(stats_round_oob["R²"][0], stats_round_oob["R²"][1], stats_round_oob["R²"][2])])

        # Save Table
        self.table_booteval = tabledata

        # Plot table
        source = ColumnDataSource(data=tabledata)
        if errorbar is True:
            columns = [TableColumn(field="evaluate", title="Evaluate"), TableColumn(field="manw_pval", title="ManW P-Value"), TableColumn(field="R2", title="R2"), TableColumn(field="auc", title="AUC"), TableColumn(field="accuracy", title="Accuracy"), TableColumn(field="precision", title="Precision"), TableColumn(field="F1score", title="F1score"), TableColumn(field="sensitivity", title="Sensitivity"), TableColumn(field="specificity", title="Specificity")]
            table_bokeh = widgetbox(DataTable(source=source, columns=columns, width=950, height=90), width=950, height=80)
        else:
            columns = [TableColumn(field="evaluate", title="Evaluate"), TableColumn(field="manw_pval", title="ManW P-Value"), TableColumn(field="R2", title="R2"), TableColumn(field="auc", title="AUC")]
            table_bokeh = widgetbox(DataTable(source=source, columns=columns, width=950, height=90), width=950, height=80)

        # Title
        if errorbar is False:
            title_bokeh = ""
        elif specificity is not False:
            title = "Specificity fixed to: {}".format(np.round(val, 2))
            title_bokeh = "<h3>{}</h3>".format(title)
        elif cutoffscore is not False:
            title = "Score cut-off fixed to: {}".format(np.round(val, 2))
            title_bokeh = "<h3>{}</h3>".format(title)
        else:
            title = "Specificity fixed to: {}".format(np.round(val, 2))
            title_bokeh = "<h3>{}</h3>".format(title)

        if title_align == "center":
            violin_bokeh.title.align = "center"
            dist_bokeh.title.align = "center"
            roc_bokeh.title.align = "center"

        # Combine table, violin plot and roc plot into one figure
        fig = layout([[violin_bokeh, dist_bokeh, roc_bokeh], [table_bokeh]])
        output_notebook()
        show(column(Div(text=title_bokeh, width=900, height=50), fig))

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
            # print("Use method calc_bootci prior to plot_projections.")
            bootx = None
        else:
            bootx = 1

        x_scores_true = deepcopy(self.model.x_scores_)
        if weight_alt is True:
            self.model.x_scores_ = self.model.x_scores_alt
            sigmoid = True
        else:
            sigmoid = False

        num_x_scores = len(self.model.x_scores_.T)

        # If there is only 1 x_score, Need to plot x_score vs. peak (as opposided to x_score[i] vs. x_score[j])
        if num_x_scores == 1:
            # Violin plot
            violin_bokeh = boxplot(self.Y_pred.flatten(), self.Y, title="", xlabel="Class", ylabel="Predicted Score", violin=True, color=["#FFCCCC", "#CCE5FF"], width=320, height=315)
            # Distribution plot
            dist_bokeh = distribution(self.Y_pred, group=self.Y, kde=True, title="", xlabel="Predicted Score", ylabel="p.d.f.", width=320, height=315, sigmoid=sigmoid)
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

                    if weight_alt is True:
                        new_range = (0.1, 1.1)

                    grid[y, x] = scatter(self.model.x_scores_[:, x].tolist(), self.model.x_scores_[:, y].tolist(), label=label_copy, group=group_copy, title="", xlabel=xlabel, ylabel=ylabel, width=width_height, height=width_height, legend=False, size=circle_size_scoreplot, label_font_size=label_font, hover_xy=False, xrange=new_range, yrange=new_range, gradient=gradient, gradient_alt=weight_alt, ci95=ci95, scatterplot=scatterplot)

                # Append each distribution curve
                for i in range(num_x_scores):
                    xlabel = "LV {} ({:0.1f}%)".format(i + 1, self.model.pctvar_[i])
                    grid[i, i] = distribution(self.model.x_scores_[:, i], group=group_copy, kde=True, title="", xlabel=xlabel, ylabel="density", width=width_height, height=width_height, label_font_size=label_font, sigmoid=sigmoid)

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
                    try:
                        scoresi = x_scores_boot[i]
                    except KeyError:
                        raise ValueError("Kevin needs to fix this. For now, increase the number of bootstraps.")
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

                    if weight_alt is True:
                        new_range = (-0.3, 1.3)

                    grid[y, x] = scatter(self.model.x_scores_[:, x].tolist(), self.model.x_scores_[:, y].tolist(), label=label_copy, group=group_copy, title="", xlabel=xlabel, ylabel=ylabel, width=width_height, height=width_height, legend=False, size=circle_size_scoreplot, label_font_size=label_font, hover_xy=False, xrange=new_range, yrange=new_range, gradient=gradient, gradient_alt=weight_alt, ci95=ci95, scatterplot=scatterplot, extraci95_x=boot_xscores[:, x].tolist(), extraci95_y=boot_xscores[:, y].tolist(), extraci95=True)

                # Append each distribution curve
                group_dist = np.concatenate((self.Y, (self.Y + 2)))

                for i in range(num_x_scores):
                    score_dist = np.concatenate((self.model.x_scores_[:, i], boot_xscores[:, i]))
                    xlabel = "LV {} ({:0.1f}%)".format(i + 1, self.model.pctvar_[i])
                    grid[i, i] = distribution(score_dist, group=group_dist, kde=True, title="", xlabel=xlabel, ylabel="density", width=width_height, height=width_height, label_font_size=label_font, sigmoid=sigmoid)

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

                    grid[x, y] = roc_plot(fpr, tpr, tpr_ci, width=width_height, height=width_height, xlabel="1-Specificity (LV{}/LV{})".format(x + 1, y + 1), ylabel="Sensitivity (LV{}/LV{})".format(x + 1, y + 1), legend=True, label_font_size=label_font, roc2=True, fpr2=fpr_boot, tpr2=tpr_boot, tpr_ci2=tpr_ci_boot)
            # Bokeh grid
            fig = gridplot(grid.tolist())

        self.model.x_scores_ = x_scores_true
        output_notebook()
        show(fig)

    def plot_projections_kfold(self, label=None, size=12, ci95=True, scatterplot=True, weight_alt=False, folds=10, n_mc=10):
        """ Plots latent variables projections against each other in a Grid format.

        Parameters
        ----------
        label : DataFrame or None, (default None)
            hovertool for scatterplot.

        size : positive integer, (default 12)
            size specifies circle size for scatterplot.
        """

        # Get copy of self
        try:
            kmodel = deepcopy(self)  # Make a copy of the model
        except TypeError:
            kmodel = copy(self)

        x_scores_true = deepcopy(self.model.x_scores_)
        if weight_alt is True:
            self.model.x_scores_ = self.model.x_scores_alt
            sigmoid = True
        else:
            sigmoid = False

        x_scores_mc = []
        crossval_idx = StratifiedKFold(n_splits=folds, shuffle=True)
        for i in tqdm(range(n_mc)):
            x_scores_cv = [None] * len(self.Y)
            for train, test in crossval_idx.split(self.X, self.Y):
                X_train = self.X[train, :]
                Y_train = self.Y[train]
                X_test = self.X[test, :]
                kmodel.train(X_train, Y_train)
                kmodel.test(X_test)
                if weight_alt is False:
                    x_scores_cv_i = kmodel.model.x_scores_
                else:
                    x_scores_cv_i = kmodel.model.x_scores_alt
                # Return value to y_pred_cv in the correct position # Better way to do this
                for (idx, val) in zip(test, x_scores_cv_i):
                    x_scores_cv[idx] = val.tolist()
            x_scores_mc.append(x_scores_cv)

        x_scores_cv = np.median(np.array(x_scores_mc), axis=0)

        x_scores_true = deepcopy(self.model.x_scores_)
        if weight_alt is True:
            self.model.x_scores_ = self.model.x_scores_alt

        num_x_scores = len(self.model.x_scores_.T)

        # If there is only 1 x_score, Need to plot x_score vs. peak (as opposided to x_score[i] vs. x_score[j])
        if num_x_scores == 1:
            # Violin plot
            violin_bokeh = boxplot(self.Y_pred.flatten(), self.Y, title="", xlabel="Class", ylabel="Predicted Score", violin=True, color=["#FFCCCC", "#CCE5FF"], width=320, height=315)
            # Distribution plot
            dist_bokeh = distribution(self.Y_pred, group=self.Y, kde=True, title="", xlabel="Predicted Score", ylabel="p.d.f.", width=320, height=315, sigmoid=sigmoid)
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
            fig = layout([[violin_bokeh, dist_bokeh, roc_bokeh], [scatter_bokeh]])

        else:

            if weight_alt is False:
                a = 'model.x_scores_'
            else:
                a = 'model.x_scores_alt'

            boot_xscores = x_scores_cv

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
                grid[i, i] = distribution(score_dist, group=group_dist, kde=True, title="", xlabel=xlabel, ylabel="density", width=width_height, height=width_height, label_font_size=label_font, sigmoid=sigmoid)

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

        self.model.x_scores_ = x_scores_true
        output_notebook()
        show(fig)

    def save_table(self, name="table.xlsx"):
        try:
            table_a = pd.DataFrame(self.table)
        except AttributeError:
            table_a = pd.DataFrame()
        try:
            table_b = pd.DataFrame(self.table_booteval)
        except AttributeError:
            table_b = pd.DataFrame()

        table = pd.concat([table_a, table_b], ignore_index=True, sort=False)
        check_type = name.split(".")
        if check_type[-1] == "xlsx":
            table.to_excel(name, index=False)
        elif check_type[-1] == "csv":
            table.to_csv(name, index=False)
        else:
            raise ValueError("name must end in .xlsx or .csv")
        print("Done! Saved table as {}".format(name))

    def pfi(self, nperm=100, metric="r2q2", mean=True):
        X = deepcopy(self.X)
        Y = deepcopy(self.Y)
        yb = self.test(X)
        mb_b = binary_metrics(Y, yb)
        mb_s = []
        for i in range(len(X.T)):
            mb_store = []
            for j in range(nperm):
                X_shuff = deepcopy(X)
                np.random.shuffle(X_shuff[i, :])
                ys = self.test(X_shuff)
                mb_s_i = binary_metrics(Y, ys)
                mb_store.append(mb_s_i)
            if mean is True:
                mb_mean = dict_mean(mb_store)
            else:
                mb_mean = dict_median(mb_store)
            mb_s.append(mb_mean)

        # Choose metric to plot
        metric_title = np.array(["ACCURACY", "AIC", "AUC", "BIC", "F1-SCORE", "PRECISION", "R²", "SENSITIVITY", "SPECIFICITY", "SSE"])
        metric_list = np.array(["acc", "aic", "auc", "bic", "f1score", "prec", "r2q2", "sens", "spec", "sse"])
        metric_idx = np.where(metric_list == metric)[0][0]

        pfi = []
        for i in mb_s:
            val_s = i[metric_title[metric_idx]]
            val_b = mb_b[metric_title[metric_idx]]
            val = val_b - val_s
            pfi.append(val)

        pfi = np.array(pfi)

        # return pfi

        # Testing out r2q2, acc, and auc
        pfi_acc = []
        for i in mb_s:
            val_s = i["ACCURACY"]
            val_b = mb_b["ACCURACY"]
            val = val_b - val_s
            pfi_acc.append(val)
        pfi_acc = np.array(pfi_acc)

        pfi_r2q2 = []
        for i in mb_s:
            val_s = i["R²"]
            val_b = mb_b["R²"]
            val = val_b - val_s
            pfi_r2q2.append(val)
        pfi_r2q2 = np.array(pfi_r2q2)

        pfi_auc = []
        for i in mb_s:
            val_s = i["AUC"]
            val_b = mb_b["AUC"]
            val = val_b - val_s
            pfi_auc.append(val)
        pfi_auc = np.array(pfi_auc)

        return pfi_acc, pfi_r2q2, pfi_auc
