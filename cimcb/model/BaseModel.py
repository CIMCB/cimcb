import numpy as np
import pandas as pd
import scipy
from abc import ABC, abstractmethod, abstractproperty
from bokeh.layouts import column, layout, widgetbox
from bokeh.models.widgets import DataTable, Div, TableColumn
from bokeh.plotting import ColumnDataSource, output_notebook, show
from copy import copy
from sklearn import metrics
from ..plot import boxplot, distribution, permutation_test, roc_calculate, roc_plot, roc_calculate_boot, roc_plot_boot
from ..utils import binary_metrics


class BaseModel(ABC):
    """Base class for models."""

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

    def permutation_test(self, nperm=100):
        """Plots permutation test figures.

        Parameters
        ----------
        nperm : positive integer, (default 100)
            Number of permutations.
        """
        fig = permutation_test(self, self.X, self.Y, nperm=nperm)
        output_notebook()
        show(fig)

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
                raise ValueError("Ytrue_test needs to have 2 groups. There is {}".format(len(np.unique(self.Y))))
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

    def booteval(self, X, Y, bootnum=100, errorbar=False, specificity=False, cutoffscore=False, title_align="left", dist_smooth=None):
        """Estimatation of the robustness and a measure of generalised predictive ability of this model using perform bootstrap aggregation.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Predictor variables, where n_samples is the number of samples and n_features is the number of predictors.

        Y : array-like, shape = [n_samples, 1]
            Response variables, where n_samples is the number of samples.

        bootnum : a positive integer, (default 100)
            The number of bootstrap samples used in the computation.
        """

        model_boot = copy(self)
        # X, Y = self.input_check(X, Y)
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
