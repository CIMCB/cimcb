import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import multiprocessing
import timeit
import time
from copy import deepcopy, copy
from itertools import combinations
from pydoc import locate
from joblib import Parallel, delayed
from bokeh.layouts import column
import importlib
from bokeh.models.widgets import DataTable, Div, TableColumn
from bokeh.layouts import column, layout, widgetbox
from sklearn.linear_model import LinearRegression
from itertools import combinations
from ..plot import scatterCI, boxplot, distribution, scatter, scatter_ellipse
from ..utils import nested_getattr
import numpy as np
import scipy
import pandas as pd
from scipy.stats import norm
import math
import multiprocessing
from copy import deepcopy
from bokeh.layouts import widgetbox, gridplot, column, row, layout
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.models import Div
from itertools import combinations
from sklearn.utils import resample
from ..plot import scatterCI, boxplot, distribution, roc_boot
from ..utils import color_scale, dict_perc, nested_getattr, dict_95ci, dict_median_scores, binary_metrics, binary_evaluation


class BaseBootstrap(ABC):
    """Base class for bootstrap: BC, BCA, and Perc."""

    @abstractmethod
    def __init__(self, model, bootnum=100, seed=None, n_cores=-1, stratify=True):
        self.X = model.X
        self.Y = model.Y
        self.name = model.__name__
        if self.name == 'cimcb.model.NN_SigmoidSigmoid' or self.name == 'cimcb.model.NN_LinearSigmoid':
            self.w1 = model.model.w1
            self.w2 = model.model.w2
        self.bootlist = model.bootlist
        self.bootnum = bootnum
        self.seed = seed
        self.bootidx = []
        self.bootstat = {}
        self.bootci = {}
        self.stratify = stratify
        self.param = model.__params__
        self.model = locate(model.__name__)
        self.test = None
        # if n_cores = -1, set n_cores to max_cores
        max_num_cores = multiprocessing.cpu_count()
        self.n_cores = n_cores
        if self.n_cores > max_num_cores:
            self.n_cores = -1
            print("Number of cores set too high. It will be set to the max number of cores in the system.", flush=True)
        if self.n_cores == -1:
            self.n_cores = max_num_cores
            print("Number of cores set to: {}".format(max_num_cores))
        self.model_orig = model

    def calc_stat(self):
        """Stores selected attributes (from self.bootlist) for the original model."""
        self.stat = {}
        #self.model_orig.train(self.model_orig.X, self.model_orig.Y)
        self.model_orig.test(self.model_orig.X, self.model_orig.Y)
        for i in self.bootlist:
            self.stat[i] = nested_getattr(self.model_orig, i)

    def calc_bootidx(self):
        """Generate indices for every resampled (with replacement) dataset."""
        np.random.seed(self.seed)
        self.bootidx = []
        self.bootidx_oob = []
        for i in range(self.bootnum):
            #bootidx_i = np.random.choice(len(self.Y), len(self.Y))
            if self.stratify == True:
                bootidx_i = resample(list(range(len(self.Y))), stratify=self.Y)
            else:
                bootidx_i = resample(list(range(len(self.Y))))
            bootidx_oob_i = np.array(list(set(range(len(self.Y))) - set(bootidx_i)))
            self.bootidx.append(bootidx_i)
            self.bootidx_oob.append(bootidx_oob_i)

    def calc_bootstat(self):
        """Trains and test model, then stores selected attributes (from self.bootlist) for each resampled dataset."""

        "Running ..."
        time.sleep(0.5)  # Sleep for 0.5 secs to finish printing

        # Start Timer
        start = timeit.default_timer()

        # Create an empty dictionary
        self.bootstat = {}
        for i in self.bootlist:
            self.bootstat[i] = []
        self.bootstat_oob = {}
        for i in self.bootlist:
            self.bootstat_oob[i] = []
        # Calculate bootstat for each bootstrap resample
        try:
            stats_loop = Parallel(n_jobs=self.n_cores)(delayed(self._calc_bootstat_loop)(i) for i in tqdm(range(self.bootnum)))
        except:
            print("TerminatedWorkerError was raised due to excessive memory usage. n_cores was reduced to 1.")
            stats_loop = Parallel(n_jobs=1)(delayed(self._calc_bootstat_loop)(i) for i in tqdm(range(self.bootnum)))
        self.stats_loop = stats_loop

        self.bootstat = {}
        for i in self.bootlist:
            self.bootstat[i] = []
        self.bootstat_oob = {}
        for i in self.bootlist:
            self.bootstat_oob[i] = []

        for i in self.stats_loop:
            ib = i[0]
            for key, value in ib.items():
                self.bootstat[key].append(value[0])

            oob = i[1]
            for key, value in oob.items():
                self.bootstat_oob[key].append(value[0])

        # Check if loadings flip
        try:
            orig = self.stat['model.x_loadings_']
            for i in range(len(self.bootstat['model.x_loadings_'])):
                check = self.bootstat['model.x_loadings_'][i]
                for j in range(len(orig.T)):
                    corr = np.corrcoef(orig[:, j], check[:, j])[1, 0]
                    if corr < 0:
                        for key, value in self.bootstat.items():
                            if key == 'model.x_loadings_':
                                value[i][:, j] = - value[i][:, j]
                                self.bootstat[key] = value
                        for key, value in self.bootstat.items():
                            if key == 'model.x_scores_':
                                value[i][:, j] = - value[i][:, j]
                                self.bootstat[key] = value
        except:
            pass

        # Stop timer
        stop = timeit.default_timer()
        self.parallel_time = (stop - start) / 60
        print("Time taken: {:0.2f} minutes with {} cores".format(self.parallel_time, self.n_cores))

    def _calc_bootstat_loop(self, i):
        """Core component of calc_ypred."""
        # Set model
        model_i = self.model(**self.param)
        # Set X and Y
        try:
            X_res = self.X[self.bootidx[i], :]
            Y_res = self.Y[self.bootidx[i]]
        except TypeError:
            X_res = []
            Y_res = self.Y[self.bootidx[i]]
            for j in self.X:
                X_res.append(j[self.bootidx[i], :])

        # Train and test
        if self.name == 'cimcb.model.NN_SigmoidSigmoid' or self.name == 'cimcb.model.NN_LinearSigmoid':
            model_i.train(X_res, Y_res, w1=self.w1, w2=self.w2)
        else:
            model_i.train(X_res, Y_res)
        model_i.test(X_res, Y_res)
        # # Get IB
        bootstatloop = {}
        for k in self.bootlist:
            bootstatloop[k] = []
        for j in self.bootlist:
            bootstatloop[j].append(nested_getattr(model_i, j))
        # # Get OOB
        bootstatloop_oob = {}
        for k in self.bootlist:
            bootstatloop_oob[k] = []
        try:
            X_res_oob = self.X[self.bootidx_oob[i], :]
            Y_res_oob = self.Y[self.bootidx_oob[i]]
        except TypeError:
            X_res_oob = []
            Y_res_oob = self.Y[self.bootidx_oob[i]]
            for j in self.X:
                X_res_oob.append(j[self.bootidx_oob[i], :])
        model_i.test(X_res_oob, Y_res_oob)
        for j in self.bootlist:
            bootstatloop_oob[j].append(nested_getattr(model_i, j))

        return [bootstatloop, bootstatloop_oob]

    def evaluate(self, parametric=True, errorbar=False, specificity=False, cutoffscore=False, title_align="left", dist_smooth=None, bc='nonparametric', label=None, legend='all', grid_line=False, smooth=0, plot_roc='data', testset=None, show_table=True, trainset=None):

        test = testset
        self.train = trainset
        if test is not None:
            bm = binary_evaluation(test[0], test[1])
            self.test = []
            for key, value in bm.items():
                self.test.append(value)

        if plot_roc == 'ci':
            ci_only = True
        else:
            ci_only = False
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

        # print(x_loc_oob_dict)
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
        violin_bokeh = boxplot(Yscore_combined, Ytrue_combined_name, xlabel="Class", ylabel="Median Predicted Score", violin=True, color=["#FFCCCC", "#CCE5FF", "#FFCCCC", "#CCE5FF"], width=320, height=315, group_name=group_name, group_name_sort=group_name_sort, title=violin_title, legend=legend_violin, label=label_violin, font_size="10pt", label_font_size="10pt", grid_line=grid_line, legend_title=True)

        # Distribution plot
        dist_bokeh = distribution(Yscore_combined, group=Ytrue_combined_name, kde=True, title="", xlabel="Median Predicted Score", ylabel="p.d.f.", width=320, height=315, padding=0.7, label_font_size="10pt", smooth=dist_smooth, group_label=label, legend_location="top_left", legend=legend_dist, grid_line=grid_line, legend_title=True, font_size="10pt")

        if self.__name__ == 'BCA':
            jackstat = self.jackstat['Y_pred']
            jackidx = self.jackidx
        else:
            jackstat = None
            jackidx = None

        roc_bokeh, auc_ib, auc_oob = roc_boot(self.Y, self.stat['Y_pred'], self.bootstat['Y_pred'], self.bootstat_oob['Y_pred'], self.bootidx, self.bootidx_oob, self.__name__, smoothval=smooth, jackstat=jackstat, jackidx=jackidx, xlabel="1-Specificity", ylabel="Sensitivity", width=320, height=315, label_font_size="10pt", legend=legend_roc, grid_line=grid_line, plot_num=0, plot=plot_roc, test=test, legend_basic=show_table, train=trainset, ci_only=ci_only)

        stats_table = pd.DataFrame(self.bootci['model.eval_metrics_'],
                                   columns=['IBLowCI', 'IBUppCI', 'IBMidCI'],
                                   index=self.model_orig.metrics_key)
        stats_table['OOBLowCI'] = np.percentile(np.array(self.bootstat_oob['model.eval_metrics_']), 2.5, axis=0)
        stats_table['OOBUppCI'] = np.percentile(np.array(self.bootstat_oob['model.eval_metrics_']), 97.5, axis=0)
        stats_table['OOBMidCI'] = np.percentile(np.array(self.bootstat_oob['model.eval_metrics_']), 50, axis=0)

        stats_table['Train'] = self.stat['model.eval_metrics_']

        if self.train is not None:
            st = binary_evaluation(self.train[0], self.train[1])
            st2 = []
            for key, value in st.items():
                st2.append(value)
            stats_table['Train'] = st2

        if self.test is not None:
            stats_table['Test'] = self.test
        else:
            stats_table['Test'] = self.stat['model.eval_metrics_']

        self.table = stats_table
        self.auc_table = [auc_ib, auc_oob]
        # Add from ROC
        self.table['Train']['AUC'] = auc_ib[2]
        self.table['IBLowCI']['AUC'] = auc_ib[0]
        self.table['IBUppCI']['AUC'] = auc_ib[1]
        self.table['Test']['AUC'] = auc_oob[2]
        self.table['OOBLowCI']['AUC'] = auc_oob[0]
        self.table['OOBUppCI']['AUC'] = auc_oob[1]
        for i in self.table:
            self.table[i][0] = np.round(self.table[i][0], 2)
            self.table[i][1] = np.round(self.table[i][1], 2)
            if self.table[i][2] > 0.01:
                self.table[i][2] = "%0.2f" % self.table[i][2]
            else:
                self.table[i][2] = "%0.2e" % self.table[i][2]

        if show_table == False:
            fig = gridplot([[violin_bokeh, dist_bokeh, roc_bokeh]])
        else:
            if self.test is not None:
                table = self.table
                tabledata = dict(
                    evaluate=[["Train (IB 95% CI)"]],
                    manw_pval=[["{}".format(table['Train'][2])]],
                    auc=[["{} ({}, {})".format(table['Train'][1], table['IBLowCI'][1], table['IBUppCI'][1])]],
                    R2=[["{} ({}, {})".format(table['Train'][0], table['IBLowCI'][0], table['IBUppCI'][0])]],
                )
                # Append test data
                tabledata["evaluate"].append(["Test (OOB 95% CI)"])
                tabledata["manw_pval"].append([table['Test'][2]])
                tabledata["auc"].append(["{} ({}, {})".format(table['Test'][1], table['OOBLowCI'][1], table['OOBUppCI'][1])]),
                tabledata["R2"].append(["{} ({}, {})".format(table['Test'][0], table['OOBLowCI'][0], table['OOBUppCI'][0])]),
                columns = [TableColumn(field="evaluate", title="Evaluate"), TableColumn(field="manw_pval", title="ManW P-Value"), TableColumn(field="R2", title="R²"), TableColumn(field="auc", title="AUC")]

                source = ColumnDataSource(data=tabledata)

                table_bokeh = widgetbox(DataTable(source=source, columns=columns, width=950, height=90), width=950, height=95)
            else:
                table = self.table
                if plot_roc == 'data':
                    tabledata = dict(
                        evaluate=[["Data (IB 95% CI)"]],
                        manw_pval=[["{}".format(table['Train'][2])]],
                        auc=[["{} ({}, {})".format(table['Train'][1], table['IBLowCI'][1], table['IBUppCI'][1])]],
                        R2=[["{} ({}, {})".format(table['Train'][0], table['IBLowCI'][0], table['IBUppCI'][0])]],
                    )
                else:
                    tabledata = dict(
                        evaluate=[["IB (95% CI)"]],
                        manw_pval=[["{}".format(table['IBMidCI'][2])]],
                        auc=[["{} ({}, {})".format(table['IBMidCI'][1], table['IBLowCI'][1], table['IBUppCI'][1])]],
                        R2=[["{} ({}, {})".format(table['IBMidCI'][0], table['IBLowCI'][0], table['IBUppCI'][0])]],
                    )

                # Append test data
                tabledata["evaluate"].append(["OOB (95% CI)"])
                tabledata["manw_pval"].append([table['OOBMidCI'][2]])
                tabledata["auc"].append(["{} ({}, {})".format(table['OOBMidCI'][1], table['OOBLowCI'][1], table['OOBUppCI'][1])]),
                tabledata["R2"].append(["{} ({}, {})".format(table['OOBMidCI'][0], table['OOBLowCI'][0], table['OOBUppCI'][0])]),

                if self.test is not None:
                    tabledata["evaluate"].append(["Test"])
                    tabledata["manw_pval"].append([table['Test'][2]])
                    tabledata["auc"].append(["{}".format(table['Test'][1])]),
                    tabledata["R2"].append(["{}".format(table['Test'][0])]),
                columns = [TableColumn(field="evaluate", title="Evaluate"), TableColumn(field="manw_pval", title="ManW P-Value"), TableColumn(field="R2", title="R2"), TableColumn(field="auc", title="AUC")]

                source = ColumnDataSource(data=tabledata)

                if self.test is not None:
                    table_bokeh = widgetbox(DataTable(source=source, columns=columns, width=950, height=140), width=950, height=140)
                else:
                    table_bokeh = widgetbox(DataTable(source=source, columns=columns, width=950, height=90), width=950, height=90)
            self.table = pd.DataFrame(tabledata)
            fig1 = gridplot([[violin_bokeh, dist_bokeh, roc_bokeh]])
            fig = layout(fig1, [table_bokeh])

        output_notebook()
        show(fig)

    def plot_loadings(self, PeakTable, peaklist=None, data=None, ylabel="Label", sort=False, sort_ci=True, grid_line=False, plot='data', x_axis_below=True):
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

        if isinstance(plot, str):
            if plot == 'data':
                mid = self.stat['model.x_loadings_']
            elif plot == 'median':
                mid = []
                for i in self.bootci['model.x_loadings_']:
                    mid.append(i[:, 2])
                mid = np.array(mid).T
            else:
                pass
        else:
            mid = plot

        # Remove rows from PeakTable if not in peaklist
        if peaklist is not None:
            PeakTable = PeakTable[PeakTable["Name"].isin(peaklist)]
        peaklabel = PeakTable[ylabel]

        a = [None] * 2

        if self.name == 'cimcb.model.NN_SigmoidSigmoid' or self.name == 'cimcb.model.NN_LinearSigmoid':
            lv_name = "Neuron"
        else:
            lv_name = "LV"

        # Plot
        plots = []
        for i in range(n_loadings):
            cii = ci_loadings[i]
            midi = mid[:, i]
            fig = scatterCI(midi,
                            ci=cii,
                            label=peaklabel,
                            hoverlabel=PeakTable[["Idx", "Name", "Label"]],
                            hline=0,
                            col_hline=True,
                            title="Loadings Plot: {} {}".format(lv_name, i + 1),
                            sort_abs=sort,
                            sort_ci=sort_ci,
                            grid_line=grid_line,
                            x_axis_below=x_axis_below)
            plots.append([fig])

        fig = layout(plots)
        output_notebook()
        show(fig)

    def plot_weights(self, PeakTable, peaklist=None, data=None, ylabel="Label", sort=False, sort_ci=True, grid_line=False, plot='data', x_axis_below=True):
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

        if isinstance(plot, str):
            if plot == 'data':
                mid = self.stat['model.x_loadings_']
            elif plot == 'median':
                mid = []
                for i in self.bootci['model.x_loadings_']:
                    mid.append(i[:, 2])
                mid = np.array(mid).T
            else:
                pass
        else:
            mid = plot

        # Remove rows from PeakTable if not in peaklist
        if peaklist is not None:
            PeakTable = PeakTable[PeakTable["Name"].isin(peaklist)]
        peaklabel = PeakTable[ylabel]

        a = [None] * 2

        if self.name == 'cimcb.model.NN_SigmoidSigmoid' or self.name == 'cimcb.model.NN_LinearSigmoid':
            lv_name = "Neuron"
        else:
            lv_name = "LV"

        # Current issue with Bokeh & Chrome, can't plot >1500 features.
        n = 1500
        if len(peaklist) > n:
            peaklist_chunk = [peaklist[i * n:(i + 1) * n] for i in range((len(peaklist) + n - 1) // n)]
            peaklabel_chunk = [peaklabel[i * n:(i + 1) * n] for i in range((len(peaklabel) + n - 1) // n)]
            grid = np.full((len(peaklist_chunk), n_loadings), None)
            for i in range(n_loadings):
                cii = ci_loadings[i]
                midi = mid[:, i]
                cii_chunk = [cii[i * n:(i + 1) * n] for i in range((len(cii) + n - 1) // n)]
                midi_chunk = [midi[i * n:(i + 1) * n] for i in range((len(midi) + n - 1) // n)]
                for j in range(len(peaklist_chunk)):
                    PeakTable_chunk = PeakTable[PeakTable["Name"].isin(peaklist_chunk[j])]
                    grid[i,j] = scatterCI(midi_chunk[j],
                                    ci=cii_chunk[j],
                                    label=peaklabel_chunk[j],
                                    hoverlabel=PeakTable_chunk[["Idx", "Name", "Label"]],
                                    hline=0,
                                    col_hline=True,
                                    title="Weights Plot: {} {}".format(lv_name, i + 1),
                                    sort_abs=sort,
                                    sort_ci=sort_ci,
                                    grid_line=grid_line,
                                    x_axis_below=x_axis_below)
            fig = gridplot(grid.tolist())

        else:
            # Plot
            plots = []
            for i in range(n_loadings):
                cii = ci_loadings[i]
                midi = mid[:, i]
                fig = scatterCI(midi,
                                ci=cii,
                                label=peaklabel,
                                hoverlabel=PeakTable[["Idx", "Name", "Label"]],
                                hline=0,
                                col_hline=True,
                                title="Weights Plot: {} {}".format(lv_name, i + 1),
                                sort_abs=sort,
                                sort_ci=sort_ci,
                                grid_line=grid_line,
                                x_axis_below=x_axis_below)
                plots.append([fig])

            fig = layout(plots)
        output_notebook()
        show(fig)

    def plot_projections(self, label=None, size=12, ci95=True, scatterplot=False, weight_alt=False, bc="nonparametric", legend='all', plot='ci', scatter_ib=True, orthog_line=True, grid_line=False, smooth=0, plot_roc='median'):
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

        if plot in ["ci", "CI"]:
            plot_num = 0
        elif plot in ["innerci", "MeanCI", "meanci"]:
            plot_num = 1
        elif plot in ["ib", "IB"]:
            plot_num = 2
        elif plot in ["oob", "OOB"]:
            plot_num = 3
        elif plot in ["all", "ALL", "All"]:
            plot_num = 4
        else:
            raise ValueError("plot has to be either 'ci', 'meanci', 'ib', 'oob', 'all'.")

        if self.name == 'cimcb.model.NN_SigmoidSigmoid' or self.name == 'cimcb.model.NN_LinearSigmoid':
            lv_name = "Neuron"
        else:
            lv_name = "LV"

        # pctvar
        pctvar_all = []
        for i in range(len(self.bootidx_oob)):
            val = self.bootstat_oob['model.pctvar_'][i]
            pctvar_all.append(val)
        #pctvar_ = np.median(pctvar_all, 0)
        pctvar_ = self.stat['model.pctvar_']

        # y_loadings_
        y_loadings_all = []
        for i in range(len(self.bootidx_oob)):
            val = self.bootstat_oob['model.y_loadings_'][i]
            y_loadings_all.append(list(val[0]))
        y_loadings_ = np.median(y_loadings_all, 0)
        y_loadings_all = np.array(y_loadings_all)

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

        # Sort by pctvar_
        order = np.argsort(pctvar_)[::-1]
        x_scores_ = x_scores_[:, order]
        x_scores_ib = x_scores_ib[:, order]
        x_scores_oob = x_scores_oob[:, order]
        pctvar_ = pctvar_[order]
        y_loadings_ = y_loadings_[order]
        y_loadings_all = y_loadings_all[:, order]
        self.x_scores_full = x_scores_ib

        if num_x_scores == 1:
            print('LV must be > 1 to plot projections')
            pass
        else:
            if scatter_ib is True:
                x_scores_full = x_scores_ib
            else:
                x_scores_full = x_scores_
            x_scores_orig = x_scores_
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
                    label_copy = pd.Series(self.Y, name='Class').apply(str)
                else:
                    newlabel = np.array(label)
                    label_copy = deepcopy(label)
                    # group_copy = deepcopy(newlabel)
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

                x_full = x_scores_full[:, x].tolist()
                y_full = x_scores_full[:, y].tolist()
                x_cv = x_scores_cv[:, x].tolist()
                y_cv = x_scores_cv[:, y].tolist()
                x_orig = x_scores_orig[:, x].tolist()
                y_orig = x_scores_orig[:, y].tolist()

                regY_full = self.Y
                regX_full = np.array([x_full, y_full]).T
                reg_stat = LinearRegression().fit(regX_full, regY_full)
                gradient = reg_stat.coef_[1] / reg_stat.coef_[0]

                gradient = y_loadings_[y] / y_loadings_[x]
                grid[y, x] = scatter_ellipse(x_orig, y_orig, x_cv, y_cv, label=label_copy, group=group_copy, title="", xlabel=xlabel, ylabel=ylabel, width=width_height, height=width_height, legend=legend_scatter, size=circle_size_scoreplot, label_font_size=label_font, hover_xy=False, xrange=new_range, yrange=new_range, gradient=gradient, ci95=True, scatterplot=scatterplot, extraci95_x=x_cv, extraci95_y=y_cv, extraci95=True, scattershow=plot_num, extraci95_x2=x_full, extraci95_y2=y_full, orthog_line=orthog_line, grid_line=grid_line, legend_title=True, font_size=label_font)

            # Append each distribution curve
            group_dist = np.concatenate((self.Y, (self.Y + 2)))

            dist_label1 = np.array(label_copy[self.Y == 0])[0]
            dist_label2 = np.array(label_copy[self.Y == 1])[0]
            dist_label = [str(dist_label1), str(dist_label2)]

            for i in range(num_x_scores):
                score_dist = np.concatenate((x_scores_full[:, i], x_scores_cv[:, i]))
                xlabel = "{} {} ({:0.1f}%)".format(lv_name, i + 1, pctvar_[i])
                grid[i, i] = distribution(score_dist, group=group_dist, kde=True, title="", xlabel=xlabel, ylabel="p.d.f.", width=width_height, height=width_height, label_font_size=label_font, legend=legend_dist, group_label=dist_label, plot_num=plot_num, grid_line=grid_line, legend_title=True, font_size=label_font)

            # Append each roc curve
            for i in range(len(comb_x_scores)):
                x, y = comb_x_scores[i]
                idx_x = order[x]
                idx_y = order[y]

                # Get Stat
                x_stat = self.stat['model.x_scores_'][:, idx_x]
                y_stat = self.stat['model.x_scores_'][:, idx_y]
                regY_stat = self.Y
                regX_stat = np.array([x_stat, y_stat]).T
                reg_stat = LinearRegression().fit(regX_stat, regY_stat)
                grad_stat = reg_stat.coef_[1] / reg_stat.coef_[0]
                theta_stat = math.atan(grad_stat)
                ypred_stat = x_stat * math.cos(theta_stat) + y_stat * math.sin(theta_stat)  # Optimal line
                stat = ypred_stat

                ypred_ib = []
                for i in range(len(self.bootstat['model.x_scores_'])):
                    x_bootstat = self.bootstat['model.x_scores_'][i][:, idx_x]
                    y_bootstat = self.bootstat['model.x_scores_'][i][:, idx_y]
                    regY_bootstat = self.Y[self.bootidx[i]]
                    regX_bootstat = np.array([x_bootstat, y_bootstat]).T
                    reg_bootstat = LinearRegression().fit(regX_bootstat, regY_bootstat)
                    grad_bootstat = reg_bootstat.coef_[1] / reg_bootstat.coef_[0]
                    theta_bootstat = math.atan(grad_bootstat)
                    ypred_ib_i = x_bootstat * math.cos(theta_bootstat) + y_bootstat * math.sin(theta_bootstat)  # Optimal line
                    ypred_ib.append(ypred_ib_i)
                bootstat = ypred_ib

                ypred_oob = []
                for i in range(len(self.bootstat_oob['model.x_scores_'])):
                    x_bootstat_oob = self.bootstat_oob['model.x_scores_'][i][:, idx_x]
                    y_bootstat_oob = self.bootstat_oob['model.x_scores_'][i][:, idx_y]
                    regY_bootstat_oob = self.Y[self.bootidx_oob[i]]
                    regX_bootstat_oob = np.array([x_bootstat_oob, y_bootstat_oob]).T
                    reg_bootstat_oob = LinearRegression().fit(regX_bootstat_oob, regY_bootstat_oob)
                    grad_bootstat_oob = reg_bootstat_oob.coef_[1] / reg_bootstat_oob.coef_[0]
                    theta_bootstat_oob = math.atan(grad_bootstat_oob)
                    ypred_oob_i = x_bootstat_oob * math.cos(theta_bootstat_oob) + y_bootstat_oob * math.sin(theta_bootstat_oob)  # Optimal line
                    ypred_oob.append(ypred_oob_i)
                bootstat_oob = ypred_oob

                if self.__name__ == 'BCA':
                    jackidx = self.jackidx
                    ypred_jackstat = []
                    for i in range(len(jackidx)):
                        x_jackstat = self.jackstat['model.x_scores_'][i][:, idx_x]
                        y_jackstat = self.jackstat['model.x_scores_'][i][:, idx_y]
                        regY_jackstat = self.Y[self.jackidx[i]]
                        regX_jackstat = np.array([x_jackstat, y_jackstat]).T
                        reg_jackstat = LinearRegression().fit(regX_jackstat, regY_jackstat)
                        grad_jackstat = reg_jackstat.coef_[1] / reg_jackstat.coef_[0]
                        theta_jackstat = math.atan(grad_jackstat)
                        ypred_jackstat_i = x_jackstat * math.cos(theta_jackstat) + y_jackstat * math.sin(theta_jackstat)  # Optimal line
                        ypred_jackstat.append(ypred_jackstat_i)
                    jackstat = ypred_jackstat
                else:
                    jackstat = None
                    jackidx = None

                grid[x, y], _, _ = roc_boot(self.Y, stat, bootstat, bootstat_oob, self.bootidx, self.bootidx_oob, self.__name__, smoothval=smooth, jackstat=jackstat, jackidx=jackidx, xlabel="1-Specificity ({}{}/{}{})".format(lv_name, x + 1, lv_name, y + 1), ylabel="Sensitivity ({}{}/{}{})".format(lv_name, x + 1, lv_name, y + 1), width=width_height, height=width_height, label_font_size=label_font, legend=legend_roc, grid_line=grid_line, plot_num=plot_num, plot=plot_roc)

            # Bokeh grid
            fig = gridplot(grid.tolist())

            output_notebook()
            show(fig)

    def plot_featureimportance(self, PeakTable, peaklist=None, ylabel="Label", sort=True, sort_ci=True, grid_line=False, plot='data', x_axis_below=True):
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

        if isinstance(plot, str):
            if plot == 'data':
                mid_coef = self.stat['model.coef_']
                mid_vip = self.stat['model.vip_']
            elif plot == 'median':
                mid_coef = self.bootci['model.coef_'][:, 2]
                mid_vip = self.bootci['model.vip_'][:, 2]
            else:
                pass
        else:
            mid_coef = plot[:, 0]
            mid_vip = plot[:, 1]

        # if plot == 'data':
        #     mid_coef = self.stat['model.coef_']
        #     mid_vip = self.stat['model.vip_']
        # else:
        #     mid_coef = self.bootci['model.coef_'][:, 2]
        #     mid_vip = self.bootci['model.vip_'][:, 2]

        if self.name == 'cimcb.model.NN_SigmoidSigmoid' or self.name == 'cimcb.model.NN_LinearSigmoid':
            name_coef = "Feature Importance: Connection Weight"
            name_vip = "Feature Importance: Garlson's Algorithm"
        else:
            name_coef = "Coefficient Plot"
            name_vip = "Variable Importance in Projection (VIP) Plot"

        # Remove rows from PeakTable if not in peaklist
        if peaklist is not None:
            PeakTable = PeakTable[PeakTable["Name"].isin(peaklist)]
        peaklabel = PeakTable[ylabel]

        # Current issue with Bokeh & Chrome, can't plot >1500 features.
        n = 1500
        if len(peaklist) > n:
            peaklist_chunk = [peaklist[i * n:(i + 1) * n] for i in range((len(peaklist) + n - 1) // n)]
            peaklabel_chunk = [peaklabel[i * n:(i + 1) * n] for i in range((len(peaklabel) + n - 1) // n)]
            mid_coef_chunk = [mid_coef[i * n:(i + 1) * n] for i in range((len(mid_coef) + n - 1) // n)]
            ci_coef_chunk = [ci_coef[i * n:(i + 1) * n] for i in range((len(ci_coef) + n - 1) // n)]
            mid_vip_chunk = [mid_vip[i * n:(i + 1) * n] for i in range((len(mid_vip) + n - 1) // n)]
            ci_vip_chunk = [ci_vip[i * n:(i + 1) * n] for i in range((len(ci_vip) + n - 1) // n)]

            grid = np.full((len(peaklist_chunk), 2), None)
            for i in range(len(peaklist_chunk)):
                PeakTable_chunk = PeakTable[PeakTable["Name"].isin(peaklist_chunk[i])]

                grid[0,i] = scatterCI(mid_coef_chunk[i], ci=ci_coef_chunk[i], label=peaklabel_chunk[i], hoverlabel=PeakTable_chunk[["Idx", "Name", "Label"]], hline=0, col_hline=True, title=name_coef, sort_abs=sort, sort_ci=sort_ci, grid_line=grid_line, x_axis_below=x_axis_below)

                if name_vip == "Variable Importance in Projection (VIP) Plot":
                    grid[1,i] = scatterCI(mid_vip_chunk[i], ci=ci_vip_chunk[i], label=peaklabel_chunk[i], hoverlabel=PeakTable_chunk[["Idx", "Name", "Label"]], hline=1, col_hline=False, title=name_vip, sort_abs=sort, sort_ci=sort_ci, sort_ci_abs=True, grid_line=grid_line, x_axis_below=x_axis_below)
                else:
                    grid[1,i] = scatterCI(mid_vip_chunk[i], ci=ci_vip_chunk[i], label=peaklabel_chunk[i], hoverlabel=PeakTable_chunk[["Idx", "Name", "Label"]], hline=np.average(self.stat['model.vip_']), col_hline=False, title=name_vip, sort_abs=sort, sort_ci_abs=True, grid_line=grid_line, x_axis_below=x_axis_below)
            fig = gridplot(grid.tolist())
        else:
            # Plot
            fig_1 = scatterCI(mid_coef, ci=ci_coef, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=0, col_hline=True, title=name_coef, sort_abs=sort, sort_ci=sort_ci, grid_line=grid_line, x_axis_below=x_axis_below)
            if name_vip == "Variable Importance in Projection (VIP) Plot":
                fig_2 = scatterCI(mid_vip, ci=ci_vip, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=1, col_hline=False, title=name_vip, sort_abs=sort, sort_ci=sort_ci, sort_ci_abs=True, grid_line=grid_line, x_axis_below=x_axis_below)
            else:
                fig_2 = scatterCI(mid_vip, ci=ci_vip, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=np.average(self.stat['model.vip_']), col_hline=False, title=name_vip, sort_abs=sort, sort_ci_abs=True, grid_line=grid_line, x_axis_below=x_axis_below)

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

        if isinstance(plot, str):
            pass
        else:
            Peaksheet["Coef_Test"] = mid_coef
            Peaksheet["VIP_Test"] = mid_vip
        return Peaksheet

    def save_results(self, name="table.xlsx"):

        table = self.table
        check_type = name.split(".")
        if check_type[-1] == "xlsx":
            table.to_excel(name)
        elif check_type[-1] == "csv":
            table.to_csv(name)
        else:
            raise ValueError("name must end in .xlsx or .csv")
        print("Done! Saved results as {}".format(name))

    @abstractmethod
    def calc_bootci(self):
        """Calculates bootstrap confidence intervals using bootci_method."""
        pass

    @abstractmethod
    def run(self):
        """Runs every function and returns bootstrap confidence intervals (a dict of arrays)."""
        pass

    @abstractmethod
    def bootci_method(self):
        """Method used to calculate boostrap confidence intervals (Refer to: BC, BCA, or Perc)."""
        pass
