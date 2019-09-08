import numpy as np
import pandas as pd
import re
import math
import multiprocessing
from abc import ABC, abstractmethod
from sklearn import metrics
from bokeh.layouts import gridplot, layout
from bokeh import events
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource, Circle, HoverTool, TapTool, LabelSet, Rect, LinearColorMapper, MultiLine, Patch, Patches, CustomJS, Text, Title
from itertools import product
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
from ..utils import color_scale, dict_perc


class BaseCrossVal(ABC):
    """Base class for crossval: kfold."""

    @abstractmethod
    def __init__(self, model, X, Y, param_dict, folds=10, n_mc=1, n_boot=0, n_cores=-1, ci=95):
        # Store basic inputs
        self.model = model
        self.X = X
        self.Y = Y
        self.num_param = len(param_dict)
        self.folds = folds
        self.n_boot = n_boot
        self.n_mc = n_mc
        # Note; if self.mc is 0, change to 1
        if self.n_mc == 0:
            self.n_mc = 1
        self.ci = ci
        # Save param_dict
        # Make sure each parameter is in a list
        for key, value in param_dict.items():
            if not isinstance(value, (list, tuple, np.ndarray)):
                param_dict[key] = [value]
        self.param_dict = param_dict
        self.param_list = list(ParameterGrid(param_dict))
        # Create a second dict, with parameters with more than 1 variable e.g. n_comp = [1, 2]
        self.param_dict2 = {}
        for key, value in param_dict.items():
            if len(value) > 1:
                self.param_dict2 = {**self.param_dict2, **{key: value}}
        self.param_list2 = list(ParameterGrid(self.param_dict2))

        # if n_cores = -1, set n_cores to max_cores
        max_num_cores = multiprocessing.cpu_count()
        self.n_cores = n_cores
        if self.n_cores > max_num_cores:
            self.n_cores = -1
            print("Number of cores set too high. It will be set to the max number of cores in the system.", flush=True)
        if self.n_cores == -1:
            self.n_cores = max_num_cores
            print("Number of cores set to: {}".format(max_num_cores))

    @abstractmethod
    def calc_ypred(self):
        """Calculates ypred full and ypred cv."""
        pass

    @abstractmethod
    def calc_ypred_epoch(self):
        """Calculates ypred full and ypred cv."""
        pass

    @abstractmethod
    def calc_stats(self):
        """Calculates binary statistics from ypred full and ypred cv."""
        pass

    def _format_table(self, stats_list):
        """Make stats pretty (pandas table -> proper names in columns)."""
        table = pd.DataFrame(stats_list).T
        param_list_string = []
        for i in range(len(self.param_list)):
            param_list_string.append(str(self.param_list[i]))
        table.columns = param_list_string
        return table

    def run(self):
        """Runs all functions prior to plot."""
        # Check that param_dict is not for epochs
        # Epoch is a special case
        print("Running ...")
        check_epoch = []
        for i in self.param_dict2.keys():
            check_epoch.append(i)
        if check_epoch == ["epochs"]:
            # Get epoch max
            epoch_list = []
            for i in self.param_list2:
                for k, v in i.items():
                    epoch_list.append(v)
            # Print and Calculate
            self.calc_ypred_epoch()
            print("returning stats at 'x' epoch interval during training until epoch={}.".format(epoch_list[-1]))
        else:
            self.calc_ypred()
        self.calc_stats()
        print("Done!")

    def plot(self, metric="r2q2", scale=1, color_scaling="tanh", rotate_xlabel=True, model="kfold", legend="bottom_right", color_beta=[10, 10, 10], ci=95, diff1_heat=True):
        """Create a full/cv plot using based on metric selected.

        Parameters
        ----------
        metric : string, (default "r2q2")
            metric has to be either "r2q2", "auc", "acc", "f1score", "prec", "sens", or "spec".
        """
        # Check model is parametric if using 'r2q2'
        if metric == "r2q2":
            if self.model.parametric is False:
                print("metric changed from 'r2q2' to 'auc' as the model is non-parametric.")
                metric = "auc"

        # Plot based on the number of parameters
        if len(self.param_dict2) == 1:
            fig = self._plot_param1(metric=metric, scale=scale, rotate_xlabel=rotate_xlabel, model=model, legend=legend, ci=ci)
        elif len(self.param_dict2) == 2:
            fig = self._plot_param2(metric=metric, scale=scale, color_scaling=color_scaling, model=model, legend=legend, color_beta=color_beta, ci=ci, diff1_heat=diff1_heat)
        else:
            raise ValueError("plot function only works for 1 or 2 parameters, there are {}.".format(len(self.param_dict2)))

        # Show plot
        output_notebook()
        show(fig)

    def _plot_param1(self, metric="r2q2", scale=1, rotate_xlabel=True, model="kfold", title_align="center", legend="bottom_right", ci=95):
        """Used for plot function if the number of parameters is 1."""

        # Get ci
        if self.n_mc > 1:
            std_list = []
            for i in range(len(self.param_list)):
                std_full_i = dict_perc(self.full_loop[i], ci=ci)
                std_cv_i = dict_perc(self.cv_loop[i], ci=ci)
                std_full_i = {k + "full": v for k, v in std_full_i.items()}
                std_cv_i = {k + "cv": v for k, v in std_cv_i.items()}
                std_cv_i["R²"] = std_full_i.pop("R²full")
                std_cv_i["Q²"] = std_cv_i.pop("R²cv")
                std_combined = {**std_full_i, **std_cv_i}
                std_list.append(std_combined)
            self.table_std = self._format_table(std_list)  # Transpose, Add headers
            self.table_std = self.table_std.reindex(index=np.sort(self.table_std.index))

        # Choose metric to plot
        metric_title = np.array(["ACCURACY", "AIC", "AUC", "BIC", "F1-SCORE", "PRECISION", "R²", "SENSITIVITY", "SPECIFICITY", "SSE"])
        metric_list = np.array(["acc", "aic", "auc", "bic", "f1score", "prec", "r2q2", "sens", "spec", "sse"])
        metric_idx = np.where(metric_list == metric)[0][0]

        # get full, cv, and diff
        full = self.table.iloc[2 * metric_idx + 1]
        cv = self.table.iloc[2 * metric_idx]
        diff = abs(full - cv)
        full_text = self.table.iloc[2 * metric_idx + 1].name
        cv_text = self.table.iloc[2 * metric_idx].name
        if metric == "r2q2":
            diff_text = "| R²-Q² |"
            y_axis_text = "R² & Q²"
            full_legend = "R²"
            cv_legend = "Q²"
        else:
            diff_text = full_text[:-4] + "diff"
            y_axis_text = full_text[:-4]
            if model == "kfold":
                full_legend = "FULL"
                cv_legend = "CV"
            else:
                full_legend = "TRAIN"
                cv_legend = "TEST"
                full_text = full_text[:-4] + "train"
                cv_text = full_text[:-5] + "test"

        # round full, cv, and diff for hovertool
        full_hover = []
        cv_hover = []
        diff_hover = []
        for j in range(len(full)):
            full_hover.append("%.2f" % round(full[j], 2))
            cv_hover.append("%.2f" % round(cv[j], 2))
            diff_hover.append("%.2f" % round(diff[j], 2))

        # get key, values (as string) from param_dict (key -> title, values -> x axis values)
        for k, v in self.param_dict2.items():
            key_title = k
            key_xaxis = k
            values = v
        values_string = [str(i) for i in values]
        values_string = []
        for i in values:
            if i == 0:
                values_string.append(str(i))
            elif 0.0001 > i:
                values_string.append("%0.2e" % i)
            elif 10000 < i:
                values_string.append("%0.2e" % i)
            else:
                values_string.append(str(i))

        # if parameter starts with n_ e.g. n_components change title to 'no. of components', xaxis to 'components'
        if key_title.startswith("n_"):
            key_xaxis = key_xaxis[2:]
            key_xaxis = key_xaxis.title()
            key_title = "no. of " + key_xaxis
        else:
            key_title = key_title.replace("_", " ")
            key_title = key_title.title()
            key_xaxis = key_title
            # if key_xaxis.endswith("s") == True:
            #     key_xaxis = key_xaxis[:-1]

        # store data in ColumnDataSource for Bokeh
        data = dict(full=full, cv=cv, diff=diff, full_hover=full_hover, cv_hover=cv_hover, diff_hover=diff_hover, values_string=values_string)
        source = ColumnDataSource(data=data)

        # fig1_yrange = (min(diff) - max(0.1 * (min(diff)), 0.07), max(diff) + max(0.1 * (max(diff)), 0.07))
        # fig1_xrange = (min(cv) - max(0.1 * (min(cv)), 0.07), max(cv) + max(0.1 * (max(cv)), 0.07))
        fig1_title = diff_text + " vs. " + cv_text

        # Plot width/height
        width = int(485 * scale)
        height = int(405 * scale)
        # fig1_yrange = (min(diff) - max(0.1 * (min(diff)), 0.07), max(diff) + max(0.1 * (max(diff)), 0.07))
        # fig1_xrange = (min(cv) - max(0.1 * (min(cv)), 0.07), max(cv) + max(0.1 * (max(cv)), 0.07))
        # x_range=(min(cv_score) - 0.03, max(cv_score) + 0.03), y_range=(min(diff_score) - 0.03, max(diff_score) + 0.03)

        # Figure 1 (DIFFERENCE (R2 - Q2) vs. Q2)
        fig1 = figure(x_axis_label=cv_text, y_axis_label=diff_text, title=fig1_title, tools="tap,pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select", plot_width=width, plot_height=height, x_range=(min(cv) - 0.03, max(cv) + 0.03), y_range=(min(diff) - 0.03, max(diff) + 0.03))

        # Figure 1: Add a line
        fig1_line = fig1.line(cv, diff, line_width=2, line_color="black", line_alpha=0.25)

        # Figure 1: Add circles (interactive click)
        fig1_circ = fig1.circle("cv", "diff", size=12, alpha=0.7, color="green", source=source)
        fig1_circ.selection_glyph = Circle(fill_color="green", line_width=2, line_color="black")
        fig1_circ.nonselection_glyph.fill_color = "green"
        fig1_circ.nonselection_glyph.fill_alpha = 0.4
        fig1_circ.nonselection_glyph.line_color = "white"

        # Figure 1: Add hovertool
        fig1.add_tools(HoverTool(renderers=[fig1_circ], tooltips=[(key_xaxis, "@values_string"), (full_text, "@full_hover"), (cv_text, "@cv_hover"), (diff_text, "@diff_hover")]))

        # Figure 1: Extra formating
        fig1.axis.major_label_text_font_size = "8pt"
        if metric is "r2q2" or metric is "auc":
            fig1.title.text_font_size = "12pt"
            fig1.xaxis.axis_label_text_font_size = "10pt"
            fig1.yaxis.axis_label_text_font_size = "10pt"
        else:
            fig1.title.text_font_size = "10pt"
            fig1.xaxis.axis_label_text_font_size = "9pt"
            fig1.yaxis.axis_label_text_font_size = "9pt"

        # Figure 2: full/cv
        fig2_title = y_axis_text + " over " + key_title
        fig2 = figure(x_axis_label=key_xaxis, y_axis_label=y_axis_text, title=fig2_title, plot_width=width, plot_height=height, x_range=pd.unique(values_string), tools="pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select")

        # Figure 2: Add Confidence Intervals if n_mc > 1
        if self.n_mc > 1:
            # get full, cv, and diff
            full_std = self.table_std.iloc[2 * metric_idx + 1]
            cv_std = self.table_std.iloc[2 * metric_idx]
            lower_ci_full = pd.Series(name=full_std.name, dtype="object")
            upper_ci_full = pd.Series(name=full_std.name, dtype="object")
            for key, values in full_std.iteritems():
                lower_ci_full[key] = values[0]
                upper_ci_full[key] = values[1]
            lower_ci_cv = pd.Series(name=cv_std.name, dtype="object")
            upper_ci_cv = pd.Series(name=cv_std.name, dtype="object")
            for key, values in cv_std.iteritems():
                lower_ci_cv[key] = values[0]
                upper_ci_cv[key] = values[1]
            # Plot as a patch
            x_patch = np.hstack((values_string, values_string[::-1]))
            y_patch_r2 = np.hstack((lower_ci_full, upper_ci_full[::-1]))
            y_patch_q2 = np.hstack((lower_ci_cv, upper_ci_cv[::-1]))
            fig2.patch(x_patch, y_patch_q2, alpha=0.10, color="blue")
            # kfold monte-carlo does not have ci for R2
            if model is not "kfold":
                fig2.patch(x_patch, y_patch_r2, alpha=0.10, color="red")

        # Figure 2: add full
        fig2_line_full = fig2.line(values_string, full, line_color="red", line_width=2)
        fig2_circ_full = fig2.circle("values_string", "full", line_color="red", fill_color="white", fill_alpha=1, size=8, source=source, legend=full_legend)
        fig2_circ_full.selection_glyph = Circle(line_color="red", fill_color="white", line_width=2)
        fig2_circ_full.nonselection_glyph.line_color = "red"
        fig2_circ_full.nonselection_glyph.fill_color = "white"
        fig2_circ_full.nonselection_glyph.line_alpha = 0.4

        # Figure 2: add cv
        fig2_line_cv = fig2.line(values_string, cv, line_color="blue", line_width=2)
        fig2_circ_cv = fig2.circle("values_string", "cv", line_color="blue", fill_color="white", fill_alpha=1, size=8, source=source, legend=cv_legend)
        fig2_circ_cv.selection_glyph = Circle(line_color="blue", fill_color="white", line_width=2)
        fig2_circ_cv.nonselection_glyph.line_color = "blue"
        fig2_circ_cv.nonselection_glyph.fill_color = "white"
        fig2_circ_cv.nonselection_glyph.line_alpha = 0.4

        # Add hovertool and taptool
        fig2.add_tools(HoverTool(renderers=[fig2_circ_full], tooltips=[(full_text, "@full_hover")], mode="vline"))
        fig2.add_tools(HoverTool(renderers=[fig2_circ_cv], tooltips=[(cv_text, "@cv_hover")], mode="vline"))
        fig2.add_tools(TapTool(renderers=[fig2_circ_full, fig2_circ_cv]))

        # Figure 2: Extra formating
        fig2.axis.major_label_text_font_size = "8pt"
        if metric is "r2q2" or metric is "auc":
            fig2.title.text_font_size = "12pt"
            fig2.xaxis.axis_label_text_font_size = "10pt"
            fig2.yaxis.axis_label_text_font_size = "10pt"
        else:
            fig2.title.text_font_size = "10pt"
            fig2.xaxis.axis_label_text_font_size = "9pt"
            fig2.yaxis.axis_label_text_font_size = "9pt"

        # Rotate
        if rotate_xlabel is True:
            fig2.xaxis.major_label_orientation = np.pi / 2

        # Figure 2: legend
        if legend == None or legend == False:
            fig2.legend.visible = False
        else:
            fig2.legend.location = legend
            fig2.legend.location = legend

        # Center title
        if title_align == "center":
            fig1.title.align = "center"
            fig2.title.align = "center"

        # Create a grid and output figures
        grid = np.full((1, 2), None)
        grid[0, 0] = fig1
        grid[0, 1] = fig2
        fig = gridplot(grid.tolist(), merge_tools=True)
        return fig

    def _plot_param2(self, metric="r2q2", xlabel=None, orientation=0, alternative=False, scale=1, heatmap_xaxis_rotate=90, color_scaling="tanh", line=False, model="kfold", title_align="center", legend="bottom_right", color_beta=[10, 10, 10], ci=95, diff1_heat=True):

        # legend always None
        legend = None

        # check color_beta
        if type(color_beta) != list:
            raise ValueError("color_beta needs to be a list of 3 values e.g. [10, 10, 10]")
        if len(color_beta) != 3:
            raise ValueError("color_beta needs to be a list of 3 values e.g. [10, 10, 10]")

        # Get ci
        if self.n_mc > 1:
            std_list = []
            for i in range(len(self.param_list)):
                std_full_i = dict_perc(self.full_loop[i], ci=ci)
                std_cv_i = dict_perc(self.cv_loop[i], ci=ci)
                std_full_i = {k + "full": v for k, v in std_full_i.items()}
                std_cv_i = {k + "cv": v for k, v in std_cv_i.items()}
                std_cv_i["R²"] = std_full_i.pop("R²full")
                std_cv_i["Q²"] = std_cv_i.pop("R²cv")
                std_combined = {**std_full_i, **std_cv_i}
                std_list.append(std_combined)
            self.table_std = self._format_table(std_list)  # Transpose, Add headers
            self.table_std = self.table_std.reindex(index=np.sort(self.table_std.index))

        metric_list = np.array(["acc", "aic", "auc", "bic", "f1score", "prec", "r2q2", "sens", "spec", "sse"])
        metric_idx = np.where(metric_list == metric)[0][0]

        # get full, cv, and diff
        full_score = self.table.iloc[2 * metric_idx + 1]
        cv_score = self.table.iloc[2 * metric_idx]
        diff_score = abs(full_score - cv_score)
        full_title = self.table.iloc[2 * metric_idx + 1].name
        cv_title = self.table.iloc[2 * metric_idx].name
        diff_title = full_title[:-4] + "diff"

        if diff1_heat == False:
            diff_heat_title = diff_title
            diff_heat_score = diff_score
        else:
            diff_heat_title = "1 - " + full_title[:-4] + "diff"
            diff_heat_score = 1 - diff_score
        y_axis_text = full_title[:-4]

        if metric is "r2q2":
            full_title = 'R²'
            cv_title = 'Q²'
            diff_title = "| R² - Q² |"
            if diff1_heat == False:
                diff_heat_title = diff_title
            else:
                diff_heat_title = "1  -   | R² - Q² |"
            y_axis_text = "R² & Q²"

        if model == "kfold":
            full_legend = "FULL"
            cv_legend = "CV"
        else:
            full_legend = "TRAIN"
            cv_legend = "TEST"
            full_title = full_title[:-4] + "train"
            cv_title = full_title[:-5] + "test"
            if metric is "r2q2":
                full_title = 'R²'
                cv_title = 'Q²'

        # round full, cv, and diff for hovertool
        full_hover = []
        cv_hover = []
        diff_hover = []
        for j in range(len(full_score)):
            full_hover.append("%.2f" % round(full_score[j], 2))
            cv_hover.append("%.2f" % round(cv_score[j], 2))
            diff_hover.append("%.2f" % round(diff_score[j], 2))

        # If n_mc > 1
        if self.n_mc > 1:
            # get full, cv, and diff
            full_std = self.table_std.iloc[2 * metric_idx + 1]
            cv_std = self.table_std.iloc[2 * metric_idx]
            lower_ci_full = pd.Series(name=full_std.name, dtype="object")
            upper_ci_full = pd.Series(name=full_std.name, dtype="object")
            for key, values in full_std.iteritems():
                lower_ci_full[key] = values[0]
                upper_ci_full[key] = values[1]
            lower_ci_cv = pd.Series(name=cv_std.name, dtype="object")
            upper_ci_cv = pd.Series(name=cv_std.name, dtype="object")
            for key, values in cv_std.iteritems():
                lower_ci_cv[key] = values[0]
                upper_ci_cv[key] = values[1]

        # Get key/values
        param_keys = []
        param_values = []
        for key, value in sorted(self.param_dict2.items()):
            param_keys.append(key)
            # value_to_string = list(map(str, value))
            # param_values.append(value_to_string)
            values_string = []
            for i in value:
                if i == 0:
                    values_string.append(str(i))
                elif 0.0001 > i:
                    values_string.append("%0.2e" % i)
                elif 10000 < i:
                    values_string.append("%0.2e" % i)
                else:
                    values_string.append(str(i))

            param_values.append(values_string)

        param_keys_title = []
        param_keys_axis = []
        for i in param_keys:
            if i.startswith("n_"):
                i_xaxis = i[2:]
                i_xaxis = i_xaxis.title()
                i_title = "no. of " + i_xaxis
            else:
                i_title = i.replace("_", " ")
                i_title = i_title.title()
                i_xaxis = i_title
            param_keys_title.append(i_title)
            param_keys_axis.append(i_xaxis)

        # Get key/value combinations
        comb = list(product(param_values[0], param_values[1]))
        key0_value = [val[0] for val in comb]
        key1_value = [val[1] for val in comb]
        key0_unique = param_values[0]
        key1_unique = param_values[1]
        table = self.table
        param_dict = self.param_dict2
        param_list = self.param_list2

        full_alpha = color_scale(full_score, method=color_scaling, beta=color_beta[0])
        cv_alpha = color_scale(cv_score, method=color_scaling, beta=color_beta[1])
        diff_alpha = color_scale(diff_heat_score, method=color_scaling, beta=color_beta[2])

        # Text for heatmaps
        full_text = []
        cv_text = []
        diff_text = []
        for i in range(len(key0_value)):
            full_text.append("%.2f" % round(full_score[i], 2))
            cv_text.append("%.2f" % round(cv_score[i], 2))
            diff_text.append("%.2f" % round(diff_score[i], 2))

        # Information for line plot
        line_key0_value = []
        for i in range(len(key0_value)):
            line_key0_value.append(key0_unique)
        line_key1_value = []
        for i in range(len(key1_value)):
            line_key1_value.append(key1_unique)

        # Extra for n_mc
        if self.n_mc > 1:
            # Information for line plot
            monte_line_key0_value = []
            for i in range(len(key0_value)):
                monte_line_key0_value.append(list(np.hstack((key0_unique, key0_unique[::-1]))))
            monte_line_key1_value = []
            for i in range(len(key1_value)):
                monte_line_key1_value.append(list(np.hstack((key1_unique, key1_unique[::-1]))))

        line0_full = []
        line0_cv = []
        for i in range(len(key0_value)):
            line0_full_i = []
            line0_cv_i = []
            for j in range(len(key0_value)):
                if key0_value[i] == key0_value[j]:
                    line0_full_i.append(full_score[j])
                    line0_cv_i.append(cv_score[j])
            line0_full.append(line0_full_i)
            line0_cv.append(line0_cv_i)

        line1_full = []
        line1_cv = []
        for i in range(len(key1_value)):
            line1_full_i = []
            line1_cv_i = []
            for j in range(len(key1_value)):
                if key1_value[i] == key1_value[j]:
                    line1_full_i.append(full_score[j])
                    line1_cv_i.append(cv_score[j])
            line1_full.append(line1_full_i)
            line1_cv.append(line1_cv_i)

        # Extra for n_mc
        if self.n_mc > 1:
            monte_line1_full = []
            monte_line1_cv = []
            for i in range(len(key1_value)):
                line1_full_i_upper = []
                line1_full_i_lower = []
                line1_cv_i_upper = []
                line1_cv_i_lower = []
                for j in range(len(key1_value)):
                    if key1_value[i] == key1_value[j]:
                        line1_full_i_upper.append(upper_ci_full[j])
                        line1_full_i_lower.append(lower_ci_full[j])
                        line1_cv_i_upper.append(upper_ci_cv[j])
                        line1_cv_i_lower.append(lower_ci_cv[j])
                monte_line1_full.append(list(np.hstack((line1_full_i_lower, line1_full_i_upper[::-1]))))
                monte_line1_cv.append(list(np.hstack((line1_cv_i_lower, line1_cv_i_upper[::-1]))))

        # Extra for n_mc
        if self.n_mc > 1:
            monte_line0_full = []
            monte_line0_cv = []
            for i in range(len(key0_value)):
                line0_full_i_upper = []
                line0_full_i_lower = []
                line0_cv_i_upper = []
                line0_cv_i_lower = []
                for j in range(len(key0_value)):
                    if key0_value[i] == key0_value[j]:
                        line0_full_i_upper.append(upper_ci_full[j])
                        line0_full_i_lower.append(lower_ci_full[j])
                        line0_cv_i_upper.append(upper_ci_cv[j])
                        line0_cv_i_lower.append(lower_ci_cv[j])
                monte_line0_full.append(list(np.hstack((line0_full_i_lower, line0_full_i_upper[::-1]))))
                monte_line0_cv.append(list(np.hstack((line0_cv_i_lower, line0_cv_i_upper[::-1]))))

        # Scatterplot color and size based on key0 and key1
        color_key0 = []
        for i in range(len(key0_value)):
            for j in range(len(key0_unique)):
                if key0_value[i] == key0_unique[j]:
                    color_key0.append(j / (len(key0_unique) - 1))

        scaler_size = preprocessing.MinMaxScaler(feature_range=(4, 20))
        size_prescale_key1 = []

        for i in range(len(key1_value)):
            for j in range(len(key1_unique)):
                if key1_value[i] == key1_unique[j]:
                    size_prescale_key1.append(j / (len(key1_unique) - 1))
        scatter_size_key1 = scaler_size.fit_transform(np.array(size_prescale_key1)[:, np.newaxis])
        scatter_size_key1 = scatter_size_key1 * scale

        # Extra
        key0_value_text = len(key1_value) * [key1_value[-1]]
        key1_value_text = len(key0_value) * [key0_value[-1]]
        line1_cv_text = []
        for i in line1_cv:
            line1_cv_text.append(i[-1])
        line0_cv_text = []
        for i in line0_cv:
            line0_cv_text.append(i[-1])
        line1_full_text = []
        for i in line1_full:
            line1_full_text.append(i[-1])
        line0_full_text = []
        for i in line0_full:
            line0_full_text.append(i[-1])

        ptext_is = ["Learning Rate", "Nodes", "Neurons", "Momentum", "Decay", "Components", "Batch Size", "Gamma", "C", "Estimators", "Max Features", "Max Depth", "Min Samples Split", "Min Samples Leaf", "Max Leaf Nodes"]
        ptext_change = ["LR", "Node", "Neur", "Mom", "Dec", "Comp", "Bat", "Gam", "C", "Est", "Feat", "Dep", "SSpl", "SLea", "LNod"]

        ptext = []
        for i in param_keys_axis:
            val = "fill"
            for j in range(len(ptext_is)):
                if i == ptext_is[j]:
                    val = ptext_change[j]
            if val == "fill":
                val = i[:3]
            ptext.append(val + " = ")

        line1_cv_score_text = []
        for i in key1_value:
            line1_cv_score_text.append(ptext[1] + i)

        line0_cv_score_text = []
        for i in key0_value:
            line0_cv_score_text.append(ptext[0] + i)

        diff_score_neg = 1 - diff_score
        # Store information in dictionary for bokeh
        data = dict(
            key0_value=key1_value,
            key1_value=key0_value,
            full_score=full_score,
            cv_score=cv_score,
            diff_score=diff_score,
            diff_heat_score=diff_heat_score,
            diff_score_neg=diff_score_neg,
            full_alpha=full_alpha,
            cv_alpha=cv_alpha,
            diff_alpha=diff_alpha,
            line_key0_value=line_key0_value,
            line_key1_value=line_key1_value,
            line0_full=line0_full,
            line0_cv=line0_cv,
            line1_full=line1_full,
            line1_cv=line1_cv,
            full_text=full_text,
            cv_text=cv_text,
            diff_text=diff_text,
            key0_value_text=key0_value_text,
            key1_value_text=key1_value_text,
            line0_cv_text=line0_cv_text,
            line1_cv_text=line1_cv_text,
            line1_cv_score_text=line1_cv_score_text,
            line1_full_text=line1_full_text,
            line0_full_text=line0_full_text,
            line0_cv_score_text=line0_cv_score_text,
        )

        if self.n_mc > 1:
            data["lower_ci_full"] = lower_ci_full
            data["upper_ci_full"] = upper_ci_full
            data["lower_ci_cv"] = lower_ci_cv
            data["upper_ci_cv"] = lower_ci_cv
            data["monte_line_key1_value"] = monte_line_key1_value
            data["monte_line_key0_value"] = monte_line_key0_value
            data["monte_line1_full"] = monte_line1_full
            data["monte_line1_cv"] = monte_line1_cv
            data["monte_line0_full"] = monte_line0_full
            data["monte_line0_cv"] = monte_line0_cv
        source = ColumnDataSource(data=data)

        # Heatmap FULL
        p1 = figure(title=full_title, tools="tap, save", x_range=key0_unique, y_range=key1_unique, x_axis_label=param_keys_axis[0], y_axis_label=param_keys_axis[1])

        p1_render = p1.rect("key1_value", "key0_value", 0.9, 0.9, color="red", alpha="full_alpha", line_color=None, source=source)

        p1_render.selection_glyph = Rect(fill_color="red", fill_alpha="full_alpha", line_width=int(3 * scale), line_color="black")
        p1_render.nonselection_glyph.fill_alpha = "full_alpha"
        p1_render.nonselection_glyph.fill_color = "red"
        p1_render.nonselection_glyph.line_color = "white"

        # Heatmap CV
        p2 = figure(title=cv_title, tools="tap, save", x_range=key0_unique, y_range=key1_unique, x_axis_label=param_keys_axis[0], y_axis_label=param_keys_axis[1])

        p2_render = p2.rect("key1_value", "key0_value", 0.9, 0.9, color="blue", alpha="cv_alpha", line_color=None, source=source)

        p2_render.selection_glyph = Rect(fill_color="blue", fill_alpha="cv_alpha", line_width=int(3 * scale), line_color="black")
        p2_render.nonselection_glyph.fill_alpha = "cv_alpha"
        p2_render.nonselection_glyph.fill_color = "blue"
        p2_render.nonselection_glyph.line_color = "white"

        # Heatmap Diff
        p3 = figure(title=diff_heat_title, tools="tap, save", x_range=key0_unique, y_range=key1_unique, x_axis_label=param_keys_axis[0], y_axis_label=param_keys_axis[1])

        p3_render = p3.rect("key1_value", "key0_value", 0.9, 0.9, color="green", alpha="diff_alpha", line_color=None, source=source)

        p3_render.selection_glyph = Rect(fill_color="green", fill_alpha="diff_alpha", line_width=int(3 * scale), line_color="black")
        p3_render.nonselection_glyph.fill_alpha = "diff_alpha"
        p3_render.nonselection_glyph.fill_color = "green"
        p3_render.nonselection_glyph.line_color = "white"

        # Extra for heatmaps
        p1.plot_width = int(320 * scale)
        p1.plot_height = int(257 * scale)
        p1.grid.grid_line_color = None
        p1.axis.axis_line_color = None
        p1.axis.major_tick_line_color = None
        p1.axis.major_label_text_font_size = str(8 * scale) + "pt"
        p1.axis.major_label_standoff = 0
        p1.xaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p1.yaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p1.title.text_font_size = str(14 * scale) + "pt"
        p1.xaxis.major_label_orientation = math.radians(heatmap_xaxis_rotate)

        p2.plot_width = int(320 * scale)
        p2.plot_height = int(257 * scale)
        p2.grid.grid_line_color = None
        p2.axis.axis_line_color = None
        p2.axis.major_tick_line_color = None
        p2.axis.major_label_text_font_size = str(8 * scale) + "pt"
        p2.axis.major_label_standoff = 0
        p2.xaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p2.yaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p2.title.text_font_size = str(14 * scale) + "pt"
        p2.xaxis.major_label_orientation = math.radians(heatmap_xaxis_rotate)

        p3.plot_width = int(320 * scale)
        p3.plot_height = int(257 * scale)
        p3.grid.grid_line_color = None
        p3.axis.axis_line_color = None
        p3.axis.major_tick_line_color = None
        p3.axis.major_label_text_font_size = str(8 * scale) + "pt"
        p3.axis.major_label_standoff = 0
        p3.xaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p3.yaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p3.title.text_font_size = str(14 * scale) + "pt"
        p3.xaxis.major_label_orientation = math.radians(heatmap_xaxis_rotate)

        text = False
        # Adding text to heatmaps
        if text is True:
            # if heatmap rect is dark, use light text and vise versa
            color_mapper_diff = LinearColorMapper(palette=["#000000", "#010101", "#fdfdfd", "#fefefe", "#ffffff"], low=0, high=1)

            label1 = LabelSet(x="key0_value", y="key1_value", text="full_text", level="glyph", x_offset=-10 * scale, y_offset=-10 * scale, source=source, render_mode="canvas", text_font_size=str(7.5 * scale) + "pt", text_color={"field": "full_alpha", "transform": color_mapper_diff})

            label2 = LabelSet(x="key0_value", y="key1_value", text="cv_text", level="glyph", x_offset=-10 * scale, y_offset=-10 * scale, source=source, render_mode="canvas", text_font_size=str(7.5 * scale) + "pt", text_color={"field": "cv_alpha", "transform": color_mapper_diff})

            label3 = LabelSet(x="key0_value", y="key1_value", text="diff_text", level="glyph", x_offset=-10 * scale, y_offset=-10 * scale, source=source, render_mode="canvas", text_font_size=str(7.5 * scale) + "pt", text_color={"field": "diff_alpha", "transform": color_mapper_diff})

            p1.add_layout(label1)
            p2.add_layout(label2)
            p3.add_layout(label3)

        p1.add_tools(HoverTool(renderers=[p1_render, p2_render, p3_render], tooltips=[(full_title, "@full_text")]))
        p2.add_tools(HoverTool(renderers=[p1_render, p2_render, p3_render], tooltips=[(cv_title, "@cv_text")]))
        p3.add_tools(HoverTool(renderers=[p1_render, p2_render, p3_render], tooltips=[(diff_title, "@diff_text")]))

        sc_title = diff_title + " vs. " + cv_title
        # Scatterplot
        p4 = figure(title=sc_title, x_axis_label=cv_title, y_axis_label=diff_title, tools="tap,pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select", x_range=(min(cv_score) - 0.03, max(cv_score) + 0.03), y_range=(min(diff_score) - 0.03, max(diff_score) + 0.03))

        color_mapper_scatter = LinearColorMapper(palette="Inferno256", low=0, high=1)

        p4_render = p4.circle("cv_score", "diff_score", size=8 * scale, alpha=0.6, color="green", source=source)
        p4_render.selection_glyph = Circle(fill_color="green", line_width=int(2 * scale), line_color="black")
        p4_render.nonselection_glyph.fill_color = "green"
        p4_render.nonselection_glyph.fill_alpha = 0.4
        p4_render.nonselection_glyph.line_color = "white"
        p4.add_tools(HoverTool(renderers=[p4_render], tooltips=[(full_title, "@full_text"), (cv_title, "@cv_text"), (diff_title, "@diff_text")]))

        p4.plot_width = int(320 * scale)
        p4.plot_height = int(257 * scale)
        p4.axis.major_label_text_font_size = str(8 * scale) + "pt"
        p4.xaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p4.yaxis.axis_label_text_font_size = str(12 * scale) + "pt"
        p4.title.text_font_size = str(14 * scale) + "pt"

        # Line plot 1
        l1_range_special = []
        if len(key0_unique) > 2:
            l1_range_special.append([" "])
        if len(key0_unique) > 5:
            l1_range_special.append([l1_range_special[-1][0] + " "])
        if len(key0_unique) > 8:
            l1_range_special.append([l1_range_special[-1][0] + " "])
        another_val = len(key0_unique) - 8
        if another_val > 0:
            for i in range(another_val):
                if i % 3 == 0:
                    l1_range_special.append([l1_range_special[-1][0] + " "])
        l1_xrange = pd.unique(key0_unique)
        l1_xrange2 = np.append(l1_xrange, l1_range_special)
        l1_title = y_axis_text + " over " + param_keys_title[0]

        y_range_min = min(cv_score) - min(cv_score) * 0.1
        y_range_max = max(full_score) + max(full_score) * 0.05

        p5 = figure(title=l1_title, x_axis_label=param_keys_axis[0], y_axis_label=y_axis_text, plot_width=int(320 * scale), plot_height=int(257 * scale), x_range=l1_xrange2, tools="pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select", y_range=(y_range_min, y_range_max))
        p5.quad(top=[1000], bottom=[-1000], left=[l1_xrange[-1]], right=[1000], color="white")

        # p5.outline_line_color = "white"
        if self.n_mc > 1:
            p5_render_patch2 = p5.patches("monte_line_key0_value", "monte_line1_cv", alpha=0, color="blue", source=source)
            p5_render_patch2.selection_glyph = Patches(fill_alpha=0.2, fill_color="blue", line_color="white")
            p5_render_patch2.nonselection_glyph.fill_alpha = 0
            p5_render_patch2.nonselection_glyph.line_color = "white"
            # kfold monte-carlo does not have ci for R2
            if model is not "kfold":
                p5_render_patch1 = p5.patches("monte_line_key0_value", "monte_line1_full", alpha=0, color="red", source=source)
                p5_render_patch1.selection_glyph = Patches(fill_alpha=0.2, fill_color="red", line_color="white")
                p5_render_patch1.nonselection_glyph.fill_alpha = 0
                p5_render_patch1.nonselection_glyph.line_color = "white"

        p5_render_1 = p5.multi_line("line_key0_value", "line1_full", line_color="red", line_width=2 * scale, source=source)
        p5_render_1.selection_glyph = MultiLine(line_color="red", line_alpha=0.8, line_width=2 * scale)
        p5_render_1.nonselection_glyph.line_color = "red"
        p5_render_1.nonselection_glyph.line_alpha = 0.05 / len(key1_unique)

        p5_render_2 = p5.circle("key1_value", "full_score", line_color="red", fill_color="white", size=8 * scale, source=source, legend=full_legend)
        p5_render_2.selection_glyph = Circle(line_color="red", fill_color="white")
        p5_render_2.nonselection_glyph.line_color = "red"
        p5_render_2.nonselection_glyph.fill_color = "white"
        p5_render_2.nonselection_glyph.line_alpha = 0.7 / len(key1_unique)

        p5_render_3 = p5.multi_line("line_key0_value", "line1_cv", line_color="blue", line_width=2 * scale, source=source)
        p5_render_3.selection_glyph = MultiLine(line_color="blue", line_alpha=0.8, line_width=2 * scale)
        p5_render_3.nonselection_glyph.line_color = "blue"
        p5_render_3.nonselection_glyph.line_alpha = 0.05 / len(key1_unique)

        p5_render_4 = p5.circle("key1_value", "cv_score", line_color="blue", fill_color="white", size=8 * scale, source=source, legend=cv_legend)
        p5_render_4.selection_glyph = Circle(line_color="blue", fill_color="white")
        p5_render_4.nonselection_glyph.line_color = "blue"
        p5_render_4.nonselection_glyph.fill_color = "white"
        p5_render_4.nonselection_glyph.line_alpha = 0.7 / len(key1_unique)

        # text
        text_here = 8 * scale
        text_line_font = str(text_here) + "pt"
        p5_render_5 = p5.text(x="key1_value_text", y="line1_cv_text", text="line1_cv_score_text", source=source, text_font_size=text_line_font, text_color="blue", x_offset=8 * scale, y_offset=6 * scale, text_alpha=0)
        p5_render_5.selection_glyph = Text(text_color="blue", text_alpha=1, text_font_size=text_line_font)
        p5_render_5.nonselection_glyph.text_alpha = 0
        p5_render_6 = p5.text(x="key1_value_text", y="line1_full_text", text="line1_cv_score_text", source=source, text_font_size=text_line_font, text_color="red", x_offset=8 * scale, y_offset=6 * scale, text_alpha=0)
        p5_render_6.selection_glyph = Text(text_color="red", text_alpha=1, text_font_size=text_line_font)
        p5_render_6.nonselection_glyph.text_alpha = 0

        p5.add_tools(HoverTool(renderers=[p5_render_2], tooltips=[(full_title, "@full_text")]))

        p5.add_tools(HoverTool(renderers=[p5_render_4], tooltips=[(cv_title, "@cv_text")]))

        p5.add_tools(TapTool(renderers=[p5_render_2, p5_render_4]))

        # Line plot 2
        l2_range_special = []
        if len(key1_unique) > 2:
            l2_range_special.append([" "])
        if len(key1_unique) > 5:
            l2_range_special.append([l2_range_special[-1][0] + " "])
        if len(key1_unique) > 8:
            l2_range_special.append([l2_range_special[-1][0] + " "])
        another_val = len(key1_unique) - 8
        if another_val > 0:
            for i in range(another_val):
                if i % 3 == 0:
                    l2_range_special.append([l2_range_special[-1][0] + " "])
        l2_xrange = pd.unique(key1_unique)
        l2_xrange2 = np.append(l2_xrange, l2_range_special)
        l1_title = y_axis_text + " over " + param_keys_title[0]
        l2_title = y_axis_text + " over " + param_keys_title[1]
        y_range_min = min(cv_score) - min(cv_score) * 0.1
        y_range_max = max(full_score) + max(full_score) * 0.05
        p6 = figure(title=l2_title, x_axis_label=param_keys_axis[1], y_axis_label=y_axis_text, plot_width=int(320 * scale), plot_height=int(257 * scale), x_range=l2_xrange2, tools="tap,pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select", y_range=(y_range_min, y_range_max))
        p6.quad(top=[1000], bottom=[-1000], left=[l2_xrange[-1]], right=[1000], color="white")

        if self.n_mc > 1:
            p6_render_patch2 = p6.patches("monte_line_key1_value", "monte_line0_cv", alpha=0, color="blue", source=source)
            p6_render_patch2.selection_glyph = Patches(fill_alpha=0.1, fill_color="blue", line_color="white")
            p6_render_patch2.nonselection_glyph.fill_alpha = 0
            p6_render_patch2.nonselection_glyph.line_color = "white"
            # kfold monte-carlo does not have ci for R2
            if model is not "kfold":
                p6_render_patch1 = p6.patches("monte_line_key1_value", "monte_line0_full", alpha=0, color="red", source=source)
                p6_render_patch1.selection_glyph = Patches(fill_alpha=0.1, fill_color="red", line_color="white")
                p6_render_patch1.nonselection_glyph.fill_alpha = 0
                p6_render_patch1.nonselection_glyph.line_color = "white"

        p6_render_1 = p6.multi_line("line_key1_value", "line0_full", line_color="red", line_width=2 * scale, source=source, legend=full_legend)
        p6_render_1.selection_glyph = MultiLine(line_color="red", line_alpha=0.8, line_width=2 * scale)
        p6_render_1.nonselection_glyph.line_color = "red"
        p6_render_1.nonselection_glyph.line_alpha = 0.05 / len(key0_unique)

        p6_render_2 = p6.circle("key0_value", "full_score", line_color="red", fill_color="white", size=8 * scale, source=source)
        p6_render_2.selection_glyph = Circle(line_color="red", fill_color="white")
        p6_render_2.nonselection_glyph.line_color = "red"
        p6_render_2.nonselection_glyph.fill_color = "white"
        p6_render_2.nonselection_glyph.line_alpha = 0.7 / len(key0_unique)

        p6_render_3 = p6.multi_line("line_key1_value", "line0_cv", line_color="blue", line_width=2 * scale, source=source, legend=cv_legend)
        p6_render_3.selection_glyph = MultiLine(line_color="blue", line_alpha=0.8, line_width=2 * scale)
        p6_render_3.nonselection_glyph.line_color = "blue"
        p6_render_3.nonselection_glyph.line_alpha = 0.05 / len(key0_unique)

        p6_render_4 = p6.circle("key0_value", "cv_score", line_color="blue", fill_color="white", size=8 * scale, source=source)
        p6_render_4.selection_glyph = Circle(line_color="blue", fill_color="white")
        p6_render_4.nonselection_glyph.line_color = "blue"
        p6_render_4.nonselection_glyph.fill_color = "white"
        p6_render_4.nonselection_glyph.line_alpha = 0.7 / len(key0_unique)

        # Text
        text_here = 8 * scale
        text_line_font = str(text_here) + "pt"
        p6_render_5 = p6.text(x="key0_value_text", y="line0_cv_text", text="line0_cv_score_text", source=source, text_font_size=text_line_font, text_color="blue", x_offset=8 * scale, y_offset=6 * scale, text_alpha=0)
        p6_render_5.selection_glyph = Text(text_color="blue", text_alpha=1, text_font_size=text_line_font)
        p6_render_5.nonselection_glyph.text_alpha = 0
        p6_render_6 = p6.text(x="key0_value_text", y="line0_full_text", text="line0_cv_score_text", source=source, text_font_size=text_line_font, text_color="red", x_offset=8 * scale, y_offset=6 * scale, text_alpha=0)
        p6_render_6.selection_glyph = Text(text_color="red", text_alpha=1, text_font_size=text_line_font)
        p6_render_6.nonselection_glyph.text_alpha = 0

        p6.add_tools(HoverTool(renderers=[p6_render_2], tooltips=[("AUC_full", "@full_text")]))

        p6.add_tools(HoverTool(renderers=[p6_render_4], tooltips=[("AUC_CV", "@cv_text")]))

        p6.add_tools(TapTool(renderers=[p6_render_2, p6_render_4]))

        # Figure: legend
        if legend == None or legend == False:
            p5.legend.visible = False
            p6.legend.visible = False
        else:
            p5.legend.location = legend
            p6.legend.location = legend

        # Center title
        if title_align == "center":
            p1.title.align = "center"
            p2.title.align = "center"
            p3.title.align = "center"
            p4.title.align = "center"
            p5.title.align = "center"
            p6.title.align = "center"

        fig = gridplot([[p1, p2, p3], [p4, p5, p6]], merge_tools=True, toolbar_location="left", toolbar_options=dict(logo=None))

        p5.xaxis.major_label_orientation = math.radians(heatmap_xaxis_rotate)
        p6.xaxis.major_label_orientation = math.radians(heatmap_xaxis_rotate)

        p1.title.text_font_size = str(12 * scale) + "pt"
        p2.title.text_font_size = str(12 * scale) + "pt"
        p3.title.text_font_size = str(12 * scale) + "pt"
        p4.title.text_font_size = str(12 * scale) + "pt"
        p5.title.text_font_size = str(12 * scale) + "pt"
        p6.title.text_font_size = str(12 * scale) + "pt"

        p1.xaxis.axis_label_text_font_size = str(10 * scale) + "pt"
        p2.xaxis.axis_label_text_font_size = str(10 * scale) + "pt"
        p3.xaxis.axis_label_text_font_size = str(10 * scale) + "pt"
        p4.xaxis.axis_label_text_font_size = str(10 * scale) + "pt"
        p5.xaxis.axis_label_text_font_size = str(10 * scale) + "pt"
        p6.xaxis.axis_label_text_font_size = str(10 * scale) + "pt"

        p1.yaxis.axis_label_text_font_size = str(10 * scale) + "pt"
        p2.yaxis.axis_label_text_font_size = str(10 * scale) + "pt"
        p3.yaxis.axis_label_text_font_size = str(10 * scale) + "pt"
        p4.yaxis.axis_label_text_font_size = str(10 * scale) + "pt"
        p5.yaxis.axis_label_text_font_size = str(10 * scale) + "pt"
        p6.yaxis.axis_label_text_font_size = str(10 * scale) + "pt"

        return fig
