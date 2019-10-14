import numpy as np
import pandas as pd
from bokeh.models import Band, HoverTool
from tqdm import tqdm
import timeit
from copy import deepcopy
from scipy.stats import norm
import time
import multiprocessing
from joblib import Parallel, delayed
from copy import deepcopy, copy
from bokeh.plotting import ColumnDataSource, figure
import scipy
from scipy import interp
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils import resample
from ..utils import binary_metrics, dict_median
#import cimcb.model


def roc_plot_boot2(ypred_ib, ypred_oob, Y, bootstat, bootidx, bootstat_oob, bootidx_oob, stat, width=320, height=315, label_font_size="10pt", xlabel="1-Specificity", ylabel="Sensitivity", legend=True, parametric=True, bc=True):

    auc_check = roc_auc_score(Y, stat)
    if auc_check > 0.5:
        pos = 1
    else:
        pos = 0
    #print(auc_check)

    fpr_linspace = np.linspace(0, 1, 100)
    fpr = fpr_linspace
    fpr_linspace_ib = fpr_linspace
    tpr_boot = []
    boot_stats = []
    auc_ib = []
    for i in range(len(bootidx)):
        # Resample and get tpr, fpr
        Yscore_res = bootstat[i]
        Ytrue = Y[bootidx[i]]
        fpr_res, tpr_res, threshold_res = metrics.roc_curve(Ytrue, Yscore_res, pos_label=pos, drop_intermediate=False)
        auc_ib.append(metrics.auc(fpr_res, tpr_res))

        # Drop intermediates when fpr=0
        tpr0_res = tpr_res[fpr_res == 0][-1]
        tpr_res = np.concatenate([[tpr0_res], tpr_res[fpr_res > 0]])
        fpr_res = np.concatenate([[0], fpr_res[fpr_res > 0]])

        # Vertical averaging... use closest fpr_res to fpr, and append the corresponding tpr
        idx = [np.abs(i - fpr_res).argmin() for i in fpr]
        tpr_list = tpr_res[idx]
        tpr_boot.append(tpr_list)

    # Get CI for tpr
    tprs_ib_low = np.percentile(tpr_boot, 2.5, axis=0)
    tprs_ib_upp = np.percentile(tpr_boot, 97.5, axis=0)
    tprs_ib_mid = np.percentile(tpr_boot, 50, axis=0)

    auc_ib_low = np.percentile(auc_ib, 2.5, axis=0)
    auc_ib_upp = np.percentile(auc_ib, 97.5, axis=0)
    auc_ib_ci = np.percentile(auc_ib, 97.5, axis=0) - np.percentile(auc_ib, 2.5, axis=0)

    Yscore_res = stat
    Ytrue = Y
    fpr_res, tpr_res, threshold_res = metrics.roc_curve(Y, stat, pos_label=pos, drop_intermediate=False)
    auc_ib_mid = metrics.auc(fpr_res, tpr_res)
    # Drop intermediates when fpr=0
    tpr0_res = tpr_res[fpr_res == 0][-1]
    tpr_res = np.concatenate([[tpr0_res], tpr_res[fpr_res > 0]])
    fpr_res = np.concatenate([[0], fpr_res[fpr_res > 0]])

    # Vertical averaging... use closest fpr_res to fpr, and append the corresponding tpr
    idx = [np.abs(i - fpr_res).argmin() for i in fpr]
    tpr_list = tpr_res[idx]
    tpr_list = np.array(tpr_list)
    #tpr_list = np.insert(tpr_list, 0, 0)
    tprs_train = deepcopy(tpr_list)
    tprs_diff =   tprs_ib_mid - tprs_train

    # Get CI for tpr
    tprs_ib_low = np.percentile(tpr_boot, 2.5, axis=0)
    tprs_ib_upp = np.percentile(tpr_boot, 97.5, axis=0)
    tprs_ib_mid = np.percentile(tpr_boot, 50, axis=0)

    if parametric is not 'null':
        tprs_ib_low = tprs_ib_low - tprs_diff
        tprs_ib_upp_old = deepcopy(tprs_ib_upp)
        tprs_ib_upp = tprs_ib_upp - tprs_diff
        tprs_ib_mid = tprs_ib_mid - tprs_diff
    if parametric is "parametric":
        low = tprs_ib_mid - tprs_ib_low
        tprs_ib_upp = tprs_ib_mid + low
        tprs_ib_upp[tprs_ib_upp > 1] = 1
    elif parametric is "nonparametric":
        low = tprs_ib_mid - tprs_ib_low
        tprs_ib_upp = tprs_ib_mid + low
        tprs_ib_upp[tprs_ib_upp > 1] = 1
        tprs_ib_upp[tprs_ib_upp_old >= 1] = 1

        for i in range(len(tprs_ib_upp)):
            if i > 1:
                if tprs_ib_upp[i] < tprs_ib_upp[i - 1]:
                    tprs_ib_upp[i] = tprs_ib_upp[i - 1]
    elif parametric is 'null':
        pass
    elif parametric is 'test':
        # Lets do a proper bias-correct based on AUC
        auc_boot = auc_ib
        auc_stat = auc_ib_mid
        nboot = len(auc_boot)
        zalpha = norm.ppf(0.05 / 2)
        obs = auc_stat  # Observed mean
        meansum = 0
        for j in range(len(auc_boot)):
            if auc_boot[j] >= obs:
                meansum = meansum + 1
        prop = meansum / nboot  # Proportion of times boot mean > obs mean
        z0 = -norm.ppf(prop)
        # new alpha
        pct1 = 100 * norm.cdf((2 * z0 + zalpha))
        pct2 = 100 * norm.cdf((2 * z0 - zalpha))
        print("Testing... ({:.2f},{:.2f})".format(pct1, pct2))
        tprs_ib_low = np.percentile(tpr_boot, pct1, axis=0)
        tprs_ib_upp = np.percentile(tpr_boot, pct2, axis=0)
    elif parametric is 'test2':

        fpr_new, stat, _ = metrics.roc_curve(Y, stat, pos_label=pos, drop_intermediate=True)
        idx_new = []
        for j in fpr_new:
            x = min(range(len(fpr_linspace)), key=lambda i: abs(fpr_linspace[i]-j))
            idx_new.append(x)

        fpr_linspace_ib = fpr_new
        bootstat = []
        for i in tpr_boot:
            bootstat.append(i[idx_new])
        bootstat = np.array(bootstat)
        bootstat = bootstat.T
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
        tprs_ib_low = boot_ci[:,0]
        tprs_ib_upp = boot_ci[:,1]
        tprs_ib_mid = stat

    else:
        raise ValueError("bc must be 'parametric', 'nonparametric', or 'null'.")


    # Add the starting 0
    tprs_ib_low = np.insert(tprs_ib_low, 0, 0)
    tprs_ib_upp = np.insert(tprs_ib_upp, 0, 0)
    tprs_ib_mid = np.insert(tprs_ib_mid, 0, 0)

    auc_ib_ci_hover = [auc_ib_ci] * len(tprs_ib_low)
    auc_ib_med_hover = [auc_ib_mid] * len(tprs_ib_low)


    tpr_boot = []
    boot_stats = []
    auc_oob = []
    for i in range(len(bootidx_oob)):
        # Resample and get tpr, fpr
        Yscore_res = bootstat_oob[i]
        Ytrue = Y[bootidx_oob[i]]
        fpr_res, tpr_res, threshold_res = metrics.roc_curve(Ytrue, Yscore_res, pos_label=pos, drop_intermediate=False)
        auc_oob.append(metrics.auc(fpr_res, tpr_res))

        # Drop intermediates when fpr=0
        tpr0_res = tpr_res[fpr_res == 0][-1]
        tpr_res = np.concatenate([[tpr0_res], tpr_res[fpr_res > 0]])
        fpr_res = np.concatenate([[0], fpr_res[fpr_res > 0]])

        # Vertical averaging... use closest fpr_res to fpr, and append the corresponding tpr
        idx = [np.abs(i - fpr_res).argmin() for i in fpr]
        tpr_list = tpr_res[idx]
        tpr_boot.append(tpr_list)

    # Get CI for tpr
    tprs_oob_low= np.percentile(tpr_boot, 2.5, axis=0)
    tprs_oob_upp = np.percentile(tpr_boot, 97.5, axis=0)
    tprs_oob_mid = np.percentile(tpr_boot, 50, axis=0)

    # Add the starting 0
    tprs_oob_low = np.insert(tprs_oob_low, 0, 0)
    tprs_oob_upp = np.insert(tprs_oob_upp, 0, 0)
    tprs_oob_mid = np.insert(tprs_oob_mid, 0, 0)

    fpr_linspace = np.insert(fpr_linspace, 0, 0)
    fpr_linspace_ib = np.insert(fpr_linspace_ib, 0, 0)

    # Get CI for cv
    auc_oob_lowci = np.percentile(auc_oob, 2.5, axis=0)
    auc_oob_uppci = np.percentile(auc_oob, 97.5, axis=0)
    auc_oob_medci = np.percentile(auc_oob, 50, axis=0)
    auc_oob_ci = (auc_oob_uppci - auc_oob_lowci) / 2
    auc_oob_ci_hover = [auc_oob_ci] * len(tprs_oob_upp)
    auc_oob_med_hover = [auc_oob_medci] * len(tprs_oob_upp)

    auc_ib_median = metrics.auc(fpr_linspace_ib, tprs_ib_mid)
    auc_oob_median = metrics.auc(fpr_linspace, tprs_oob_mid)
    roc_title = "AUC: {} ({})".format(np.round(auc_ib_median, 2), np.round(auc_oob_median, 2))

    # specificity and ci-interval for HoverTool
    spec = 1 - fpr_linspace
    ci = (tprs_ib_upp - tprs_ib_low) / 2

    data = {"x": fpr_linspace_ib, "y": tprs_ib_mid, "lowci": tprs_ib_low, "uppci": tprs_ib_upp, "spec": spec, "ci": ci, "auc_ib_ci_hover":auc_ib_ci_hover, "auc_ib_med_hover":auc_ib_med_hover}

    source = ColumnDataSource(data=data)

    fig = figure(title="", plot_width=width, plot_height=height, x_axis_label=xlabel, y_axis_label=ylabel, x_range=(-0.06, 1.06), y_range=(-0.06, 1.06))
    fig.line([0, 1], [0, 1], color="black", line_dash="dashed", line_width=4, legend="Equal Distribution Line")
    figline = fig.line("x", "y", color="green", line_width=4, alpha=0.6, legend="IB (AUC = {:.2f} +/- {:.2f})".format(auc_ib_mid,auc_ib_ci), source=source)
    fig.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111} (+/- @ci{1.111})"), ("AUC", "@auc_ib_med_hover{1.111} (+/- @auc_ib_ci_hover{1.111})")]))

    # Figure: add 95CI band
    figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="green", source=source)
    fig.add_layout(figband)

        # specificity and ci-interval for HoverTool
    spec2 = 1 - fpr_linspace
    ci2 = (tprs_oob_upp - tprs_oob_low) / 2

    data2 = {"x": fpr_linspace, "y": tprs_oob_mid, "lowci": tprs_oob_low, "uppci": tprs_oob_upp, "spec": spec2, "ci": ci2, "auc_oob_ci_hover":auc_oob_ci_hover, "auc_oob_med_hover":auc_oob_med_hover}

    source2 = ColumnDataSource(data=data2)

    figline2 = fig.line("x", "y", color="orange", line_width=4, alpha=0.6, legend="OOB (AUC = {:.2f} +/- {:.2f})".format(auc_oob_medci, auc_oob_ci), source=source2)
    fig.add_tools(HoverTool(renderers=[figline2], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111} (+/- @ci{1.111})"), ("AUC", "@auc_oob_med_hover{1.111} (+/- @auc_oob_ci_hover{1.111})")]))

    # Figure: add 95CI band
    figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="orange", source=source2)
    fig.add_layout(figband)


    fig.legend.location = "bottom_right"
    fig.legend.label_text_font_size = "10pt"
    if legend is False:
        fig.legend.visible = False

    return fig


def roc_plot_cv(Y_predfull, Y_predcv, Ytrue, width=450, height=350, xlabel="1-Specificity", ylabel="Sensitivity", legend=True, label_font_size="13pt", show_title=True, title_font_size="13pt", title=""):

    auc_check = roc_auc_score(Ytrue, Y_predfull)
    if auc_check > 0.5:
        pos = 1
    else:
        pos = 0

    fprf, tprf, thresholdf = metrics.roc_curve(Ytrue, Y_predfull, pos_label=pos, drop_intermediate=False)
    specf = 1 - fprf
    auc_full = metrics.auc(fprf, tprf)
    auc_full_hover = [auc_full] * len(tprf)

    # Figure
    data = {"x": fprf, "y": tprf, "spec": specf, "aucfull": auc_full_hover}

    source = ColumnDataSource(data=data)
    fig = figure(title=title, plot_width=width, plot_height=height, x_axis_label=xlabel, y_axis_label=ylabel, x_range=(-0.06, 1.06), y_range=(-0.06, 1.06))

    # Figure: add line
    fig.line([0, 1], [0, 1], color="black", line_dash="dashed", line_width=2.5, legend="Equal Distribution Line")
    figline = fig.line("x", "y", color="green", line_width=3.5, alpha=0.6, legend="FULL (AUC = {:.2f})".format(auc_full), source=source)
    fig.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111}"), ("AUC", "@aucfull")]))

    # ADD CV
    # bootstrap using vertical averaging

    # fpr, tpr with drop_intermediates for fpr = 0 (useful for plot... since we plot specificity on x-axis, we don't need intermediates when fpr=0)
    fpr = fprf
    tpr = tprf
    tpr0 = tpr[fpr == 0][-1]
    tpr = np.concatenate([[tpr0], tpr[fpr > 0]])
    fpr = np.concatenate([[0], fpr[fpr > 0]])
    tpr_boot = []
    boot_stats = []
    auc_cv = []
    for i in range(len(Y_predcv)):
        # Resample and get tpr, fpr
        Yscore_res = Y_predcv[i]
        fpr_res, tpr_res, threshold_res = metrics.roc_curve(Ytrue, Yscore_res, pos_label=pos, drop_intermediate=False)
        auc_cv.append(metrics.auc(fpr_res, tpr_res))
        # Drop intermediates when fpr=0
        tpr0_res = tpr_res[fpr_res == 0][-1]
        tpr_res = np.concatenate([[tpr0_res], tpr_res[fpr_res > 0]])
        fpr_res = np.concatenate([[0], fpr_res[fpr_res > 0]])

        # Vertical averaging... use closest fpr_res to fpr, and append the corresponding tpr
        idx = [np.abs(i - fpr_res).argmin() for i in fpr]
        tpr_list = tpr_res[idx]
        tpr_boot.append(tpr_list)

    # Get CI for tpr
    tpr_lowci = np.percentile(tpr_boot, 2.5, axis=0)
    tpr_uppci = np.percentile(tpr_boot, 97.5, axis=0)
    tpr_medci = np.percentile(tpr_boot, 50, axis=0)



    # Add the starting 0
    tpr = np.insert(tpr, 0, 0)
    fpr = np.insert(fpr, 0, 0)
    tpr_lowci = np.insert(tpr_lowci, 0, 0)
    tpr_uppci = np.insert(tpr_uppci, 0, 0)
    tpr_medci = np.insert(tpr_medci, 0, 0)

    # Get CI for cv
    auc_lowci = np.percentile(auc_cv, 2.5, axis=0)
    auc_uppci = np.percentile(auc_cv, 97.5, axis=0)
    auc_medci = np.percentile(auc_cv, 50, axis=0)
    auc_ci = (auc_uppci - auc_lowci) / 2
    auc_ci_hover = [auc_ci] * len(tpr_medci)
    auc_med_hover = [auc_medci] * len(tpr_medci)

    # Concatenate tpr_ci
    tpr_ci = np.array([tpr_lowci, tpr_uppci, tpr_medci])
    # specificity and ci-interval for HoverTool
    spec2 = 1 - fpr
    ci2 = (tpr_uppci - tpr_lowci) / 2

    data2 = {"x": fpr, "y": tpr_medci, "lowci": tpr_lowci, "uppci": tpr_uppci, "spec": spec2, "ci": ci2, "auc_ci_hover":auc_ci_hover, "auc_med_hover":auc_med_hover}

    source2 = ColumnDataSource(data=data2)

    figline = fig.line("x", "y", color="orange", line_width=3.5, alpha=0.6, legend="CV (AUC = {:.2f} +/- {:.2f})".format(auc_medci, auc_ci,), source=source2)
    fig.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111} (+/- @ci{1.111})"), ("AUC", "@auc_med_hover{1.111} (+/- @auc_ci_hover{1.111})")]))

    # Figure: add 95CI band
    figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="orange", source=source2)
    fig.add_layout(figband)

    # Change font size
    if show_title is True:
        fig.title.text = "AUC FULL ({}) & AUC CV ({} +/- {})".format(np.round(auc_full,2), np.round(auc_medci,2), np.round(auc_ci,2))
        fig.title.text_font_size = title_font_size
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    # Edit legend
    fig.legend.location = "bottom_right"
    # fig.legend.label_text_font_size = "1pt"
    # fig.legend.label_text_font = "1pt"
    if legend is False:
        fig.legend.visible = False

    return fig


def roc_plot(fpr, tpr, tpr_ci, median=False, width=450, height=350, xlabel="1-Specificity", ylabel="Sensitivity", legend=True, label_font_size="13pt", title="", errorbar=False, roc2=False, fpr2=None, tpr2=None, tpr_ci2=None):
    """Creates a rocplot using Bokeh.

    Parameters
    ----------
    fpr : array-like, shape = [n_samples]
        False positive rates. Calculate using roc_calculate.

    tpr : array-like, shape = [n_samples]
        True positive rates. Calculate using roc_calculate.

    tpr_ci : array-like, shape = [n_samples, 2]
        True positive rates 95% confidence intervals [lowci, uppci]. Calculate using roc_calculate.
    """

    # Get CI
    tpr_lowci = tpr_ci[0]
    tpr_uppci = tpr_ci[1]
    tpr_medci = tpr_ci[2]
    auc = metrics.auc(fpr, tpr)

    if median == True:
        tpr = tpr_medci

    # specificity and ci-interval for HoverTool
    spec = 1 - fpr
    ci = (tpr_uppci - tpr_lowci) / 2

    if roc2 is True:
        tpr_lowci2 = tpr_ci2[0]
        tpr_uppci2 = tpr_ci2[1]
        tpr_medci2 = tpr_ci2[2]
        auc = metrics.auc(fpr2, tpr2)

        if median == True:
            tpr2 = tpr_medci2

        # specificity and ci-interval for HoverTool
        spec2 = 1 - fpr2
        ci2 = (tpr_uppci2 - tpr_lowci2) / 2

    # Figure
    data = {"x": fpr, "y": tpr, "lowci": tpr_lowci, "uppci": tpr_uppci, "spec": spec, "ci": ci}

    source = ColumnDataSource(data=data)
    fig = figure(title=title, plot_width=width, plot_height=height, x_axis_label=xlabel, y_axis_label=ylabel, x_range=(-0.06, 1.06), y_range=(-0.06, 1.06))

    # Figure: add line
    if roc2 is False:
        fig.line([0, 1], [0, 1], color="black", line_dash="dashed", line_width=2.5, legend="Equal Distribution Line")
        figline = fig.line("x", "y", color="green", line_width=3.5, alpha=0.6, legend="ROC Curve (Train)", source=source)
        fig.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111} (+/- @ci{1.111})")]))

        # Figure: add 95CI band
        figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="green", source=source)
        fig.add_layout(figband)

    if roc2 is True:

        fig.line([0, 1], [0, 1], color="black", line_dash="dashed", line_width=2.5, legend="Equal Distribution Line")
        figline = fig.line("x", "y", color="green", line_width=3.5, alpha=0.6, legend="ROC Curve (Model)", source=source)
        fig.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111} (+/- @ci{1.111})")]))

        # Figure: add 95CI band
        figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="green", source=source)
        fig.add_layout(figband)

        data2 = {"x2": fpr2, "y2": tpr2, "lowci2": tpr_lowci2, "uppci2": tpr_uppci2, "spec2": spec2, "ci2": ci2}
        source2 = ColumnDataSource(data=data2)
        figline = fig.line("x2", "y2", color="orange", line_width=3.5, alpha=0.6, legend="ROC Curve (Bootstrap IB)", source=source2)
        fig.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity (Boot)", "@spec2{1.111}"), ("Sensitivity (Boot)", "@y2{1.111} (+/- @ci2{1.111})")]))

        # Figure: add 95CI band
        figband = Band(base="x2", lower="lowci2", upper="uppci2", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="orange", source=source2)
        fig.add_layout(figband)

    # Figure: add errorbar  spec =  1 - fpr
    if errorbar is not False:
        idx = np.abs(fpr - (1 - errorbar)).argmin()  # this find the closest value in fpr to errorbar fpr
        fpr_eb = fpr[idx]
        tpr_eb = tpr[idx]
        tpr_lowci_eb = tpr_lowci[idx]
        tpr_uppci_eb = tpr_uppci[idx]

        # Edge case: If this is a perfect roc curve, and specificity >= 1, make sure error_bar is at (0,1) not (0,0)
        if errorbar >= 1:
            for i in range(len(fpr)):
                if fpr[i] == 0 and tpr[i] == 1:
                    fpr_eb = 0
                    tpr_eb = 1
                    tpr_lowci_eb = 1
                    tpr_uppci_eb = 1

        roc_whisker_line = fig.multi_line([[fpr_eb, fpr_eb]], [[tpr_lowci_eb, tpr_uppci_eb]], line_alpha=1, line_color="black")
        roc_whisker_bot = fig.multi_line([[fpr_eb - 0.03, fpr_eb + 0.03]], [[tpr_lowci_eb, tpr_lowci_eb]], line_color="black")
        roc_whisker_top = fig.multi_line([[fpr_eb - 0.03, fpr_eb + 0.03]], [[tpr_uppci_eb, tpr_uppci_eb]], line_alpha=1, line_color="black")
        fig.circle([fpr_eb], [tpr_eb], size=8, fill_alpha=1, line_alpha=1, line_color="black", fill_color="white")

    # Change font size
    fig.title.text_font_size = "11pt"
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size
    fig.legend.label_text_font = "10pt"

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    # Edit legend
    fig.legend.location = "bottom_right"
    fig.legend.label_text_font_size = "10pt"
    if legend is False:
        fig.legend.visible = False

    return fig


def roc_calculate(Ytrue, Yscore, bootnum=1000, metric=None, val=None, parametric=True):
    """Calculates required metrics for the roc plot function (fpr, tpr, and tpr_ci).

    Parameters
    ----------
    Ytrue : array-like, shape = [n_samples]
        Binary label for samples (0s and 1s)

    Yscore : array-like, shape = [n_samples]
        Predicted y score for samples

    Returns
    ----------------------------------
    fpr : array-like, shape = [n_samples]
        False positive rates.

    tpr : array-like, shape = [n_samples]
        True positive rates.

    tpr_ci : array-like, shape = [n_samples, 2]
        True positive rates 95% confidence intervals [lowci, uppci].
    """

    # Get fpr, tpr
    try:
        fpr, tpr, threshold = metrics.roc_curve(Ytrue, Yscore, pos_label=1, drop_intermediate=False)
    except ValueError:
        raise ValueError("You need to lower the learning_rate! This is a common issue when using the ‘mean_squared_error’ loss function called exploding gradients. 'At an extreme, the values of weights can become so large as to overflow and result in NaN values' (REF: https://machinelearningmastery.com/exploding-gradients-in-neural-networks/).")

    # fpr, tpr with drop_intermediates for fpr = 0 (useful for plot... since we plot specificity on x-axis, we don't need intermediates when fpr=0)
    tpr0 = tpr[fpr == 0][-1]
    tpr = np.concatenate([[tpr0], tpr[fpr > 0]])
    fpr = np.concatenate([[0], fpr[fpr > 0]])

    # if metric is provided, calculate stats
    if metric is not None:
        specificity, sensitivity, threshold = get_spec_sens_cuttoff(Ytrue, Yscore, metric, val)
        stats = get_stats(Ytrue, Yscore, specificity, parametric)
        stats["val_specificity"] = specificity
        stats["val_sensitivity"] = specificity
        stats["val_cutoffscore"] = threshold

    # bootstrap using vertical averaging
    tpr_boot = []
    boot_stats = []
    for i in range(bootnum):
        # Resample and get tpr, fpr
        Ytrue_res, Yscore_res = resample(Ytrue, Yscore)
        fpr_res, tpr_res, threshold_res = metrics.roc_curve(Ytrue_res, Yscore_res, pos_label=1, drop_intermediate=False)

        # Drop intermediates when fpr=0
        tpr0_res = tpr_res[fpr_res == 0][-1]
        tpr_res = np.concatenate([[tpr0_res], tpr_res[fpr_res > 0]])
        fpr_res = np.concatenate([[0], fpr_res[fpr_res > 0]])

        # Vertical averaging... use closest fpr_res to fpr, and append the corresponding tpr
        idx = [np.abs(i - fpr_res).argmin() for i in fpr]
        tpr_list = tpr_res[idx]
        tpr_boot.append(tpr_list)

        # if metric is provided, calculate stats
        if metric is not None:
            stats_res = get_stats(Ytrue_res, Yscore_res, specificity, parametric)
            boot_stats.append(stats_res)

    # Get CI for bootstat
    if metric is not None:
        bootci_stats = {}
        for i in boot_stats[0].keys():
            stats_i = [k[i] for k in boot_stats]
            stats_i = np.array(stats_i)
            stats_i = stats_i[~np.isnan(stats_i)]  # Remove nans
            try:
                lowci = np.percentile(stats_i, 2.5)
                uppci = np.percentile(stats_i, 97.5)
            except IndexError:
                lowci = np.nan
                uppci = np.nan
            bootci_stats[i] = [lowci, uppci]

    # Get CI for tpr
    tpr_lowci = np.percentile(tpr_boot, 2.5, axis=0)
    tpr_uppci = np.percentile(tpr_boot, 97.5, axis=0)
    tpr_medci = np.percentile(tpr_boot, 50, axis=0)

    # Add the starting 0
    tpr = np.insert(tpr, 0, 0)
    fpr = np.insert(fpr, 0, 0)
    tpr_lowci = np.insert(tpr_lowci, 0, 0)
    tpr_uppci = np.insert(tpr_uppci, 0, 0)
    tpr_medci = np.insert(tpr_medci, 0, 0)

    # Concatenate tpr_ci
    tpr_ci = np.array([tpr_lowci, tpr_uppci, tpr_medci])

    if metric is None:
        return fpr, tpr, tpr_ci
    else:
        return fpr, tpr, tpr_ci, stats, bootci_stats


def roc_plot_boot(fpr_ib, tpr_ib_ci, fpr_oob, tpr_oob_ci, width=450, height=350, xlabel="1-Specificity", ylabel="Sensitivity", legend=True, label_font_size="13pt", title="", errorbar=False):
    """Creates a rocplot using Bokeh.

    Parameters
    ----------
    fpr : array-like, shape = [n_samples]
        False positive rates. Calculate using roc_calculate.

    tpr : array-like, shape = [n_samples]
        True positive rates. Calculate using roc_calculate.

    tpr_ci : array-like, shape = [n_samples, 2]
        True positive rates 95% confidence intervals [lowci, uppci]. Calculate using roc_calculate.
    """

    # Get CI
    tpr_ib = tpr_ib_ci[0]
    tpr = tpr_ib
    tpr_ib_lowci = tpr_ib_ci[1]
    tpr_ib_uppci = tpr_ib_ci[2]
    fpr = np.insert(fpr_ib[0], 0, 0)
    auc_ib = metrics.auc(fpr, tpr_ib)

    # add oob median CI
    tpr_oob = tpr_oob_ci[0]
    tpr_oob_lowci = tpr_oob_ci[1]
    tpr_oob_uppci = tpr_oob_ci[2]
    auc_oob = metrics.auc(fpr, tpr_oob)

    # specificity and ci-interval for HoverTool
    spec = 1 - fpr
    ci = (tpr_oob_uppci - tpr_oob_lowci) / 2

    # Figure
    data = {"x": fpr, "y": tpr_ib, "lowci": tpr_ib_lowci, "uppci": tpr_ib_uppci, "spec": spec, "ci": ci, "x_oob": fpr, "y_oob": tpr_oob, "lowci_oob": tpr_oob_lowci, "uppci_oob": tpr_oob_uppci}
    source = ColumnDataSource(data=data)
    fig = figure(title=title, plot_width=width, plot_height=height, x_axis_label=xlabel, y_axis_label=ylabel, x_range=(-0.06, 1.06), y_range=(-0.06, 1.06))

    # Figure: add line
    fig.line([0, 1], [0, 1], color="black", line_dash="dashed", line_width=2.5, legend="Equal Distribution Line")
    figline = fig.line("x", "y", color="green", line_width=3.5, alpha=0.6, legend="ROC Curve (IB)", source=source)
    fig.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111} (+/- @ci{1.111})")]))

    # Figure: add 95CI band
    figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="green", source=source)
    fig.add_layout(figband)

    # Figure: add oob line
    figline2 = fig.line("x_oob", "y_oob", color="red", line_width=3.5, alpha=0.6, legend="ROC Curve (OOB)", source=source)
    figband2 = Band(base="x_oob", lower="lowci_oob", upper="uppci_oob", level="underlay", fill_alpha=0.1, line_width=1, line_color="black", fill_color="red", source=source)
    fig.add_layout(figband2)

    # Figure: add errorbar  spec =  1 - fpr
    if errorbar is not False:
        idx = np.abs(fpr - (1 - errorbar)).argmin()  # this find the closest value in fpr to errorbar fpr
        fpr_eb = fpr[idx]
        tpr_eb = tpr[idx]
        tpr_lowci_eb = tpr_ib_lowci[idx]
        tpr_uppci_eb = tpr_ib_uppci[idx]

        # Edge case: If this is a perfect roc curve, and specificity >= 1, make sure error_bar is at (0,1) not (0,0)
        if errorbar >= 1:
            for i in range(len(fpr)):
                if fpr[i] == 0 and tpr[i] == 1:
                    fpr_eb = 0
                    tpr_eb = 1
                    tpr_lowci_eb = 1
                    tpr_uppci_eb = 1

        roc_whisker_line = fig.multi_line([[fpr_eb, fpr_eb]], [[tpr_lowci_eb, tpr_uppci_eb]], line_alpha=1, line_width=2, line_color="darkgreen")
        roc_whisker_bot = fig.multi_line([[fpr_eb - 0.03, fpr_eb + 0.03]], [[tpr_lowci_eb, tpr_lowci_eb]], line_width=2, line_color="darkgreen")
        roc_whisker_top = fig.multi_line([[fpr_eb - 0.03, fpr_eb + 0.03]], [[tpr_uppci_eb, tpr_uppci_eb]], line_width=2, line_alpha=1, line_color="darkgreen")
        fig.circle([fpr_eb], [tpr_eb], size=8, fill_alpha=1, line_alpha=1, line_color="black", fill_color="white")

    # Figure: add errorbar  spec =  1 - fpr
    if errorbar is not False:
        idx = np.abs(fpr - (1 - errorbar)).argmin()  # this find the closest value in fpr to errorbar fpr
        fpr_eb2 = fpr[idx]
        tpr_eb2 = tpr_oob[idx]
        tpr_lowci_eb2 = tpr_oob_lowci[idx]
        tpr_uppci_eb2 = tpr_oob_uppci[idx]

        # Edge case: If this is a perfect roc curve, and specificity >= 1, make sure error_bar is at (0,1) not (0,0)
        if errorbar >= 1:
            for i in range(len(fpr)):
                if fpr[i] == 0 and tpr_oob[i] == 1:
                    fpr_eb2 = 0
                    tpr_eb2 = 1
                    tpr_lowci_eb2 = 1
                    tpr_uppci_eb2 = 1

        roc_whisker_line2 = fig.multi_line([[fpr_eb2, fpr_eb2]], [[tpr_lowci_eb2, tpr_uppci_eb2]], line_width=2, line_alpha=1, line_color="darkred")
        roc_whisker_bot2 = fig.multi_line([[fpr_eb2 - 0.03, fpr_eb2 + 0.03]], [[tpr_lowci_eb2, tpr_lowci_eb2]], line_width=2, line_color="darkred")
        roc_whisker_top2 = fig.multi_line([[fpr_eb2 - 0.03, fpr_eb2 + 0.03]], [[tpr_uppci_eb2, tpr_uppci_eb2]], line_width=2, line_alpha=1, line_color="darkred")
        fig.circle([fpr_eb2], [tpr_eb2], size=8, fill_alpha=1, line_alpha=1, line_color="black", fill_color="white")

    # Change font size
    fig.title.text_font_size = "11pt"
    fig.xaxis.axis_label_text_font_size = label_font_size
    fig.yaxis.axis_label_text_font_size = label_font_size
    fig.legend.label_text_font = "10pt"

    # Extra padding
    fig.min_border_left = 20
    fig.min_border_right = 20
    fig.min_border_top = 20
    fig.min_border_bottom = 20

    # Edit legend
    fig.legend.location = "bottom_right"
    fig.legend.label_text_font_size = "10pt"
    if legend is False:
        fig.legend.visible = False
    return fig


def roc_calculate_boot(model, Xtrue, Ytrue, Yscore, bootnum=1000, metric=None, val=None, parametric=True, n_cores=-1):
    """Calculates required metrics for the roc plot function (fpr, tpr, and tpr_ci).

    Parameters
    ----------
    Ytrue : array-like, shape = [n_samples]
        Binary label for samples (0s and 1s)

    Yscore : array-like, shape = [n_samples]
        Predicted y score for samples

    Returns
    ----------------------------------
    fpr : array-like, shape = [n_samples]
        False positive rates.

    tpr : array-like, shape = [n_samples]
        True positive rates.

    tpr_ci : array-like, shape = [n_samples, 2]
        True positive rates 95% confidence intervals [lowci, uppci].
    """
    # if n_cores = -1, set n_cores to max_cores
    max_num_cores = multiprocessing.cpu_count()
    n_cores = n_cores
    if n_cores > max_num_cores:
        n_cores = -1
        print("Number of cores set too high. It will be set to the max number of cores in the system.", flush=True)
    if n_cores == -1:
        n_cores = max_num_cores
        print("Number of cores set to: {}".format(max_num_cores))

    time.sleep(0.5)  # Sleep for 0.5 secs to finish printing

    # Start Timer
    start = timeit.default_timer()

    # model copy
    model_name = model.__name__
    model_boot = eval(model_name)

    # Get fpr, tpr
    fpr, tpr, threshold = metrics.roc_curve(Ytrue, Yscore, pos_label=1, drop_intermediate=False)

    # fpr, tpr with drop_intermediates for fpr = 0 (useful for plot... since we plot specificity on x-axis, we don't need intermediates when fpr=0)
    tpr0 = tpr[fpr == 0][-1]
    tpr = np.concatenate([[tpr0], tpr[fpr > 0]])
    fpr = np.concatenate([[0], fpr[fpr > 0]])

    # if metric is provided, calculate stats
    if metric is not None:
        specificity, sensitivity, threshold = get_spec_sens_cuttoff(Ytrue, Yscore, metric, val)
        stats = get_stats(Ytrue, Yscore, specificity, parametric)
        stats["val_specificity"] = specificity
        stats["val_sensitivity"] = specificity
        stats["val_cutoffscore"] = threshold

    # Check if multiple multiblock
    if len(Xtrue) == 2:
        mb_split = len(Xtrue[0].T)
        Xtrue = np.concatenate((Xtrue[0], Xtrue[1]), axis=1)
    else:
        mb_split = 0

    # bootstrap using vertical averaging to linspace
    mean_fpr = np.linspace(0, 1, 1000)
    # understand location
    x_loc = pd.DataFrame(Xtrue)
    x0_loc = list(x_loc[Ytrue == 0].index)
    x1_loc = list(x_loc[Ytrue == 1].index)
    x_loc_ib_dict = {k: [] for k in list(x_loc.index)}
    x_loc_oob_dict = {k: [] for k in list(x_loc.index)}
    # stratified resample
    x0 = Xtrue[Ytrue == 0]
    x1 = Xtrue[Ytrue == 1]
    x0_idx = list(range(len(x0)))
    x1_idx = list(range(len(x1)))

    # input for parallel
    class para_class:
        def __init__(self, x0_idx, x1_idx, model_boot, x0, x1, metric, specificity, parametric, mean_fpr, n_cores, x0_loc, x1_loc, params, mb_split):
            self.x0_idx = x0_idx
            self.x1_idx = x1_idx
            self.model_boot = model_boot
            self.x0 = x0
            self.x1 = x1
            self.metric = metric
            self.specificity = specificity
            self.parametric = parametric
            self.mean_fpr = mean_fpr
            self.n_cores = n_cores
            self.x0_loc = x0_loc
            self.x1_loc = x1_loc
            self.params = params
            self.mb_split = mb_split

        def _roc_calculate_boot_loop(self, i):
            val = _roc_calculate_boot_loop(self)
            return val

    params = model.__params__

    self = para_class(x0_idx, x1_idx, model_boot, x0, x1, metric, specificity, parametric, mean_fpr, n_cores, x0_loc, x1_loc, params, mb_split)

    para_output = Parallel(n_jobs=self.n_cores)(delayed(self._roc_calculate_boot_loop)(i) for i in tqdm(range(bootnum)))

    tpr_ib = []
    fpr_ib = []
    stat_ib_boot = []
    median_ib = []
    tpr_oob = []
    fpr_oob = []
    stat_oob_boot = []
    median_oob = []
    manw_pval = []
    temp_x_loc_ib_dict = []
    temp_x_loc_oob_dict = []

    for i in para_output:
        tpr_ib.append(i[0])
        fpr_ib.append(i[1])
        stat_ib_boot.append(i[2])
        median_ib.append(i[3])
        tpr_oob.append(i[4])
        fpr_oob.append(i[5])
        stat_oob_boot.append(i[6])
        median_oob.append(i[7])
        manw_pval.append(i[8])
        temp_x_loc_ib_dict.append(i[9])
        temp_x_loc_oob_dict.append(i[10])

    for i in temp_x_loc_ib_dict:
        for j in i:
            x_loc_ib_dict[j[0]].append(j[1])

    for i in temp_x_loc_oob_dict:
        for j in i:
            x_loc_oob_dict[j[0]].append(j[1])

    # Get CI for bootstat ib
    if metric is not None:
        stat_ib = {}
        for i in stat_ib_boot[0].keys():
            stats_i = [k[i] for k in stat_ib_boot]
            stats_i = np.array(stats_i)
            stats_i = stats_i[~np.isnan(stats_i)]  # Remove nans
            try:
                lowci = np.percentile(stats_i, 2.5)
                medci = np.percentile(stats_i, 50)
                uppci = np.percentile(stats_i, 97.5)
            except IndexError:
                lowci = np.nan
                medci = np.nan
                uppci = np.nan
            stat_ib[i] = [medci, lowci, uppci]

    # Get CI for bootstat oob
    if metric is not None:
        stat_oob = {}
        for i in stat_oob_boot[0].keys():
            stats_i = [k[i] for k in stat_oob_boot]
            stats_i = np.array(stats_i)
            stats_i = stats_i[~np.isnan(stats_i)]  # Remove nans
            try:
                lowci = np.percentile(stats_i, 2.5)
                medci = np.percentile(stats_i, 50)
                uppci = np.percentile(stats_i, 97.5)
            except IndexError:
                lowci = np.nan
                medci = np.nan
                uppci = np.nan
            stat_oob[i] = [medci, lowci, uppci]

    # Get CI for tpr
    tpr_oob_lowci = np.percentile(tpr_oob, 2.5, axis=0)
    tpr_oob_medci = np.percentile(tpr_oob, 50, axis=0)
    tpr_oob_uppci = np.percentile(tpr_oob, 97.5, axis=0)

    # Add the starting 0
    tpr_oob_lowci = np.insert(tpr_oob_lowci, 0, 0)
    tpr_oob_medci = np.insert(tpr_oob_medci, 0, 0)
    tpr_oob_uppci = np.insert(tpr_oob_uppci, 0, 0)

    # Concatenate tpr_ci
    tpr_oob_ci = np.array([tpr_oob_medci, tpr_oob_lowci, tpr_oob_uppci])

    # Get CI for tpr
    tpr_ib_lowci = np.percentile(tpr_ib, 2.5, axis=0)
    tpr_ib_medci = np.percentile(tpr_ib, 50, axis=0)
    tpr_ib_uppci = np.percentile(tpr_ib, 97.5, axis=0)

    # Add the starting 0
    tpr_ib_lowci = np.insert(tpr_ib_lowci, 0, 0)
    tpr_ib_medci = np.insert(tpr_ib_medci, 0, 0)
    tpr_ib_uppci = np.insert(tpr_ib_uppci, 0, 0)

    # Get median score per boot
    median_y_ib = x_loc_ib_dict
    median_y_oob = x_loc_oob_dict

    # Concatenate tpr_ci
    tpr_ib_ci = np.array([tpr_ib_medci, tpr_ib_lowci, tpr_ib_uppci])

    # Stop timer
    stop = timeit.default_timer()
    self.parallel_time = (stop - start) / 60
    print("Time taken: {:0.2f} minutes with {} cores".format(self.parallel_time, self.n_cores))

    if metric is None:
        return fpr, tpr, tpr_ci
    else:
        return fpr_ib, tpr_ib_ci, stat_ib, median_ib, fpr_oob, tpr_oob_ci, stat_oob, median_oob, stats, median_y_ib, median_y_oob, manw_pval


def _roc_calculate_boot_loop(self):
    "loop using joblib"

    x0_idx = self.x0_idx
    x1_idx = self.x1_idx
    params = self.params
    model_boot = self.model_boot(**params)
    x0 = self.x0
    x1 = self.x1
    metric = self.metric
    specificity = self.specificity
    parametric = self.parametric
    mean_fpr = self.mean_fpr
    x0_loc = self.x0_loc
    x1_loc = self.x1_loc
    mb_split = self.mb_split

    # resample
    x0_idx_ib = resample(x0_idx)
    x1_idx_ib = resample(x1_idx)
    x0_idx_oob = list(set(x0_idx) - set(x0_idx_ib))
    x1_idx_oob = list(set(x1_idx) - set(x1_idx_ib))
    # get x
    x0_ib = x0[x0_idx_ib]
    x1_ib = x1[x1_idx_ib]
    x0_oob = x0[x0_idx_oob]
    x1_oob = x1[x1_idx_oob]
    x_ib = np.concatenate((x0_ib, x1_ib))
    x_oob = np.concatenate((x0_oob, x1_oob))
    # get y
    y0_ib = np.zeros(len(x0_idx_ib))
    y1_ib = np.ones(len(x1_idx_ib))
    y0_oob = np.zeros(len(x0_idx_oob))
    y1_oob = np.ones(len(x1_idx_oob))
    y_ib = np.concatenate((y0_ib, y1_ib))
    y_oob = np.concatenate((y0_oob, y1_oob))

    # Check if multiblock
    if self.mb_split > 0:
        x_ib_0 = x_ib[:, :mb_split]
        x_ib_1 = x_ib[:, mb_split:]
        x_ib = [x_ib_0, x_ib_1]
        x_oob_0 = x_oob[:, :mb_split]
        x_oob_1 = x_oob[:, mb_split:]
        x_oob = [x_oob_0, x_oob_1]

    # train and test model
    ypred_ib = model_boot.train(x_ib, y_ib)
    ypred_oob = model_boot.test(x_oob)
    # get median ypred per group
    ypred_ib_0 = ypred_ib[: len(x0_idx_ib)]
    ypred_ib_1 = ypred_ib[len(x0_idx_ib):]
    k_median_ib = [np.median(ypred_ib_0), np.median(ypred_ib_1)]  # 1

    # get ib fpr, tpr, stats
    fpri, tpri, _ = metrics.roc_curve(y_ib, ypred_ib, pos_label=1, drop_intermediate=False)
    k_fpr_ib = mean_fpr
    k_tpr_ib = interp(mean_fpr, fpri, tpri)

    # tpr_ib[-1][0] = 0.0
    # if metric is provided, calculate stats
    if metric is not None:
        stats_resi = get_stats(y_ib, ypred_ib, specificity, parametric)
        k_stat_ib_boot = stats_resi

    # get median ypred per group
    ypred_oob_0 = ypred_oob[: len(x0_idx_oob)]
    ypred_oob_1 = ypred_oob[len(x0_idx_oob):]
    k_median_oob = [np.median(ypred_oob_0), np.median(ypred_oob_1)]

    # get oob
    fpro, tpro, _ = metrics.roc_curve(y_oob, ypred_oob, pos_label=1, drop_intermediate=False)
    k_fpr_oob = mean_fpr
    k_tpr_oob = interp(mean_fpr, fpro, tpro)
    # k_tpr_oob[-1][0] = 0.0

    # if metric is provided, calculate stats
    if metric is not None:
        stats_reso = get_stats(y_oob, ypred_oob, specificity, parametric)
        k_stat_oob_boot = stats_reso
    # manu
    try:
        manw_pval_ib = scipy.stats.mannwhitneyu(ypred_ib_0, ypred_ib_1, alternative="two-sided")[1]
    except ValueError:
        manw_pval_ib = 0
    try:
        manw_pval_oob = scipy.stats.mannwhitneyu(ypred_oob_0, ypred_oob_1, alternative="two-sided")[1]
    except ValueError:
        manw_pval_oob = 0
    k_manw_pval = [manw_pval_ib, manw_pval_oob]

    # get average ypred
    k_x_loc_ib_dict = []
    for i in range(len(ypred_ib_0)):
        idx_res = x0_idx_ib[i]
        idx_true = x0_loc[idx_res]
        k_x_loc_ib_dict.append([idx_true, ypred_ib_0[i]])
    for i in range(len(ypred_ib_1)):
        idx_res = x1_idx_ib[i]
        idx_true = x1_loc[idx_res]
        k_x_loc_ib_dict.append([idx_true, ypred_ib_1[i]])

    # get average ypred
    k_x_loc_oob_dict = []
    for i in range(len(ypred_oob_0)):
        idx_res = x0_idx_oob[i]
        idx_true = x0_loc[idx_res]
        k_x_loc_oob_dict.append([idx_true, ypred_oob_0[i]])
    for i in range(len(ypred_oob_1)):
        idx_res = x1_idx_oob[i]
        idx_true = x1_loc[idx_res]
        k_x_loc_oob_dict.append([idx_true, ypred_oob_1[i]])

    return [k_tpr_ib, k_fpr_ib, k_stat_ib_boot, k_median_ib, k_tpr_oob, k_fpr_oob, k_stat_oob_boot, k_median_oob, k_manw_pval, k_x_loc_ib_dict, k_x_loc_oob_dict]


def get_sens_spec(Ytrue, Yscore, cuttoff_val):
    """Get sensitivity and specificity from cutoff value."""
    Yscore_round = np.where(np.array(Yscore) > cuttoff_val, 1, 0)
    tn, fp, fn, tp = metrics.confusion_matrix(Ytrue, Yscore_round).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def get_sens_cuttoff(Ytrue, Yscore, specificity_val):
    """Get sensitivity and cuttoff value from specificity."""
    fpr0 = 1 - specificity_val
    fpr, sensitivity, thresholds = metrics.roc_curve(Ytrue, Yscore, pos_label=1, drop_intermediate=False)
    idx = np.abs(fpr - fpr0).argmin()  # this find the closest value in fpr to fpr0
    # Check that this is not a perfect roc curve
    # If it is perfect, allow sensitivity = 1, rather than 0
    if specificity_val == 1 and sensitivity[idx] == 0:
        for i in range(len(fpr)):
            if fpr[i] == 1 and sensitivity[i] == 1:
                return 1, 0.5
    return sensitivity[idx], thresholds[idx]


def get_spec_sens_cuttoff(Ytrue, Yscore, metric, val):
    """Return specificity, sensitivity, cutoff value provided the metric and value used."""
    if metric == "specificity":
        specificity = val
        sensitivity, threshold = get_sens_cuttoff(Ytrue, Yscore, val)
    elif metric == "cutoffscore":
        threshold = val
        sensitivity, specificity = get_sens_spec(Ytrue, Yscore, val)
    return specificity, sensitivity, threshold


def get_stats(Ytrue, Yscore, specificity, parametric):
    """Calculates binary metrics given the specificity."""
    sensitivity, cutoffscore = get_sens_cuttoff(Ytrue, Yscore, specificity)
    stats = binary_metrics(Ytrue, Yscore, cut_off=cutoffscore, parametric=parametric)
    return stats
