import numpy as np
import pandas as pd
from bokeh.models import Band, HoverTool
from tqdm import tqdm
import timeit
import warnings
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
from ..utils import binary_metrics, dict_median, smooth
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label
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
from ..utils import binary_evaluation


def roc(Y, stat, test=None, bootnum=100, legend=True, grid_line=False, label_font_size="10pt", xlabel="1 - Specificity", ylabel="Sensitivity", width=320, height=315, method='BCA', plot='data', legend_basic=False):

    # Set positive
    auc_check = roc_auc_score(Y, stat)
    if auc_check > 0.5:
        pos = 1
    else:
        pos = 0

    # Set Linspace for FPR
    fpr_linspace = np.linspace(0, 1, 1000)  # Make it 1000

    # Calculate for STAT
    fpr_stat, tpr_stat, _ = metrics.roc_curve(Y, stat, pos_label=pos, drop_intermediate=False)
    auc_stat = metrics.auc(fpr_stat, tpr_stat)

    # Drop intermediates when fpr = 0
    tpr_stat = interp(fpr_linspace, fpr_stat, tpr_stat)
    tpr_list = tpr_stat

    # tpr0_stat = tpr_stat[fpr_stat == 0][-1]
    # tpr_stat = np.concatenate([[tpr0_stat], tpr_stat[fpr_stat > 0]])
    # fpr_stat = np.concatenate([[0], fpr_stat[fpr_stat > 0]])

    # # Vertical averaging
    # idx = [np.abs(i - fpr_stat).argmin() for i in fpr_linspace]
    # tpr_list = np.array(tpr_stat[idx])

    binary_stats_train_dict = binary_evaluation(Y, stat)
    binary_stats_train = []
    for key, value in binary_stats_train_dict.items():
      binary_stats_train.append(value)
    binary_stats_train = np.array(binary_stats_train)

    binary_stats_train_boot = []
    tpr_bootstat = []

    if bootnum > 1:
      for i in range(bootnum):
        bootidx = resample(list(range(len(Y))), stratify=Y)  # Default stratified
        # Get Yscore and Y for each bootstrap and calculate
        Yscore_boot = stat[bootidx]
        Ytrue_boot = Y[bootidx]
        fpr_boot, tpr_boot, _ = metrics.roc_curve(Ytrue_boot, Yscore_boot, pos_label=pos, drop_intermediate=False)
        auc_boot = metrics.auc(fpr_boot, tpr_boot)
        if auc_boot < 0.5:
          fpr_boot, tpr_boot, _ = metrics.roc_curve(Ytrue_boot, Yscore_boot, pos_label=abs(1 - pos), drop_intermediate=False)

        bstat_loop = binary_evaluation(Ytrue_boot, Yscore_boot)
        bstat_list = []
        for key, value in bstat_loop.items():
            bstat_list.append(value)
        binary_stats_train_boot.append(bstat_list)
        # Drop intermediates when fpr = 0
        tpr0_boot = tpr_boot[fpr_boot == 0][-1]
        tpr_boot = np.concatenate([[tpr0_boot], tpr_boot[fpr_boot > 0]])
        fpr_boot = np.concatenate([[0], fpr_boot[fpr_boot > 0]])

        # Vertical averaging
        idx = [np.abs(i - fpr_boot).argmin() for i in fpr_linspace]
        tpr_bootstat.append(np.array(tpr_boot[idx]))
    binary_stats_train_boot = np.array(binary_stats_train_boot)

    if bootnum > 1:
      if method == 'BCA':
        binary_stats_jack_boot = []
        jackidx = []
        base = np.arange(0, len(Y))
        for i in base:
            jack_delete = np.delete(base, i)
            jackidx.append(jack_delete)

        tpr_jackstat = []
        for i in jackidx:
          # Get Yscore and Y for each bootstrap and calculate
          Yscore_jack = stat[i]
          Ytrue_jack = Y[i]
          fpr_jack, tpr_jack, _ = metrics.roc_curve(Ytrue_jack, Yscore_jack, pos_label=pos, drop_intermediate=False)
          auc_jack = metrics.auc(fpr_jack, tpr_jack)
          if auc_boot < 0.5:
            fpr_jack, tpr_jack, _ = metrics.roc_curve(Ytrue_jack, Yscore_jack, pos_label=abs(1 - pos), drop_intermediate=False)

          jstat_loop = binary_evaluation(Ytrue_jack, Yscore_jack)
          jstat_list = []
          for key, value in jstat_loop.items():
              jstat_list.append(value)
          binary_stats_jack_boot.append(jstat_list)

          # Drop intermediates when fpr = 0
          tpr0_jack = tpr_boot[fpr_boot == 0][-1]
          tpr_jack = np.concatenate([[tpr0_jack], tpr_jack[fpr_jack > 0]])
          fpr_jack = np.concatenate([[0], fpr_jack[fpr_jack > 0]])

          # Vertical averaging
          idx = [np.abs(i - fpr_jack).argmin() for i in fpr_linspace]
          tpr_jackstat.append(np.array(tpr_jack[idx]))
        binary_stats_jack_boot = np.array(binary_stats_jack_boot)

    if bootnum > 1:
      if method == 'BCA':
          tpr_ib = bca_method(tpr_bootstat, tpr_list, tpr_jackstat)
          tpr_ib = np.concatenate((np.zeros((1, 3)), tpr_ib), axis=0)  # Add starting 0
          stat_ib = bca_method(binary_stats_train_boot, binary_stats_train, binary_stats_jack_boot)
      elif method == 'Per':
          tpr_ib = per_method(tpr_bootstat, tpr_list)
          tpr_ib = np.concatenate((np.zeros((1, 3)), tpr_ib), axis=0)  # Add starting 0
          stat_ib = per_method(binary_stats_train_boot, binary_stats_train)
          stat_ib = list(stat_ib)
      elif method == 'CPer':
          tpr_ib = cper_method(tpr_bootstat, tpr_list)
          tpr_ib = np.concatenate((np.zeros((1, 3)), tpr_ib), axis=0)  # Add starting 0
          stat_ib = cper_method(binary_stats_train_boot, binary_stats_train)
          stat_ib = list(stat_ib)
      else:
        raise ValueError("bootmethod has to be 'BCA', 'Perc', or 'CPer'.")

      #stat_ib = np.array(stat_ib).T
      #print(stat_ib)
      # ROC up
      # for i in range(len(tpr_ib.T)):
      #     for j in range(1, len(tpr_ib)):
      #         if tpr_ib[j, i] < tpr_ib[j - 1, i]:
      #             tpr_ib[j, i] = tpr_ib[j - 1, i]

      # Get tpr mid
      if method != 'Per':
          tpr_ib[:, 2] = (tpr_ib[:, 0] + tpr_ib[:, 1]) / 2
          for i in range(len(stat_ib)):
            stat_ib[i][2] = binary_stats_train[i]
    else:
      tpr_ib = []
      tpr_ib.append(tpr_list)
      tpr_ib.append(tpr_list)
      tpr_ib.append(tpr_list)
      tpr_ib = np.array(tpr_ib)
      tpr_ib = tpr_ib.T
      tpr_ib = np.concatenate((np.zeros((1, 3)), tpr_ib), axis=0)  # Add starting 0
      tpr_ib = np.concatenate((tpr_ib, np.ones((1, 3))), axis=0)  # Add end 1
      binary_stats_train_dict = binary_evaluation(Y, stat)
      binary_stats_train = []
      for key, value in binary_stats_train_dict.items():
        binary_stats_train.append(value)
      stat_ib = []
      stat_ib.append(binary_stats_train)
      stat_ib.append(binary_stats_train)
      stat_ib.append(binary_stats_train)

    # Test if available
    if test is not None:
      test_y = test[0]
      test_ypred = test[1]
      fpr_test, tpr_test, _ = metrics.roc_curve(test_y, test_ypred, pos_label=pos, drop_intermediate=False)
      auc_test = metrics.auc(fpr_test, tpr_test)

      binary_stats_test_dict = binary_evaluation(test_y, test_ypred)
      binary_stats_test = []
      for key, value in binary_stats_test_dict.items():
        binary_stats_test.append(value)
      stat_ib.append(binary_stats_test)

      # Drop intermediates when fpr = 0
      tpr_test = interp(fpr_linspace, fpr_test, tpr_test)
      tpr_test = np.insert(tpr_test, 0, 0)  # Add starting 0
      tpr_test = np.concatenate([tpr_test, [1]])


      # Drop intermediates when fpr = 0
      # tpr0_test = tpr_test[fpr_test == 0][-1]
      # tpr_test = np.concatenate([[tpr0_test], tpr_test[fpr_test > 0]])
      # fpr_test = np.concatenate([[0], fpr_test[fpr_test > 0]])

      # # Vertical averaging
      # idx_test = [np.abs(i - fpr_test).argmin() for i in fpr_linspace]
      # tpr_test = tpr_test[idx_test]
      # tpr_test = np.insert(tpr_test, 0, 0)  # Add starting 0

    fpr_linspace = np.insert(fpr_linspace, 0, 0)  # Add starting 0
    fpr_linspace  = np.concatenate((fpr_linspace, [1]))  # Add end 1

    # if 'data' plot original data instead of median
    if plot == 'data':
      tpr_list_linspace = np.concatenate([[0], tpr_list])  # Add starting 0
      tpr_list_linspace = np.concatenate([tpr_list_linspace, [1]])  # Add starting 0
      tpr_ib[:, 2] = tpr_list_linspace
    elif plot == 'median':
      pass
    else:
      raise ValueError("plot must be 'data' or 'median'")

    # Check upper limit / lower limit
    for i in tpr_ib:
      if i[0] > i[2]:
        i[0] = i[2]
      if i[1] < i[2]:
        i[1] = i[2]

    # Calculate AUC
    auc_ib_low = metrics.auc(fpr_linspace, tpr_ib[:, 0])
    auc_ib_upp = metrics.auc(fpr_linspace, tpr_ib[:, 1])
    auc_ib_mid = metrics.auc(fpr_linspace, tpr_ib[:, 2])
    auc_ib = np.array([auc_ib_low, auc_ib_upp, auc_ib_mid])

    # Plot
    spec = 1 - fpr_linspace
    ci_ib = (tpr_ib[:, 1] - tpr_ib[:, 0]) / 2
    ci_oob = (tpr_ib[:, 1] - tpr_ib[:, 0]) / 2

    fig = figure(title="",
                 plot_width=width,
                 plot_height=height,
                 x_axis_label=xlabel,
                 y_axis_label=ylabel,
                 x_range=(-0.06, 1.06),
                 y_range=(-0.06, 1.06))
    fig.line([0, 1], [0, 1], color="black", line_dash="dashed", alpha=0.8, line_width=1)  # Equal Distribution Line

    # Plot IB
    data_ib = {"x": fpr_linspace,
               "y": tpr_ib[:, 2],
               "lowci": tpr_ib[:, 0],
               "uppci": tpr_ib[:, 1],
               "spec": spec,
               "ci": ci_ib}
    source_ib = ColumnDataSource(data=data_ib)

    # Line IB
    if bootnum > 1:
        if legend_basic == True:
          legend_ib = "Train"
        else:
          legend_ib = "Train (AUC = {:.2f} +/- {:.2f})".format(auc_ib[2], (auc_ib[1] - auc_ib[0]) / 2)
        figline_ib = fig.line("x",
                              "y",
                              color="green",
                              line_width=2.5,
                              alpha=0.7,
                              legend=legend_ib,
                              source=source_ib)
        fig.add_tools(HoverTool(renderers=[figline_ib],
                                tooltips=[("Specificity", "@spec{1.111}"),
                                          ("Sensitivity", "@y{1.111} (+/- @ci{1.111})"), ]))
        # CI Band IB
        figband_ib = Band(base="x",
                          lower="lowci",
                          upper="uppci",
                          level="underlay",
                          fill_alpha=0.1,
                          line_width=0.5,
                          line_color="black",
                          fill_color="green",
                          source=source_ib)
        fig.add_layout(figband_ib)
    else:
        if legend_basic == True:
            legend_ib = "Train"
        else:
            legend_ib = "Train (AUC = {:.2f})".format(auc_ib[2])

        figline_ib = fig.line("x",
                              "y",
                              color="green",
                              line_width=2.5,
                              alpha=0.7,
                              legend=legend_ib,
                              source=source_ib)
        fig.add_tools(HoverTool(renderers=[figline_ib],
                                tooltips=[("Specificity", "@spec{1.111}"),
                                          ("Sensitivity", "@y{1.111} (+/- @ci{1.111})"), ]))

    # Line Test
    if test is not None:
      if legend_basic == True:
        legend_oob = "Test"
      else:
        legend_oob = "Test (AUC = {:.2f})".format(auc_test)

      # Plot IB
      data_test = {"x": fpr_linspace,
                 "y": tpr_test,
                 "spec": spec}
      source_test = ColumnDataSource(data=data_test)


      figline_test = fig.line("x",
                            "y",
                            color="orange",
                            line_width=2.5,
                            alpha=0.7,
                            legend=legend_oob,
                            source=source_test)
      fig.add_tools(HoverTool(renderers=[figline_test],
                              tooltips=[("Specificity", "@spec{1.111}"),
                                          ("Sensitivity", "@y{1.111}"), ]))

    if grid_line == False:
        fig.xgrid.visible = False
        fig.ygrid.visible = False

    fig.legend.visible =  False

    if legend == True:
      if legend_basic == True:
        fig.legend.visible =  True
        fig.legend.location = "bottom_right"
      else:
        if test is None:
            oob_text = "Train (AUC = {:.2f} +/- {:.2f})".format(auc_ib[2], (auc_ib[1] - auc_ib[0])/2)

            oob_text_add = Label(x=0.38, y=0.02,
                             text=oob_text, render_mode='css', text_font_size= '9pt')

            fig.add_layout(oob_text_add)


            fig.quad(top=0.12, bottom=0, left=0.30, right=1, color='white', alpha=1,line_color='black')

            fig.circle(0.34,0.06,color='green',size=8)

        else:
            ib_text = "Train (AUC = {:.2f} +/- {:.2f})".format(auc_ib[2], (auc_ib[1] - auc_ib[0])/2)
            oob_text = "Test (AUC = {:.2f})".format(auc_test)
            ib_text_add = Label(x=0.38, y=0.10,
                               text=ib_text, render_mode='canvas', text_font_size= '9pt')

            fig.add_layout(ib_text_add)

            oob_text_add = Label(x=0.38, y=0.02,
                             text=oob_text, render_mode='canvas', text_font_size= '9pt')

            fig.add_layout(oob_text_add)


            fig.quad(top=0.20, bottom=0, left=0.30, right=1, color='white', alpha=1,line_color='black')

            fig.circle(0.34,0.14,color='green',size=8)
            fig.circle(0.34,0.06,color='purple',size=8)

    if legend_basic == True:
      return fig, stat_ib
    else:
      return fig

def roc_boot(Y,
             stat,
             bootstat,
             bootstat_oob,
             bootidx,
             bootidx_oob,
             method,
             smoothval=0,
             jackstat=None,
             jackidx=None,
             xlabel="1 - Specificity",
             ylabel="Sensitivity",
             width=320,
             height=315,
             label_font_size="10pt",
             legend=True,
             grid_line=False,
             plot_num=0,
             plot='data',
             test=None,
             legend_basic=False,
             train=None,
             ci_only=False):

    # Set positive
    auc_check = roc_auc_score(Y, stat)
    if auc_check > 0.5:
        pos = 1
    else:
        pos = 0

    # Set Linspace for FPR
    fpr_linspace = np.linspace(0, 1, 1000)  # Make it 1000

    # Calculate for STAT
    fpr_stat, tpr_stat, _ = metrics.roc_curve(Y, stat, pos_label=pos, drop_intermediate=False)
    auc_stat = metrics.auc(fpr_stat, tpr_stat)
    tpr_stat = interp(fpr_linspace, fpr_stat, tpr_stat)
    tpr_list = tpr_stat

    # Calculate for BOOTSTAT (IB)
    pos_loop = []
    tpr_bootstat = []
    for i in range(len(bootidx)):
        # Get Yscore and Y for each bootstrap and calculate
        Yscore_boot = bootstat[i]
        Ytrue_boot = Y[bootidx[i]]
        fpr_boot, tpr_boot, _ = metrics.roc_curve(Ytrue_boot, Yscore_boot, pos_label=pos, drop_intermediate=False)
        auc_boot = metrics.auc(fpr_boot, tpr_boot)
        if auc_boot > 0.5:
            pos_loop.append(pos)
        else:
          fpr_boot, tpr_boot, _ = metrics.roc_curve(Ytrue_boot, Yscore_boot, pos_label=abs(1 - pos), drop_intermediate=False)
          pos_loop.append(abs(1 - pos))

        # Drop intermediates when fpr = 0
        tpr0_boot = tpr_boot[fpr_boot == 0][-1]
        tpr_boot = np.concatenate([[tpr0_boot], tpr_boot[fpr_boot > 0]])
        fpr_boot = np.concatenate([[0], fpr_boot[fpr_boot > 0]])

        # Vertical averaging
        idx = [np.abs(i - fpr_boot).argmin() for i in fpr_linspace]
        tpr_bootstat.append(np.array(tpr_boot[idx]))

#         tpr_boot = interp(fpr_linspace, fpr_boot, tpr_boot)
#         tpr_bootstat.append(tpr_boot)

    if method == 'BCA':
        tpr_jackstat = []
        for i in range(len(jackidx)):
            # Get Yscore and Y for each bootstrap and calculate
            Yscore_jack = jackstat[i]
            Ytrue_jack = Y[jackidx[i]]
            fpr_jack, tpr_jack, _ = metrics.roc_curve(Ytrue_jack, Yscore_jack, pos_label=pos, drop_intermediate=False)
            auc_jack = metrics.auc(fpr_jack, tpr_jack)
#             if auc_jack < 0.5:
#                 fpr_jack, tpr_jack, _ = metrics.roc_curve(Ytrue_jack, Yscore_jack, pos_label=abs(1 - pos), drop_intermediate=False)

            # Drop intermediates when fpr = 0
            tpr0_jack = tpr_jack[fpr_jack == 0][-1]
            tpr_jack = np.concatenate([[tpr0_jack], tpr_jack[fpr_jack > 0]])
            fpr_jack = np.concatenate([[0], fpr_jack[fpr_jack > 0]])

            # Vertical averaging
            idx = [np.abs(i - fpr_jack).argmin() for i in fpr_linspace]
            tpr_jackstat.append(np.array(tpr_jack[idx]))

    #save_stat = [tpr_bootstat, tpr_list, tpr_jackstat, fpr_linspace]
    if method == 'BCA':
        tpr_ib = bca_method(tpr_bootstat, tpr_list, tpr_jackstat)

    if method == 'Per':
        tpr_ib = per_method(tpr_bootstat, tpr_list)

    if method == 'CPer':
        tpr_ib = cper_method(tpr_bootstat, tpr_list)

    tpr_ib = np.array(tpr_ib)
    # ROC up
    if method != 'Per':
      for i in range(len(tpr_ib.T)):
          for j in range(1, len(tpr_ib)):
              if tpr_ib[j, i] < tpr_ib[j - 1, i]:
                  tpr_ib[j, i] = tpr_ib[j - 1, i]

    # # Check upper limit / lower limit
    if method != 'Per':
        for i in range(len(tpr_ib)):
            if tpr_ib[i][0] > tpr_list[i]:
                tpr_ib[i][0] = tpr_list[i]
            if tpr_ib[i][1] < tpr_list[i]:
                tpr_ib[i][1] = tpr_list[i]

    tpr_ib = np.concatenate((np.zeros((1, 3)), tpr_ib), axis=0)  # Add starting 0
    tpr_ib = np.concatenate((tpr_ib, np.ones((1, 3))), axis=0)  # Add end 1

    # Get tpr mid
    if method != 'Per':
        tpr_ib[:, 2] = (tpr_ib[:, 0] + tpr_ib[:, 1]) / 2

    #print('testing.')

    # Calculate for OOB
    auc_bootstat_oob = []
    tpr_bootstat_oob = []
    for i in range(len(bootidx_oob)):
         # Get Yscore and Y for each bootstrap oob and calculate
        Yscore_boot_oob = bootstat_oob[i]
        Ytrue_boot_oob = Y[bootidx_oob[i]]
        fpr_boot_oob, tpr_boot_oob, _ = metrics.roc_curve(Ytrue_boot_oob, Yscore_boot_oob, pos_label=pos, drop_intermediate=False)
        auc_boot_oob = metrics.auc(fpr_boot_oob, tpr_boot_oob)
        # if auc_boot_oob < 0.5:
        #   fpr_boot_oob, tpr_boot_oob, _ = metrics.roc_curve(Ytrue_boot_oob, Yscore_boot_oob, pos_label=abs(1-pos_loop[i]), drop_intermediate=False)
        auc_boot_oob = metrics.auc(fpr_boot_oob, tpr_boot_oob)
        auc_bootstat_oob.append(auc_boot_oob)

        # Drop intermediates when fpr = 0
        tpr0_boot_oob = tpr_boot_oob[fpr_boot_oob == 0][-1]
        tpr_boot_oob = np.concatenate([[tpr0_boot_oob], tpr_boot_oob[fpr_boot_oob > 0]])
        fpr_boot_oob = np.concatenate([[0], fpr_boot_oob[fpr_boot_oob > 0]])

        # Vertical averaging
        idx_oob = [np.abs(i - fpr_boot_oob).argmin() for i in fpr_linspace]
        tpr_bootstat_oob.append(np.array(tpr_boot_oob[idx_oob]))
        #tpr_boot_oob = interp(fpr_linspace, fpr_boot_oob, tpr_boot_oob)
        #tpr_bootstat_oob.append(tpr_boot_oob)

    # Get CI for tpr
    tpr_oob_lowci = np.percentile(tpr_bootstat_oob, 2.5, axis=0)
    tpr_oob_medci = np.percentile(tpr_bootstat_oob, 50, axis=0)
    tpr_oob_uppci = np.percentile(tpr_bootstat_oob, 97.5, axis=0)
    tpr_oob = np.array([tpr_oob_lowci, tpr_oob_uppci, tpr_oob_medci]).T

    #tpr_oob = per_method(tpr_bootstat_oob, tpr_list)
    auc_oob = per_method(auc_bootstat_oob, auc_stat)
    tpr_oob = np.concatenate((np.zeros((1, 3)), tpr_oob), axis=0)  # Add starting 0
    tpr_oob = np.concatenate((tpr_oob, np.ones((1, 3))), axis=0)  # Add end 1

    # ROC up
    if method != 'Per':
      for i in range(len(tpr_oob.T)):
          for j in range(1, len(tpr_oob)):
              if tpr_oob[j, i] < tpr_oob[j - 1, i]:
                  tpr_oob[j, i] = tpr_oob[j - 1, i]

    # Test if available
    if test is not None:
      test_y = test[0]
      test_ypred = test[1]
      fpr_test, tpr_test, _ = metrics.roc_curve(test_y, test_ypred, pos_label=pos, drop_intermediate=False)
      auc_test = metrics.auc(fpr_test, tpr_test)

      # Drop intermediates when fpr = 0
      # tpr0_test= tpr_test[fpr_test == 0][-1]
      # tpr_test = np.concatenate([[tpr0_test], tpr_test[fpr_test > 0]])
      # fpr_test = np.concatenate([[0], fpr_test[fpr_test > 0]])

      # # Vertical averaging
      # idx_test = [np.abs(i - fpr_test).argmin() for i in fpr_linspace]
      # tpr_test = tpr_test[idx_test]

      tpr_test = interp(fpr_linspace, fpr_test, tpr_test)
      tpr_test = np.insert(tpr_test, 0, 0) # Add starting 0
      tpr_test = np.concatenate((tpr_test,[1]))
      tpr_oob[:, 2] = tpr_test

    # if 'data' plot original data instead of median
    if train is not None:
      fpr_stat, tpr_stat, _ = metrics.roc_curve(train[0], train[1], pos_label=pos, drop_intermediate=False)
      tpr_stat = interp(fpr_linspace, fpr_stat, tpr_stat)
      tpr_list = tpr_stat
    if plot == 'data':
      tpr_list_linspace = np.concatenate([[0], tpr_list])  # Add starting 0
      tpr_list_linspace = np.concatenate([tpr_list_linspace,[1]])  # Add starting 0
      tpr_ib[:,2] = tpr_list_linspace
    elif plot == 'median':
      pass
    else:
      pass
    # else:
    #   raise ValueError("plot must be 'data' or 'median'")


    fpr_linspace = np.insert(fpr_linspace, 0, 0)  # Add starting 0
    fpr_linspace  = np.concatenate((fpr_linspace, [1]))  # Add end 1

    # Calculate AUC
    auc_ib_low = metrics.auc(fpr_linspace, tpr_ib[:, 0])
    auc_ib_upp = metrics.auc(fpr_linspace, tpr_ib[:, 1])
    auc_ib_mid = metrics.auc(fpr_linspace, tpr_ib[:, 2])
    auc_ib = np.array([auc_ib_low, auc_ib_upp, auc_ib_mid])
    auc_oob_low = metrics.auc(fpr_linspace, tpr_oob[:, 0])
    auc_oob_upp = metrics.auc(fpr_linspace, tpr_oob[:, 1])
    auc_oob_mid = metrics.auc(fpr_linspace, tpr_oob[:, 2])
    auc_oob = np.array([auc_oob_low, auc_oob_upp, auc_oob_mid])

    # print(auc_ib)
    # print(auc_oob)
    # print("AUC IB {} ({},{})".format(auc_ib[2], auc_ib[0], auc_ib[1]))
    # print("AUC OOB {} ({},{})".format(auc_oob[2], auc_oob[0], auc_oob[1]))

    # Smooth if set
    if smoothval > 1:
        tpr_ib[:, 0] = smooth(tpr_ib[:, 0], smoothval)
        tpr_ib[:, 1] = smooth(tpr_ib[:, 1], smoothval)
        tpr_ib[:, 2] = smooth(tpr_ib[:, 2], smoothval)
        tpr_oob[:, 0] = smooth(tpr_oob[:, 0], smoothval)
        tpr_oob[:, 1] = smooth(tpr_oob[:, 1], smoothval)
        tpr_oob[:, 2] = smooth(tpr_oob[:, 2], smoothval)
        tpr_test = smooth(tpr_test, smoothval)

    # Plot
    spec = 1 - fpr_linspace
    ci_ib = (tpr_ib[:, 1] - tpr_ib[:, 0]) / 2
    ci_oob = (tpr_ib[:, 1] - tpr_ib[:, 0]) / 2

    fig = figure(title="",
                 plot_width=width,
                 plot_height=height,
                 x_axis_label=xlabel,
                 y_axis_label=ylabel,
                 x_range=(-0.06, 1.06),
                 y_range=(-0.06, 1.06))
    fig.line([0, 1], [0, 1], color="black", line_dash="dashed", alpha=0.8, line_width=1)

    # Plot IB
    data_ib = {"x": fpr_linspace,
               "y": tpr_ib[:, 2],
               "lowci": tpr_ib[:, 0],
               "uppci": tpr_ib[:, 1],
               "spec": spec,
               "ci": ci_ib}
    source_ib = ColumnDataSource(data=data_ib)

    # Line IB
    if plot_num in [0, 1, 2, 4]:

        if legend_basic == True:
          legend_text = "Train"
        else:
          legend_text = "IB (AUC = {:.2f} +/- {:.2f})".format(auc_ib[2], (auc_ib[1] - auc_ib[0]) / 2)

        if ci_only == False:
          figline_ib = fig.line("x",
                                "y",
                                color="green",
                                line_width=2.5,
                                alpha=0.7,
                                legend=legend_text,
                                source=source_ib)
          fig.add_tools(HoverTool(renderers=[figline_ib],
                                  tooltips=[("Specificity", "@spec{1.111}"),
                                            ("Sensitivity", "@y{1.111} (+/- @ci{1.111})"), ]))
        # CI Band IB
        figband_ib = Band(base="x",
                          lower="lowci",
                          upper="uppci",
                          level="underlay",
                          fill_alpha=0.1,
                          line_width=0.5,
                          line_color="black",
                          fill_color="green",
                          source=source_ib)
        fig.add_layout(figband_ib)

        figlegend_ib = fig.rect([10],[20],[5],[5], color="green", fill_alpha=0.1, line_width=0.5, line_color="grey", legend="IB (95% CI)")

    # Plot OOB
    data_oob = {"x": fpr_linspace,
                "y": tpr_oob[:, 2],
                "lowci": tpr_oob[:, 0],
                "uppci": tpr_oob[:, 1],
                "spec": spec,
                "ci": ci_oob}
    source_oob = ColumnDataSource(data=data_oob)

    # Line OOB
    if plot_num in [0, 1, 3, 4]:

        if legend_basic == True:
          legend_text = "Test"
        else:
          legend_text = "OOB (AUC = {:.2f} +/- {:.2f})".format(auc_oob[2], (auc_oob[1] - auc_oob[0]) / 2)

        if ci_only == False:
          figline = fig.line("x",
                             "y",
                             color="orange",
                             line_width=2.5,
                             alpha=0.7,
                             legend=legend_text,
                             source=source_oob)
          fig.add_tools(HoverTool(renderers=[figline],
                                  tooltips=[("Specificity", "@spec{1.111}"),
                                            ("Sensitivity", "@y{1.111} (+/- @ci{1.111})"), ]))

        # CI Band OOB
        figband_oob = Band(base="x",
                           lower="lowci",
                           upper="uppci",
                           level="underlay",
                           fill_alpha=0.1,
                           line_width=0.5,
                           line_color="black",
                           fill_color="orange",
                           source=source_oob)
        fig.add_layout(figband_oob)

        figlegend_ib = fig.rect([10],[20],[5],[5], color="orange", fill_alpha=0.1, line_width=0.5, line_color="grey", legend="OOB (95% CI)")

    # Line Test
    # if test is not None:

    #   if legend_basic == True:
    #     legend_text = "Test"
    #   else:
    #     legend_text = "Test (AUC = {:.2f})".format(auc_test)
    #   # Plot IB
    #   data_test = {"x": fpr_linspace,
    #              "y": tpr_test,
    #              "spec": spec}
    #   source_test = ColumnDataSource(data=data_test)


    #   figline_test = fig.line("x",
    #                         "y",
    #                         color="purple",
    #                         line_width=2.5,
    #                         alpha=0.8,
    #                         legend=legend_text,
    #                         line_dash="dashed",
    #                         source=source_test)
    #   fig.add_tools(HoverTool(renderers=[figline_test],
    #                           tooltips=[("Specificity", "@spec{1.111}"),
    #                                       ("Sensitivity", "@y{1.111}"), ]))

    if grid_line == False:
        fig.xgrid.visible = False
        fig.ygrid.visible = False

    # Legend Manually because of bokeh issue

    ib_text = "IB (AUC = {:.2f} +/- {:.2f})".format(auc_ib[2], (auc_ib[1] - auc_ib[0])/2)
    oob_text = "OOB (AUC = {:.2f} +/- {:.2f})".format(auc_oob[2], (auc_oob[1] - auc_oob[0])/2)

    fig.legend.visible =  False

    if legend_basic == True:
      fig.legend.location = "bottom_right"
      fig.legend.visible = True
    else:
      if test is not None:
        if legend == True:

          ib_text_add = Label(x=0.38, y=0.18,
                                     text=ib_text, render_mode='canvas', text_font_size= '9pt')

          fig.add_layout(ib_text_add)

          oob_text_add = Label(x=0.38, y=0.10,
                           text=oob_text, render_mode='canvas', text_font_size= '9pt')

          fig.add_layout(oob_text_add)

          test_text = "Test (AUC = {:.2f})".format(auc_test)
          test_text_add = Label(x=0.38, y=0.02,
            text=test_text, render_mode='canvas', text_font_size= '9pt')

          fig.add_layout(test_text_add)


          fig.quad(top=0.28, bottom=0, left=0.30, right=1, color='white', alpha=1,  line_color='lightgrey')

          fig.circle(0.34,0.22,color='green',size=8)
          fig.circle(0.34,0.14,color='orange',size=8)
          fig.circle(0.34,0.06,color='purple',size=8)
      else:
        if legend == True:
            if plot_num in [0,1,4]:
                if width == 320:
                  ib_text_add = Label(x=0.38, y=0.10,
                                     text=ib_text, render_mode='canvas', text_font_size= '9pt')

                  fig.add_layout(ib_text_add)

                  oob_text_add = Label(x=0.38, y=0.02,
                                   text=oob_text, render_mode='canvas', text_font_size= '9pt')

                  fig.add_layout(oob_text_add)


                  fig.quad(top=0.20, bottom=0, left=0.30, right=1, color='white', alpha=1,  line_color='lightgrey')

                  fig.circle(0.34,0.14,color='green',size=8)
                  fig.circle(0.34,0.06,color='orange',size=8)
                elif width == 475:
                    ib_text_add = Label(x=0.52, y=0.15,
                                     text=ib_text, render_mode='canvas', text_font_size= '10pt')

                    fig.add_layout(ib_text_add)

                    oob_text_add = Label(x=0.52, y=0.05,
                                     text=oob_text, render_mode='canvas', text_font_size= '10pt')

                    fig.add_layout(oob_text_add)


                    fig.quad(top=0.25, bottom=0, left=0.42, right=1, color='white', alpha=0.4,  line_color='lightgrey')

                    fig.circle(0.47,0.17,color='green',size=8)
                    fig.circle(0.47,0.07,color='orange',size=8)
                elif width == 316:
                    ib_text_add = Label(x=0.22, y=0.15,
                                     text=ib_text, render_mode='canvas', text_font_size= '10pt')

                    fig.add_layout(ib_text_add)

                    oob_text_add = Label(x=0.22, y=0.05,
                                     text=oob_text, render_mode='canvas', text_font_size= '10pt')

                    fig.add_layout(oob_text_add)


                    fig.quad(top=0.25, bottom=0, left=0.12, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.17,0.18,color='green',size=8)
                    fig.circle(0.17,0.08,color='orange',size=8)

                elif width == 237:
                    ib_text_1 = "IB (AUC = {:.2f}".format(auc_ib[2])
                    ib_text_2 = "+/- {:.2f})".format((auc_ib[1] - auc_ib[0])/2)

                    oob_text_1 = "OOB (AUC ="
                    oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_oob[2],(auc_oob[1] - auc_oob[0])/2)


                    ib_text_add_1 = Label(x=0.38, y=0.28,
                                     text=ib_text_1, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(ib_text_add_1)

                    ib_text_add_2 = Label(x=0.38, y=0.19,
                                     text=ib_text_2, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(ib_text_add_2)

                    oob_text_add_1 = Label(x=0.38, y=0.09,
                                     text=oob_text_1, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(oob_text_add_1)

                    oob_text_add_2 = Label(x=0.38, y=0.00,
                                     text=oob_text_2, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(oob_text_add_2)


                    fig.quad(top=0.4, bottom=0, left=0.20, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.27,0.30,color='green',size=8)
                    fig.circle(0.27,0.10,color='orange',size=8)
                elif width == 190:
                    ib_text_1 = "IB (AUC ="
                    ib_text_2 = "{:.2f} +/- {:.2f})".format(auc_ib[2], (auc_ib[1] - auc_ib[0])/2)

                    oob_text_1 = "OOB (AUC ="
                    oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_oob[2],(auc_oob[1] - auc_oob[0])/2)


                    ib_text_add_1 = Label(x=0.28, y=0.32,
                                     text=ib_text_1, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(ib_text_add_1)

                    ib_text_add_2 = Label(x=0.28, y=0.23,
                                     text=ib_text_2, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(ib_text_add_2)

                    oob_text_add_1 = Label(x=0.28, y=0.09,
                                     text=oob_text_1, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(oob_text_add_1)

                    oob_text_add_2 = Label(x=0.28, y=0.00,
                                     text=oob_text_2, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(oob_text_add_2)


                    fig.quad(top=0.47, bottom=0, left=0.12, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.20,0.30,color='green',size=8)
                    fig.circle(0.20,0.10,color='orange',size=8)
                elif width == 158:
                    ib_text_1 = "IB (AUC ="
                    ib_text_2 = "{:.2f} +/- {:.2f})".format(auc_ib[2], (auc_ib[1] - auc_ib[0])/2)

                    oob_text_1 = "OOB (AUC ="
                    oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_oob[2],(auc_oob[1] - auc_oob[0])/2)


                    ib_text_add_1 = Label(x=0.28, y=0.32,
                                     text=ib_text_1, render_mode='canvas', text_font_size= '6pt')

                    fig.add_layout(ib_text_add_1)

                    ib_text_add_2 = Label(x=0.28, y=0.23,
                                     text=ib_text_2, render_mode='canvas', text_font_size= '6pt')

                    fig.add_layout(ib_text_add_2)

                    oob_text_add_1 = Label(x=0.28, y=0.09,
                                     text=oob_text_1, render_mode='canvas', text_font_size= '6pt')

                    fig.add_layout(oob_text_add_1)

                    oob_text_add_2 = Label(x=0.28, y=0.00,
                                     text=oob_text_2, render_mode='canvas', text_font_size= '6pt')

                    fig.add_layout(oob_text_add_2)


                    fig.quad(top=0.47, bottom=0, left=0.12, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.20,0.30,color='green',size=8)
                    fig.circle(0.20,0.10,color='orange',size=8)
                elif width == 135:
                    ib_text_1 = "IB (AUC ="
                    ib_text_2 = "{:.2f} +/- {:.2f})".format(auc_ib[2], (auc_ib[1] - auc_ib[0])/2)

                    oob_text_1 = "OOB (AUC ="
                    oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_oob[2],(auc_oob[1] - auc_oob[0])/2)


                    ib_text_add_1 = Label(x=0.28, y=0.32,
                                     text=ib_text_1, render_mode='canvas', text_font_size= '5pt')

                    fig.add_layout(ib_text_add_1)

                    ib_text_add_2 = Label(x=0.28, y=0.23,
                                     text=ib_text_2, render_mode='canvas', text_font_size= '5pt')

                    fig.add_layout(ib_text_add_2)

                    oob_text_add_1 = Label(x=0.28, y=0.09,
                                     text=oob_text_1, render_mode='canvas', text_font_size= '5pt')

                    fig.add_layout(oob_text_add_1)

                    oob_text_add_2 = Label(x=0.28, y=0.00,
                                     text=oob_text_2, render_mode='canvas', text_font_size= '5pt')

                    fig.add_layout(oob_text_add_2)


                    fig.quad(top=0.47, bottom=0, left=0.12, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.20,0.30,color='green',size=8)
                    fig.circle(0.20,0.10,color='orange',size=8)
                else:
                    fig.legend.location = "bottom_right"
                    fig.legend.visible = True
            elif plot_num == 2:
                if width == 475:
                    ib_text_add = Label(x=0.52, y=0.03,
                                     text=ib_text, render_mode='canvas', text_font_size= '10pt')

                    fig.add_layout(ib_text_add)

                    fig.quad(top=0.10, bottom=0, left=0.42, right=1, color='white', alpha=0.4,  line_color='lightgrey')

                    fig.circle(0.47,0.05,color='green',size=8)
                elif width == 316:
                    ib_text_add = Label(x=0.30, y=0.02,
                                     text=ib_text, render_mode='canvas', text_font_size= '10pt')

                    fig.add_layout(ib_text_add)

                    fig.quad(top=0.10, bottom=0, left=0.20, right=1, color='white', alpha=0.4,  line_color='lightgrey')

                    fig.circle(0.25,0.05,color='green',size=8)
                elif width == 237:
                    ib_text_1 = "IB (AUC = {:.2f}".format(auc_ib[2])
                    ib_text_2 = "+/- {:.2f})".format((auc_ib[1] - auc_ib[0])/2)


                    ib_text_add_1 = Label(x=0.38, y=0.09,
                                     text=ib_text_1, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(ib_text_add_1)

                    ib_text_add_2 = Label(x=0.38, y=0.00,
                                     text=ib_text_2, render_mode='canvas', text_font_size= '6.8pt')


                    fig.add_layout(ib_text_add_2)

                    fig.quad(top=0.2, bottom=0, left=0.20, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.27,0.10,color='green',size=8)
                elif width == 190:
                    ib_text_1 = "IB (AUC ="
                    ib_text_2 = "{:.2f} +/- {:.2f})".format(auc_ib[2], (auc_ib[1] - auc_ib[0])/2)


                    ib_text_add_1 = Label(x=0.28, y=0.09,
                                     text=ib_text_1, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(ib_text_add_1)

                    ib_text_add_2 = Label(x=0.28, y=0.00,
                                     text=ib_text_2, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(ib_text_add_2)


                    fig.quad(top=0.24, bottom=0, left=0.12, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.20,0.10,color='green',size=8)
                elif width == 158:
                    ib_text_1 = "IB (AUC ="
                    ib_text_2 = "{:.2f}+/- {:.2f})".format(auc_ib[2], (auc_ib[1] - auc_ib[0])/2)


                    ib_text_add_1 = Label(x=0.28, y=0.09,
                                     text=ib_text_1, render_mode='canvas', text_font_size= '6pt')

                    fig.add_layout(ib_text_add_1)

                    ib_text_add_2 = Label(x=0.28, y=0,
                                     text=ib_text_2, render_mode='canvas', text_font_size= '6pt')

                    fig.add_layout(ib_text_add_2)


                    fig.quad(top=0.24, bottom=0, left=0.12, right=1, color='white', alpha=1,   line_color='lightgrey')

                    fig.circle(0.20,0.10,color='green',size=8)
                elif width == 135:
                    ib_text_1 = "IB (AUC ="
                    ib_text_2 = "{:.2f} +/- {:.2f})".format(auc_ib[2], (auc_ib[1] - auc_ib[0])/2)


                    ib_text_add_1 = Label(x=0.28, y=0.09,
                                     text=ib_text_1, render_mode='canvas', text_font_size= '5pt')

                    fig.add_layout(ib_text_add_1)

                    ib_text_add_2 = Label(x=0.28, y=0.00,
                                     text=ib_text_2, render_mode='canvas', text_font_size= '5pt')

                    fig.add_layout(ib_text_add_2)


                    fig.quad(top=0.24, bottom=0, left=0.12, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.20,0.10,color='green',size=8)
                else:
                    fig.legend.location = "bottom_right"
                    fig.legend.visible = True


            elif plot_num == 3:
                if width == 475:
                    oob_text_add = Label(x=0.52, y=0.03,
                                     text=oob_text, render_mode='canvas', text_font_size= '10pt')

                    fig.add_layout(oob_text_add)


                    fig.quad(top=0.10, bottom=0, left=0.42, right=1, color='white', alpha=0.4,  line_color='lightgrey')

                    fig.circle(0.47,0.05,color='orange',size=8)
                    # fig.circle(0.47,0.07,color='orange',size=8)
                elif width == 316:

                    oob_text_add = Label(x=0.22, y=0.02,
                                     text=oob_text, render_mode='canvas', text_font_size= '10pt')

                    fig.add_layout(oob_text_add)


                    fig.quad(top=0.10, bottom=0, left=0.12, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.17,0.05,color='orange',size=8)
                elif width == 237:

                    oob_text_1 = "OOB (AUC ="
                    oob_text_2 = "{:.2f}+/- {:.2f})".format(auc_oob[2],(auc_oob[1] - auc_oob[0])/2)


                    oob_text_add_1 = Label(x=0.38, y=0.09,
                                     text=oob_text_1, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(oob_text_add_1)

                    oob_text_add_2 = Label(x=0.38, y=0.00,
                                     text=oob_text_2, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(oob_text_add_2)


                    fig.quad(top=0.2, bottom=0, left=0.20, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.27,0.10,color='orange',size=8)
                elif width == 190:

                    oob_text_1 = "OOB (AUC ="
                    oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_oob[2],(auc_oob[1] - auc_oob[0])/2)

                    oob_text_add_1 = Label(x=0.28, y=0.09,
                                     text=oob_text_1, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(oob_text_add_1)

                    oob_text_add_2 = Label(x=0.28, y=0.00,
                                     text=oob_text_2, render_mode='canvas', text_font_size= '6.8pt')

                    fig.add_layout(oob_text_add_2)


                    fig.quad(top=0.24, bottom=0, left=0.12, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.20,0.10,color='orange',size=8)
                elif width == 158:

                    oob_text_1 = "OOB (AUC ="
                    oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_oob[2],(auc_oob[1] - auc_oob[0])/2)



                    oob_text_add_1 = Label(x=0.28, y=0.09,
                                     text=oob_text_1, render_mode='canvas', text_font_size= '6pt')

                    fig.add_layout(oob_text_add_1)

                    oob_text_add_2 = Label(x=0.28, y=0.00,
                                     text=oob_text_2, render_mode='canvas', text_font_size= '6pt')

                    fig.add_layout(oob_text_add_2)


                    fig.quad(top=0.24, bottom=0, left=0.12, right=1, color='white', alpha=1,  line_color='lightgrey')

                    fig.circle(0.20,0.10,color='orange',size=8)
                elif width == 135:

                    oob_text_1 = "OOB (AUC ="
                    oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_oob[2],(auc_oob[1] - auc_oob[0])/2)


                    oob_text_add_1 = Label(x=0.28, y=0.09,
                                     text=oob_text_1, render_mode='canvas', text_font_size= '5pt')

                    fig.add_layout(oob_text_add_1)

                    oob_text_add_2 = Label(x=0.28, y=0.00,
                                     text=oob_text_2, render_mode='canvas', text_font_size= '5pt')

                    fig.add_layout(oob_text_add_2)


                    fig.quad(top=0.24, bottom=0, left=0.12, right=1, color='white', alpha=1, line_color='lightgrey')

                    fig.circle(0.20,0.10,color='orange',size=8)
                else:
                    fig.legend.location = "bottom_right"
                    fig.legend.visible = True

    if train is None:
      return fig, auc_ib, auc_oob
    else:
      return fig, auc_ib, auc_oob


def roc_cv(Y_predfull, Y_predcv, Ytrue, width=450, height=350, xlabel="1-Specificity", ylabel="Sensitivity", legend=True, label_font_size="13pt", show_title=True, title_font_size="13pt", title="", plot_num=0, grid_line=False):

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
    # fig.line([0, 1], [0, 1], color="black", line_dash="dashed", line_width=2.5, legend="Equal Distribution Line")
    fig.line([0, 1], [0, 1], color="black", line_dash="dashed", alpha=0.8, line_width=1)
    if plot_num in [0, 1, 2, 4]:
        figline = fig.line("x", "y", color="green", line_width=2.5, alpha=0.8, legend="FULL (AUC = {:.2f})".format(auc_full), source=source)
        fig.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111}")]))
    else:
        pass
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

    data2 = {"x": fpr, "y": tpr_medci, "lowci": tpr_lowci, "uppci": tpr_uppci, "spec": spec2, "ci": ci2}

    source2 = ColumnDataSource(data=data2)

    if plot_num in [0, 1, 3, 4]:
        figline = fig.line("x", "y", color="orange", line_width=2.5, alpha=0.8, legend="CV (AUC = {:.2f} +/- {:.2f})".format(auc_medci, auc_ci,), source=source2)
        fig.add_tools(HoverTool(renderers=[figline], tooltips=[("Specificity", "@spec{1.111}"), ("Sensitivity", "@y{1.111} (+/- @ci{1.111})")]))

        # Figure: add 95CI band
        figband = Band(base="x", lower="lowci", upper="uppci", level="underlay", fill_alpha=0.1, line_width=0.5, line_color="black", fill_color="orange", source=source2)
        fig.add_layout(figband)
    else:
        pass
    # Change font size
    if show_title is True:
        fig.title.text = "AUC FULL ({}) & AUC CV ({} +/- {})".format(np.round(auc_full, 2), np.round(auc_medci, 2), np.round(auc_ci, 2))
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
    # if legend is False:
    #     fig.legend.visible = False

    if grid_line == False:
        fig.xgrid.visible = False
        fig.ygrid.visible = False

    # Legend Manually because of bokeh issue
    auc_full = np.round(auc_full, 2)
    auc_cv1 = np.round(auc_medci, 2)
    auc_cv2 = np.round(auc_ci, 2)
    ib_text = "FULL (AUC = {:.2f})".format(auc_full)
    oob_text = "CV (AUC = {:.2f} +/- {:.2f})".format(auc_cv1, auc_cv2)

    fig.legend.visible =  False
    if legend == True:
        if plot_num in [0,1,4]:
            if width == 475:
                ib_text_add = Label(x=0.52, y=0.15,
                                 text=ib_text, render_mode='canvas', text_font_size= '10pt')

                fig.add_layout(ib_text_add)

                oob_text_add = Label(x=0.52, y=0.05,
                                 text=oob_text, render_mode='canvas', text_font_size= '10pt')

                fig.add_layout(oob_text_add)


                fig.quad(top=0.25, bottom=0, left=0.42, right=1, color='white', alpha=0.4,line_color='lightgrey')

                fig.circle(0.47,0.17,color='green',size=8)
                fig.circle(0.47,0.07,color='orange',size=8)
            elif width == 316:
                ib_text_add = Label(x=0.30, y=0.15,
                                 text=ib_text, render_mode='canvas', text_font_size= '10pt')

                fig.add_layout(ib_text_add)

                oob_text_add = Label(x=0.30, y=0.05,
                                 text=oob_text, render_mode='canvas', text_font_size= '10pt')

                fig.add_layout(oob_text_add)


                fig.quad(top=0.25, bottom=0, left=0.20, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.25,0.18,color='green',size=8)
                fig.circle(0.25,0.08,color='orange',size=8)
            elif width == 237:
                ib_text_add = Label(x=0.30, y=0.15,
                                 text=ib_text, render_mode='canvas', text_font_size= '6.4pt')

                fig.add_layout(ib_text_add)

                oob_text_add = Label(x=0.30, y=0.05,
                                 text=oob_text, render_mode='canvas', text_font_size= '6.4pt')

                fig.add_layout(oob_text_add)


                fig.quad(top=0.25, bottom=0, left=0.20, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.25,0.18,color='green',size=8)
                fig.circle(0.25,0.08,color='orange',size=8)
            elif width == 190:
                ib_text_1 = "FULL (AUC ="
                ib_text_2 = "{:.2f})".format(auc_full)

                oob_text_1 = "CV (AUC ="
                oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_cv1, auc_cv2)


                ib_text_add_1 = Label(x=0.28, y=0.32,
                                 text=ib_text_1, render_mode='canvas', text_font_size= '6.8pt')

                fig.add_layout(ib_text_add_1)

                ib_text_add_2 = Label(x=0.28, y=0.23,
                                 text=ib_text_2, render_mode='canvas', text_font_size= '6.8pt')

                fig.add_layout(ib_text_add_2)

                oob_text_add_1 = Label(x=0.28, y=0.09,
                                 text=oob_text_1, render_mode='canvas', text_font_size= '6.8pt')

                fig.add_layout(oob_text_add_1)

                oob_text_add_2 = Label(x=0.28, y=0.00,
                                 text=oob_text_2, render_mode='canvas', text_font_size= '6.8pt')

                fig.add_layout(oob_text_add_2)


                fig.quad(top=0.47, bottom=0, left=0.12, right=1, color='white', alpha=1, line_color='lightgrey')

                fig.circle(0.20,0.30,color='green',size=8)
                fig.circle(0.20,0.10,color='orange',size=8)
            elif width == 158:
                ib_text_1 = "FULL (AUC ="
                ib_text_2 = "{:.2f})".format(auc_full)

                oob_text_1 = "CV (AUC ="
                oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_cv1, auc_cv2)


                ib_text_add_1 = Label(x=0.28, y=0.32,
                                 text=ib_text_1, render_mode='canvas', text_font_size= '6pt')

                fig.add_layout(ib_text_add_1)

                ib_text_add_2 = Label(x=0.28, y=0.23,
                                 text=ib_text_2, render_mode='canvas', text_font_size= '6pt')

                fig.add_layout(ib_text_add_2)

                oob_text_add_1 = Label(x=0.28, y=0.09,
                                 text=oob_text_1, render_mode='canvas', text_font_size= '6pt')

                fig.add_layout(oob_text_add_1)

                oob_text_add_2 = Label(x=0.28, y=0.00,
                                 text=oob_text_2, render_mode='canvas', text_font_size= '6pt')

                fig.add_layout(oob_text_add_2)


                fig.quad(top=0.47, bottom=0, left=0.12, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.20,0.30,color='green',size=8)
                fig.circle(0.20,0.10,color='orange',size=8)
            elif width == 135:
                ib_text_1 = "FULL (AUC ="
                ib_text_2 = "{:.2f})".format(auc_full)

                oob_text_1 = "CV (AUC ="
                oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_cv1, auc_cv2)


                ib_text_add_1 = Label(x=0.28, y=0.32,
                                 text=ib_text_1, render_mode='canvas', text_font_size= '5pt')

                fig.add_layout(ib_text_add_1)

                ib_text_add_2 = Label(x=0.28, y=0.23,
                                 text=ib_text_2, render_mode='canvas', text_font_size= '5pt')

                fig.add_layout(ib_text_add_2)

                oob_text_add_1 = Label(x=0.28, y=0.09,
                                 text=oob_text_1, render_mode='canvas', text_font_size= '5pt')

                fig.add_layout(oob_text_add_1)

                oob_text_add_2 = Label(x=0.28, y=0.00,
                                 text=oob_text_2, render_mode='canvas', text_font_size= '5pt')

                fig.add_layout(oob_text_add_2)


                fig.quad(top=0.47, bottom=0, left=0.12, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.20,0.30,color='green',size=8)
                fig.circle(0.20,0.10,color='orange',size=8)
            else:
                fig.legend.location = "bottom_right"
                fig.legend.visible = True
        elif plot_num == 2:
            if width == 475:
                ib_text_add = Label(x=0.52, y=0.03,
                                 text=ib_text, render_mode='canvas', text_font_size= '10pt')

                fig.add_layout(ib_text_add)

                fig.quad(top=0.10, bottom=0, left=0.42, right=1, color='white', alpha=0.4,line_color='lightgrey')

                fig.circle(0.47,0.05,color='green',size=8)
            elif width == 316:
                ib_text_add = Label(x=0.40, y=0.02,
                                 text=ib_text, render_mode='canvas', text_font_size= '10pt')

                fig.add_layout(ib_text_add)

                fig.quad(top=0.12, bottom=0, left=0.30, right=1, color='white', alpha=0.4,line_color='lightgrey')

                fig.circle(0.35,0.05, color='green',size=8)
            elif width == 237:
                ib_text_1 = "FULL (AUC ="
                ib_text_2 = "{:.2f})".format(auc_full)


                ib_text_add_1 = Label(x=0.38, y=0.09,
                                 text=ib_text_1, render_mode='canvas', text_font_size= '6.8pt')

                fig.add_layout(ib_text_add_1)

                ib_text_add_2 = Label(x=0.38, y=0.00,
                                 text=ib_text_2, render_mode='canvas', text_font_size= '6.8pt')


                fig.add_layout(ib_text_add_2)

                fig.quad(top=0.21, bottom=0, left=0.20, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.27,0.10,color='green',size=8)
            elif width == 190:
                ib_text_1 = "FULL (AUC ="
                ib_text_2 = "{:.2f})".format(auc_full)


                ib_text_add_1 = Label(x=0.28, y=0.09,
                                 text=ib_text_1, render_mode='canvas', text_font_size= '6.8pt')

                fig.add_layout(ib_text_add_1)

                ib_text_add_2 = Label(x=0.28, y=0.00,
                                 text=ib_text_2, render_mode='canvas', text_font_size= '6.8pt')

                fig.add_layout(ib_text_add_2)


                fig.quad(top=0.25, bottom=0, left=0.12, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.20,0.10,color='green',size=8)
            elif width == 158:
                ib_text_1 = "FULL (AUC ="
                ib_text_2 = "{:.2f})".format(auc_full)


                ib_text_add_1 = Label(x=0.28, y=0.09,
                                 text=ib_text_1, render_mode='canvas', text_font_size= '6pt')

                fig.add_layout(ib_text_add_1)

                ib_text_add_2 = Label(x=0.28, y=0,
                                 text=ib_text_2, render_mode='canvas', text_font_size= '6pt')

                fig.add_layout(ib_text_add_2)


                fig.quad(top=0.25, bottom=0, left=0.12, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.20,0.10,color='green',size=8)
            elif width == 135:
                ib_text_1 = "FULL (AUC ="
                ib_text_2 = "{:.2f})".format(auc_full)


                ib_text_add_1 = Label(x=0.28, y=0.09,
                                 text=ib_text_1, render_mode='canvas', text_font_size= '5pt')

                fig.add_layout(ib_text_add_1)

                ib_text_add_2 = Label(x=0.28, y=0.00,
                                 text=ib_text_2, render_mode='canvas', text_font_size= '5pt')

                fig.add_layout(ib_text_add_2)


                fig.quad(top=0.25, bottom=0, left=0.12, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.20,0.10,color='green',size=8)
            else:
                fig.legend.location = "bottom_right"
                fig.legend.visible = True


        elif plot_num == 3:
            if width == 475:
                oob_text_add = Label(x=0.52, y=0.03,
                                 text=oob_text, render_mode='canvas', text_font_size= '10pt')

                fig.add_layout(oob_text_add)


                fig.quad(top=0.10, bottom=0, left=0.42, right=1, color='white', alpha=0.4,line_color='lightgrey')

                fig.circle(0.47,0.05,color='orange',size=8)
                # fig.circle(0.47,0.07,color='orange',size=8)
            elif width == 316:

                oob_text_add = Label(x=0.27, y=0.02,
                                 text=oob_text, render_mode='canvas', text_font_size= '10pt')

                fig.add_layout(oob_text_add)


                fig.quad(top=0.11, bottom=0, left=0.17, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.22,0.05,color='orange',size=8)
            elif width == 237:

                oob_text_1 = "CV (AUC ="
                oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_cv1, auc_cv2)


                oob_text_add_1 = Label(x=0.38, y=0.09,
                                 text=oob_text_1, render_mode='canvas', text_font_size= '6.8pt')

                fig.add_layout(oob_text_add_1)

                oob_text_add_2 = Label(x=0.38, y=0.00,
                                 text=oob_text_2, render_mode='canvas', text_font_size= '6.8pt')

                fig.add_layout(oob_text_add_2)


                fig.quad(top=0.21, bottom=0, left=0.20, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.27,0.10,color='orange',size=8)
            elif width == 190:

                oob_text_1 = "CV (AUC ="
                oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_cv1, auc_cv2)

                oob_text_add_1 = Label(x=0.28, y=0.09,
                                 text=oob_text_1, render_mode='canvas', text_font_size= '6.8pt')

                fig.add_layout(oob_text_add_1)

                oob_text_add_2 = Label(x=0.28, y=0.00,
                                 text=oob_text_2, render_mode='canvas', text_font_size= '6.8pt')

                fig.add_layout(oob_text_add_2)


                fig.quad(top=0.24, bottom=0, left=0.12, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.20,0.10,color='orange',size=8)
            elif width == 158:

                oob_text_1 = "CV (AUC ="
                oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_cv1, auc_cv2)



                oob_text_add_1 = Label(x=0.28, y=0.09,
                                 text=oob_text_1, render_mode='canvas', text_font_size= '6pt')

                fig.add_layout(oob_text_add_1)

                oob_text_add_2 = Label(x=0.28, y=0.00,
                                 text=oob_text_2, render_mode='canvas', text_font_size= '6pt')

                fig.add_layout(oob_text_add_2)


                fig.quad(top=0.24, bottom=0, left=0.12, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.20,0.10,color='orange',size=8)
            elif width == 135:
                oob_text_1 = "CV (AUC ="
                oob_text_2 = "{:.2f} +/- {:.2f})".format(auc_cv1, auc_cv2)


                oob_text_add_1 = Label(x=0.28, y=0.09,
                                 text=oob_text_1, render_mode='canvas', text_font_size= '5pt')

                fig.add_layout(oob_text_add_1)

                oob_text_add_2 = Label(x=0.28, y=0.00,
                                 text=oob_text_2, render_mode='canvas', text_font_size= '5pt')

                fig.add_layout(oob_text_add_2)


                fig.quad(top=0.24, bottom=0, left=0.12, right=1, color='white', alpha=1,line_color='lightgrey')

                fig.circle(0.20,0.10,color='orange',size=8)
            else:
                fig.legend.location = "bottom_right"
                fig.legend.visible = True

    return fig


def per_method(bootstat, stat):
    """Calculates bootstrap confidence intervals using the percentile bootstrap interval."""
    if stat.ndim == 1:
        boot_ci = []
        # Calculate bootci for each component (peak), and append it to bootci
        for i in range(len(bootstat[0])):
            bootstat_i = [item[i] for item in bootstat]
            lower_ci = np.percentile(bootstat_i, 2.5)
            upper_ci = np.percentile(bootstat_i, 97.5)
            mid_ci = np.percentile(bootstat_i, 50)
            boot_ci.append([lower_ci, upper_ci, mid_ci])
        boot_ci = np.array(boot_ci)
    elif stat.ndim == 0:
        lower_ci = np.percentile(bootstat, 2.5)
        upper_ci = np.percentile(bootstat, 97.5)
        mid_ci = np.percentile(bootstat, 50)
        boot_ci = [lower_ci, upper_ci, mid_ci]
        boot_ci = np.array(boot_ci)
    # Recursive component (to get ndim = 1, and append)
    else:
        ncomp = stat.shape[1]
        boot_ci = []
        for k in range(ncomp):
            bootstat_k = []
            for j in range(len(bootstat)):
                bootstat_k.append(bootstat[j][:, k])
            boot_ci_k = per_method(bootstat_k, stat[:, k])
            boot_ci.append(boot_ci_k)
        boot_ci = np.array(boot_ci)
    return boot_ci


def cper_method(bootstat, stat):
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
            boot_ci_k = cper_method(bootstat_k, stat[:, k])
            boot_ci.append(boot_ci_k)
        boot_ci = np.array(boot_ci)
    return boot_ci


def bca_method(bootstat, stat, jackstat):
    """Calculates bootstrap confidence intervals using the bias-corrected and accelerated bootstrap interval."""
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
        z0 = -norm.ppf(prop, loc=0, scale=1)

        # new alpha
        jmean = np.mean(jackstat, axis=0)
        num = np.sum((jmean - jackstat) ** 3, axis=0)
        den = np.sum((jmean - jackstat) ** 2, axis=0)
        ahat = num / (6 * den ** (3 / 2))

        # Ignore warnings as they are delt with at line 123 with try/except
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zL = z0 + norm.ppf(0.05 / 2, loc=0, scale=1)
            pct1 = 100 * norm.cdf((z0 + zL / (1 - ahat * zL)))
            zU = z0 + norm.ppf((1 - 0.05 / 2), loc=0, scale=1)
            pct2 = 100 * norm.cdf((z0 + zU / (1 - ahat * zU)))
            zM = z0 + norm.ppf((0.5), loc=0, scale=1)
            pct3 = 100 * norm.cdf((z0 + zM / (1 - ahat * zM)))
            # pct3 = (pct1 + pct2) / 2
            # for i in range(len(pct3)):
            #     if np.isnan(pct3[i]) == True:
            #         pct3[i] = (pct2[i] + pct1[i]) / 2

        boot_ci = []
        for i in range(len(pct1)):
            bootstat_i = [item[i] for item in bootstat]
            try:
                append_low = np.percentile(bootstat_i, pct1[i])
                append_upp = np.percentile(bootstat_i, pct2[i])
                append_mid = np.percentile(bootstat_i, pct3[i])
            except ValueError:
                # Use BC (CPerc) as there is no skewness
                pct1 = 100 * norm.cdf((2 * z0 + zalpha))
                pct2 = 100 * norm.cdf((2 * z0 - zalpha))
                pct2 = 100 * norm.cdf((2 * z0))
                append_low = np.percentile(bootstat_i, pct1[i])
                append_upp = np.percentile(bootstat_i, pct2[i])
                append_mid = np.percentile(bootstat_i, pct2[i])
            boot_ci.append([append_low, append_upp, append_mid])

    # Recursive component (to get ndim = 1, and append)
    else:
        ncomp = stat.shape[1]
        boot_ci = []
        for k in range(ncomp):
            var = []
            var_jstat = []
            for j in range(len(bootstat)):
                var.append(bootstat[j][:, k])
            for m in range(len(jackstat)):
                var_jstat.append(jackstat[m][:, k])
            var_boot = bca_method(var, stat[:, k], var_jstat)
            boot_ci.append(var_boot)
        boot_ci = np.array(boot_ci)

    return boot_ci


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
