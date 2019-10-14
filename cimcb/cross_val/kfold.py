import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from copy import deepcopy, copy
import timeit
import time
import multiprocessing
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from .BaseCrossVal import BaseCrossVal
from ..utils import binary_metrics, multiclass_metrics, dict_perc, dict_median


class kfold(BaseCrossVal):
    """ Exhaustitive search over param_dict calculating binary metrics.

    Parameters
    ----------
    model : object
        This object is assumed to store bootlist attributes in .model (e.g. modelPLS.model.x_scores_).

    X : array-like, shape = [n_samples, n_features]
        Predictor variables, where n_samples is the number of samples and n_features is the number of predictors.

    Y : array-like, shape = [n_samples, 1]
        Response variables, where n_samples is the number of samples.

    param_dict : dict
        List of attributes to calculate and return bootstrap confidence intervals.

    folds: : a positive integer, (default 10)
        The number of folds used in the computation.

    bootnum : a positive integer, (default 100)
        The number of bootstrap samples used in the computation for the plot.

    Methods
    -------
    Run: Runs all necessary methods prior to plot.

    Plot: Creates a R2/Q2 plot.
    """

    def __init__(self, model, X, Y, param_dict, folds=5, n_mc=1, n_boot=0, n_cores=-1, ci=95):
        super().__init__(model=model, X=X, Y=Y, param_dict=param_dict, folds=folds, n_mc=n_mc, n_boot=n_boot, n_cores=n_cores, ci=ci)
        self.crossval_idx = StratifiedKFold(n_splits=folds, shuffle=True)

    def calc_ypred(self):
        """Calculates ypred full and ypred cv."""

        time.sleep(0.5)  # Sleep for 0.5 secs to finish printing

        # Start Timer
        start = timeit.default_timer()

        # FULL
        full = Parallel(n_jobs=self.n_cores)(delayed(self._calc_full_loop)(i) for i in tqdm(range(len(self.param_list)), desc="1/2"))
        self.ypred_full = []
        self.x_scores_full = []
        self.y_loadings_ = []
        self.pctvar_ = []
        self.w1 = []
        self.w2 = []
        for i in range(len(self.param_list)):
            self.ypred_full.append(full[i][0])
            self.x_scores_full.append(full[i][1])
            self.y_loadings_.append(full[i][2])
            self.pctvar_.append(full[i][3])
            self.w1.append(full[i][4])
            self.w2.append(full[i][5])

        # for i in tqdm(range(len(self.param_list)), desc="1/2"):
        #     parami = self.param_list[i]
        #     model_i = self.model(**parami)
        #     # model_i.set_params(parami)
        #     # Full
        #     model_i.train(self.X, self.Y)
        #     ypred_full_i = model_i.test(self.X)
        #     self.ypred_full.append(ypred_full_i)
        #     self.x_scores_full.append(model_i.model.x_scores_)
        #     self.y_loadings_.append(model_i.model.y_loadings_)
        #     self.pctvar_.append(model_i.model.pctvar_)
        #     if model_i.__name__ == "cimcb.model.NN_SigmoidSigmoid":
        #         self.w1.append(model_i.model.w1)
        #         self.w2.append(model_i.model.w2)
        #     else:
        #         self.w1.append([0])
        #         self.w2.append([0])

        self.loop_w1 = self.w1 * self.n_mc
        self.loop_w2 = self.w2 * self.n_mc

        # Actual loop CV including Monte-Carlo reps
        self.loop_mc = self.param_list * self.n_mc
        ypred = Parallel(n_jobs=self.n_cores)(delayed(self._calc_cv_loop)(i) for i in tqdm(range(len(self.loop_mc)), desc="2/2"))

        # Split ypred into full / cv and put in final format
        # Format :::> self.ypred_full -> parameter_type -> monte-carlo -> y_true / y_pred
        self.ypred_cv = [[] for i in range(len(self.param_list))]
        self.x_scores_cv = [[] for i in range(len(self.param_list))]
        self.loop_mc_numbers = list(range(len(self.param_list))) * self.n_mc
        for i in range(len(self.loop_mc)):
            j = self.loop_mc_numbers[i]  # Location to append to
            self.ypred_cv[j].append(ypred[i][0])
            self.x_scores_cv[j].append(ypred[i][1])

        # Stop timer
        stop = timeit.default_timer()
        self.parallel_time = (stop - start) / 60
        print("Time taken: {:0.2f} minutes with {} cores".format(self.parallel_time, self.n_cores))

    def _calc_full_loop(self, i):
        parami = self.param_list[i]
        model_i = self.model(**parami)
        # model_i.set_params(parami)
        # Full
        model_i.train(self.X, self.Y)
        ypred_full_i = model_i.test(self.X)
        ypred_full = ypred_full_i
        x_scores_full = model_i.model.x_scores_
        y_loadings_ = model_i.model.y_loadings_
        pctvar_ = model_i.model.pctvar_
        if model_i.__name__ == "cimcb.model.NN_SigmoidSigmoid":
            w1 = model_i.model.w1
            w2 = model_i.model.w2
        else:
            w1 = 0
            w2 = 0
        return [ypred_full, x_scores_full, y_loadings_, pctvar_, w1, w2]

    def _calc_cv_loop(self, i):
        """Core component of calc_ypred."""
        # Set hyper - parameters
        params_i = self.loop_mc[i]
        model_i = self.model()
        model_i.set_params(params_i)
        # Full
        model_i.train(self.X, self.Y)
        if model_i.__name__ == "cimcb.model.NN_SigmoidSigmoid":
            model_i.train(self.X, self.Y, w1=self.loop_w1[i], w2=self.loop_w2[i])
        else:
            model_i.train(self.X, self.Y)

        # CV (for each fold)
        ypred_cv_i, x_scores_cv = self._calc_cv_ypred(model_i, self.X, self.Y)
        ypred_cv = ypred_cv_i
        return [ypred_cv_i, x_scores_cv]

    def calc_stats(self):
        """Calculates binary statistics from ypred full and ypred cv."""
        # Calculate for each parameter and append
        stats_list = []
        std_list = []
        self.full_loop = []
        self.cv_loop = []
        for i in range(len(self.param_list)):
            full_loop = []
            cv_loop = []

            # Get all monte-carlo
            for k in range(len(self.ypred_cv[i])):
                cv_mc = binary_metrics(self.Y, self.ypred_cv[i][k], parametric=self.model.parametric)
                cv_loop.append(cv_mc)

            # Average binary metrics
            stats_full_i = binary_metrics(self.Y, self.ypred_full[i], parametric=self.model.parametric)
            stats_cv_i = dict_median(cv_loop)

            # Rename columns
            stats_full_i = {k + "full": v for k, v in stats_full_i.items()}
            stats_cv_i = {k + "cv": v for k, v in stats_cv_i.items()}
            stats_cv_i["R²"] = stats_full_i.pop("R²full")
            stats_cv_i["Q²"] = stats_cv_i.pop("R²cv")

            # Combine and append
            stats_combined = {**stats_full_i, **stats_cv_i}
            stats_list.append(stats_combined)

            # Save loop -> full_loop is a placeholder
            self.full_loop.append(cv_loop)
            self.cv_loop.append(cv_loop)

            # Keep std if n_mc > 1
            if self.n_mc > 1:
                std_full_i = dict_perc(cv_loop, ci=self.ci)
                std_cv_i = dict_perc(cv_loop, ci=self.ci)
                std_full_i = {k + "full": v for k, v in std_full_i.items()}
                std_cv_i = {k + "cv": v for k, v in std_cv_i.items()}
                std_cv_i["R²"] = std_full_i.pop("R²full")
                std_cv_i["Q²"] = std_cv_i.pop("R²cv")
                std_combined = {**std_full_i, **std_cv_i}
                std_list.append(std_combined)

        self.table = self._format_table(stats_list)  # Transpose, Add headers
        self.table = self.table.reindex(index=np.sort(self.table.index))
        if self.n_mc > 1:
            self.table_std = self._format_table(std_list)  # Transpose, Add headers
            self.table_std = self.table_std.reindex(index=np.sort(self.table_std.index))
        return self.table

    def _calc_cv_ypred(self, model_i, X, Y):
        """Method used to calculate ypred cv."""
        ypred_cv_i = [None] * len(Y)
        x_scores_cv_i = [None] * len(Y)

        if len(X) == len(Y):
            for train, test in self.crossval_idx.split(X, Y):
                X_train = X[train, :]
                Y_train = Y[train]
                X_test = X[test, :]
                model_i.train(X_train, Y_train)
                ypred_cv_i_j = model_i.test(X_test)
                # Return value to y_pred_cv in the correct position # Better way to do this
                for (idx, val) in zip(test, ypred_cv_i_j):
                    ypred_cv_i[idx] = val.tolist()

                # Calc x_scores_cv is applicable
                if "model.x_scores_" in model_i.bootlist:
                    x_scores_cv_i_j = model_i.model.x_scores_
                    for (idx, val) in zip(test, x_scores_cv_i_j):
                        x_scores_cv_i[idx] = val.tolist()
        return ypred_cv_i, x_scores_cv_i

    def plot(self, metric="r2q2", scale=1, color_scaling="tanh", rotate_xlabel=True, legend=True, color_beta=[10, 10, 10], ci=95, diff1_heat=True, method='ratio', style=1, alt=True):
        super().plot(metric=metric, scale=scale, color_scaling=color_scaling, rotate_xlabel=rotate_xlabel, legend=legend, model="kfold", color_beta=color_beta, ci=ci, diff1_heat=diff1_heat, style=style, method=method, alt=alt)
