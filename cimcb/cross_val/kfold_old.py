import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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

        # Actual loop including Monte-Carlo reps
        self.loop_mc = self.param_list * self.n_mc
        ypred = Parallel(n_jobs=self.n_cores)(delayed(self._calc_ypred_loop)(i) for i in tqdm(range(len(self.loop_mc))))

        # Split ypred into full / cv and put in final format
        # Format :::> self.ypred_full -> parameter_type -> monte-carlo -> y_true / y_pred
        self.ypred_full = [[] for i in range(len(self.param_list))]
        self.ypred_cv = [[] for i in range(len(self.param_list))]
        self.x_scores_full = [[] for i in range(len(self.param_list))]
        self.x_scores_cv = [[] for i in range(len(self.param_list))]
        self.loop_mc_numbers = list(range(len(self.param_list))) * self.n_mc
        for i in range(len(self.loop_mc)):
            j = self.loop_mc_numbers[i]  # Location to append to
            self.ypred_full[j].append(ypred[i][0])
            self.ypred_cv[j].append(ypred[i][1])
            self.x_scores_full[j].append(ypred[i][2])
            self.x_scores_cv[j].append(ypred[i][3])

        # Stop timer
        stop = timeit.default_timer()
        self.parallel_time = (stop - start) / 60
        print("Time taken: {:0.2f} minutes with {} cores".format(self.parallel_time, self.n_cores))

    def calc_ypred_epoch(self):
        """Calculates ypred full and ypred cv for each epoch (edge case)."""

        time.sleep(0.5)  # Sleep for 0.5 secs to finish printing

        # Start Timer
        start = timeit.default_timer()

        # Set param to the max -> Actual loop including Monte-Carlo reps
        epoch_param = [self.param_list[-1]]
        self.loop_mc = epoch_param * self.n_mc
        ypred, x_scores_ = Parallel(n_jobs=self.n_cores)(delayed(self._calc_ypred_loop_epoch)(i) for i in tqdm(range(len(self.loop_mc))))
        self.x = ypred
        # Get epoch list
        self.epoch_list = []
        for m in self.param_list2:
            for t, v in m.items():
                self.epoch_list.append(v - 1)

        self.x_scores_ = x_scores_
        # Split ypred into full / cv and put in final format
        # Format :::> self.ypred_full -> parameter_type -> monte-carlo -> y_true / y_pred
        # Note, we need to pull out the specific epochs from the model
        self.ypred_full = [[] for i in range(len(self.epoch_list))]
        self.ypred_cv = [[] for i in range(len(self.epoch_list))]
        for i in range(len(self.loop_mc)):
            for j in range(len(self.epoch_list)):
                self.ypred_full[j].append(ypred[i][0][j])
                self.ypred_cv[j].append(ypred[i][1][j])

        # Stop timer
        stop = timeit.default_timer()
        self.parallel_time = (stop - start) / 60
        print("Time taken: {:0.2f} minutes with {} cores".format(self.parallel_time, self.n_cores))

    def _calc_ypred_loop(self, i):
        """Core component of calc_ypred."""
        # Set hyper - parameters
        params_i = self.loop_mc[i]
        model_i = self.model()
        model_i.set_params(params_i)
        # Full
        model_i.train(self.X, self.Y)
        ypred_full_i = model_i.test(self.X)
        ypred_full = ypred_full_i
        # Calc x_scores_full is applicable
        x_scores_full = [None] * len(self.Y)
        if "model.x_scores_" in model_i.bootlist:
            x_scores_full = model_i.model.x_scores_
        # CV (for each fold)
        ypred_cv_i, x_scores_cv = self._calc_cv_ypred(model_i, self.X, self.Y)
        ypred_cv = ypred_cv_i
        return [ypred_full, ypred_cv, x_scores_full, x_scores_cv]

    def _calc_ypred_loop_epoch(self, i):
        """Core component of calc_ypred."""
        # Put ypred into standard format
        epoch_list = []
        for i in self.param_list2:
            for k, v in i.items():
                epoch_list.append(v - 1)
        # Set hyper-parameters
        param = self.loop_mc[-1]
        model_i = self.model(**param)
        # Full
        model_i.train(self.X, self.Y, epoch_ypred=True, epoch_xtest=self.X)
        ypred_full_i = model_i.epoch.Y_test
        ypred_full = []
        for i in range(len(epoch_list)):
            actual_epoch = epoch_list[i]
            ypred_full.append(ypred_full_i[actual_epoch])
        # CV
        # if Y is one-hot encoded, flatten it for Stratified Kfold
        try:
            if len(Y[0]) > 1:
                dummy = pd.DataFrame(Y)
                stratY = dummy.idxmax(axis=1)
            else:
                stratY = Y
        except TypeError:
            stratY = Y
        # split
        if len(self.X) == len(self.Y):
            fold_split = []
            for train, test in self.crossval_idx.split(self.X, stratY):
                fold_split.append((train, test))
        else:
            fold_split = []
            for train, test in self.crossval_idx.split(self.X[0], stratY):
                fold_split.append((train, test))
        # Split into train and test
        ypred_cv_i = np.zeros((len(self.Y), len(epoch_list)))
        for i in range(len(fold_split)):
            train, test = fold_split[i]
            if len(self.X) == len(self.Y):
                X_train = self.X[train, :]
                Y_train = self.Y[train]
                X_test = self.X[test, :]
                Y_test = self.Y[test]
            else:
                # Multiblock
                X0 = self.X[0]
                X1 = self.X[1]
                Y = self.Y
                X0_train = X0[train, :]
                X1_train = X1[train, :]
                X_train = [X0_train, X1_train]
                Y_train = Y[train]
                X0_test = X0[test, :]
                X1_test = X1[test, :]
                X_test = [X0_test, X1_test]
            # Full
            model_i.train(X_train, Y_train, epoch_ypred=True, epoch_xtest=X_test)
            ypred_cv_i_j = model_i.epoch.Y_test
            for j in range(len(epoch_list)):
                ypred_ypred = ypred_cv_i_j[epoch_list[j]]
                for (idx, val) in zip(test, ypred_ypred):
                    ypred_cv_i[idx, j] = val.tolist()
        ypred_cv = []
        for i in range(len(ypred_cv_i.T)):
            ypred_cv.append(ypred_cv_i[:, i])
        return [ypred_full, ypred_cv]

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
            for k in range(len(self.ypred_full[i])):

                # Check if binary
                try:
                    if len(self.Y[0]) > 1:
                        multiclass = True
                    else:
                        multiclass = False
                except TypeError:
                    multiclass = False

                if multiclass == True:
                    full_mc = multiclass_metrics(self.Y, self.ypred_full[i][k], parametric=self.model.parametric)
                    cv_mc = multiclass_metrics(self.Y, self.ypred_cv[i][k], parametric=self.model.parametric)
                else:
                    full_mc = binary_metrics(self.Y, self.ypred_full[i][k], parametric=self.model.parametric)
                    cv_mc = binary_metrics(self.Y, self.ypred_cv[i][k], parametric=self.model.parametric)
                full_loop.append(full_mc)
                cv_loop.append(cv_mc)

            # Average binary metrics
            stats_full_i = full_loop[0]
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
        if self.n_mc > 1:
            self.table_std = self._format_table(std_list)  # Transpose, Add headers
        return self.table

    def _calc_cv_ypred(self, model_i, X, Y):
        """Method used to calculate ypred cv."""
        ypred_cv_i = [None] * len(Y)
        x_scores_cv_i = [None] * len(Y)

        # if Y is one-hot encoded, flatten it for Stratified Kfold
        try:
            if len(Y[0]) > 1:
                dummy = pd.DataFrame(Y)
                stratY = dummy.idxmax(axis=1)
            else:
                stratY = Y
        except TypeError:
            stratY = Y

        if len(X) == len(Y):
            for train, test in self.crossval_idx.split(X, stratY):
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

        else:
            # Multiblock study
            X0 = X[0]
            X1 = X[1]
            for train, test in self.crossval_idx.split(X0, stratY):
                X0_train = X0[train, :]
                X1_train = X1[train, :]
                X_train = [X0_train, X1_train]
                Y_train = Y[train]
                X0_test = X0[test, :]
                X1_test = X1[test, :]
                X_test = [X0_test, X1_test]
                model_i.train(X_train, Y_train)
                ypred_cv_i_j = model_i.test(X_test)
                # Return value to y_pred_cv in the correct position # Better way to do this
                for (idx, val) in zip(test, ypred_cv_i_j):
                    ypred_cv_i[idx] = val.tolist()

        return ypred_cv_i, x_scores_cv_i

    def plot(self, metric="r2q2", scale=1, color_scaling="tanh", rotate_xlabel=True, legend="bottom_right", color_beta=[10, 10, 10], ci=95, diff1_heat=True):
        super().plot(metric=metric, scale=scale, color_scaling=color_scaling, rotate_xlabel=rotate_xlabel, legend=legend, model="kfold", color_beta=color_beta, ci=ci, diff1_heat=diff1_heat)
