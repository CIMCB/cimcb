import numpy as np
from joblib import Parallel, delayed
import timeit
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from .BaseCrossVal import BaseCrossVal
from ..utils import binary_metrics, dict_perc, dict_median


class holdout(BaseCrossVal):
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

    split: : a positive number between 0 and 1, (default 0.8)
        The split for the train and test split.

    Methods
    -------
    Run: Runs all necessary methods prior to plot.

    Plot: Creates a R2/Q2 plot.
    """

    def __init__(self, model, X, Y, param_dict, folds=None, n_mc=1, n_boot=0, n_cores=-1, ci=95, test_size=0.2, stratify=True):

        super().__init__(model=model, X=X, Y=Y, param_dict=param_dict, folds=folds, n_mc=n_mc, n_boot=n_boot, n_cores=n_cores, ci=ci)

        if folds is not None:
            print("You are using holdout not kfold, so folds has no effect.")

        # Save holdout specific inputs
        self.test_size = test_size
        if stratify is True:
            self.stratify = Y
        else:
            self.stratify = None

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
        self.loop_mc_numbers = list(range(len(self.param_list))) * self.n_mc
        for i in range(len(self.loop_mc)):
            j = self.loop_mc_numbers[i]  # Location to append to
            self.ypred_full[j].append(ypred[i][0])
            self.ypred_cv[j].append(ypred[i][1])

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
        ypred = Parallel(n_jobs=self.n_cores)(delayed(self._calc_ypred_loop_epoch)(i) for i in tqdm(range(len(self.loop_mc))))

        # Get epoch list
        self.epoch_list = []
        for m in self.param_list2:
            for t, v in m.items():
                self.epoch_list.append(v - 1)

        # Split ypred into full / cv and put in final format
        # Format :::> self.ypred_full -> parameter_type -> monte-carlo -> y_true / y_pred
        # Note, we need to pull out the specific epochs from the model
        self.ypred_full = [[] for i in range(len(self.epoch_list))]
        self.ypred_cv = [[] for i in range(len(self.epoch_list))]
        for i in range(len(self.loop_mc)):
            for j in range(len(self.epoch_list)):
                actual_epoch = self.epoch_list[j]
                self.ypred_full[j].append([ypred[i][0][0][0], ypred[i][0][0][1][actual_epoch]])
                self.ypred_cv[j].append([ypred[i][1][0][0], ypred[i][1][0][1][actual_epoch]])

        # Stop timer
        stop = timeit.default_timer()
        self.parallel_time = (stop - start) / 60
        print("Time taken: {:0.2f} minutes with {} cores".format(self.parallel_time, self.n_cores))

    def _calc_ypred_loop(self, i):
        """Core component of calc_ypred."""
        # Set x and y
        if len(self.X) == len(self.Y):
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=self.test_size, stratify=self.stratify)
        else:
            X0_train, X0_test, X1_train, X1_test, Y_train, Y_test = train_test_split(self.X[0], self.X[1], self.Y, test_size=self.test_size, stratify=self.stratify)
            X_train = [X0_train, X1_train]
            X_test = [X0_test, X1_test]

        # Set hyper - parameters
        params_i = self.loop_mc[i]
        model_i = self.model()
        model_i.set_params(params_i)
        # Split into train and test
        ypred_full_i = model_i.train(X_train, Y_train)
        ypred_cv_i = model_i.test(X_test)
        # Get ypred full cv
        ypred_full = [Y_train, ypred_full_i]
        ypred_cv = [Y_test, ypred_cv_i]
        return [ypred_full, ypred_cv]

    def _calc_ypred_loop_epoch(self, i):
        """Core component of calc_ypred_epoch."""
        # Set inputs
        Y_full = []
        Y_cv = []
        if len(self.X) == len(self.Y):
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=self.test_size, stratify=self.stratify)
        else:
            X0_train, X0_test, X1_train, X1_test, Y_train, Y_test = train_test_split(self.X[0], self.X[1], self.Y, test_size=self.test_size, stratify=self.stratify)
            X_train = [X0_train, X1_train]
            X_test = [X0_test, X1_test]
        # Set hyper - parameters
        params_i = self.loop_mc[i]
        model_i = self.model()
        model_i.set_params(params_i)
        # Train model with epoch_ypred=True
        model_i.train(X_train, Y_train, epoch_ypred=True, epoch_xtest=X_test)
        Y_full_split = model_i.epoch.Y_train
        Y_full.append([Y_train, Y_full_split])
        Y_cv_split = model_i.epoch.Y_test
        Y_cv.append([Y_test, Y_cv_split])
        return [Y_full, Y_cv]

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
                full_mc = binary_metrics(self.ypred_full[i][k][0], self.ypred_full[i][k][1], parametric=self.model.parametric)
                cv_mc = binary_metrics(self.ypred_cv[i][k][0], self.ypred_cv[i][k][1], parametric=self.model.parametric)
                full_loop.append(full_mc)
                cv_loop.append(cv_mc)

            # Average binary metrics
            stats_full_i = dict_median(full_loop)
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
            self.full_loop.append(full_loop)
            self.cv_loop.append(cv_loop)

            # Keep std if n_mc > 1
            if self.n_mc > 1:
                std_full_i = dict_perc(full_loop, ci=self.ci)
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

    def plot(self, metric="r2q2", scale=1, color_scaling="tanh", rotate_xlabel=True, legend="bottom_right", color_beta=[10, 10, 10], ci=95, diff1_heat=True):
        super().plot(metric=metric, scale=scale, color_scaling=color_scaling, rotate_xlabel=rotate_xlabel, legend=legend, model="holdout", color_beta=color_beta, ci=ci, diff1_heat=diff1_heat)
