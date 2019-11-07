import numpy as np
import warnings
from tqdm import tqdm
from scipy.stats import norm
import time
import timeit
from joblib import Parallel, delayed
from .BaseBootstrap import BaseBootstrap
from ..utils import nested_getattr


class BCA(BaseBootstrap):
    """ Returns bootstrap confidence intervals using the bias-corrected and accelerated boostrap interval.

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
    """

    def __init__(self, model, bootnum=100, seed=None, n_cores=-1, stratify=True):
        super().__init__(model=model, bootnum=bootnum, seed=seed, n_cores=n_cores, stratify=stratify)
        self.jackidx = []
        self.jackstat = {}
        self.__name__ = "BCA"

    def calc_stat(self):
        super().calc_stat()

    def calc_jackidx(self):
        """Generate indices for every resampled (using jackknife technique) dataset."""
        self.jackidx = []
        base = np.arange(0, len(self.Y))
        for i in base:
            jack_delete = np.delete(base, i)
            self.jackidx.append(jack_delete)

    def calc_jackstat(self):
        """Trains and test model, then stores selected attributes (from self.bootlist) for each resampled (using jackknife technique) dataset."""
        self.jackstat = {}
        for i in self.bootlist:
            self.jackstat[i] = []
        try:
            stats_loop = Parallel(n_jobs=self.n_cores)(delayed(self._calc_jackstat_loop)(i) for i in tqdm(range(len(self.jackidx)), desc="2/2"))
        except:
            print("TerminatedWorkerError was raised due to excessive memory usage. n_cores was reduced to 1.")
            stats_loop = Parallel(n_jobs=1)(delayed(self._calc_jackstat_loop)(i) for i in tqdm(range(len(self.jackidx)), desc="2/2"))
            
        self.jackstat_test = stats_loop
        for i in stats_loop:
            i_dict = i[0]
            for key, value in i_dict.items():
                self.jackstat[key].append(value[0])

    def _calc_jackstat_loop(self, i):
        """Core component of calc_ypred."""
        # Set model
        model_i = self.model(**self.param)
        # Set X and Y
        X_res = self.X[self.jackidx[i], :]
        Y_res = self.Y[self.jackidx[i]]
        # Train and test
        if self.name == 'cimcb.model.NN_SigmoidSigmoid' or self.name == 'cimcb.model.NN_LinearSigmoid':
            model_i.train(X_res, Y_res, w1=self.w1, w2=self.w2)
        else:
            model_i.train(X_res, Y_res)
        model_i.test(X_res, Y_res)
        # # Get IB
        jackstatloop = {}
        for k in self.bootlist:
            jackstatloop[k] = []
        for j in self.bootlist:
            jackstatloop[j].append(nested_getattr(model_i, j))

        return [jackstatloop]

    def calc_bootidx(self):
        super().calc_bootidx()

    def calc_bootstat(self):
        """Trains and test model, then stores selected attributes (from self.bootlist) for each resampled dataset."""

        # Create an empty dictionary
        self.bootstat = {}
        for i in self.bootlist:
            self.bootstat[i] = []
        self.bootstat_oob = {}
        for i in self.bootlist:
            self.bootstat_oob[i] = []
        # Calculate bootstat for each bootstrap resample
        try:
            stats_loop = Parallel(n_jobs=self.n_cores)(delayed(self._calc_bootstat_loop)(i) for i in tqdm(range(self.bootnum), desc="1/2"))
        except:
            print("TerminatedWorkerError was raised due to excessive memory usage. n_cores was reduced to 1.")
            stats_loop = Parallel(n_jobs=1)(delayed(self._calc_bootstat_loop)(i) for i in tqdm(range(self.bootnum), desc="1/2"))
            
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

    def calc_bootci(self):
        self.bootci = {}
        for i in self.bootlist:
            try:
                self.bootci[i] = self.bootci_method(self.bootstat[i], self.stat[i], self.jackstat[i])
            except:
                pass

    def run(self):
        "Running ..."
        time.sleep(0.2)  # Sleep for 0.5 secs to finish printing

        # Start Timer
        start = timeit.default_timer()

        self.calc_stat()
        self.calc_jackidx()
        self.calc_bootidx()
        self.calc_bootstat()
        self.calc_jackstat()
        self.calc_bootci()

        time.sleep(0.2)  # Sleep for 0.5 secs to finish printing

        # Stop timer
        stop = timeit.default_timer()
        self.parallel_time = (stop - start) / 60
        print("Time taken: {:0.2f} minutes with {} cores".format(self.parallel_time, self.n_cores))

    @staticmethod
    def bootci_method(bootstat, stat, jackstat):
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
                #pct3 = (pct1 + pct2) / 2
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
            boot_ci = np.array(boot_ci)
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
                var_boot = BCA.bootci_method(var, stat[:, k], var_jstat)
                boot_ci.append(var_boot)
            boot_ci = np.array(boot_ci)

        return boot_ci
