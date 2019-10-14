import numpy as np
import warnings
from tqdm import tqdm
from scipy.stats import norm
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

    def __init__(self, model, X, Y, bootlist, bootnum=100, seed=None):
        super().__init__(model=model, X=X, Y=Y, bootlist=bootlist, bootnum=bootnum, seed=seed)
        self.stat = {}
        self.jackidx = []
        self.jackstat = {}

    def calc_stat(self):
        """Stores selected attributes (from self.bootlist) for the original model."""
        self.stat = {}
        for i in self.bootlist:
            self.stat[i] = nested_getattr(self.model, i)

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
        for i in tqdm(self.jackidx, desc="Jackknife Resample"):
            X_res = self.X[i, :]
            Y_res = self.Y[i]
            self.model.train(X_res, Y_res)
            for i in self.bootlist:
                self.jackstat[i].append(nested_getattr(self.model, i))

    def calc_bootidx(self):
        super().calc_bootidx()

    def calc_bootstat(self):
        super().calc_bootstat()

    def calc_bootci(self):
        self.bootci = {}
        for i in self.bootlist:
            self.bootci[i] = self.bootci_method(self.bootstat[i], self.stat[i], self.jackstat[i])

    def run(self):
        self.calc_stat()
        self.calc_bootidx()
        self.calc_jackidx()
        self.calc_bootstat()
        self.calc_jackstat()
        self.calc_bootci()
        return self.bootci

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

            boot_ci = []
            for i in range(len(pct1)):
                bootstat_i = [item[i] for item in bootstat]
                try:
                    append_low = np.percentile(bootstat_i, pct1[i])
                    append_upp = np.percentile(bootstat_i, pct2[i])
                except ValueError:
                    # USE BC if BCA is not possible
                    pct1 = 100 * norm.cdf((2 * z0 + zalpha))
                    pct2 = 100 * norm.cdf((2 * z0 - zalpha))
                    append_low = np.percentile(bootstat_i, pct1[i])
                    append_upp = np.percentile(bootstat_i, pct2[i])
                boot_ci.append([append_low, append_upp])
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
