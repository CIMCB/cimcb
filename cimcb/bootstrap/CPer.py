import numpy as np
import scipy
import pandas as pd
from scipy.stats import norm
import math
import multiprocessing
from copy import deepcopy
from bokeh.layouts import widgetbox, gridplot, column, row, layout
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from .BaseBootstrap import BaseBootstrap
from itertools import combinations
from ..plot import scatterCI, boxplot, distribution, scatter, scatter_ellipse
from ..utils import nested_getattr, dict_95ci, dict_median_scores


class CPer(BaseBootstrap):
    """ Returns bootstrap confidence intervals using the bias-corrected boostrap interval.

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
        To return bootci, initalise then use method run().
    """

    def __init__(self, model, bootnum=100, seed=None, n_cores=-1, stratify=True):
        super().__init__(model=model, bootnum=bootnum, seed=seed, n_cores=n_cores, stratify=stratify)
        self.__name__ = "CPer"

    def calc_stat(self):
        super().calc_stat()

    def calc_bootidx(self):
        super().calc_bootidx()

    def calc_bootstat(self):
        super().calc_bootstat()

    def calc_bootci(self):
        self.bootci = {}
        for i in self.bootlist:
            self.bootci[i] = self.bootci_method(self.bootstat[i], self.stat[i])

    def run(self):
        self.calc_stat()
        self.calc_bootidx()
        self.calc_bootstat()
        self.calc_bootci()

    @staticmethod
    def bootci_method(bootstat, stat):
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
                boot_ci_k = CPer.bootci_method(bootstat_k, stat[:, k])
                boot_ci.append(boot_ci_k)
            boot_ci = np.array(boot_ci)
        return boot_ci
