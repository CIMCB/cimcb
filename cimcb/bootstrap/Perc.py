import numpy as np
from .BaseBootstrap import BaseBootstrap
from ..utils import nested_getattr


class Perc(BaseBootstrap):
    """ Returns bootstrap confidence intervals using the percentile boostrap interval.

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

    def __init__(self, model, X, Y, bootlist, bootnum=100, seed=None):
        super().__init__(model=model, X=X, Y=Y, bootlist=bootlist, bootnum=bootnum, seed=seed)

    def calc_stat(self):
        """Stores selected attributes (from self.bootlist) for the original model."""
        self.stat = {}
        for i in self.bootlist:
            self.stat[i] = nested_getattr(self.model, i)

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
        return self.bootci

    @staticmethod
    def bootci_method(bootstat, stat):
        """Calculates bootstrap confidence intervals using the percentile bootstrap interval."""
        if bootstat[0].ndim == 1:
            boot_ci = []
            # Calculate bootci for each component (peak), and append it to bootci
            for i in range(len(bootstat[0])):
                bootstat_i = [item[i] for item in bootstat]
                lower_ci = np.percentile(bootstat_i, 2.5)
                upper_ci = np.percentile(bootstat_i, 97.5)
                boot_ci.append([lower_ci, upper_ci])
            boot_ci = np.array(boot_ci)

        # Recursive component (to get ndim = 1, and append)
        else:
            ncomp = stat.shape[1]
            boot_ci = []
            for k in range(ncomp):
                bootstat_k = []
                for j in range(len(bootstat)):
                    bootstat_k.append(bootstat[j][:, k])
                boot_ci_k = Perc.bootci_method(bootstat_k, stat[:, k])
                boot_ci.append(boot_ci_k)
            boot_ci = np.array(boot_ci)
        return boot_ci
