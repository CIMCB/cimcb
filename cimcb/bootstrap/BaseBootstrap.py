import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from copy import deepcopy, copy
from ..utils import nested_getattr


class BaseBootstrap(ABC):
    """Base class for bootstrap: BC, BCA, and Perc."""

    @abstractmethod
    def __init__(self, model, X, Y, bootlist, bootnum=100, seed=None):
        try:
            self.model = deepcopy(model)  # Make a copy of the model
        except TypeError:
            self.model = copy(model)
        self.X = X
        self.Y = Y
        self.bootlist = bootlist
        self.bootnum = bootnum
        self.seed = seed
        self.bootidx = []
        self.bootstat = {}
        self.bootci = {}

    def calc_bootidx(self):
        """Generate indices for every resampled (with replacement) dataset."""
        np.random.seed(self.seed)
        self.bootidx = []
        for i in range(self.bootnum):
            bootidx_i = np.random.choice(len(self.Y), len(self.Y))
            self.bootidx.append(bootidx_i)

    def calc_bootstat(self):
        """Trains and tests model, then stores selected attributes (from self.bootlist) for each resampled dataset."""
        # Create an empty dictionary
        self.bootstat = {}
        for i in self.bootlist:
            self.bootstat[i] = []
        # Calculate bootstat for each bootstrap resample
        for i in tqdm(self.bootidx, desc="Bootstrap Resample"):
            X_res = self.X[i, :]
            Y_res = self.Y[i]
            self.model.train(X_res, Y_res)
            self.model.test(X_res)
            for j in self.bootlist:
                self.bootstat[j].append(nested_getattr(self.model, j))

    @abstractmethod
    def calc_bootci(self):
        """Calculates bootstrap confidence intervals using bootci_method."""
        pass

    @abstractmethod
    def run(self):
        """Runs every function and returns bootstrap confidence intervals (a dict of arrays)."""
        pass

    @abstractmethod
    def bootci_method(self):
        """Method used to calculate boostrap confidence intervals (Refer to: BC, BCA, or Perc)."""
        pass
