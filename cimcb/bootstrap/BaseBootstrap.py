import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import multiprocessing
import timeit
import time
from copy import deepcopy, copy
from itertools import combinations
from pydoc import locate
from joblib import Parallel, delayed
import importlib
from ..utils import nested_getattr


class BaseBootstrap(ABC):
    """Base class for bootstrap: BC, BCA, and Perc."""

    @abstractmethod
    def __init__(self, model, bootnum=100, seed=None, n_cores=-1):
        self.X = model.X
        self.Y = model.Y
        self.name = model.__name__
        if self.name == 'cimcb.model.NN_SigmoidSigmoid':
            self.w1 = model.model.w1
            self.w2 = model.model.w2
        self.bootlist = model.bootlist
        self.bootnum = bootnum
        self.seed = seed
        self.bootidx = []
        self.bootstat = {}
        self.bootci = {}
        self.param = model.__params__
        self.model = locate(model.__name__)

        # if n_cores = -1, set n_cores to max_cores
        max_num_cores = multiprocessing.cpu_count()
        self.n_cores = n_cores
        if self.n_cores > max_num_cores:
            self.n_cores = -1
            print("Number of cores set too high. It will be set to the max number of cores in the system.", flush=True)
        if self.n_cores == -1:
            self.n_cores = max_num_cores
            print("Number of cores set to: {}".format(max_num_cores))

        self.stat = {}
        for i in self.bootlist:
            self.stat[i] = nested_getattr(model, i)

    def calc_bootidx(self):
        """Generate indices for every resampled (with replacement) dataset."""
        np.random.seed(self.seed)
        self.bootidx = []
        self.bootidx_oob = []
        for i in range(self.bootnum):
            bootidx_i = np.random.choice(len(self.Y), len(self.Y))
            bootidx_oob_i = np.array(list(set(range(len(self.Y))) - set(bootidx_i)))
            self.bootidx.append(bootidx_i)
            self.bootidx_oob.append(bootidx_oob_i)

    def calc_bootstat(self):
        """Trains and test model, then stores selected attributes (from self.bootlist) for each resampled dataset."""

        "Running ..."
        time.sleep(0.5)  # Sleep for 0.5 secs to finish printing

        # Start Timer
        start = timeit.default_timer()

        # Create an empty dictionary
        self.bootstat = {}
        for i in self.bootlist:
            self.bootstat[i] = []
        self.bootstat_oob = {}
        for i in self.bootlist:
            self.bootstat_oob[i] = []
        # Calculate bootstat for each bootstrap resample
        stats_loop = Parallel(n_jobs=self.n_cores)(delayed(self._calc_bootstat_loop)(i) for i in tqdm(range(self.bootnum)))
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

        # Stop timer
        stop = timeit.default_timer()
        self.parallel_time = (stop - start) / 60
        print("Time taken: {:0.2f} minutes with {} cores".format(self.parallel_time, self.n_cores))

    def _calc_bootstat_loop(self, i):
        """Core component of calc_ypred."""
        # Set model
        model_i = self.model(**self.param)
        # Set X and Y
        X_res = self.X[self.bootidx[i], :]
        Y_res = self.Y[self.bootidx[i]]
        # Train and test
        if self.name == 'cimcb.model.NN_SigmoidSigmoid':
            model_i.train(X_res, Y_res, w1=self.w1, w2=self.w2)
        else:
            model_i.train(X_res, Y_res)
        model_i.test(X_res)
        # # Get IB
        bootstatloop = {}
        for k in self.bootlist:
            bootstatloop[k] = []
        for j in self.bootlist:
            bootstatloop[j].append(nested_getattr(model_i, j))
        # # Get OOB
        bootstatloop_oob = {}
        for k in self.bootlist:
            bootstatloop_oob[k] = []
        X_res_oob = self.X[self.bootidx_oob[i], :]
        model_i.test(X_res_oob)
        for j in self.bootlist:
            bootstatloop_oob[j].append(nested_getattr(model_i, j))
        return [bootstatloop, bootstatloop_oob]

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
