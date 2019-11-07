import math
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import combinations
from sklearn.cross_decomposition import PLSRegression
from bokeh.plotting import output_notebook, show
from bokeh.layouts import gridplot
from bokeh.plotting import ColumnDataSource, figure
from .BaseModel import BaseModel
from ..plot import scatter, distribution, boxplot
from abc import ABC, abstractmethod, abstractproperty
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import scipy
import collections
import math
from scipy.stats import logistic
from itertools import combinations
from copy import deepcopy, copy
from bokeh.layouts import widgetbox, gridplot, column, row, layout
from bokeh.models import HoverTool, Band
from bokeh.models.widgets import DataTable, Div, TableColumn
from bokeh.models.annotations import Title
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from scipy import interp
from sklearn import metrics
from sklearn.utils import resample
from ..plot import scatter, scatterCI, boxplot, distribution, permutation_test
from ..utils import binary_metrics, binary_evaluation


class PLS_NIPALS(BaseModel):
    """ Partial least-squares regression using the SIMPLS algorithm.

    Parameters
    ----------
    n_components : int, (default 2)
        Number of components to keep.

    Methods
    -------
    train : Fit model to data.

    test : Apply model to test data.

    evaluate : Evaluate model.

    calc_bootci : Calculate bootstrap intervals for plot_featureimportance.

    plot_featureimportance : Plot coefficient and Variable Importance in Projection (VIP).

    plot_permutation_test : Perform a permutation test and plot.
    """

    parametric = True
    # bootlist = ["model.vip_", "model.coef_"]  # list of metrics to bootstrap
    # bootlist = ["model.vip_", "model.coef_", "model.x_loadings_", "model.x_scores_", "Y_pred", "model.pctvar_", "model.y_loadings_"]  # list of metrics to bootstrap
    bootlist = ["model.vip_", "model.coef_", "model.x_loadings_", "model.x_scores_", "Y_pred", "model.pctvar_", "model.y_loadings_", "model.metrics"]

    def __init__(self, n_components=2):
        self.model = PLSRegression(n_components=n_components)  # Should change this to an empty model
        self.n_component = n_components
        self.k = n_components

        self.__name__ = 'cimcb.model.PLS_NIPALS'
        self.__params__ = {'n_components': n_components}

    def set_params(self, params):
        self.__init__(**params)

    def train(self, X, Y):
        """ Fit the PLS model, save additional stats (as attributes) and return Y predicted values.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Predictor variables, where n_samples is the number of samples and n_features is the number of predictors.

        Y : array-like, shape = [n_samples, 1]
            Response variables, where n_samples is the number of samples.

        Returns
        -------
        y_pred_train : array-like, shape = [n_samples, 1]
            Predicted y score for samples.
        """
        # Error check
        # X, Y = self.input_check(X, Y)

        # Fit model
        self.model.fit(X, Y)

        # Calculate vip, pctvar (Explained variance in X) and flatten coef_ for future use
        # meanX = np.mean(X, axis=0)
        # X0 = X - meanX
        # self.model.pctvar_ = sum(abs(self.model.x_loadings_) ** 2) / sum(sum(abs(X0) ** 2)) * 100
        # self.model.vip_ = vip(self.model)
        # self.model.coef_ = self.model.coef_.flatten()
        y_pred_train = self.model.predict(X).flatten()

        self.model.pctvar_ = []
        for i in range(self.n_component):
            Y_pred = np.dot(self.model.x_scores_[:, i].reshape(-1, 1), self.model.y_loadings_[:, i].reshape(-1, 1).T) * Y.std(axis=0, ddof=1) + Y.mean(axis=0)
            explainedvar = r2_score(Y, Y_pred) * 100
            self.model.pctvar_.append(explainedvar)
        self.model.pctvar_ = np.array(self.model.pctvar_)

        # T = self.model.x_scores_
        # W = self.model.x_weights_
        # Q = self.model.y_loadings_
        # w0, w1 = W.shape
        # s = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
        # s_sum = np.sum(s, axis=0)
        # w_norm = np.array([(W[:, i] / np.linalg.norm(W[:, i]))
        #                    for i in range(w1)])
        # self.model.vip_ = np.sqrt(w0 * np.sum(s * w_norm.T ** 2, axis=1) / s_sum)

        t = self.model.x_scores_
        w = self.model.x_weights_
        q = self.model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        self.model.vip_ = vips
        # Calculate and return Y predicted value
        y_pred_train = self.model.predict(X).flatten()
        self.model.coef_ = self.model.coef_.flatten()

        self.model.y_loadings_ = self.model.y_weights_
        self.model.x_scores = t
        self.Y_pred = y_pred_train  # Y_pred vs. Y_pred_train
        self.Y_true = Y
        self.X = X
        self.Y = Y  # Y vs. Y_true

        self.metrics_key = []
        self.model.eval_metrics_ = []
        bm = binary_evaluation(Y, y_pred_train)
        for key, value in bm.items():
            self.model.eval_metrics_.append(value)
            self.metrics_key.append(key)

        return y_pred_train

    def test(self, X, Y=None):
        """Calculate and return Y predicted value.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Test variables, where n_samples is the number of samples and n_features is the number of predictors.

        Returns
        -------
        y_pred_test : array-like, shape = [n_samples, 1]
            Predicted y score for samples.
        """
        # Convert to X to numpy array if a DataFrame
        if isinstance(X, pd.DataFrame or pd.Series):
            X = np.array(X)

        # Overwrite x_scores_ from model.fit with using test X (or do model.x_scores_test_) ?
        self.model.x_scores_ = self.model.transform(X)
        # Calculate and return Y predicted value
        y_pred_test = self.model.predict(X).flatten()
        self.Y_pred = y_pred_test

        if Y is not None:
            self.metrics_key = []
            self.model.eval_metrics_ = []
            bm = binary_evaluation(Y, y_pred_test)
            for key, value in bm.items():
                self.model.eval_metrics_.append(value)
                self.metrics_key.append(key)

            self.model.eval_metrics_ = np.array(self.model.eval_metrics_)
        return y_pred_test
