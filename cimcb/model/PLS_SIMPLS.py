import numpy as np
import pandas as pd
from .BaseModel import BaseModel


class PLS_SIMPLS(BaseModel):
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

    booteval : Bootstrap evaluation.
    """

    parametric = True  # Calculate R2/Q2 for cross_val

    def __init__(self, n_components=2):
        self.model = lambda: None
        self.n_component = n_components

        self.__name__ = 'cimcb.model.PLS_SIMPLS'
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
        X, Y = self.input_check(X, Y)

        # Calculates and store attributes of PLS SIMPLS
        Xscores, Yscores, Xloadings, Yloadings, Weights, Beta = self.pls_simpls(X, Y, ncomp=self.n_component)
        self.model.x_scores_ = Xscores
        self.model.y_scores_ = Yscores
        self.model.x_loadings_ = Xloadings
        self.model.y_loadings_ = Yloadings
        self.model.x_weights_ = Weights
        self.model.beta_ = Beta

        # Calculate pctvar, flatten coef_ and vip for future use
        meanX = np.mean(X, axis=0)
        X0 = X - meanX
        self.model.pctvar_ = sum(abs(self.model.x_loadings_) ** 2) / sum(sum(abs(X0) ** 2)) * 100
        self.model.coef_ = Beta[1:]
        W0 = Weights / np.sqrt(np.sum(Weights ** 2, axis=0))
        sumSq = np.sum(Xscores ** 2, axis=0) * np.sum(Yloadings ** 2, axis=0)
        self.model.vip_ = np.sqrt(len(Xloadings) * np.sum(sumSq * W0 ** 2, axis=1) / np.sum(sumSq, axis=0))

        # Calculate and return Y predicted value
        newX = np.insert(X, 0, np.ones(len(X)), axis=1)
        y_pred_train = np.matmul(newX, Beta)

        # Storing X, Y, and Y_pred
        self.X = X
        self.Y = Y
        self.Y_pred = y_pred_train
        return y_pred_train

    def test(self, X):
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

        self.model.x_scores_ = np.dot(X, self.model.x_weights_)

        # Calculate and return Y predicted value
        newX = np.insert(X, 0, np.ones(len(X)), axis=1)
        y_pred_test = np.matmul(newX, self.model.beta_)
        return y_pred_test

    @staticmethod
    def pls_simpls(X, Y, ncomp=2):
        """PLS SIMPLS method. Refer to https://doi.org/10.1016/0169-7439(93)85002-X"""

        # Error check that X and Y match
        n, dx = X.shape
        ny = len(Y)
        if ny != n:
            raise ValueError("X and Y must have the same number of rows")

        # Error check for ncomp < maxncomp
        maxncomp = min(n - 1, dx)
        if ncomp > maxncomp:
            raise ValueError("ncomp must be less than or equal to {} for these data.".format(maxncomp))

        # Center both predictors and response
        meanX = np.mean(X, axis=0)
        meanY = np.mean(Y, axis=0)
        X0 = X - meanX
        Y0 = Y - meanY
        n, dx = X0.shape
        dy = 1

        # Empty arrays for loadings, scores, and weights
        Xloadings = np.zeros([dx, ncomp])
        Yloadings = np.zeros([dy, ncomp])
        Xscores = np.zeros([n, ncomp])
        Yscores = np.zeros([n, ncomp])
        Weights = np.zeros([dx, ncomp])

        # An orthonormal basis for the X loadings
        V = np.zeros([dx, ncomp])
        Cov = np.matmul(X0.T, Y0)
        Cov = Cov.reshape(len(Cov), 1)

        for i in range(ncomp):
            # Find unit length ti=X0*ri and ui=Y0*ci whose covariance, ri'*X0'*Y0*ci, is jointly maximized, subject to ti'*tj=0 for j=1:(i-1).
            ri, si, ci = np.linalg.svd(Cov)
            ri = ri[:, 0]
            si = si[0]
            ci = ci[0]
            ti = np.matmul(X0, ri)
            normti = np.linalg.norm(ti)
            ti = ti / normti
            qi = si * ci / normti

            Xloadings[:, i] = np.matmul(X0.T, ti)
            Yloadings[:, i] = qi
            Xscores[:, i] = ti
            Yscores[:, i] = (Y0 * qi).tolist()
            Weights[:, i] = ri / normti  #

            # Update the orthonormal basis with modified Gram Schmidt
            vi = Xloadings[:, i]
            for repeat in range(2):
                for j in range(i):
                    vj = V[:, j]
                    vi = vi - np.matmul(vj.T, vi) * vj

            vi = vi / np.linalg.norm(vi)
            V[:, i] = vi

            # Deflate Cov
            vim = vi * np.matmul(vi.T, Cov)
            Cov = Cov - vim.reshape(len(vim), 1)
            Vi = V[:, 0: i + 1]
            Vim = np.dot(Vi, np.matmul(Vi.T, Cov)).flatten()
            Cov = Cov - Vim.reshape(len(Vim), 1)

        # Use modified Gram-Schmidt, repeated twice.
        for i in range(ncomp):
            ui = Yscores[:, i]
            for repeat in range(2):
                for j in range(i):
                    tj = Xscores[:, j]
                    ui = ui - np.dot(np.matmul(tj.T, ui), tj)
            Yscores[:, i] = ui

        Beta = np.matmul(Weights, Yloadings.T)
        Beta_add = meanY - np.dot(meanX, Beta)
        Beta = np.insert(Beta, 0, Beta_add)
        return Xscores, Yscores, Xloadings, Yloadings, Weights, Beta
