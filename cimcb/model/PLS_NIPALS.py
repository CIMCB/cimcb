import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from .BaseModel import BaseModel


class PLS_NIPALS(BaseModel):
    """ Partial least-squares regression using the NIPALS algorithm.

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
        self.model = PLSRegression(n_components=n_components)
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
        X, Y = self.input_check(X, Y)

        # Fit model
        self.model.fit(X, Y)

        # Calculate pctvar (Explained variance in X)
        self.model.pctvar_ = (sum(abs(self.model.x_scores_) ** 2) / sum(sum(abs(X) ** 2)) * 100)

        T = self.model.x_scores_
        W = self.model.x_weights_
        Q = self.model.y_loadings_
        w0, w1 = W.shape
        s = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
        s_sum = np.sum(s, axis=0)
        w_norm = np.array([(W[:, i] / np.linalg.norm(W[:, i]))
                           for i in range(w1)])
        self.model.vip_ = np.sqrt(w0 * np.sum(s * w_norm.T ** 2, axis=1) / s_sum)

        # Calculate and return Y predicted value
        y_pred_train = self.model.predict(X)

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

        # Overwrite x_scores_
        self.model.x_scores_ = self.model.transform(X)

        # Calculate and return Y predicted value
        y_pred_test = self.model.predict(X)
        return y_pred_test
