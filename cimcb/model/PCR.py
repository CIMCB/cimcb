import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from .BaseModel import BaseModel


class PCR(BaseModel):
    """ Principal component regression.

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
    bootlist = ["model.coef_"]  # list of metrics to bootstrap

    def __init__(self, n_components=2):
        self.model = PCA(n_components=n_components)
        self.regrmodel = LinearRegression()
        self.k = n_components

        self.__name__ = 'cimcb.model.PCR'
        self.__params__ = {'n_components': n_components}

    def set_params(self, params):
        self.__init__(**params)

    def train(self, X, Y):
        """ Fit the PCR model, save additional stats (as attributes) and return Y predicted values.

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

        # Ensure array and error check
        X, Y = self.input_check(X, Y)

        # Fit the model
        self.model.fit(X)
        self.model.x_scores_ = self.model.transform(X)
        self.regrmodel.fit(self.model.x_scores_, Y)

        # Save x_loadings, coef, pctvar, x_weights, y_loadings and vip
        self.model.x_loadings_ = self.model.components_.T
        self.model.coef_ = np.dot(self.model.x_loadings_, self.regrmodel.coef_)
        self.model.pctvar_ = self.model.explained_variance_
        self.model.x_weights_ = self.model.components_.T
        self.model.y_loadings_ = self.regrmodel.coef_.reshape(1, len(self.regrmodel.coef_))

        # Calculate and return Y prediction value
        y_pred_train = self.regrmodel.predict(self.model.x_scores_).flatten()

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

        # Calculate and return Y predicted value
        newX = self.model.transform(X)
        y_pred_test = self.regrmodel.predict(newX).flatten()
        return y_pred_test
