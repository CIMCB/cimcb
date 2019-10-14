import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from .BaseModel import BaseModel


class SVM(BaseModel):
    """Support Vector Machine"""

    parametric = True
    bootlist = None  # list of metrics to bootstrap

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma="auto", tol=0.001, max_iter=-1):
        self.model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, probability=True, tol=tol, max_iter=max_iter)
        self.k = None

        self.__name__ = 'cimcb.model.SVM'
        self.__params__ = {'C': C, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'tol': tol, 'max_iter': max_iter}

    def set_params(self, params):
        self.__init__(**params)

    def train(self, X, Y):
        """ Fit the SVM model, save additional stats (as attributes) and return Y predicted values.

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
        self.model.fit(X, Y)

        # Predict_proba was designed for multi-groups...
        # This makes it sure that y_pred is correct
        y_pred = self.model.predict_proba(X)
        pred_0 = roc_auc_score(Y, y_pred[:, 0])
        pred_1 = roc_auc_score(Y, y_pred[:, 1])
        if pred_0 > pred_1:
            self.pred_index = 0
        else:
            self.pred_index = 1

        # Calculate and return Y prediction value
        y_pred_train = np.array(self.model.predict_proba(X)[:, self.pred_index])

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
        y_pred_test = np.array(self.model.predict_proba(X)[:, self.pred_index])
        return y_pred_test
