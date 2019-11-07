import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from .BaseModel import BaseModel
from ..utils import binary_metrics, binary_evaluation


class RF(BaseModel):
    """Random forest"""

    parametric = True
    bootlist = None  # list of metrics to bootstrap

    bootlist = ["Y_pred", "model.eval_metrics_"]  # list of metrics to bootstrap

    def __init__(self, n_estimators=100, max_features="auto", max_depth=None, criterion="gini", min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, n_jobs=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, n_jobs=n_jobs)
        self.k = n_estimators

        self.__name__ = 'cimcb.model.RF'
        self.__params__ = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'criterion': criterion, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_leaf_nodes': max_leaf_nodes, 'n_jobs': n_jobs}

    def set_params(self, params):
        self.__init__(**params)

    def train(self, X, Y):
        """ Fit the RF model, save additional stats (as attributes) and return Y predicted values.

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

        self.model.y_loadings_ = np.array([0, 0, 0])
        self.model.x_scores_ = np.array([0, 0, 0])
        self.model.pctvar_ = np.array([0, 0, 0])

        # Storing X, Y, and Y_pred
        self.X = X
        self.Y = Y
        self.Y_pred = y_pred_train

        self.metrics_key = []
        self.model.eval_metrics_ = []
        bm = binary_evaluation(Y, y_pred_train)
        for key, value in bm.items():
            self.model.eval_metrics_.append(value)
            self.metrics_key.append(key)

        self.Y_train = Y
        self.Y_pred_train = y_pred_train

        self.model.eval_metrics_ = np.array(self.model.eval_metrics_)

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

        # Calculate and return Y predicted value
        y_pred_test = np.array(self.model.predict_proba(X)[:, self.pred_index])

        if Y is not None:
            self.metrics_key = []
            self.model.eval_metrics_ = []
            bm = binary_evaluation(Y, y_pred_test)
            for key, value in bm.items():
                self.model.eval_metrics_.append(value)
                self.metrics_key.append(key)

            self.model.eval_metrics_ = np.array(self.model.eval_metrics_)

        self.Y_pred = y_pred_test
        return y_pred_test
