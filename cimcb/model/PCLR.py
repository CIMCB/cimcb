import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from .BaseModel import BaseModel
from sklearn.metrics import roc_auc_score
from ..utils import binary_metrics, binary_evaluation


class PCLR(BaseModel):
    """ Principal component logistic regression.

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
    bootlist = ["model.coef_", "Y_pred", "model.eval_metrics_"]  # list of metrics to bootstrap

    def __init__(self, n_components=2):
        self.model = PCA(n_components=n_components)
        self.regrmodel = LogisticRegression(solver="liblinear")
        self.k = n_components

        self.__name__ = 'cimcb.model.PCLR'
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
        self.model.coef_ = np.dot(self.model.x_loadings_, self.regrmodel.coef_.flatten())
        self.model.pctvar_ = self.model.explained_variance_
        self.model.x_weights_ = self.model.components_.T
        self.model.y_loadings_ = self.regrmodel.coef_.reshape(1, len(self.regrmodel.coef_.flatten()))

        # Calculate and return Y prediction value
        #y_pred_train = self.regrmodel.predict(self.model.x_scores_).flatten()
        # Predict_proba was designed for multi-groups...
        # This makes it sure that y_pred is correct
        y_pred = self.regrmodel.predict_proba(self.model.x_scores_)
        pred_0 = roc_auc_score(Y, y_pred[:, 0])
        pred_1 = roc_auc_score(Y, y_pred[:, 1])
        if pred_0 > pred_1:
            self.pred_index = 0
        else:
            self.pred_index = 1

        # Calculate and return Y prediction value
        y_pred_train = np.array(self.regrmodel.predict_proba(self.model.x_scores_)[:, self.pred_index])

        self.Y_train = Y
        self.Y_pred_train = y_pred_train
        self.Y_pred = y_pred_train
        self.X = X
        self.Y = Y
        self.metrics_key = []
        self.model.eval_metrics_ = []
        bm = binary_evaluation(Y, y_pred_train)
        for key, value in bm.items():
            self.model.eval_metrics_.append(value)
            self.metrics_key.append(key)

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
        newX = self.model.transform(X)
        #y_pred_test = self.regrmodel.predict(newX).flatten()
        # Calculate and return Y predicted value
        y_pred_test = np.array(self.regrmodel.predict_proba(newX)[:, self.pred_index])

        # Calculate and return Y predicted value
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
