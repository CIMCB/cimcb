import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from .BaseModel import BaseModel


class RF(BaseModel):
    """Random forest.

    Parameters
    ----------
    max_depth : int or None, (default None)
        Maximum depth allowed for each tree

    min_samples_leaf : int or float, (default 2)
        Minimum number of samples required at each leaf node after a split in a tree. This value can either be an integer or a fraction of the total number of samples.

    n_estimators : int, (default 100)
        Number of trees in the forest.

    max_features : int, float, string or None, (default "sqrt”)
        Number of features considered at each split in a tree.

    criterion: string, (default "gini")
        Function used to measure the quality of the split in a tree. This is 'gini' for Gini impurity or 'entropy' for information gain

    min_samples_split : int or float, (default 2)
        Minimum number of samples required for a split in a tree. This value can either be an integer or a fraction of the total number of samples

    max_leaf_nodes : int or None, (default None)
        Maximum number of leaf nodes in a tree.

    Methods
    -------
    train : Fit model to data.

    test : Apply model to test data.

    evaluate : Evaluate model.

    booteval : Bootstrap evaluation.
    """

    parametric = True  # Calculate R2/Q2 for cross_val

    def __init__(self, n_estimators=100, max_features="auto", max_depth=None, criterion="gini", min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, n_jobs=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, n_jobs=n_jobs)

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