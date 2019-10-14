import numpy as np
from keras.callbacks import Callback
from keras.optimizers import SGD
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from .BaseModel import BaseModel
from ..utils import YpredCallback


class NN_LinearLogit_Sklearn(BaseModel):
    """2 Layer linear-linear neural network using Keras"""

    parametric = False
    bootlist = None

    def __init__(self, n_nodes=2, epochs2=200, learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, loss="binary_crossentropy", batch_size=None, verbose=0):
        self.n_nodes = n_nodes
        self.verbose = verbose
        self.n_epochs = epochs2
        self.k = n_nodes
        self.batch_size = batch_size
        self.loss = loss
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.optimizer = "sgd"

    def train(self, X, Y, epoch_ypred=False, epoch_xtest=None):
        """ Fit the neural network model, save additional stats (as attributes) and return Y predicted values.

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

        # If batch-size is None:
        if self.batch_size is None:
            self.batch_size = len(X)

        # Ensure array and error check
        X, Y = self.input_check(X, Y)

        self.model = MLPClassifier(hidden_layer_sizes=(self.n_nodes,),
                                   activation='identity',
                                   solver=self.optimizer,
                                   learning_rate_init=self.learning_rate,
                                   momentum=self.momentum,
                                   batch_size=self.batch_size,
                                   nesterovs_momentum=False,
                                   max_iter=self.n_epochs)

        # Fit
        self.model.fit(X, Y)

        y_pred_train = self.model.predict(X)

        # Storing X, Y, and Y_pred
        self.Y_pred = y_pred_train
        self.X = X
        self.Y = Y
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
        y_pred_test = self.model.predict(X)
        return y_pred_test
