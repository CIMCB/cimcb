import numpy as np
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from .BaseModel import BaseModel
from ..utils import YpredCallback


class NN_SigmoidSigmoid(BaseModel):
    """2 Layer sigmoid-sigmoid neural network using Keras.

    Parameters
    ----------
    n_neurons : int, (default 2)
        Number of neurons in the hidden layer.

    epochs : int, (default 200)
        Number of iterations in the model training

    learning_rate : float, (default 0.01)
        The parameter that controls the step-size in updating the weights.

    momentum : float, (default 0.0)
        Value that alters the learning rate schedule, whereby increasing the learning rate when the error cost gradient continue in the same direction.

    decay: float, (default 0.0)
        Value that alters the learning rate schedule, whereby decreasing the learning rate after each epoch/iteration.

    loss : string, (default "binary_crossentropy")
        Function used to calculate the error of the model during the model training process known as backpropagation.

    Methods
    -------
    train : Fit model to data.

    test : Apply model to test data.

    evaluate : Evaluate model.

    booteval : Bootstrap evaluation.
    """

    parametric = True  # Calculate R2/Q2 for cross_val

    def __init__(self, n_neurons=2, epochs=200, learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, loss="binary_crossentropy", batch_size=None, verbose=0):
        self.n_neurons = n_neurons
        self.verbose = verbose
        self.n_epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov)

        self.__name__ = 'cimcb.model.NN_SigmoidSigmoid'
        self.__params__ = {'n_neurons': n_neurons, 'epochs': epochs, 'learning_rate': learning_rate, 'momentum': momentum, 'decay': decay, 'nesterov': nesterov, 'loss': loss, 'batch_size': batch_size, 'verbose': verbose}

    def set_params(self, params):
        self.__init__(**params)

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
        self.X = X
        self.Y = Y

        # If batch-size is None:
        if self.batch_size is None:
            self.batch_size = len(X)

        self.model = Sequential()
        self.model.add(Dense(self.n_neurons, activation="sigmoid", input_dim=len(X.T)))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])

        # If epoch_ypred is True, calculate ypred for each epoch
        if epoch_ypred is True:
            self.epoch = YpredCallback(self.model, X, epoch_xtest)
        else:
            self.epoch = Callback()

        # Fit
        self.model.fit(X, Y, epochs=self.n_epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=[self.epoch])
        y_pred_train = self.model.predict(X).flatten()

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
        y_pred_test = self.model.predict(X).flatten()
        return y_pred_test
