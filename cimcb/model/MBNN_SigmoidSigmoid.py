import numpy as np
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, LSTM, concatenate
from keras.layers import Dense
from .BaseModel import BaseModel
from ..utils import YpredCallback


class MBNN_SigmoidSigmoid(BaseModel):
    """2 Layer logistic-logistic neural network using Keras"""

    parametric = True
    bootlist = None

    def __init__(self, n_neurons_l1=2, n_neurons_l2=2, epochs=200, learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, loss="binary_crossentropy", batch_size=None, verbose=0):
        self.n_neurons_l1 = n_neurons_l1
        self.n_neurons_l2 = n_neurons_l2
        self.verbose = verbose
        self.n_epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov)

        self.__name__ = 'cimcb.model.MBNN_SigmoidSigmoid'
        self.__params__ = {'n_neurons_l1': n_neurons_l1, 'n_neurons_l2': n_neurons_l2, 'epochs': epochs, 'learning_rate': learning_rate, 'momentum': momentum, 'decay': decay, 'nesterov': nesterov, 'loss': loss, 'batch_size': batch_size, 'verbose': verbose}

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

        # If batch-size is None:
        if self.batch_size is None:
            self.batch_size = len(X)

        X1 = X[0]
        X2 = X[1]

        # Layer for X1
        input_X1 = Input(shape=(len(X1.T),))
        layer1_X1 = Dense(self.n_neurons_l1, activation="sigmoid")(input_X1)
        layer1_X1 = Model(inputs=input_X1, outputs=layer1_X1)

        # Layer for X2
        input_X2 = Input(shape=(len(X2.T),))
        layer1_X2 = Dense(self.n_neurons_l1, activation="sigmoid")(input_X2)
        layer1_X2 = Model(inputs=input_X2, outputs=layer1_X2)

        # Concatenate
        concat = concatenate([layer1_X1.output, layer1_X2.output])
        model_concat = Dense(self.n_neurons_l2, activation="sigmoid")(concat)
        model_concat = Dense(1, activation="sigmoid")(model_concat)

        self.model = Model(inputs=[layer1_X1.input, layer1_X2.input], outputs=model_concat)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])

        # If epoch_ypred is True, calculate ypred for each epoch
        if epoch_ypred is True:
            self.epoch = YpredCallback(self.model, X, epoch_xtest)
        else:
            self.epoch = Callback()

        # Fit
        self.model.fit([X1, X2], Y, epochs=self.n_epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=[self.epoch])

        # Not sure about the naming scheme (trying to match PLS)
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
