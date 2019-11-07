import numpy as np
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, LSTM, concatenate
from keras.layers import Dense
from .BaseModel import BaseModel
from ..utils import YpredCallback, binary_evaluation


class MBNN_SigmoidSigmoid(BaseModel):
    """2 Layer logistic-logistic neural network using Keras"""

    parametric = True
    bootlist = ["Y_pred", "model.eval_metrics_"]  # list of metrics to bootstrap

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
            batch_size = len(X)
        else:
            batch_size = self.batch_size

        #X = np.array(X)
        X1 = X[0]
        X2 = X[1]

        # Layers in loop
        layer1 = []
        for i in X:
            input_i = Input(shape=(len(i.T),))
            layer1_i = Dense(self.n_neurons_l1, activation="sigmoid")(input_i)
            layer1_i = Model(inputs=input_i, outputs=layer1_i)
            layer1.append(layer1_i)

        # Concatenate
        concat = concatenate([i.output for i in layer1])
        model_concat = Dense(self.n_neurons_l2, activation="sigmoid")(concat)
        model_concat = Dense(1, activation="sigmoid")(model_concat)

        self.model = Model(inputs=[i.input for i in layer1], outputs=model_concat)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])

        self.metrics_key = []
        self.model.eval_metrics_ = []

        self.model.pfi_acc_ = np.zeros((1, len(Y)))
        self.model.pfi_r2q2_ = np.zeros((1, len(Y)))
        self.model.pfi_auc_ = np.zeros((1, len(Y)))
        self.model.vip_ = np.zeros((1, len(Y)))
        self.model.coef_ = np.zeros((1, len(Y)))

        self.model.y_loadings_ = np.array([0, 0, 0])
        self.model.x_scores_ = np.array([0, 0, 0])
        self.model.x_loadings_ = np.array([0, 0, 0])
        self.model.pctvar_ = np.array([0, 0, 0])

        # Fit
        self.model.fit(X, Y, epochs=self.n_epochs, batch_size=batch_size, verbose=self.verbose)

        # Not sure about the naming scheme (trying to match PLS)
        y_pred_train = self.model.predict(X).flatten()


        self.model.eval_metrics_ = []
        bm = binary_evaluation(Y, y_pred_train)
        for key, value in bm.items():
            self.model.eval_metrics_.append(value)
            self.metrics_key.append(key)
        self.model.eval_metrics_ = np.array(self.model.eval_metrics_)

        # Storing X, Y, and Y_pred
        self.Y_train = Y
        self.Y_pred_train = y_pred_train
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
