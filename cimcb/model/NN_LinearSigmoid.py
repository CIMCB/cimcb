import numpy as np
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from scipy.stats import logistic
from copy import deepcopy, copy
from sklearn.metrics import r2_score
from keras import backend as K
from keras.constraints import max_norm, non_neg, min_max_norm, unit_norm
from .BaseModel import BaseModel
from ..utils import YpredCallback


class NN_LinearSigmoid(BaseModel):
    """2 Layer linear-logistic neural network using Keras"""

    parametric = True
    bootlist = ["model.vip_", "model.coef_", "model.x_loadings_", "model.x_scores_", "Y_pred"]  # list of metrics to bootstrap

    def __init__(self, n_neurons=2, epochs=200, learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, loss="binary_crossentropy", batch_size=None, verbose=0):
        self.n_neurons = n_neurons
        self.verbose = verbose
        self.n_epochs = epochs
        self.k = n_neurons
        self.batch_size = batch_size
        self.loss = loss
        self.decay = decay
        self.nesterov = nesterov
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov)

        self.__name__ = 'cimcb.model.NN_LinearSigmoid'
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

        # # If using Keras, set tf to 1 core
        # config = K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8, allow_soft_placement=True)
        # session = tf.Session(config=config)
        # K.set_session(session)

        # If batch-size is None:
        if self.batch_size is None:
            self.batch_size = len(X)

        self.model = Sequential()
        self.model.add(Dense(self.n_neurons, activation="linear", input_dim=len(X.T)))
        self.model.add(Dense(1, activation="sigmoid", kernel_initializer='ones'))
        #self.model.add(Dense(1, activation="sigmoid", kernel_initializer='ones', kernel_constraint=non_neg()))
        #self.model.add(Dense(1, activation="sigmoid", kernel_constraint=non_neg()))
        #self.model.add(Dense(1, activation="sigmoid", kernel_constraint=unit_norm(1)))
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])

        # If epoch_ypred is True, calculate ypred for each epoch
        if epoch_ypred is True:
            self.epoch = YpredCallback(self.model, X, epoch_xtest)
        else:
            self.epoch = Callback()

        #print("Before: {}".format(self.model.layers[1].get_weights()[0].flatten()))
        # print("Before: {}".format(self.model.layers[1].get_weights()[0]))
        # Fit
        self.model.fit(X, Y, epochs=self.n_epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=[self.epoch])

        self.model.pctvar_ = pctvar_calc(self.model, X, Y)
        #print("After: {} .... {}".format(self.model.layers[1].get_weights()[0].flatten(), self.model.pctvar_))

        layer1_weight = self.model.layers[0].get_weights()[0]
        layer1_bias = self.model.layers[0].get_weights()[1]
        layer2_weight = self.model.layers[1].get_weights()[0]
        layer2_bias = self.model.layers[1].get_weights()[1]

        # Coef vip
        self.model.vip_ = garson(layer1_weight, layer2_weight.flatten())
        self.model.coef_ = connectionweight(layer1_weight, layer2_weight.flatten())

        # Not sure about the naming scheme (trying to match PLS)
        self.model.x_loadings_ = layer1_weight
        self.model.x_scores_ = np.matmul(X, self.model.x_loadings_) + layer1_bias
        self.model.x_scores_alt = self.model.x_scores_
        self.model.y_loadings_ = layer2_weight
        self.model.y_scores = np.matmul(self.model.x_scores_alt, self.model.y_loadings_) + layer2_bias
        y_pred_train = self.model.predict(X).flatten()

        # Sort by pctvar
        order = np.argsort(self.model.pctvar_)[::-1]
        self.model.x_scores_ = self.model.x_scores_[:, order]
        self.model.x_loadings_ = self.model.x_loadings_[:, order]
        self.model.y_loadings_ = self.model.y_loadings_[order]

        self.model.y_loadings_ = self.model.y_loadings_.T
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

        layer1_weight = self.model.layers[0].get_weights()[0]
        layer1_bias = self.model.layers[0].get_weights()[1]
        layer2_weight = self.model.layers[1].get_weights()[0]
        layer2_bias = self.model.layers[1].get_weights()[1]

        self.model.x_scores_ = np.matmul(X, self.model.x_loadings_) + layer1_bias
        self.model.x_scores_alt = self.model.x_scores_
        #self.model.y_scores = np.matmul(self.model.x_scores_alt, self.model.y_loadings_) + layer2_bias
        y_pred_test = self.model.predict(X).flatten()
        self.Y_pred = y_pred_test

        return y_pred_test


def pctvar_calc(model, X, Y):
    x1 = X
    w1 = model.layers[0].get_weights()[0]
    b1 = model.layers[0].get_weights()[1]
    w2 = model.layers[1].get_weights()[0]
    b2 = model.layers[1].get_weights()[1]

    x2 = logistic.cdf(np.matmul(x1, w1) + b1)

    pctvar = []
    if len(w2) == 1:
        y = logistic.cdf(np.matmul(x2, w2) + b2)
        r2_i = r2_score(Y, y)
        pctvar.append(r2_i)
    else:
        for i in range(len(w2)):
            w2_i = deepcopy(w2)
            w2_i[~i] = 0
            y = logistic.cdf(np.matmul(x2, w2_i) + b2)
            r2_i = r2_score(Y, y)
            pctvar.append(r2_i)

    pct = np.array(pctvar)
    return pct


def garson(A, B):
    """
    Computes Garson's algorithm
    A = matrix of weights of input-hidden layer (rows=input & cols=hidden)
    B = vector of weights of hidden-output layer
    """
    B = np.diag(B)

    # connection weight through the different hidden node
    cw = np.dot(A, B)

    # weight through node (axis=0 is column; sum per input feature)
    cw_h = abs(cw).sum(axis=0)

    # relative contribution of input neuron to outgoing signal of each hidden neuron
    # sum to find relative contribution of input neuron
    rc = np.divide(abs(cw), abs(cw_h))
    rc = rc.sum(axis=1)

    # normalize to 100% for relative importance
    ri = rc / rc.sum()
    return(ri)


def connectionweight(A, B):
    """
    Computes Garson's algorithm
    A = matrix of weights of input-hidden layer (rows=input & cols=hidden)
    B = vector of weights of hidden-output layer
    """
    #B = np.diag(B)

    # connection weight through the different hidden node
    cw = np.dot(A, B)

    return cw
