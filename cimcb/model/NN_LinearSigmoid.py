import numpy as np
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from scipy.stats import logistic
from copy import deepcopy, copy
from sklearn.metrics import r2_score, explained_variance_score
from keras import backend as K
from bokeh.plotting import output_notebook, show
from keras.constraints import max_norm, non_neg, min_max_norm, unit_norm
from ..plot import permutation_test
from .BaseModel import BaseModel
from ..utils import YpredCallback, binary_metrics, binary_evaluation


class NN_LinearSigmoid(BaseModel):
    """2 Layer linear-logistic neural network using Keras"""

    parametric = True
    bootlist = ["model.vip_", "model.coef_", "model.x_loadings_", "model.x_scores_", "Y_pred", "model.pctvar_", "model.y_loadings_", "model.pfi_acc_", "model.pfi_r2q2_", "model.pfi_auc_", "model.eval_metrics_"]  # list of metrics to bootstrap

    def __init__(self, n_neurons=2, epochs=200, learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False, loss="binary_crossentropy", batch_size=None, verbose=0, pfi_metric="r2q2", pfi_nperm=0, pfi_mean=True, seed=None):
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
        self.pfi_metric = pfi_metric
        self.pfi_nperm = pfi_nperm
        self.pfi_mean = pfi_mean
        self.optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=nesterov)
        self.compiled = False
        self.seed = seed
        self.__name__ = 'cimcb.model.NN_LinearSigmoid'
        self.__params__ = {'n_neurons': n_neurons, 'epochs': epochs, 'learning_rate': learning_rate, 'momentum': momentum, 'decay': decay, 'nesterov': nesterov, 'loss': loss, 'batch_size': batch_size, 'verbose': verbose, 'seed': seed}

    def set_params(self, params):
        self.__init__(**params)

    def train(self, X, Y, epoch_ypred=False, epoch_xtest=None, w1=False, w2=False):
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
        self.X = X
        self.Y = Y

        # If epoch_ypred is True, calculate ypred for each epoch
        if epoch_ypred is True:
            self.epoch = YpredCallback(self.model, X, epoch_xtest)
        else:
            self.epoch = Callback()

        if self.compiled == False:
            np.random.seed(self.seed)
            self.model = Sequential()
            self.model.add(Dense(self.n_neurons, activation="linear", input_dim=len(X.T)))
            self.model.add(Dense(1, activation="sigmoid"))
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])
            self.model.w1 = self.model.layers[0].get_weights()
            self.model.w2 = self.model.layers[1].get_weights()
            self.compiled == True
        else:
            self.model.layers[0].set_weights(self.model.w1)
            self.model.layers[1].set_weights(self.model.w2)
        #print("Before: {}".format(self.model.layers[1].get_weights()[0].flatten()))
        # print("Before: {}".format(self.model.layers[1].get_weights()[0]))

        if w1 != False:
            self.model.layers[0].set_weights(w1)
            self.model.w1 = w1
        if w2 != False:
            self.model.layers[1].set_weights(w2)
            self.model.w2 = w2

        # Fit
        self.model.fit(X, Y, epochs=self.n_epochs, batch_size=self.batch_size, verbose=self.verbose)

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
        # if self.compiled == False:
        #     if w1 == False:
        #         if w2 == False:
        #             order = np.argsort(self.model.pctvar_)[::-1]
        #             self.model.x_scores_ = self.model.x_scores_[:, order]
        #             self.model.x_loadings_ = self.model.x_loadings_[:, order]
        #             self.model.y_loadings_ = self.model.y_loadings_[order]
        #             self.model.y_loadings_ = self.model.y_loadings_.T
        #             self.model.pctvar_ = self.model.pctvar_[order]
        #             self.model.w1[0] = self.model.w1[0][:, order]
        #             self.model.w2[0] = self.model.w2[0][order]
        #     self.compiled = True

        self.model.y_loadings_ = layer2_weight.T

        # Calculate pfi
        if self.pfi_nperm == 0:
            self.model.pfi_acc_ = np.zeros((1, len(Y)))
            self.model.pfi_r2q2_ = np.zeros((1, len(Y)))
            self.model.pfi_auc_ = np.zeros((1, len(Y)))
        else:
            pfi_acc, pfi_r2q2, pfi_auc = self.pfi(nperm=self.pfi_nperm, metric=self.pfi_metric, mean=self.pfi_mean)
            self.model.pfi_acc_ = pfi_acc
            self.model.pfi_r2q2_ = pfi_r2q2
            self.model.pfi_auc_ = pfi_auc

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

        layer1_weight = self.model.layers[0].get_weights()[0]
        layer1_bias = self.model.layers[0].get_weights()[1]
        layer2_weight = self.model.layers[1].get_weights()[0]
        layer2_bias = self.model.layers[1].get_weights()[1]

        self.model.x_scores_ = np.matmul(X, layer1_weight) + layer1_bias
        self.model.x_scores_alt = self.model.x_scores_
        #self.model.y_scores = np.matmul(self.model.x_scores_alt, self.model.y_loadings_) + layer2_bias
        y_pred_test = self.model.predict(X).flatten()
        self.Y_pred = y_pred_test

        # Calculate and return Y predicted value
        if Y is not None:
            self.metrics_key = []
            self.model.eval_metrics_ = []
            bm = binary_evaluation(Y, y_pred_test)
            for key, value in bm.items():
                self.model.eval_metrics_.append(value)
                self.metrics_key.append(key)

            self.model.eval_metrics_ = np.array(self.model.eval_metrics_)
        return y_pred_test


def pctvar_calc(model, X, Y):
    x1 = X
    w1 = model.layers[0].get_weights()[0]
    b1 = model.layers[0].get_weights()[1]
    w2 = model.layers[1].get_weights()[0]
    b2 = model.layers[1].get_weights()[1]

    x2 = np.matmul(x1, w1) + b1

    pctvar = []
    if len(w2) == 1:
        y = logistic.cdf(np.matmul(x2, w2) + b2)
        #r2_i = r2_score(Y, y) * 100
        r2_i = explained_variance_score(Y, y) * 100
        pctvar.append(r2_i)
    else:
        for i in range(len(w2)):
            x2 = logistic.cdf(np.matmul(x1, w1[:, i]) + b1[i])
            x2 = np.reshape(x2, (-1, 1))
            y = logistic.cdf(np.matmul(x2, w2[i]) + b2)
            r2_i = explained_variance_score(Y, y) * 100
            pctvar.append(r2_i)

    # # Alternative (same result)
    # for i in range(len(w2)):
    #         w2_i = deepcopy(w2)
    #         w2_i[~i] = 0
    #         y = logistic.cdf(np.matmul(x2, w2_i))
    #         #r2_i = r2_score(Y, y) * 100
    #         r2_i = explained_variance_score(Y, y) * 100
    #         pctvar.append(r2_i)

    pct = np.array(pctvar)
    # convert to reltive explained variance
    pct = pct / np.sum(pct) * 100
    return pct


def garson(A, B):
    B = np.diag(B)
    cw = np.dot(A, B)
    cw_h = abs(cw).sum(axis=0)
    rc = np.divide(abs(cw), abs(cw_h))
    rc = rc.sum(axis=1)
    #ri = rc / rc.sum()
    return(rc)


def connectionweight(A, B):
    cw = np.dot(A, B)
    return cw
