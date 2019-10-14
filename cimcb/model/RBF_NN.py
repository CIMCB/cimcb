import numpy
import numpy as np
import math
from sklearn.cluster import KMeans
from .BaseModel import BaseModel


class RBF_NN(BaseModel):
    """Radial basis function neural network"""

    parametric = True
    bootlist = None

    def __init__(self, n_clusters=8, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.k = n_clusters

        self.__name__ = 'cimcb.model.RBF_NN'
        self.__params__ = {'n_clusters': n_clusters, 'max_iter': max_iter}

    def set_params(self, params):
        self.__init__(**params)

    def train(self, X, Y):
        """ Fit the rbf-nn model, save additional stats (as attributes) and return Y predicted values.

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

        km = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter)
        km.fit(X)
        cent = km.cluster_centers_

        self.model = KMeans

        # Determine the value of sigma
        max = 0
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                d = numpy.linalg.norm(cent[i] - cent[j])
                if d > max:
                    max = d
        d = max
        sigma = d / math.sqrt(2 * self.n_clusters)

        # Set up G matrix
        shape = X.shape
        row = shape[0]
        column = self.n_clusters
        G = numpy.empty((row, column), dtype=float)
        for i in range(row):
            for j in range(column):
                dist = numpy.linalg.norm(X[i] - cent[j])
                G[i][j] = math.exp(-math.pow(dist, 2) / math.pow(2 * sigma, 2))

        # Find W
        GTG = numpy.dot(G.T, G)
        GTG_inv = numpy.linalg.inv(GTG)
        fac = numpy.dot(GTG_inv, G.T)
        W = numpy.dot(fac, Y)
        self.cent = cent
        self.W = W
        self.G = G
        self.sigma = sigma
        y_pred_train = np.dot(G, W)
        self.xcols_num = len(X.T)
        cent2 = []
        for i in range(len(self.cent.T)):
            something = []
            for j in range(len(self.cent)):
                something.append(self.cent[j][i])
            cent2.append(something)

        self.cent2 = np.array(cent2)

        self.vi = np.dot(self.cent2, self.W)

        self.model.x_scores_ = self.G
        self.model.y_loadings_ = self.W.reshape(1, len(self.W))
        self.model.pctvar_ = np.ones((1, len(self.model.y_loadings_[0])))
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

        # Set up G matrix
        shape = X.shape
        row = shape[0]
        column = self.n_clusters
        G = numpy.empty((row, column), dtype=float)
        for i in range(row):
            for j in range(column):
                dist = numpy.linalg.norm(X[i] - self.cent[j])
                G[i][j] = math.exp(-math.pow(dist, 2) / math.pow(2 * self.sigma, 2))
        y_pred_test = numpy.dot(G, self.W)
        return y_pred_test
