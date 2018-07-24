import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import scipy.stats as stats
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn import svm
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import kneighbors_graph
import time
import scipy.sparse as sparse
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger


class ForwardProcess(object):

    def __init__(self, X0, T, alpha, beta, corr, N, m):
        """
        Parameters
        ==========
        S0             : array
                          positive, initial Stock Value
        alpha              : lambda function of t and X
                          drift
        T               : lambda function of t and X
                          Maturity time
        beta           : lambda function
                          volatility
        corr : correlation

         Returns
        =======
        Forward_diffusion : class
        """
        self.X0 = X0
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.corr = corr
        self.dim = len(X0)
        self.N = N
        self.m = m

        def generate_one_forward_diffusion(X0, T, m, dim, corr, alpha, beta):
            dt = T / m
            X = np.zeros((m + 1, dim))
            dB = np.zeros((m + 1, dim))
            X[0] = X0
            C = np.linalg.cholesky(corr)
            for t in range(1, m + 1):
                rand = np.random.standard_normal(dim)
                rand_int = np.dot(C, rand)
                X[t] = X[t - 1] + alpha(t - 1, X[t - 1]) * dt + beta(t - 1, X[t - 1]) * (dt ** 0.5)* rand_int
                dB[t] = (dt ** 0.5) * rand_int
            return X, dB

        def generate_N_forward_diffusion(X0, T, m, dim, corr, alpha, beta, N):
            """

            :param N: Number of particles generated for each asset
            :param m: number of time step
            :return: 3 dimensional matrix of assets and 3 dimensional matrix of its brownian motions
            """
            X = np.zeros([m + 1, N, dim])
            dB = np.zeros([m + 1, N, dim])
            # Generate N*dim assets
            for i in range(self.N):
                X_int, dB_int = generate_one_forward_diffusion(X0, T, m, dim, corr, alpha, beta)
                X[:, i, :] = X_int
                dB[:, i, :] = dB_int
            return X, dB

        self.X, self.dB = generate_N_forward_diffusion(self.X0, self.T, self.m, self.dim, self.corr, self.alpha,
                                                       self.beta, self.N)


class GBM(ForwardProcess):

    def __init__(self, X0, T, mu, sigma, N, m, alpha=None, beta=None, corr=None):
        self.mu = mu
        self.sigma = sigma
        self.alpha = lambda t, x: self.mu * x
        self.beta = lambda t, x: sigma * x
        ForwardProcess.__init__(self, X0, T, self.alpha, self.beta, corr, N, m)


class BSDE(object):
    def __init__(self, driver, xi, fwd_process):
        """

        :param driver: lambda function
                        f(t, X_t, Y_t, Z_t)
        :param xi: lambda function
                    final condition, Y_T = xi(X_T)
        :param fwd_process : ForwardProcess object
        """
        self.driver = driver
        self.xi = xi
        self.fwd_process = fwd_process

    def Regression(self, regression='RandomForest', RF_n_tree=100, RF_max_leaf_nodes=20, n_picard=5):
        """

        :param method: regression method : available : RandomForest, GradientBoosting, Mesh
        :param RF_n_tree:
        :param RF_max_leaf_nodes:
        :param n_picard:
        :return:
        """
        m = self.fwd_process.m
        N = self.fwd_process.N
        T = self.fwd_process.T
        p = self.fwd_process.dim
        dt = T / m
        regressor = None
        if regression == 'RandomForest':
            regressor = RandomForestRegressor(n_estimators=RF_n_tree,
                                       max_leaf_nodes=RF_max_leaf_nodes,
                                       oob_score=False,
                                       n_jobs=-1)

        elif regression == 'svm':
            regressor = svm.SVR()

        elif regression == 'gbr':
            regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')

        elif regression == 'mesh':
            create_mesh = []

        X, dB = self.fwd_process.X, self.fwd_process.dB

        Y = self.xi(X[-1])
        for t in range(m - 1, 0, -1):
            X_in = X[t]
            Z = np.zeros([N, p])
            # Regression for Z
            for k in range(p):
                regressor.fit(X_in, Y * dB[t, :, k])
                Z[:, k] = regressor.predict(X_in) * (1. / dt)

            regressor.fit(X_in, Y)
            J = regressor.predict(X_in)
            Y_inc = J + self.driver(t, X_in, Y, Z) * dt

            for __ in range(n_picard):
                for k in range(p):
                    regressor.fit(X_in, (Y - Y_inc) * dB[t, :, k])
                    Z[:, k] = regressor.predict(X_in) * (1. / dt)
                Y_inc = J + self.driver(t, X_in, Y, Z) * dt

            Y[:] = Y_inc
            # plt.plot(X, Z, 'r.')
            # plt.show()

        Y_opt = np.mean(Y)
        return Y_opt


