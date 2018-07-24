import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import scipy.stats as stats
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
import copy
from sklearn import svm
import parmap
import time
from bokeh.plotting import figure, show
import scipy
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from concurrent.futures import Executor, ProcessPoolExecutor

plt.rcParams['figure.figsize'] = (7, 7)  # Make the figures a bit bigger


class Option(object):

    AMERICAN = 'AMERICAN'
    EUROPEAN = 'EUROPEAN'
    ASIAN = 'ASIAN'


def hypercube(x_data, y_data, x, n_hc):
    # Some Verif
    if len(x_data) != len(y_data):
        print("data arrays have differents lengths")
    x_data = sorted(x_data)

    # we take an n*log(n) as most values are located
    x_data = x_data * np.log(x_data)
    x = x * np.log(x)
    # plt.plot(x_data, y_data, 'r.')
    # plt.show()
    # Size of datas
    N_data = len(x_data)
    N = len(x)
    delta_hc = np.floor_divide(N_data, n_hc)
    x_data_min = x_data[0]
    x_data_max = x_data[N_data - 1]
    # Lets compute the hypercube output of this data according to a linearspace
    y_hc = np.zeros(n_hc)
    y = np.zeros(N)
    y_hc[0] = np.mean(y_data[0:delta_hc])
    for i in range(1, n_hc - 1):
        y_hc[i] = np.mean(y_data[(i * delta_hc):((i + 1) * delta_hc)])
    y_hc[n_hc - 1] = np.mean(y_data[(n_hc - 1) * delta_hc:])

    for i in range(N):
        p = np.searchsorted(x_data, x[i])
        p = np.floor_divide(p, delta_hc)
        if p == 0:
            y[i] = y_hc[p]
        else:
            y[i] = y_hc[p]
    return y


class BSDE(object):
    def __init__(self, S0, K, T, mu, sigma, q):
        '''

        Parameters
        ==========
        S0             : float
                          positive, initial Stock Value
        mu              : float
                          drift
        K               : float
                          Strike price
        T               : float
                          Maturity time
        sigma           : float
                          volatility
         Returns
        =======
        BSDE : class
        '''
        self.S0 = S0
        self.K = K
        self.T = T
        self.mu = mu
        self.sigma = sigma
        self.q = q

    def generate_paths(self, r, N, m, mode='delta_B'):
        if mode == 'delta_B':
            dt = self.T / m
            S = np.zeros((m + 1, N))
            dB = np.zeros((m + 1, N))
            S[0] = self.S0
            for t in range(1, m + 1):
                X = np.random.standard_normal(size=N)
                S[t] = S[t - 1] * np.exp((self.mu - self.sigma * self.sigma / 2) * dt
                                         + self.sigma * np.sqrt(dt) * X)
                dB[t] = np.sqrt(dt) * X
            return (S, dB)
        elif mode == 'B':
            dt = self.T / m
            S = np.zeros((m + 1, N))
            B = np.zeros((m + 1, N))
            S[0] = self.S0
            for t in range(1, m + 1):
                X = np.random.standard_normal(size=N)
                S[t] = S[t - 1] * np.exp((self.mu - self.sigma * self.sigma / 2) * dt
                                         + self.sigma * np.sqrt(dt) * X)
                B[t] = np.sqrt(t) * X
            return (S, B)

    def get_price_lsm(self, R, r, N, m, K1=95., K2=105., deg=8, oPayoff="call", oType="European", n_picard=10):
        '''
        Function to generate stock paths.

        Parameters
        ==========

        r               : float
                          lending interest rate
        R               : float
                          borrowing interest rate
        N               : int
                          Number of paths generated
        m               : int
                          number of steps
        d               : int
                          polynomial fit degree

        Returns
        =======
        Y_opt : float
                Price of the European option
        '''
        # Time-step
        dt = self.T / m
        # Discount factor
        df = 1. / (1 + r * dt)
        theta = (self.mu - r) / self.sigma
        # S, dB
        S, dB = self.generate_paths(r, N, m)
        # price of the option at time T = Initialization for a call
        if oPayoff == "call":
            Y = np.maximum(S[-1] - self.K, 0)
        elif oPayoff == "put":
            Y = np.maximum(self.K - S[-1], 0)
        elif oPayoff == "call combination":
            Y = np.maximum(S[-1] - K1, 0) - 2 * np.maximum(S[-1] - K2, 0)
        elif oPayoff == "put combination":
            Y = np.maximum(K1 - S[-1], 0) - 2 * np.maximum(K2 - S[-1], 0)
        elif oPayoff == "call average":
            Y = np.maximum(S.mean(axis=0) - self.K, 0)

        if (oType == 'European'):
            # Iteration over time backwardly
            for t in range(m - 1, 0, -1):
                X = S[t]
                # Regression for Z
                reg1 = np.polyfit(X, Y * dB[t], deg)
                Z = np.polyval(reg1, X) * (1. / dt)

                if n_picard > 0 :
                    for __ in range(n_picard):
                        reg = np.polyfit(X, Y - theta * Z * dt - np.minimum(Y - (1. / self.sigma) * Z, 0) * (R - r) * dt,
                                         deg)
                        poly_y = np.poly1d(reg)
                        J = poly_y(X)
                        Y_inc = df * J
                        poly_z = np.polyder(poly_y)
                        Z = self.sigma * poly_z(X) * X

                else:
                    reg = np.polyfit(X, Y - theta * Z * dt - np.minimum(Y - (1. / self.sigma) * Z, 0) * (R - r) * dt,
                                     deg)
                    poly_y = np.poly1d(reg)
                    J = poly_y(X)
                    Y_inc = df * J
                # # regression for Y
                # reg = np.polyfit(X, Y, deg)
                # J = np.polyval(reg, X)
                #
                # # Y = np.polyval(reg,X)-Y*r*dt-theta*Z*dt+np.minimum(Y-(1/sigma)*Z,0)*(R-r)*dt
                # Y_inc = df * (J - theta * Z * dt - np.minimum(Y - (1. / self.sigma) * Z, 0) * (R - r) * dt)

                Y[:] = Y_inc
                # plt.plot(X, Z, 'r.')
                # plt.show()
            Y_opt = df * np.mean(Y)
            return Y_opt

        if (oType == 'American'):
            # Iteration over time backwardly
            for t in range(m - 1, 0, -1):
                X = S[t]
                # Regression for Z
                reg1 = np.polyfit(X, Y * dB[t], deg)
                Z = (1. / dt) * np.polyval(reg1, X)
                # print (np.mean( Y * dB[t]) / dt - np.mean(Z), np.std(Z))
                # regression for Y
                reg = np.polyfit(X, Y, deg)
                J = np.polyval(reg, X)
                # Y = np.polyval(reg,X)-Y*r*dt-theta*Z*dt+np.minimum(Y-(1/sigma)*Z,0)*(R-r)*dt
                Y = np.maximum(df * (J - theta * Z * dt -
                                     np.minimum(J - (1. / self.sigma) * Z, 0) *
                                     (R - r) * dt), np.maximum(S[t] - self.K, 0))

                # plt.plot(Y,Z, 'r.')
                # plt.show()
            Y_opt = df * np.mean(Y)
            Z_opt = df * np.mean(Z)
            # print (np.mean(Z), np.var(Z))
            return (Y_opt)

    def get_price_derivative(self, R, r, N, m, K1=95., K2=105., deg=5, oPayoff="call", oType="European", n_picard=5,
                             l=None, use_display= False, n_neighbors=None):
        '''
        Function to generate stock paths.

        Parameters
        ==========

        r               : float
                          lending interest rate
        R               : float
                          borrowing interest rate
        N               : int
                          Number of paths generated
        m               : int
                          number of steps
        d               : int
                          polynomial fit degree

        Returns
        =======
        Y_opt : float
                Price of the European option
        '''

        # Time-step
        dt = self.T / m
        # Discount factor
        df = 1. / (1 + r * dt)
        theta = -(r - self.mu) / self.sigma
        # S, dB
        S, dB = self.generate_paths(r, N, m)
        # price of the option at time T = Initialization for a call
        if oPayoff == "call":
            Y = np.maximum(S[-1] - self.K, 0)
        elif oPayoff == "put":
            Y = np.maximum(self.K - S[-1], 0)
        elif oPayoff == "call combination":
            Y = np.maximum(S[-1] - K1, 0) - 2 * np.maximum(S[-1] - K2, 0)
        elif oPayoff == "put combination":
            Y = np.maximum(K1 - S[-1], 0) - 2 * np.maximum(K2 - S[-1], 0)
        elif oPayoff == "call average":
            Y = np.maximum(S.mean(axis=0) - self.K, 0)

        Y_inc = np.zeros(N)
        if oType == 'European':
            # Iteration over time backwardly
            for t in range(m - 1, 0, -1):
                X = S[t]
                Z = np.zeros(N)

                if r != R or r != self.mu:
                    # first Regression of Z using polynomial least square
                    reg1 = np.polyfit(X, Y * dB[t], deg)
                    Z = np.polyval(reg1, X) * (1. / dt)

                # plt.plot(X, Z,'r.')
                # plt.show()

                # list of nearest neighbors
                if n_neighbors is not None:
                    NN = kneighbors_graph(X.reshape(N, 1), n_neighbors,
                                          mode='distance').nonzero()
                else:
                    raise Exception('give a number of neighbors')

                x = NN[0]
                y = NN[1]

                # W is the weight matrix, i.e in w(i,j) = exp(-(x_j - x_i)^2 / 2.l^2) / sum_axis_0(exp(-(x_j - x_i)^2 / 2.l^2))
                W = scipy.sparse.lil_matrix((N, N))
                W[x, y] = np.exp(- (X[y] - X[x]) ** 2 / (2 * l ** 2))

                # As (i, i) not in NN, we add it manually
                W.setdiag(np.ones(N))

                #Sum over each row and update the weight
                sum_weights = W.sum(axis=0)
                W = W.dot(scipy.sparse.diags(np.array(1 / sum_weights)[0]))

                # E[Y_(t+dt)_j|F_t] = f(X_j) = sum(w(i, j)*Y(i))
                expected_Y = W.dot(Y)

                if r != R or r != self.mu:
                    # derivated weights w'(i,j) = - (x_j - x_i) / l**2  w(i, j)
                    derivated_weights = scipy.sparse.lil_matrix((N, N))
                    derivated_weights[x, y] = - (X[y] - X[x]) / l ** 2 * np.exp(- (X[y] - X[x]) ** 2 / (2 * l ** 2))
                    sum_derivated_weights = scipy.sparse.diags(np.array(derivated_weights.sum(axis=0))[0])

                    # Z[j] = sigma * f'(X_j) = sigma * sum(a(i,j)*Y(i))
                    # where a(i,j) = (w'(i,j)*sum(w(i,j)) - w(i,j)*sum(w'(i,j))) / (sum(w(i,j)))^2
                    f_prime = (derivated_weights.dot(
                        scipy.sparse.diags(np.array(sum_weights)[0])) - W.dot(
                        sum_derivated_weights)) / np.square(sum_weights)
                    f_prime = f_prime.sum(axis=0)


                for __ in range(n_picard):
                    Y_inc = df * (expected_Y - theta * Z * dt - np.minimum(Y - (1. / self.sigma) * Z, 0) * (R - r) * dt)
                    if r != R or r != self.mu:
                        Z = self.sigma * X * np.array(self.sigma * np.multiply(f_prime, Y_inc))[0]

                if use_display:
                    p = figure(width=250, plot_height=250, title='Z against X')
                    p.circle(X, Z, color='navy')
                    show(p)
                Y[:] = Y_inc
            Y_opt = df * np.mean(Y)
            return Y_opt

        if (oType == 'American'):
            # Iteration over time backwardly
            for t in range(m - 1, 0, -1):
                X = S[t]
                # Regression for Z
                reg1 = np.polyfit(X, Y * dB[t], deg)
                Z = (1. / dt) * np.polyval(reg1, X)
                # print (np.mean( Y * dB[t]) / dt - np.mean(Z), np.std(Z))
                # regression for Y
                reg = np.polyfit(X, Y, deg)
                J = np.polyval(reg, X)
                # Y = np.polyval(reg,X)-Y*r*dt-theta*Z*dt+np.minimum(Y-(1/sigma)*Z,0)*(R-r)*dt
                Y = np.maximum(df * (J - theta * Z * dt -
                                     np.minimum(J - (1. / self.sigma) * Z, 0) *
                                     (R - r) * dt), np.maximum(S[t] - self.K, 0))

                # plt.plot(Y,Z, 'r.')
                # plt.show()
            Y_opt = df * np.mean(Y)
            Z_opt = df * np.mean(Z)
            # print (np.mean(Z), np.var(Z))
            return (Y_opt)

    def get_price_RF(self, R, r, N, m, K1=95., K2=105., oPayoff="call", RF_n_tree=100, RF_max_leaf_nodes=100,
                     RF_max_features='auto',
                     RF_max_depth=None,
                     RF_min_samples_split=2,
                     RF_min_samples_leaf=1,
                     RF_warm_start=False,
                     oType='European',
                     n_picard=10,
                     regression='RF'):
        '''
        Get price using Random Forest

        Parameters
        ==========
        r               : float
                          lending interest rate
        R               : float
                          borrowing interest rate
        N               : int
                          Number of paths generated
        m               : int

        RF_n_estimators : int
                          Number of trees generated by the Random Forest regression

        RF_max_leaf_    : int
        nodes             Maximum number of leafs for every branch of the tree generated

        :param oPayoff : string
                         Type of the payoff, can be 'call', 'put', ...

        Returns
        =======
        Y_opt : float
                Price of the European option with different interest rates
        '''
        # Time-step
        dt = self.T / m
        # Discount factor
        df = 1 / (1 + r * dt)
        theta = -(r - self.mu) / self.sigma
        # S, dB
        S, dB = self.generate_paths(r, N, m)
        # price of the option at time T = Initialization for a call
        if oPayoff == "call":
            Y = np.maximum(S[-1] - self.K, 0)
        elif oPayoff == "put":
            Y = np.maximum(self.K - S[-1], 0)
        elif oPayoff == "call combination":
            Y = np.maximum(S[-1] - K1, 0) - 2 * np.maximum(S[-1] - K2, 0)
        elif oPayoff == "put combination":
            Y = np.maximum(K1 - S[-1], 0) - 2 * np.maximum(K2 - S[-1], 0)
        elif oPayoff == "call average":
            Y = np.maximum(S.mean(axis=0) - self.K, 0)

        if regression == 'RF':
            rf = RandomForestRegressor(n_estimators=RF_n_tree,
                                       max_features=RF_max_features,
                                       max_leaf_nodes=RF_max_leaf_nodes,
                                       oob_score=False,
                                       max_depth=RF_max_depth,
                                       min_samples_split=RF_min_samples_split,
                                       min_samples_leaf=RF_min_samples_leaf,
                                       warm_start=RF_warm_start,
                                       n_jobs=-1)

        elif regression == 'svm':
            rf = svm.SVR()

        elif regression == 'gbr':
            rf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')

        if oType == 'European':

            # Iteration over time backwardly
            for t in range(m - 1, 0, -1):
                X = S[t]
                X = X[:, None]


                # Regression for Z only if R != r
                if r != R or r!= self.mu:
                    rf.fit(X, Y * dB[t])
                    Z = rf.predict(X) * (1. / dt)
                    driver_measurable = - theta * Z
                    driver_stoch = - r * Y - (R - r) * np.minimum(Y - (1. / self.sigma) * Z, 0)

                    # regression for Y
                    rf.fit(X, Y)
                    J = rf.predict(X)
                    # Y = np.polyval(reg,X)-Y*r*dt-t heta*Z*dt+np.minimum(Y-(1/sigma)*Z,0)*(R-r)*dt
                    Y_inc = J + driver_measurable * dt + driver_stoch * dt

                    for __ in range(n_picard):
                        rf.fit(X, (Y - Y_inc) * dB[t])
                        Z = rf.predict(X) * (1. / dt)
                        driver_measurable = - theta * Z
                        driver_stoch = - r * Y - (R - r) * np.minimum(Y - (1. / self.sigma) * Z, 0)
                        Y_inc = J + driver_measurable * dt + driver_stoch * dt
                    Y[:] = Y_inc
                else:
                    # regression for Y
                    rf.fit(X, Y)
                    J = rf.predict(X)
                    Y = df * J

            Y_opt = df * np.mean(Y)
            return Y_opt

        elif oType == 'American':
            # Iteration over time backwardly
            for t in range(m - 1, 0, -1):
                X = S[t]
                X = X[:, None]
                rf = RandomForestRegressor(n_estimators=RF_n_tree,
                                           max_leaf_nodes=RF_max_leaf_nodes,
                                           oob_score=False,
                                           n_jobs=-1)

                # Regression for Z
                rf.fit(X, Y * dB[t])
                Z = rf.predict(X) * (1. / dt)

                # regression for Y
                rf.fit(X, Y)
                J = rf.predict(X)
                # Y = np.polyval(reg,X)-Y*r*dt-theta*Z*dt+np.minimum(Y-(1/sigma)*Z,0)*(R-r)*dt
                Y = np.maximum(df * (J - theta * Z * dt -
                                     np.minimum(Y - (1. / self.sigma) * Z, 0) *
                                     (R - r) * dt), np.maximum(S[t] - self.K, 0))

            Y_opt = df * Y.sum() / N
            return Y_opt

    def get_price_hc(self, R, r, N, m, delta=10, K1=95., K2=105., n_hc=20, oPayoff="call", oType="European", n_picard=0):
        """
        Function to generate stock paths.

        Parameters
        ==========

        r               : float
                          lending interest rate
        R               : float
                          borrowing interest rate
        N               : int
                          Number of paths generated
        m               : int
                          number of steps
        d               : int
                          polynomial fit degree

        Return
        =======
        Y_opt : float
                Price of the European option
        """
        # Time-step
        dt = self.T / m
        # Discount factor
        df = 1. / (1 + r * dt)
        theta = -(r - self.mu) / self.sigma
        # S, dB
        S, dB = self.generate_paths(r, N, m)
        # price of the option at time T = Initialization for a call
        if oPayoff == "call":
            Y = np.maximum(S[-1] - self.K, 0)
        elif oPayoff == "put":
            Y = np.maximum(self.K - S[-1], 0)
        elif oPayoff == "call combination":
            Y = np.maximum(S[-1] - K1, 0) - 2 * np.maximum(S[-1] - K2, 0)
        elif oPayoff == "put combination":
            Y = np.maximum(K1 - S[-1], 0) - 2 * np.maximum(K2 - S[-1], 0)

        if oType == 'European':
            # Iteration over time backwardly
            for t in range(m - 1, 0, -1):
                X = S[t]
                # Regression for Z
                # Z = hypercube(X, Y * dB[t], X, n_hc)* (1. / dt)
                # plt.plot(X,Y,'r.')
                # plt.show()
                import maths
                Z = 1 / dt * np.array(maths.hc_regression(X, delta, X, Y * dB[t]))
                # regression for Y
                J = np.array(maths.hc_regression(X, delta, X, Y))
                # J = hypercube(X, Y, X, n_hc)

                # Y = np.polyval(reg,X)-Y*r*dt-theta*Z*dt+np.minimum(Y-(1/sigma)*Z,0)*(R-r)*dt
                Y_inc = df * (J - theta * Z * dt - np.minimum(Y - (1. / self.sigma) * Z, 0) * (R - r) * dt)
                for __ in range(n_picard):
                    Z = hypercube(X, (Y - Y_inc) * dB[t], X, n_hc) * (1. / dt)

                    Y_inc = df * (J - theta * Z * dt +
                                  np.minimum(Y - (1. / self.sigma) * Z, 0) *
                                  (R - r) * dt)
                Y[:] = Y_inc
                # plt.plot(X, Z, 'r.')
                # plt.show()
            Y_opt = df * np.mean(Y)
        if oType == 'American':
            # Iteration over time backwardly
            for t in range(m - 1, 0, -1):
                X = S[t]
                # Regression for Z
                Z = self.hypercube(X, Y * dB[t], X, n_hc)

                # regression for Y
                J = self.hypercube(X, Y, X, n_hc)

                # Y = np.polyval(reg,X)-Y*r*dt-theta*Z*dt+np.minimum(Y-(1/sigma)*Z,0)*(R-r)*dt
                Y = np.maximum(df * (J - theta * Z * dt -
                                     np.minimum(J - (1. / self.sigma) * Z, 0) *
                                     (R - r) * dt), np.maximum(S[t] - self.K, 0))

                # plt.plot(Y,Z, 'r.')
                # plt.show()
            Y_opt = df * np.mean(Y)
            Z_opt = df * np.mean(Z)
            # print (np.mean(Z), np.var(Z))
        return Y_opt

    def get_cva(self, N, m, r, RF_n_tree, RF_max_leaf_nodes, beta):
        # Time-step
        dt = self.T / m
        # beta = lambda_c * (1 - R)
        # S, dB
        S, dB = self.generate_paths(r, N, m)
        # Iteration over the paths"
        f = lambda x: 1 if x > 1 else 0
        Y = np.zeros(N)
        for i in range(N):
            Y[i] = 1 - 2 * f(S[- 1, i])

        for t in range(m - 1, 0, -1):
            X = S[t]
            X = X[:, None]
            rf = RandomForestRegressor(n_estimators=RF_n_tree,
                                       max_leaf_nodes=RF_max_leaf_nodes,
                                       oob_score=False,
                                       n_jobs=-1,
                                       bootstrap=True)

            # regression for Y
            rf.fit(X, Y)
            J = rf.predict(X)
            for i in range(N):
                Y[i] = 1 / (1 + beta * dt * (1 - f(J[i] + 1))) * J[i]

        Y_opt = np.mean(Y)
        return (Y_opt)

    def get_price_mesh(self, R, r, N, m,
                       mode='NN', n_neighbors=10,
                       K1=95., K2=105., oPayoff="call",
                       oType="European", n_picard=0,
                       use_variance_reduction=False,
                       display_plot=False):
        """
        Approach
        ========
        Call T(i,j,t) the transition from S_i at time "t" to S_j at time "t+dt". Then:
        1) P(S_[t+dt] = S_j) \approx (1/N) * sum_i T(i,j,t)  \def P^[t+dt]_j
        2) E[X^[t+dt]_j|S^t_i] \approx (1/N) * sum_j T(i,j,t) * X^[t+dt]_j / P^[t+dt]_j  (basic importance sampling)
        3) Consequently, define
            W(i,j,t) =  T(i,j,t) / P^[t+dt]_j
        and we have
            E[ (something) | filtration_t] = (1/N) * dot_product(W, something)
        4) remark: one can slightly reduce the variance by further normalizing the rows

        """

        def do_BSDE_mesh_update(Y, Y_inc, Z, dB, W, cv, cv_expectation, mode='all'):
            """
            W       : weight matrix in usual mesh method
            S_t     : Stock price at time "t"
            S_t_dt  : Stock price at time "t + dt"
            dB      : Brownian increments
            """
            # Regression for Z_t:
            #
            # Z_t = E( Y_[t+dt] * dW_t | filtration_t)  /dt
            #
            if mode == 'all':
                if use_variance_reduction == True:
                    # print("I am here")
                    # use the fact that E[ S_t_dt * dW | F_t] = sigma * S * dt + (higher order term)
                    # cv1 = W * S_t_dt * dB  # control variate
                    cv_std = np.std(cv, axis=1)
                    cv_mean = np.mean(cv, axis=1)
                    cv_centred[:] = (cv - cv_mean[:, np.newaxis]) / cv_std[:, np.newaxis]

                    YdB[:] = W * ((Y - Y_inc) * dB[t])
                    YdB_std[:] = np.std(YdB, axis=1)
                    YdB_mean[:] = np.mean(YdB, axis=1)
                    YdB_centred[:] = (YdB - YdB_mean[:, np.newaxis]) / YdB_std[:, np.newaxis]

                    corr = np.mean(cv_centred * YdB_centred, axis=1)
                    Z[:] = np.mean(YdB - (cv - cv_expectation[:, np.newaxis]) * corr[:, np.newaxis], axis=1) * (1. / dt)
                else:
                    Z[:] = (1. / N) * np.dot(W, (Y - Y_inc) * dB[t]) * (1. / dt)
                driver_measurable = - theta * Z
                driver_stoch = - r * Y - (R - r) * np.minimum(Y - (1. / self.sigma) * Z, 0)

                # Regression for Y_t
                #
                # Y_t = E(Y_[t+dt] + dt * driver | filtration_t)
                #
                Y_inc[:] = (1. / N) * np.dot(W, Y + dt * driver_stoch) + dt * driver_measurable
                # Z[:] = numerical_diff(S[t,:], Y_inc) * S[t,:] * self.sigma
                # print("Inside Z={}".format(Z))
                # print("Inside Y_inc={}".format(Y_inc))
            else:
                if use_variance_reduction == True:
                    # print("I am here")
                    # use the fact that E[ S_t_dt * dW | F_t] = sigma * S * dt + (higher order term)
                    # cv1 = W * S_t_dt * dB  # control variate
                    cv_std = np.std(cv, axis=1)
                    cv_mean = np.mean(cv, axis=1)
                    cv_centred[:] = (cv - cv_mean[:, np.newaxis]) / cv_std[:, np.newaxis]

                    YdB[:] = W * ((Y - Y_inc) * dB[t])
                    YdB_std[:] = np.std(YdB, axis=1)
                    YdB_mean[:] = np.mean(YdB, axis=1)
                    YdB_centred[:] = (YdB - YdB_mean[:, np.newaxis]) / YdB_std[:, np.newaxis]

                    corr = np.mean(cv_centred * YdB_centred, axis=1)
                    Z[:] = np.mean(YdB - (cv - cv_expectation[:, np.newaxis]) * corr[:, np.newaxis], axis=1) * (1. / dt)
                else:
                    Z[:] = (1. / N) * np.dot(W, (Y - Y_inc) * dB[t]) * (1. / dt)

        # Time-step
        dt = self.T / m
        # Discount factor
        df = 1 / (1 + r * dt)
        theta = -(r - self.mu) / self.sigma
        drift_dt = (self.mu - 0.5 * self.sigma ** 2) * dt
        sigma_sqt = self.sigma * np.sqrt(dt)
        gauss_normalization = np.sqrt(2 * np.pi) * sigma_sqt
        # S, dB
        # S, B = self.generate_paths(r, N, m, mode='B')
        S, dB = self.generate_paths(r, N, m)

        # price of the option at time T = Initialization for a call
        if oPayoff == "call":
            Y = np.maximum(S[-1] - self.K, 0)
        elif oPayoff == "put":
            Y = np.maximum(self.K - S[-1], 0)
        elif oPayoff == "call combination":
            Y = np.maximum(S[-1] - K1, 0) - 2 * np.maximum(S[-1] - K2, 0)
        elif oPayoff == "put combination":
            Y = np.maximum(K1 - S[-1], 0) - 2 * np.maximum(K2 - S[-1], 0)
        elif oPayoff == 'call average':
            Y = np.maximum(S.mean(axis=0) - self.K, 0)

        if mode == 'all':
            if (oType == 'European'):
                # log_S follows a drifted Brownian motion with
                # drift = mu - sigma**2/2
                # volatility = sigma
                # weight matrix
                W = np.zeros([N, N])
                transition_matrix = np.zeros([N, N])
                marginal_vector = np.zeros(N)
                dist_matrix = np.zeros([N, N])
                log_S_start, log_S_end = np.zeros(N), np.zeros(N)
                Z, Y_inc = np.zeros(N), np.zeros(N)
                cv, cv_centred = np.zeros([N, N]), np.zeros([N, N])
                cv_std, cv_mean, cv_expectation = np.zeros(N), np.zeros(N), np.zeros(N)
                YdB, YdB_centred = np.zeros([N, N]), np.zeros([N, N])
                YdB_std, YdB_mean = np.zeros(N), np.zeros(N)
                # Iteration over time backwardly
                for t in range(m - 1, 0, -1):
                    # work on logscale
                    log_S_end = np.log(S[t + 1, :])
                    log_S_start = np.log(S[t, :])
                    # distances matrix
                    dist_matrix = -np.subtract.outer(log_S_start, log_S_end)
                    # transition densities matrix
                    transition_matrix = np.exp(
                        -0.5 * np.square(dist_matrix - drift_dt) / sigma_sqt ** 2) / gauss_normalization
                    # marginals
                    marginal_vector = np.mean(transition_matrix, axis=0)

                    if display_plot:
                        plt.subplot(3, m, t + 1)
                        plt.plot(log_S_end, marginal_vector, "r.")
                        plt.title("Marginal Density at time {}".format(str(t * dt)))

                    # weight matrix
                    W = transition_matrix / marginal_vector
                    # normalize the rows
                    # W_row_sums = W.sum(axis=1)
                    # W = W / W_row_sums[:, np.newaxis]

                    # initial guess for Y_inc is set to zero
                    Y_inc[:] = 0.
                    # use control variate
                    cv[:, :] = W * dB[t] * dB[t]
                    cv_expectation[:] = dt
                    # do BSDE update
                    do_BSDE_mesh_update(Y, Y_inc, Z, dB[t], W,
                                        cv, cv_expectation)
                    # print("After Z={}".format(Z))
                    # print("After Y_inc={}".format(Y_inc))


                    for __ in range(n_picard):
                        cv[:, :] = W * Z * dB[t]
                        cv_expectation[:] = 0
                        do_BSDE_mesh_update(Y, Y_inc, Z, dB[t], W,
                                            cv, cv_expectation)
                    Y[:] = Y_inc

                    if display_plot:
                        plt.subplot(4, m, m + t + 1)
                        plt.plot(S[t, :], Y, "b.")
                        plt.title("Price at time".format(str(t * dt)))

                        plt.subplot(4, m, 2 * m + t + 1)
                        plt.plot(S[t, :], Z, "b.")
                        plt.title("Z at time".format(str(t * dt)))


                        # plt.plot(S[t+1,:], Z_, "r.")

                        # plt.plot(X, Z, 'r.')
                        # plt.show()
            Y_opt = df * np.mean(Y)
            return (Y_opt)

        elif mode == 'NN':
            if (oType == 'European'):
                # log_S follows a drifted Brownian motion with
                # drift = mu - sigma**2/2
                # volatility = sigma
                # weight matrix
                Z, Y_inc = np.zeros(N), np.zeros(N)
                cv, cv_centred = np.zeros([N, N]), np.zeros([N, N])
                cv_std, cv_mean, cv_expectation = np.zeros(N), np.zeros(N), np.zeros(N)
                YdB, YdB_centred = np.zeros([N, N]), np.zeros([N, N])
                YdB_std, YdB_mean = np.zeros(N), np.zeros(N)
                # Iteration over time backwardly
                for t in range(m - 1, 0, -1):
                    # work on logscale
                    log_S_end = np.log(S[t + 1, :])
                    log_S_start = np.log(S[t, :])
                    # distances matrix
                    dist_matrix = - np.subtract.outer(log_S_start, log_S_end)
                    # transition densities matrix
                    transition_matrix = np.exp(
                        -0.5 * np.square(dist_matrix - drift_dt) / sigma_sqt ** 2) / gauss_normalization
                    # marginals
                    marginal_vector = np.mean(transition_matrix, axis=0)

                    if display_plot:
                        plt.subplot(3, m, t + 1)
                        plt.plot(log_S_end, marginal_vector, "r.")
                        plt.title("Marginal Density at time {}".format(str(t * dt)))

                    # weight matrix
                    W = transition_matrix / marginal_vector
                    # normalize the rows
                    NN = kneighbors_graph(log_S_end.reshape(N, 1), n_neighbors,
                                          mode='distance').nonzero()
                    x = NN[0]
                    y = NN[1]
                    W_bar = copy.deepcopy(W)
                    W_bar[x, y] = 0
                    np.fill_diagonal(W_bar, 0.)
                    W = W - W_bar
                    W_sparse = sparse.csr_matrix(W)

                    # Regression for Z_t:
                    #
                    # Z_t = E( Y_[t+dt] * dW_t | filtration_t)  /dt
                    #

                    # 1st version
                    # if use_variance_reduction == True:
                    #     # use the fact that ****
                    #     for j in range(N):
                    #         list_neighbors = NN[1][j:(j + n_neighbors)].tolist() + [j]
                    #         Y_nn = np.zeros(n_neighbors + 1)
                    #         W_nn = W[j, list_neighbors]
                    #         dB_nn = dB[t, list_neighbors]
                    #         cv[j, :] = W_nn * dB_nn * dB_nn
                    #         cv_expectation = dt
                    #         do_BSDE_mesh_update(Y, Y_nn, Z, dB[t], W,
                    #                             cv, cv_expectation)
                    #     Y_inc[:] = 0.
                    #     # use control variate
                    #     cv[:, :] = W * dB[t] * dB[t]
                    #     cv_expectation[:] = dt
                    #     # do BSDE update
                    #     do_BSDE_mesh_update(Y, Y_inc, Z, dB[t], W,
                    #                         cv, cv_expectation)
                    #     # print("After Z={}".format(Z))
                    #     # print("After Y_inc={}".format(Y_inc))
                    #     for j in range(N):
                    #         list_neighbors = NN[1][j:(j + n_neighbors)].tolist() + [j]
                    #         Y_nn = Y[list_neighbors]
                    #         W_nn = W[j, list_neighbors]
                    #         dB_nn = dB[t, list_neighbors]
                    #         cv1 = W_nn * (S[t, list_neighbors] * dB_nn)  # control variate
                    #         cv1_std = np.std(cv1, axis=1)
                    #         cv1_mean = np.mean(cv1, axis=1)
                    #         cv1_centred = (cv1 - cv1_mean[:, np.newaxis]) / cv1_std[:, np.newaxis]
                    #
                    #         YdB = W_nn * (Y_nn * dB_nn)
                    #         YdB_std = np.std(YdB, axis=1)
                    #         YdB_mean = np.mean(YdB, axis=1)
                    #         YdB_centred = (YdB - YdB_mean[:, np.newaxis]) / YdB_std[:, np.newaxis]
                    #
                    #         corr = np.mean(cv1_centred * YdB_centred, axis=1)
                    #         Z[j] = np.mean(YdB - (cv1 - 1.) * corr[:, np.newaxis], axis=1) * (1. / dt)
                    #         Y_inc[j] = df * ((1. / (n_neighbors + 1)) * np.dot(W_nn, Y_nn) + driver_measurable * dt)
                    #
                    #         # Z_ = (1. / N) * np.dot(W, Y * dB[t]) * (1. / dt)
                    #         # print("diff = {}".format( np.sqrt(np.mean(np.square(Z-Z_)))) )
                    # else:
                    if R != r:
                        Z = (1. / N) * W_sparse.dot(Y * dB[t]) * (1. / dt)
                    else:
                        Z = 0.

                    driver_measurable = -theta * Z
                    driver_stoch = - r * Y - (R - r) * np.minimum(Y - (1. / self.sigma) * Z, 0)
                    Y_inc = (1. / N) * W_sparse.dot(Y + dt * driver_stoch) + dt * driver_measurable
                    # follwing is Majdi's version
                    # delta_B = -np.subtract.outer(B[t,:],B[t+1,:])
                    # Z = (1. / n) * np.sum(W * (delta_B * Y[:,np.newaxis]), axis=1) * (1./ dt)

                    # Regression for Y_t
                    #
                    # Y_t = E(Y_[t+dt] + dt * driver | filtration_t)
                    #

                    # driver = -theta * Z - r * Y - (R-r) * np.minimum(Y - (1. / self.sigma) * Z, 0)
                    #   decompose driver as
                    #   driver = driver_measurable + driver_stoch
                    # where "driver_measurable" is already F_t measurable


                    # Y_inc = np.dot(W, Y + dt * driver)
                    for __ in range(n_picard):
                        # update Z
                        # 1sr version
                        if R != r:
                            Z = (1. / N) * W_sparse.dot((Y - Y_inc) * dB[t]) * (1. / dt)
                        else:
                            Z = 0.
                        # Majdi's version
                        # delta_B = -np.subtract.outer(B[t,:],B[t+1,:])
                        # Z = np.sum(W * delta_B, axis=1) * (1./ dt)
                        # update Z
                        driver_measurable = - theta * Z
                        driver_stoch = - r * Y - (R - r) * np.minimum(Y - (1. / self.sigma) * Z, 0)
                        Y_inc = (1. / N) * W_sparse.dot(Y + dt * driver_stoch) + dt * driver_measurable
                    Y = Y_inc

                    if display_plot:
                        plt.subplot(3, m, m + t + 1)
                        plt.plot(S[t, :], Y, "b.")
                        plt.title("Price at time".format(str(t * dt)))

                        plt.subplot(3, m, 2 * m + t + 1)
                        plt.plot(S[t, :], Z, "b.")
                        plt.title("Z at time".format(str(t * dt)))

                        # plt.plot(X, Z, 'r.')
                        # plt.show()
            Y_opt = df * np.mean(Y)
        return Y_opt
      
def Z_computation(j, X, Y, dB, dt, rf, t):
    dB_j = dB[t, :, j]
    rf.fit(X, Y * dB_j)
    Z_int = rf.predict(X) * (1. / dt)
    return Z_int

class BsdeHD(object):
    def __init__(self, T, K, M, mu, Q, sigma, S_init, r, R):
        self.T = T
        self.K = K
        self.r = r
        self.R = R
        self.M = M
        self.mu = mu
        self.Q = Q
        self.sigma = sigma
        self.S_init = S_init
        self.p = len(self.M)

    @staticmethod
    def Z_compute(sigma, X, N, l, Z, W_marg, n_neighbors):
        for k in range(len(Z)):
            X_k = X[:, k]
            NN_k = kneighbors_graph(X_k.reshape(N, 1), n_neighbors,
                                    mode='distance').nonzero()
            x = NN_k[0]
            y = NN_k[1]
            grad_W_k = scipy.sparse.lil_matrix((N, N))
            grad_W_k[x, y] = - (X_k[y] - X_k[x]) / l ** 2 * np.exp(- (X_k[y] - X_k[x]) ** 2 / (2 * l ** 2))
            marg_grad_W_k = scipy.sparse.diags(np.array(grad_W_k.sum(axis=0))[0])
            grad_k = (grad_W_k * scipy.sparse.diags(np.array(W_marg)[0]) - W * marg_grad_W_k) / np.square(
                W_marg)
            Z[:, k] = sigma * X_k * np.array(grad_k.sum(axis=0)[0])
            return Z.sum(axis=1)

    def generate_correlated_paths(self, m):
        dt = self.T / m
        S = np.zeros((m + 1, self.p))
        dB = np.zeros((m + 1, self.p))
        S[0] = self.S_init
        C = np.linalg.cholesky(self.M)
        for t in range(1, m + 1):
            rand = np.random.standard_normal(self.p)
            rand_int = np.dot(C, rand)
            S[t] = S[t - 1] * np.exp((self.mu - self.Q - self.sigma ** 2 / 2) * dt +
                                     (dt ** 0.5) * self.sigma * rand_int)
            dB[t] = np.sqrt(dt) * rand_int
        return S, dB

    def generate_multiple_asset(self, N, m):
        S = np.zeros([m + 1, N, self.p])
        dB = np.zeros([m + 1, N, self.p])
        # Generate N*p assets
        for i in range(N):
            S_int, dB_int = self.generate_correlated_paths(m)
            S[:, i, :] = S_int
            dB[:, i, :] = dB_int
        return (S, dB)

    def exercise_matrix(self, S, N, m, oType, oPayoff, K1=95., K2=105.):
        payoff = np.zeros((m + 1, N))
        if (oPayoff == 'geometric'):
            if (oType == 'call'):
                for j in range(N):
                    for t in range(m + 1):
                        pay_int = np.prod(S[t, j, :]) ** (1. / self.p)
                        pay = max(pay_int - self.K, 0)
                        payoff[t, j] = pay
            elif (oType == 'put'):
                for j in range(N):
                    for i in range(m + 1):
                        pay_int = np.prod(S[i, j, :]) ** (1. / self.p)
                        pay = max(self.K - pay_int, 0)
                        payoff[i, j] = pay
            elif (oType == 'call combination'):
                for j in range(N):
                    for t in range(m + 1):
                        pay_int = np.prod(S[t, j, :]) ** (1. / self.p)
                        pay = max(pay_int - K1, 0) - 2 * max(pay_int - K2, 0)
                        payoff[t, j] = pay
        elif (oPayoff == 'max'):
            if (oType == 'call'):
                for j in range(N):
                    for t in range(m + 1):
                        pay_int = max(S[t, j, :])
                        pay = max(pay_int - self.K, 0)
                        payoff[t, j] = pay
            elif (oType == 'put'):
                for j in range(N):
                    for t in range(m + 1):
                        pay_int = pay_int = max(S[t, j, :])
                        pay = max(self.K - pay_int, 0)
                        payoff[t, j] = pay
            else:
                print("Enter the type of the option : call or put")
        elif (oPayoff == 'average'):
            if (oType == 'call'):
                for j in range(N):
                    for t in range(m + 1):
                        pay_int = np.mean(S[t, j, :])
                        pay = max(pay_int - self.K, 0)
                        payoff[t, j] = pay
            elif (oType == 'put'):
                for j in range(N):
                    for t in range(m + 1):
                        pay_int = np.mean(S[t, j, :])
                        pay = max(self.K - pay_int, 0)
                        payoff[t, j] = pay

        else:
            print("You did not enter a possible type of payoff.")
        return (payoff)

    def get_price_derivative(self, R, r, N, m, RF_n_estimators= 100, RF_max_leaf_nodes= 50, K1=95., K2=105.,
                             option_type='call', option_payoff='geometric',
                             oType='European', n_picard=10, l=1, n_neighbors=None):
        '''
        Function to generate stock paths.

        Parameters
        ==========

        r               : float
                          lending interest rate
        R               : float
                          borrowing interest rate
        N               : int
                          Number of paths generated
        m               : int
                          number of steps
        d               : int
                          polynomial fit degree

        Returns
        =======
        Y_opt : float
                Price of the European option
        '''
        # Time-step
        dt = self.T / m
        # Discount factor
        df = 1 / (1 + self.r * dt)
        theta = (self.mu - self.r) / self.sigma
        # N simulations of p underlying assets
        S, dB = self.generate_multiple_asset(N, m)
        # Matrix of exercise prices
        h = self.exercise_matrix(S, N, m, option_type, option_payoff)
        # price of the option at time T = Initialization
        Y = h[-1]
        rf = RandomForestRegressor(n_estimators=RF_n_estimators,
                                   max_leaf_nodes=RF_max_leaf_nodes,
                                   oob_score=False,
                                   n_jobs=-1)
        if (oType == 'European'):
            # Iteration over time backwardly
            for t in range(m - 1, 0, -1):
                # Creation of the matrix N*p to regress"
                X = S[t, :, :]
                Z = np.zeros([N, self.p])
                Z_sum = np.zeros(N)

                if self.r != self.R:
                    # Regression for Z
                    for k in range (self.p):
                        dB_k = dB[t, :, k]
                        rf.fit(X, Y * dB_k)
                        Z[:, k] = rf.predict(X) * (1. / dt)
                    Z_sum = Z.sum(axis=1)

                if n_neighbors is not None:
                    NN = NearestNeighbors(n_neighbors).fit(X)
                    distances, indices = NN.kneighbors(X)
                else:
                    raise Exception('Give a number of neighbors according to the {} particles'.format(N))

                weights = np.exp(- distances ** 2 / (2 * l ** 2))
                # W is the weight matrix, i.e in w(i,j) = exp(-(x_j - x_i)^2 / 2.l^2) / sum_axis_0(exp(-(x_j - x_i)^2 / 2.l^2))
                row = np.repeat(list(range(N)), n_neighbors)
                col = indices.reshape(N * n_neighbors)
                W = scipy.sparse.coo_matrix((weights.reshape(N * n_neighbors), (row, col)),
                                            shape=(N, N))
                W_marg = W.sum(axis=0)
                W = W.dot(scipy.sparse.diags(np.array(1 / W_marg)[0]))
                expected_Y = W.dot(Y)


                # marg_grad_W = grad_W.sum(axis=0)
                if n_picard > 0 :
                    for __ in range (n_picard):
                        Y_inc = df * (
                        expected_Y - theta * Z_sum - (self.R - self.r) * np.minimum(Y - (1. / self.sigma) * Z_sum, 0))

                        for k in range(self.p):
                            X_k = X[:, k]
                            NN_k = kneighbors_graph(X_k.reshape(N, 1), n_neighbors,
                                                    mode='distance').nonzero()
                            x = NN_k[0]
                            y = NN_k[1]
                            grad_W_k = scipy.sparse.lil_matrix((N, N))
                            grad_W_k[x, y] = - (X_k[y] - X_k[x]) / l ** 2 * np.exp(- (X_k[y] - X_k[x]) ** 2 / (2 * l ** 2))
                            marg_grad_W_k = scipy.sparse.diags(np.array(grad_W_k.sum(axis=0))[0])
                            grad_k = (grad_W_k * scipy.sparse.diags(np.array(W_marg)[0]) - W * marg_grad_W_k) / np.square(W_marg)
                            Z[:, k] = self.sigma * X_k * np.array(grad_k.sum(axis=0)[0])
                        Z_sum = Z.sum(axis=1)
                else:
                    Y_inc = df * (
                        expected_Y - theta * Z_sum - (self.R - self.r) * np.minimum(Y - (1. / self.sigma) * Z_sum, 0))

                Y[:] = Y_inc
                # plt.plot(X, Z, 'r.')
                # plt.show()
            Y_opt = df * np.mean(Y)
            return Y_opt

        if (oType == 'American'):
            # Iteration over time backwardly
            for t in range(m - 1, 0, -1):
                X = S[t]
                # Regression for Z
                reg1 = np.polyfit(X, Y * dB[t], deg)
                Z = (1. / dt) * np.polyval(reg1, X)
                # print (np.mean( Y * dB[t]) / dt - np.mean(Z), np.std(Z))
                # regression for Y
                reg = np.polyfit(X, Y, deg)
                J = np.polyval(reg, X)
                # Y = np.polyval(reg,X)-Y*r*dt-theta*Z*dt+np.minimum(Y-(1/sigma)*Z,0)*(R-r)*dt
                Y = np.maximum(df * (J - theta * Z * dt -
                                     np.minimum(J - (1. / self.sigma) * Z, 0) *
                                     (R - r) * dt), np.maximum(S[t] - self.K, 0))

                # plt.plot(Y,Z, 'r.')
                # plt.show()
            Y_opt = df * np.mean(Y)
            Z_opt = df * np.mean(Z)
            # print (np.mean(Z), np.var(Z))
            return (Y_opt)

    def get_price(self, N, m, RF_n_estimators=100, RF_max_leaf_nodes=50,
                  option_type='call',
                  RF_max_features='auto',
                  RF_max_depth=None,
                  RF_min_samples_split=2,
                  RF_min_samples_leaf=1,
                  RF_warm_start=False,
                  option_payoff='geometric', oType='European', n_picard=0,
                  regression='RF', 
                  n_threads=2):

        # Time-step
        dt = self.T / m
        # Discount factor
        df = 1 / (1 + self.r * dt)
        theta = (self.mu - self.r) / self.sigma
        # N simulations of p underlying assets
        S, dB = self.generate_multiple_asset(N, m)
        # Matrix of exercise prices
        h = self.exercise_matrix(S, N, m, option_type, option_payoff)
        # price of the option at time T = Initialization
        Y = h[-1]

        if regression == 'RF':
            rf = RandomForestRegressor(n_estimators=RF_n_estimators,
                                       max_features=RF_max_features,
                                       max_leaf_nodes=RF_max_leaf_nodes,
                                       oob_score=False,
                                       max_depth=RF_max_depth,
                                       min_samples_split=RF_min_samples_split,
                                       min_samples_leaf=RF_min_samples_leaf,
                                       warm_start=RF_warm_start,
                                       n_jobs=-1)
        elif regression == 'svm':
            rf = svm.SVR()

        elif regression == 'gbr':
            rf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')

        if (oType == 'European'):

            # Iteration over the paths"
            for t in range(m - 1, 0, -1):
                # Creation of the matrix N*p to regress"
                X = S[t, :, :]
                Z = np.zeros(N)

                if self.r != self.R:
                    # Regression for Z
                    
                    z_int = parmap.map(Z_computation, list(range(self.p)), X, Y, dB, dt, rf, t, processes= n_threads)
                    Z = np.sum(z_int, axis=0)
                    # regression for Y
                    rf.fit(X, Y)
                    J = rf.predict(X)
                    Y_new = df * (J - theta * Z * dt -
                                  np.minimum(Y - (1. / self.sigma) * Z, 0) *
                                  (self.R - self.r) * dt)

                    for __ in range(n_picard):
                        z_int = parmap.map(Z_computation, list(range(self.p)), X, Y - Y_new, dB, dt, rf, t, processes= n_threads)
                        Z = np.sum(z_int, axis=0)
                        Y_new = df * (J - theta * Z * dt -
                                      np.minimum(Y - (1. / self.sigma) * Z, 0) *
                                      (self.R - self.r) * dt)
                    Y[:] = Y_new

                else:
                    # regression for Y
                    rf.fit(X, Y)
                    J = rf.predict(X)
                    Y = df * J

            Y_opt = df * np.mean(Y)

        elif (oType == 'American'):

            # Iteration over the paths"
            for t in range(m - 1, 0, -1):
                # Creation of the matrix N*p to regress"
                X = np.zeros([N, self.p])
                Z = np.zeros(N)
                for i in range(N):
                    for j in range(self.p):
                        X[i, j] = S[i][t, j]

                rf = RandomForestRegressor(n_estimators=RF_n_estimators,
                                           max_leaf_nodes=RF_max_leaf_nodes,
                                           oob_score=False,
                                           n_jobs=-1)

                # Regression for Z
                for k in range(self.p):
                    dB_k = np.zeros(N)
                    for i in range(N):
                        dB_k[i] = dB[i][t, k]
                    rf.fit(X, Y * dB_k)
                    Z_int = rf.predict(X) * (1. / dt)
                    Z = Z + Z_int
                # regression for Y
                rf.fit(X, Y)
                J = rf.predict(X)
                Y = np.maximum(df * (J + theta * Z * dt -
                                     np.minimum(Y - (1. / self.sigma) * Z, 0) *
                                     (self.R - self.r) * dt), h[t])
            Y_opt = df * np.mean(Y)
        return (Y_opt)

    def labordere(self, N, m, RF_n_estimators, RF_max_leaf_nodes):

        # Time-step
        dt = self.T / m
        c = 0.15
        alpha = 0.2
        # N simulations of p underlying assets
        S, dB = self.generate_multiple_asset(N, m)
        # price of the option at time T = Initialization
        Y = np.zeros(N)
        for i in range(N):
            Y[i] = np.cos(np.sum(S[-1, i, :]))
        a = (3 * self.p + 1) / (2 * self.p)
        b = np.zeros(self.p)
        for i in range(self.p):
            b[i] = (1 / self.p) * (1 + (i + 1) / self.p)

        # Iteration over the paths"
        for t in range(m - 1, 0, -1):
            df = np.exp(alpha * (m - t) * dt)
            # Creation of the matrix N*p to regress"
            X = S[t, :, :]
            Z = np.zeros([N, self.p])

            rf = RandomForestRegressor(n_estimators=RF_n_estimators,
                                       max_leaf_nodes=RF_max_leaf_nodes,
                                       oob_score=False,
                                       n_jobs=-1)

            # Regression for Z
            for j in range(self.p):
                dB_j = np.zeros(N)
                dB_j = dB[t, :, j]
                rf.fit(X, Y * dB_j)
                Z_int = rf.predict(X) * (1. / dt)
                Z[:, j] = Z_int

            # regression for Y
            rf.fit(X, Y)
            J = rf.predict(X)
            mean_int = np.dot(X, np.ones(self.p))
            Y_new = J + dt * np.cos(mean_int) * (
                alpha + 0.5 * self.sigma ** 2 + c * np.sin(mean_int) * a * df) * df + c * np.dot(Z, b) * J

            Y[:] = Y_new
        Y_opt = np.mean(Y)

        return (Y_opt)

    def get_price_mesh(self, N, m,
                       option_type='call',
                       option_payoff='geometric',
                       mode='all',
                       n_neighbors=10,
                       K1=95.,
                       K2=105.,
                       oPayoff="call",
                       oType="European", n_picard=0,
                       use_variance_reduction=False,
                       display_plot=False):
        """
        Approach
        ========
        Call T(i,j,t) the transition from S_i at time "t" to S_j at time "t+dt". Then:
        1) P(S_[t+dt] = S_j) \approx (1/N) * sum_i T(i,j,t)  \def P^[t+dt]_j
        2) E[X^[t+dt]_j|S^t_i] \approx (1/N) * sum_j T(i,j,t) * X^[t+dt]_j / P^[t+dt]_j  (basic importance sampling)
        3) Consequently, define
            W(i,j,t) =  T(i,j,t) / P^[t+dt]_j
        and we have
            E[ (something) | filtration_t] = (1/N) * dot_product(W, something)
        4) remark: one can slightly reduce the variance by further normalizing the rows

        """
        # Time-step
        dt = self.T / m
        # Discount factor
        df = 1 / (1 + self.r * dt)
        theta = -(self.r - self.mu) / self.sigma
        drift_dt = (self.mu - 0.5 * self.sigma ** 2) * dt
        sigma_sqt = self.sigma * np.sqrt(dt)
        gauss_normalization = np.sqrt(2 * np.pi) * sigma_sqt
        # S, dB
        S, dB = self.generate_multiple_asset(N, m)
        # price of the option at time T = Initialization for a call
        # Matrix of exercise prices
        h = self.exercise_matrix(S, N, m, option_type, option_payoff)
        # price of the option at time T = Initialization
        Y = h[-1]

        if mode == 'all':
            if (oType == 'European'):
                # log_S follows a drifted Brownian motion with
                # drift = mu - sigma**2/2
                # volatility = sigma
                # weight matrix
                W = np.zeros([N, N, self.p])
                transition_3d_matrix = np.zeros([N, N, self.p])
                marginal_matrix = np.zeros([N, self.p])
                dist_matrix = np.zeros([N, N])
                log_S_start = np.zeros(N)
                log_S_end = np.zeros(N)
                Z = np.zeros(N)
                Y_inc = np.zeros(N)
                # Iteration over time backwardly
                for t in range(m - 1, 0, -1):
                    # work on logscale
                    # distances matrix
                    for l in range(self.p):
                        log_S_end = np.log(S[t + 1, :, l])
                        log_S_start = np.log(S[t, :, l])
                        dist_matrix = - np.subtract.outer(log_S_start, log_S_end)
                        # transition densities matrix
                        transition_3d_matrix[:, :, l] = np.exp(
                            -0.5 * np.square(dist_matrix - drift_dt) / sigma_sqt ** 2) / gauss_normalization
                        # marginals
                    if display_plot:
                        plt.subplot(3, m, t + 1)
                        plt.plot(log_S_end, marginal_matrix, "r.")
                        plt.title("Marginal Density at time {}".format(str(t * dt)))
                    W_joint = np.prod(transition_3d_matrix, axis=2)
                    marginal_vector = np.mean(W_joint, axis=0)
                    marginal_matrix = np.mean(transition_3d_matrix, axis=0)
                    # weight matrix
                    W_joint = W_joint / marginal_vector
                    W = transition_3d_matrix / marginal_matrix
                    # normalize the rows
                    # W_row_sums = W.sum(axis=1)
                    # W = W / W_row_sums[:, np.newaxis]

                    # Regression for Z_t:
                    #
                    # Z_t = E( Y_[t+dt] * dW_t | filtration_t)  /dt
                    #

                    # 1st version
                    if use_variance_reduction == True:
                        # use the fact that ****
                        cv1 = W * (S[t, :] * dB[t])  # control variate
                        cv1_std = np.std(cv1, axis=1)
                        cv1_mean = np.mean(cv1, axis=1)
                        cv1_centred = (cv1 - cv1_mean[:, np.newaxis]) / cv1_std[:, np.newaxis]
                        YdB = W * (Y * dB[t])
                        YdB_std = np.std(YdB, axis=1)
                        YdB_mean = np.mean(YdB, axis=1)
                        YdB_centred = (YdB - YdB_mean[:, np.newaxis]) / YdB_std[:, np.newaxis]

                        corr = np.mean(cv1_centred * YdB_centred, axis=1)
                        Z = np.mean(YdB - (cv1 - 1.) * corr[:, np.newaxis], axis=1) * (1. / dt)
                        Z_ = (1. / N) * np.dot(W, Y * dB[t]) * (1. / dt)
                        # print("diff = {}".format( np.sqrt(np.mean(np.square(Z-Z_)))) )

                    else:
                        if self.R != self.r:
                            for l in range(self.p):
                                Z = Z + (1. / N) * np.dot(W[:, :, l], Y * dB[t, :, l]) * (1. / dt)
                        else:
                            Z = 0.
                    # follwing is Majdi's version
                    # delta_B = -np.subtract.outer(B[t,:],B[t+1,:])
                    # Z = (1. / n) * np.sum(W * (delta_B * Y[:,np.newaxis]), axis=1) * (1./ dt)

                    # Regression for Y_t
                    #
                    # Y_t = E(Y_[t+dt] + dt * driver | filtration_t)
                    #

                    # driver = -theta * Z - r * Y - (R-r) * np.minimum(Y - (1. / self.sigma) * Z, 0)
                    #   decompose driver as
                    #   driver = driver_measurable + driver_stoch
                    # where "driver_measurable" is already F_t measurable


                    driver_measurable = - theta * Z
                    driver_stoch = - self.r * Y - (self.R - self.r) * np.minimum(Y - (1. / self.sigma) * Z, 0)
                    Y_inc = (1. / N) * np.dot(W_joint, Y + dt * driver_stoch) + dt * driver_measurable

                    # Y_inc = np.dot(W, Y + dt * driver)
                    for __ in range(n_picard):
                        # update Z
                        # 1sr version
                        if self.R != self.r:
                            for l in range(self.p):
                                Z = Z + (1. / N) * np.dot(W[:, :, l], (Y - Y_inc) * dB[t, :, l]) * (1. / dt)
                        else:
                            Z = 0.
                        # Majdi's version
                        # delta_B = -np.subtract.outer(B[t,:],B[t+1,:])
                        # Z = np.sum(W * delta_B, axis=1) * (1./ dt)
                        # update Z
                        driver_measurable = - theta * Z
                        driver_stoch = - self.r * Y - (self.R - self.r) * np.minimum(Y - (1. / self.sigma) * Z, 0)
                        Y_inc = (1. / N) * np.dot(W_joint, Y + dt * driver_stoch) + dt * driver_measurable
                    Y = Y_inc

                    if display_plot:
                        plt.subplot(3, m, m + t + 1)
                        plt.plot(S[t, :], Y, "b.")
                        plt.title("Price at time".format(str(t * dt)))

                        plt.subplot(3, m, 2 * m + t + 1)
                        plt.plot(S[t, :], Z, "b.")
                        plt.title("Z at time".format(str(t * dt)))

                        # plt.plot(X, Z, 'r.')
                        # plt.show()
            # Z = np.dot(dB[0], Y) * (1. / dt)
            # driver = -theta * Z - self.r * Y - (self.R - self.r) * np.minimum(Y - (1. / self.sigma) * Z, 0)
            # Y_opt = np.mean(Y + dt * driver)
            Y_opt = df * np.mean(Y)
            return (Y_opt)

        if mode == 'NN':
            if (oType == 'European'):
                # log_S follows a drifted Brownian motion with
                # drift = mu - sigma**2/2
                # volatility = sigma
                # weight matrix
                W = np.zeros([N, N, self.p])
                transition_3d_matrix = np.zeros([N, N, self.p])
                marginal_matrix = np.zeros([N, self.p])
                dist_matrix = np.zeros([N, N])
                log_S_start = np.zeros(N)
                log_S_end = np.zeros(N)
                Z = np.zeros(N)
                Y_inc = np.zeros(N)
                # Iteration over time backwardly
                for t in range(m - 1, 0, -1):
                    V = []
                    NN_Z = []
                    # work on logscale
                    # distances matrix
                    for l in range(self.p):
                        log_S_end = np.log(S[t + 1, :, l])
                        log_S_start = np.log(S[t, :, l])
                        V.append(log_S_end)
                        NN_Z.append(kneighbors_graph(log_S_end.reshape(N, 1), n_neighbors,
                                                     mode='distance').nonzero())
                        dist_matrix = - np.subtract.outer(log_S_start, log_S_end)
                        # transition densities matrix
                        transition_3d_matrix[:, :, l] = np.exp(
                            -0.5 * np.square(dist_matrix - drift_dt) / sigma_sqt ** 2) / gauss_normalization
                        # marginals
                    if display_plot:
                        plt.subplot(3, m, t + 1)
                        plt.plot(log_S_end, marginal_matrix, "r.")
                        plt.title("Marginal Density at time {}".format(str(t * dt)))
                    W_joint = np.prod(transition_3d_matrix, axis=2)
                    marginal_vector = np.mean(W_joint, axis=0)
                    marginal_matrix = np.mean(transition_3d_matrix, axis=0)
                    # weight matrix
                    W_joint = W_joint / marginal_vector
                    W = transition_3d_matrix / marginal_matrix
                    # normalize the rows
                    V = np.asarray(V).reshape(N, self.p)
                    NN = kneighbors_graph(V, n_neighbors,
                                          mode='distance').nonzero()
                    x = NN[0]
                    y = NN[1]
                    W_bar = copy.deepcopy(W_joint)
                    W_bar[x, y] = 0
                    np.fill_diagonal(W_bar, 0.)
                    W_joint = W_joint - W_bar
                    W_joint_sparse = sparse.csr_matrix(W_joint)

                    # Regression for Z_t:
                    #
                    # Z_t = E( Y_[t+dt] * dW_t | filtration_t)  /dt
                    #

                    # 1st version
                    if use_variance_reduction == True:
                        # use the fact that ****
                        cv1 = W * (S[t, :] * dB[t])  # control variate
                        cv1_std = np.std(cv1, axis=1)
                        cv1_mean = np.mean(cv1, axis=1)
                        cv1_centred = (cv1 - cv1_mean[:, np.newaxis]) / cv1_std[:, np.newaxis]

                        YdB = W * (Y * dB[t])
                        YdB_std = np.std(YdB, axis=1)
                        YdB_mean = np.mean(YdB, axis=1)
                        YdB_centred = (YdB - YdB_mean[:, np.newaxis]) / YdB_std[:, np.newaxis]

                        corr = np.mean(cv1_centred * YdB_centred, axis=1)
                        Z = np.mean(YdB - (cv1 - 1.) * corr[:, np.newaxis], axis=1) * (1. / dt)
                        Z_ = (1. / N) * np.dot(W, Y * dB[t]) * (1. / dt)
                        # print("diff = {}".format( np.sqrt(np.mean(np.square(Z-Z_)))) )

                    else:
                        for l in range(self.p):
                            x_l = NN_Z[l][0]
                            y_l = NN_Z[l][1]
                            W_l = W[:, :, l]
                            W_bar_l = copy.deepcopy(W_l)
                            W_bar_l[x_l, y_l] = 0
                            np.fill_diagonal(W_bar_l, 0.)
                            W_l = W_l - W_bar_l
                            W_sparse_l = sparse.csr_matrix(W_l)
                            Z_l = 1 / N * W_l.dot(Y * dB[t, :, l]) * (1 / dt)
                            Z = Z + Z_l

                        alpha = - theta * Z - (self.R - self.r) * np.minimum(Y - (1. / self.sigma) * Z, 0)
                        Y_inc = df * ((1. / N) * W_joint_sparse.dot(Y) + alpha * dt)

                    # follwing is Majdi's version
                    # delta_B = -np.subtract.outer(B[t,:],B[t+1,:])
                    # Z = (1. / n) * np.sum(W * (delta_B * Y[:,np.newaxis]), axis=1) * (1./ dt)

                    # Regression for Y_t
                    #
                    # Y_t = E(Y_[t+dt] + dt * driver | filtration_t)
                    #

                    # driver = -theta * Z - r * Y - (R-r) * np.minimum(Y - (1. / self.sigma) * Z, 0)
                    #   decompose driver as
                    #   driver = driver_measurable + driver_stoch
                    # where "driver_measurable" is already F_t measurable


                    # Y_inc = np.dot(W, Y + dt * driver)
                    for __ in range(n_picard):
                        # update Z
                        # 1sr version
                        for l in range(self.p):
                            x_l = NN_Z[l][0]
                            y_l = NN_Z[l][1]
                            W_l = W[:, :, l]
                            W_bar_l = copy.deepcopy(W_l)
                            W_bar_l[x_l, y_l] = 0
                            np.fill_diagonal(W_bar_l, 0.)
                            W_l = W_l - W_bar_l
                            W_sparse_l = sparse.csr_matrix(W_l)
                            Z_l = 1 / N * W_l.dot((Y - Y_inc) * dB[t, :, l]) * (1 / dt)
                            Z = Z + Z_l

                        driver_measurable = - theta * Z
                        driver_stoch = - self.r * Y - (self.R - self.r) * np.minimum(Y - (1. / self.sigma) * Z, 0)
                        Y_inc = (1. / N) * np.dot(W_joint_sparse, Y + dt * driver_stoch) + dt * driver_measurable
                    Y = Y_inc

                    if display_plot:
                        plt.subplot(3, m, m + t + 1)
                        plt.plot(S[t, :], Y, "b.")
                        plt.title("Price at time".format(str(t * dt)))

                        plt.subplot(3, m, 2 * m + t + 1)
                        plt.plot(S[t, :], Z, "b.")
                        plt.title("Z at time".format(str(t * dt)))

                        # plt.plot(X, Z, 'r.')
                        # plt.show()
            Y_opt = df * np.mean(Y)
            return Y_opt

    def get_comparison_bs(self, strike, dim):
        d = self.sigma ** 2 * (1 - 1 / dim) / 2 + self.Q
        d1 = 1 / (self.sigma * np.sqrt(self.T / dim)) * (
            np.log(self.S_init / strike) + (self.R + 0.5 * self.sigma ** 2 / dim - d) * self.T)
        d2 = d1 - self.sigma * np.sqrt(self.T / dim)
        dscnt = np.exp(-self.R * self.T)
        call = -norm.cdf(d2) * strike * dscnt + norm.cdf(d1) * self.S_init * np.exp(- d * self.T)
        return call


class Touzi(object):
    def __init__(self, T, S_init, k, alpha, c, m, sigma, mu, rho):
        self.T = T
        self.S_init = S_init
        self.k = k
        self.alpha = alpha
        self.c = c
        self.m = m
        self.sigma = sigma
        self.mu = mu
        self.rho = rho

    def diffusion_touzi(self):
        dt = self.T / self.m
        S = np.zeros((self.m + 1, 2))
        dB = np.zeros((self.m + 1, 2))
        S[0] = self.S_init
        for t in range(1, self.m + 1):
            rand = np.random.standard_normal(2)
            S[t, 0] = (S[t - 1, 0] + self.k * self.alpha * dt + self.c * np.sqrt(S[t - 1, 0]) * rand[0] * np.sqrt(
                dt) + 0.25 * self.c * self.c * dt * (rand[0] ** 2 - 1)) / (1 + self.k * dt)
            S[t, 1] = S[t - 1, 1] + self.sigma * np.sqrt(dt) * rand[1]
            dB[t] = np.sqrt(dt) * rand
        return S, dB

    def generate_multiple_diff(self, N):
        S = np.zeros([self.m + 1, N, 2])
        dB = np.zeros([self.m + 1, N, 2])
        for i in range(N):
            S_int, dB_int = self.diffusion_touzi()
            S[:, i, :] = S_int
            dB[:, i, :] = dB_int
        return (S, dB)

    def portfolio_opt(self, eta, eps, N, RF_n_estimators=100, RF_max_leaf_nodes=50, M=40):
        dt = self.T / self.m
        f = lambda x: - np.exp(- eta * x)
        S, dB = self.generate_multiple_diff(N)
        U = f(S[-1, :, 0])
        Z = np.array([eta * np.exp(- eta * S[-1, 0]), 0.])

        for t in range((self.m - 1), 0, -1):
            X = S[t, :, :]
            rf = RandomForestRegressor(n_estimators=RF_n_estimators,
                                       max_leaf_nodes=RF_max_leaf_nodes,
                                       oob_score=False,
                                       n_jobs=-1)

            rf.fit(X, U * dB[t, :, 0])
            Z_1 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, U * dB[t, :, 1])
            Z_2 = 1 / (dt * self.sigma) * rf.predict(X)

            Z = np.array([Z_1, Z_2])

            rf.fit(X, Z_1 * dB[t, :, 0])
            Gamma_1_1 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, Z_1 * dB[t, :, 1])
            Gamma_1_2 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, Z_2 * dB[t, :, 1])
            Gamma_2_2 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, U)
            J = rf.predict(X)

            for j in range(N):
                f = lambda theta: - (0.5 * theta ** 2 * max(X[j, 1], eps) * Gamma_1_1[j] + theta * (
                    self.mu * Z_1[j] + self.rho * self.c * max(X[j, 1], eps) * Gamma_1_2[j]))
                # for theta in np.linspace(eps, M, 10):
                #     print("value of the function a time {} for particle {} with theta {} is {}".format(t, j, theta,
                #                                                                                        f(theta)))
                opt_theta = scipy.optimize.minimize_scalar(f, (eps, M))
                U[j] = J[j] + (-0.5 * self.sigma * (Gamma_1_1[j] + Gamma_2_2[j]) - 0.5 * self.sigma ** 2 + f(
                    opt_theta.x)) * dt

        u_opt = np.mean(U)
        return u_opt


class Touzi_5D(object):
    def __init__(self, T, X_init, k1, k2, b, xi, alpha, c1, c2, m1, m2, sigma, sigma1, mu1, mu2, rho):
        self.T = T
        self.X_init = X_init
        self.k1 = k1
        self.k2 = k2
        self.m1 = m1
        self.m2 = m2
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.mu1 = mu1
        self.sigma = sigma
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.rho = rho
        self.b = b
        self.xi = xi

    def forward_5d(self, m):
        dt = self.T / m
        X = np.zeros((m + 1, 5))
        dB = np.zeros((m + 1, 5))
        X[0] = self.X_init
        for t in range(1, m + 1):
            rand = np.random.standard_normal(5)
            X[t, 0] = X[t - 1, 0] + self.sigma * np.sqrt(dt) * rand[0]
            X[t, 1] = self.b + np.exp(-self.k * dt) * (X[t - 1, 1] - self.b) + self.xi * np.sqrt(
                (1 - np.exp(-2 * self.k * dt)) / (2 * self.k)) * rand[1]
            X[t, 2] = X[t - 1, 2] * np.exp((self.mu1 - 0.5 * self.sigma1 ** 2 * X[t - 1, 2] ** -1 * X[t - 1, 3]) * dt +
                                           self.sigma1 * X[t - 1, 2] ** -0.5 * np.sqrt(X[t - 1, 3] * rand[2]))
            X[t, 3] = (X[t - 1, 3] + self.k1 * self.m1 * dt + self.c1 * np.sqrt(X[t - 1, 3]) * rand[3] * np.sqrt(
                dt) + 0.25 * self.c1 ** 2 * dt * (rand[3] ** 2 - 1)) / (1 + self.k1 * dt)
            X[t, 4] = (X[t - 1, 4] + self.k2 * self.m2 * dt + self.c2 * np.sqrt(X[t - 1, 4]) * rand[4] * np.sqrt(
                dt) + 0.25 * self.c2 ** 2 * dt * (rand[4] ** 2 - 1)) / (1 + self.k2 * dt)
            dB[t] = np.sqrt(dt) * rand
        return X, dB

    def generate_multiple_diff(self, N, m):
        S = np.zeros([m + 1, N, 5])
        dB = np.zeros([m + 1, N, 5])
        for i in range(N):
            S_int, dB_int = self.forward_5d(m)
            S[:, i, :] = S_int
            dB[:, i, :] = dB_int
        return (S, dB)

    def portfolio_opt(self, m, eta, N, RF_n_estimators, RF_max_leaf_nodes):
        dt = self.T / m
        f = lambda x: - np.exp(- eta * x)
        S, dB = self.generate_multiple_diff(N, m)
        U = f(S[-1, :, 0])
        Z = np.array([eta * np.exp(- eta * S[-1, 0]), 0., 0., 0., 0.])

        for t in range((m - 1), 0, -1):
            X = S[t, :, :]
            rf = RandomForestRegressor(n_estimators=RF_n_estimators,
                                       max_leaf_nodes=RF_max_leaf_nodes,
                                       oob_score=False,
                                       n_jobs=-1)

            rf.fit(X, U * dB[t, :, 0])
            Z_0 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, U * dB[t, :, 1])
            Z_1 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, U * dB[t, :, 2])
            Z_2 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, U * dB[t, :, 3])
            Z_3 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, U * dB[t, :, 4])
            Z_4 = 1 / (dt * self.sigma) * rf.predict(X)

            Z = np.array([Z_0, Z_1, Z_2, Z_3, Z_4])

            rf.fit(X, Z_0 * dB[t, :, 0])
            Gamma_0_0 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, Z_1 * dB[t, :, 1])
            Gamma_1_1 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, Z_2 * dB[t, :, 2])
            Gamma_2_2 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, Z_3 * dB[t, :, 3])
            Gamma_3_3 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, Z_4 * dB[t, :, 4])
            Gamma_4_4 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, Z_0 * dB[t, :, 2])
            Gamma_0_2 = 1 / (dt * self.sigma) * rf.predict(X)

            rf.fit(X, U)
            J = rf.predict(X)

            U = J + (-0.5 * self.sigma * (
                Gamma_0_0 + Gamma_1_1 + Gamma_2_2 + Gamma_3_3 + Gamma_4_4) + 0.5 * self.sigma ** 2 * Gamma_1_1 -
                     Z_1 * X[:, 0] * X[:, 1] + (
                         (self.mu2 - self.mu1) * Z_1 + self.sigma1 ** 2 * X[:, 3] * Gamma_0_2) ** 2 / (
                         2 * self.sigma1 ** 2 * X[:, 3] * Gamma_0_0 * X[:, 2] ** -1)
                     + ((self.mu2 - self.mu1) * Z_1) ** 2 / (2 * self.sigma2 ** 2 * X[:, 4] * Gamma_0_0)) * dt

        u_opt = np.mean(U)
        return u_opt
