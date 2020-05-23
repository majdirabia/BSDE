from BSDE import *
import time

T = 1.
K = 100.
S0 = 100.
sigma = 0.4
p = 7
M = np.eye(p)
r = 0.03
mu = 0.03
R = 0.03
Q = 0.05
N =10000
m = 10

test = BsdeHD(T, K, M, mu, Q, sigma, S0, r, R)
print (test.get_price(N, m, RF_n_estimators=100, RF_max_leaf_nodes=2000,
                     oType='American'))