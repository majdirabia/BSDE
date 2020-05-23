import BSDE
import numpy as np

T = 1.
S_init = np.array([0.3, 1.])
mu = 0.15
c = 0.2
k = 0.1
alpha = 0.3
rho = 0
sigma = 0.6
m = 6
eps = 0.01
eta = 0.
N = 100
M = 10

a = []
# for __ in range (10):
test = BSDE.Touzi(T, S_init, k, alpha, c, m, sigma, mu, rho)
print (test.portfolio_opt(1, eps, N, RF_n_estimators=100, RF_max_leaf_nodes=50, M=M))



# for eta_int in [3.5, 3.6, 3.7, 3.8, 3.9]:
#     a = []
#     print (eta_int)
#     for __ in range (10):
#         test = BSDE.Touzi(T, S_init, k, alpha, c, m, sigma, mu, rho)
#         a.append(test.portfolio_opt(eta_int, eps, N, RF_n_estimators=10, RF_max_leaf_nodes=5, M=M))
#     a = np.array(a)
#
#     print(a.mean())