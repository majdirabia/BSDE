from BSDE import *
T = 1
m = 10
K = 100.
r = 0.
R = 0.
p = 10
M = np.eye(p)
S_init = 0.5
mu = 0.
sigma = 1 / np.sqrt(p)
N = 1000
Q = 0.
RF_n_trees = 100
RF_max_leaf_nodes = 50
test_hd = BsdeHD(T, K, M, mu, Q, sigma, S_init, r, R)
res = test_hd.labordere(N,m,RF_n_trees, RF_max_leaf_nodes)
print(res)
