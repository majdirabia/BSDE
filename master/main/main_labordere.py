from BSDE import *
T = 4
m = 6
K = 100.
r = 0.
R = 0.
p = 10
M = np.eye(p)
S_init = 1.
mu = 0.
sigma = 0.2
N = 10000
Q = 0.
RF_n_trees = 100
RF_max_leaf_nodes = 50
beta = 0.01
#test_hd = BsdeHD(T, K, M, mu, Q, sigma, S_init, r, R)
#res = test_hd.labordere(N,m,RF_n_trees, RF_max_leaf_nodes)
#print(res)

a = np.zeros(10)
for i in range (10):
    test = BSDE(S_init, K, T, mu, sigma, Q)
    a[i] = test.get_cva(N, m, r, RF_n_trees, RF_max_leaf_nodes, beta)

min_a = min(a)
max_a = max(a)
mean_a = np.mean(a)
std_a = np.std(a)
print ("mean = " + str(mean_a))
print ("std = " + str(std_a))
print ("min = " + str(min_a))
print ("max = " + str(max_a))