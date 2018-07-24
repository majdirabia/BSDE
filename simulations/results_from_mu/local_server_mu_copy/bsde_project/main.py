from BSDE import *


import time
times = []
dim = []
for p in range (1,21,2):
    start = time.time()
    T = 3
    m = 8
    K = 100.
    r = 0.05
    R = 0.05
    M = np.eye(p)
    S_init = 100.
    mu = 0.05
    sigma = 0.2
    N = 4000
    Q = 0.1
    RF_n_trees = 200
    RF_max_leaf_nodes = 100
    test_hd = BsdeHD(T, K, M, mu, Q, sigma, S_init, r, R)
    test_hd.get_price(N,m, RF_n_trees,RF_max_leaf_nodes,
                      option_type = 'call', option_payoff = 'max', oType= 'European', n_picard= 5)
    elapsed = time.time() - start
    times.append(elapsed)
    dim.append(p)


#test = BSDE(S_init, K, T, mu, sigma, Q)
#print(test.get_price_hc(R, r, N, m, n_hc=10, n_picard=0))
#print(test.get_price_lsm(R, r, N, m, n_picard=0))
#X = 100 + np.sqrt(20) * np.random.standard_normal(20)
#Y = 7 + 0.2 * np.random.standard_normal(20)
#y = hypercube(X,Y,[1,2,3],20)
#print (y)
#print(test_hd.get_comparison_bs(K, R, sigma, S_init, T, p, Q))
