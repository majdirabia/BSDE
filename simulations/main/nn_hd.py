from BSDE import *
import time
import cProfile

T = 0.5
m = 4
p = 7
K = 100.
r = 0.04
R = 0.06
M = np.eye(p)
S_init = 100.
mu = 0.06
sigma = 0.2
N = 2
n_neighbors = 1
Q = 0

start = time.time()
M_run = 1
a = np.zeros(M_run)
for i in range (M_run):
    test_hd = BsdeHD(T, K, M, mu, Q, sigma, S_init, r, R)
    a[i] = test_hd.get_price_derivative(R, r, N, m,
                                             option_type='call', option_payoff='geometric',
                                             oType='European', n_picard=1, n_neighbors=n_neighbors, l=0.1)
    elapsed = time.time() - start
print (round(elapsed, 2))
min_a = min(a)
max_a = max(a)
mean_a = np.mean(a)
std_a = np.std(a)
print ("mean = " + str(mean_a))
print ("std = " + str(std_a))
print ("min = " + str(min_a))
print ("max = " + str(max_a))
print(''.format())

# elapsed = time.time() - start
# price_bs = test_hd.get_comparison_bs(K, p)
# print ('{} within {}s compared to B-S price of {}'.format(round(price_derivative, 4), round(elapsed, 2), price_bs))
