from BSDE import *
import time
import cProfile

T = 1.
K = 100
S0 = 100
sigma = 0.2
r = 0.01
mu = 0.06
R = 0.01
q = 0.
m = 6
N = 1000
l = 1.
deg = 5
n_picard = 4
n_neighbors = 1
M_run = 20
test = BSDE(S0, K, T, mu, sigma, q)
print (test.get_price_lsm(R, r, N, m, deg=deg,
                          n_picard=n_picard, oPayoff = "call average"))
print (test.get_price_RF(R, r, N, m, n_picard=n_picard, oPayoff = "call average"))

print (test.get_price_mesh(R, r, N, m, mode='all',n_picard=n_picard, oPayoff = "call average"))

