import BSDE
import numpy as np

T = 1.
S_init = np.array([1., 0.3])
mu = 0.15
c = 0.2
k = 0.1
alpha = 0.3
rho = 0
sigma = 1.2
m = 6



test = BSDE.Touzi(T, S_init, k, alpha, c, m, sigma, mu, rho)
print (test.portfolio_opt(0.5, 1000, 100, 50))