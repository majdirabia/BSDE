from BSDE import *
import cProfile
# after your program ends

dim = 7
X0 = 100 + np.zeros(dim)
T = 0.5
r = 0.04
R = 0.06
mu = 0.06 + np.zeros_like(X0)
sigma = 0.2 + np.zeros(dim)
alpha = lambda t, x: mu * x
beta = lambda t, x: sigma * x 
corr = np.eye(dim)
N = 100000
m = 6
theta = (mu - r) / sigma
driver = lambda t, x, y, z: - z.dot(theta) - r * y - (R -r) * np.minimum(y - np.sum(z / sigma, axis=1),0)


# pr=cProfile.Profile()
# pr.enable()

start=time.time()

K = 100
xi = lambda x: np.maximum(np.prod(x, axis=1) ** (1 / x.shape[1]) - K, 0)
fwd = ForwardProcess(X0, T, alpha, beta, corr, N, m)

K1 = 105.
K2 = 95.

xi_asian_option = lambda x : np.maximum(x.mean(axis=1) - K2, 0)

xi_cc = lambda x: np.maximum(x - K2, 0) - 2 * np.maximum(x - K1, 0)
bsde = BSDE(driver, xi, fwd)
test = bsde.Regression(RF_n_tree=100, RF_max_leaf_nodes=20, n_picard=1)
print(test)

print (round(time.time() - start, 2))

# M_run = 1
#
# a = np.zeros(M_run)
# timing = np.zeros_like(a)
# for i in range(M_run):
#     bsde = BSDE(driver, xi, fwd)
#     a[i] = bsde.Regression(RF_n_tree=100, RF_max_leaf_nodes=20, n_picard=20)
#     print (a[i])
# print(a.mean())
#
#
# pr.disable()
# pr.print_stats(sort="time")