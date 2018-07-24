

from BSDE import *
import cProfile
import warnings
warnings.filterwarnings("ignore")
import time

T = 0.5
m_time_discretization = 6
K = 100
S0 = 100
sigma = 0.2
r = 0.04
p = 7
M = np.eye(p)
N_particles = 100
n_neighbors = 10
mu = 0.06
R = 0.06
q = 0.



l = range(10, 1000, 20)
a = np.zeros(len(l))
# for j, i in enumerate(l):
#     price_mesh_fast = test.get_price_mesh(N_particles, m_time_discretization,
#                                           r, R, n_picard=0,
#                                           display_plot=False, use_variance_reduction=False, n_neighbors=i, mode='NN')
#     a[j] = price_mesh_fast
#     print("Mesh Fast for {} neighbors = {}".format(i, price_mesh_fast))
#
# plt.plot(l, a, 'r.')
# plt.show()

pr = cProfile.Profile()
pr.enable()
#
# price = test.get_price_mesh(N_particles, m_time_discretization,
#                             r, R, n_picard=0,
#                             display_plot=False, use_variance_reduction=False,
#                             n_neighbors=999, mode='NN')
#

# print("NN = {}".format(price))
# test = BSDE(S0, K, T, mu, sigma, q)
# price_lsm = test.get_price_lsm(R, r, N_particles, m_time_discretization, n_picard=10)
# print("lsm derivatives techniques = {}".format(price_lsm))

# price_derivative = test.get_price_derivative(R, r, N_particles, m_time_discretization, n_picard=0, n_derivatives=10,
#                                              l=10)
# print(" derivatives techniques  = {} with l = {}".format(price_derivative, 1))
# for l in range(1, 100, 2):
#     price_derivative = test.get_price_derivative(R, r, N_particles, m_time_discretization, n_picard=0, n_derivatives=10,
#                                                  l=l)
#     print(" derivatives techniques  = {} with l = {}".format(price_derivative, l))
M_run = 1
#
a = np.zeros(M_run)
for i in range (M_run):
    test = BSDE(S0, K, T, mu, sigma, q)
    a[i] = test.get_price_derivative(R, r, N_particles, m_time_discretization, l=1, n_neighbors=n_neighbors)
min_a = min(a)
pr.disable()
pr.print_stats(sort="time")
max_a = max(a)
mean_a = np.mean(a)
std_a = np.std(a)
print ("mean = " + str(mean_a))
print ("std = " + str(std_a))
print ("min = " + str(min_a))
print ("max = " + str(max_a))

# test = BSDE(S0, K, T, mu, sigma, q)
# price_derivative = test.get_price_derivative(R, r, N_particles, m_time_discretization, l=1., n_neighbors=n_neighbors,
#                                              use_display=False)
# # print (price_derivative)
# N_in = [10, 50, 100, 200, 300, 400, 500, 1000, 2000]
# plot_N = np.zeros(len(N_in))
# timing = np.zeros_like(plot_N)
# for i, N in enumerate(N_in):
#     start = time.time()
#     test = BSDE(S0, K, T, mu, sigma, q)
#     plot_N[i] = test.get_price_derivative(R, r, N, m_time_discretization, l=1, n_neighbors=np.floor(N / 10)+1)
#     elapsed = time.time()
#     timing [i] = elapsed
# plt.plot(N_in, timing, 'r.')
# plt.show()
