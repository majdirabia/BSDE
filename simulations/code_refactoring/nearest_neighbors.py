from BSDE import *

T = 0.5
m_time_discretization = 10
K = 100
S0 = 100
sigma = 0.2
r = 0.06
p = 7
M = np.eye(p)

N_particles = 100
mu = 0.06
R = 0.06
q = 0.
RF_n_trees = 100
RF_max_leaf_nodes = 50

test = BSDE(S0, K, T, mu, sigma, q)
#price_mesh_slow = test.get_price_mesh(N_particles, m_paths, r, R, mode='all neighbours')
#print("Mesh Slow = {}".format(price_mesh_slow))

plt.rcParams['figure.figsize'] = (7*m_time_discretization,7*2) # Make the figures a bit

price_mesh_fast = test.get_price_mesh(N_particles, m_time_discretization,
                                          r, R, n_picard = 0,mode = 'NN',
                                           display_plot = False, use_variance_reduction = False, n_neighbors=90)

print("Mesh Fast = {}".format(price_mesh_fast))



# for i in range (10, 1000, 70):
#     price_mesh_fast = test.get_price_mesh(N_particles, m_time_discretization,
#                                           r, R, n_picard = 0,
#                                           display_plot = True, use_variance_reduction = False, n_neighbors = i)
#     print("Mesh Fast = {}".format(price_mesh_fast))

price_LSM = test.get_price_lsm(R, r, N_particles, m_time_discretization, deg=5, n_picard = 10)
print("LSM = {}".format(price_LSM))

#print(test.get_price_hc(R, r, N, m, n_hc=10, n_picard=0))
#print(test.get_price_lsm(R, r, N, m, n_picard=0))
#X = 100 + np.sqrt(20) * np.random.standard_normal(20)
#Y = 7 + 0.2 * np.random.standard_normal(20)
#y = hypercube(X,Y,[1,2,3],20)
#print (y)
#print(test_hd.get_comparison_bs(K, R, sigma, S_init, T, p, Q))

# S, dB = diffusion_touzi(T, S_init, 0.1, 0.3, 0.2, 6, 0.6)
# print (S, dB)