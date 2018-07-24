from BSDE import *
import time
import warnings
warnings.filterwarnings('ignore')

T = 0.25
K = 100.
S0 = 100.
p = 1
M = np.eye(p)
sigma = 0.2
r = 0.01
mu = 0.05
R = 0.06
Q = 0.

M_run = 20
m = [4, 6, 8, 10, 12]
N = [100, 1000, 4000]
n_picard = 3


# a = np.zeros([len(deg), len(N), M_run])
# timing = np.zeros_like(a)
# for i, deg_int in enumerate(deg):
#     for j, N_int in enumerate(N):
#         for k in range(M_run):
#             start = time.time()
#             test = BSDE(S0, K, T, mu, sigma, Q)
#             a[i, j, k] = test.get_price_lsm(R, r, N_int, m, deg=deg_int, n_picard=10, oPayoff='call combination')
#             elapsed = time.time() - start
#             timing[i, j, k] = round(elapsed, 3)
#         print (a[i, j, :].mean())
#
a = np.zeros([len(m), len(N), M_run])
timing = np.zeros_like(a)
for i, m_int in enumerate(m):
    for j, N_int in enumerate(N):
        for k in range(M_run):
            start = time.time()
            test = BSDE(S0, K, T, mu, sigma, Q)
            a[i, j, k] = test.get_price_mesh(R, r, N_int, m_int,  n_picard=n_picard, mode='all', oPayoff='call combination')
            elapsed = time.time() - start
            timing[i, j, k] = round(elapsed, 3)
        print (a[i, j, :].mean())
#
# import pandas as pd
#
# df = pd.DataFrame(a)
# df.to_csv('lsm.csv')
#
matrix = a.mean(axis=2)
matrix = matrix.T
matrix = matrix.round(3)
std_mat = a.std(axis=2)
std_mat = std_mat.T
std_mat = std_mat.round(3)
new_mat = np.chararray(shape=[len(N), len(m)], itemsize=100)
for i in range(len(N)):
    for j in range(len(m)):
        msg = "{} ({})".format(matrix[i, j], std_mat[i, j])

        new_mat[i, j] = msg

import tabulate
print (tabulate.tabulate(new_mat.T, headers=N, tablefmt="latex"))
