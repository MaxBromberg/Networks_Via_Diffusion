import numpy as np

import sys
sys.path.append('../')

import graph
import utility_funcs
import plotter

num_nodes = 60

G = graph.Graph(num_nodes=num_nodes, edge_conservation_coefficient=0.5, selectivity=0.5,
                reinforcement_info_score_coupling=True, positive_eff_dist_and_reinforcement_correlation=False)
# G.uniform_random_edge_init()
# G.sparse_random_edge_init(nonzero_edges_per_node=3, connected=True)
G.scale_free_edge_init(degree_exponent=1.2, min_k=1, equal_edge_weights=True, connected=True)
# G.nx_scale_free_edge_init(degree_exponent=1.2, min_k=1)
# G.nx_scale_free_edge_init_unconnected()
# G.barabasi_albert_edge_init(num_edges_per_new_node=int(num_nodes/10))

print(f'np.round(G.A[-1], 3): \n {np.round(G.A[-1], 3)}')

inv_one_minus_A = np.linalg.inv(np.identity(G.A.shape[1]) - G.A[-1])
zeros = 0
for value in inv_one_minus_A.flatten():
    if value == 0: zeros += 1
print(f'{zeros} zeros in inverse initial 1-A')

output_dir = "/home/maqz/Desktop/"
csv_name = 'sparse.csv'
# utility_funcs.A_to_csv(G.A[-1], output_dir=output_dir, csv_name=csv_name, delimiter=" ")

utility_funcs.sum_matrix_signs(G.A[-1])
utility_funcs.print_row_column_sums(G.A[-1])

G.simulate(num_runs=1000, multiple_path=False)
t = 1
# A = G.RWED(G.A[-t])
A = G.A[-t]
np.printoptions(suppress=True)
print(f'A: \n{np.round(A, 3)}')
utility_funcs.sum_matrix_signs(A)
utility_funcs.print_row_column_sums(A)
plotter.plot_network(G)
# plotter.plot_eff_dist(G, all_to_all=True)
# plotter.plot_eff_dist(G)

# RWED, non_numpy_RWED = G.RWED(G.A[-t]), G.RWED(G.A[-t], via_numpy=False)
# print(f'G.RWED(G.A[-1]):\n {np.round(RWED, 2)} \n')
# print(f'G.RWED(G.A[-1], via_numpy=False):\n {np.round(non_numpy_RWED, 2)}')
# RWED_diffs = [diff for diff in list(RWED.flatten() - non_numpy_RWED.flatten()) if diff != 0]
# RWED_diffs = RWED - non_numpy_RWED
# print(f'RWED differences from computational library (numpy.linalg - scipy.sparse):  \n{RWED_diffs}')

