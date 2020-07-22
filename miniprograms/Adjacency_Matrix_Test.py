import numpy as np

import sys
sys.path.append('../')

import graph
import utility_funcs

num_nodes = 10

G = graph.Graph(num_nodes=num_nodes, edge_conservation_coefficient=0.5, selectivity=0.5,
                reinforcement_info_score_coupling=True, positive_eff_dist_and_reinforcement_correlation=False)
# G.uniform_random_edge_init()
G.sparse_random_edge_init(nonzero_edges_per_node=3, connected=True)
# G.scale_free_edge_init(degree_exponent=1.4, min_k=1, equal_edge_weights=True, connected=True)
# G.nx_scale_free_edge_init(degree_exponent=1.4, min_k=1)
# G.nx_scale_free_edge_init_unconnected()
# G.barabasi_albert_edge_init(num_nodes, 4)

output_dir = "/home/maqz/Desktop/"
csv_name = 'sparse.csv'
print(G.A[-1])
utility_funcs.A_to_csv(G.A[-1], output_dir=output_dir, csv_name=csv_name, delimiter=" ")
# G.simulate(num_runs=1000, multiple_path=False)
# print(G.A[-1])


# plotter.plot_eff_dist(G, all_to_all=True)
# print(f'G.RWED(G.A[-1]):\n {np.round(G.RWED(G.A[-1]), 2)} \n')
# print(f'G.RWED(G.A[-1], via_numpy=False):\n {np.round(G.RWED(G.A[-1], via_numpy=False), 2)}')

np.printoptions(suppress=True)
print(np.round(G.A[-1], 3))
row_sums = []
column_sums = []
for node in range(G.A[-1].shape[0]):
    row_sums.append(sum(G.A[-1][node, :]))
    column_sums.append(sum(G.A[-1][:, node]))
    if np.isclose(sum(G.A[-1][node, :]), 0): print(f'Row {node} sums to zero')
    if np.isclose(sum(G.A[-1][:, node]), 0): print(f'Column {node} sums to zero')

print(f'Row sums:\n {np.array(row_sums)}')
print(f'Column sums:\n {np.round(np.array(column_sums), 2)}')
