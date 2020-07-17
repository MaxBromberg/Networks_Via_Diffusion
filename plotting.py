import graph
import plotter
import numpy as np

num_nodes = 10
num_runs = 301
outgoing_edges_per_node = int(num_nodes/3)
value_per_nugget = 1  # does not matter at present, as effective distances and hence node values are based on eff_dis

examined_node = 2
rounding = 3

# edge_weighting_exp: tunes between only y (edge) dependence (at 0) and x (eff_dist) dependence at (at 1)

G = graph.Graph(num_nodes, edge_conservation_coefficient=0.2, selectivity=0.8, reinforcement_info_score_coupling=False, positive_eff_dist_and_reinforcement_correlation=False)
# G.sparse_random_edge_init(outgoing_edges_per_node)
G.uniform_random_edge_init()
G.simulate(num_runs, eff_dist_delta_param=1, constant_source_node=1, equilibrium_distance=200, multiple_path=False, verbose=True)
# plotter.plot_weight_histogram(G, num_bins=100, show=True)
# plotter.plot_effective_distance_histogram(G.get_eff_dist(multiple_path=True), num_bins=100)
# plotter.plot_adjacency_matrix_as_heatmap(G, show=True)

# plotter.plot_node_edges(G, examined_node, num_nodes, num_runs, value_per_nugget, show=True, save_fig=False)
# plotter.plot_ave_node_values(G, value_per_nugget)
# plotter.plot_ave_node_values(G, value_per_nugget, as_efficiency=True)
# plotter.plot_node_values(G, node='all', show=True, save_fig=False)
# plotter.plot_edge_stds(G, examined_node, num_nodes, value_per_nugget, show=True, all_nodes=True, save_fig=False)
# plotter.plot_global_eff_dist(G)  #, fit='average')
# plotter.plot_edge_sum(G)

# plotter.plot_network(G, value_per_nugget, show=True, save_fig=False)
# plotter.plot_single_network(G, num_runs, source_weighting=True)
plotter.plot_network(G, nodes_sized_by_eff_distance=False)
plotter.plot_degree_histogram(G)
# plotter.animate_network_evolution(G, num_runs_per_fig=5, gif_duration_in_sec=10, verbose=True)

print(f'G.eff_dist_diff: {G.eff_dist_diff()}')

# Only convert to nx_graphs if the plots are needed. Takes a bit
# print(f'Converting numpy graphs to nx_graphs for network observables calculations:')
# nx_graphs = G.convert_history_to_list_of_nx_graphs(verbose=False)
# plotter.plot_clustering_coefficients(nx_graphs)
# plotter.plot_ave_neighbor_degree(nx_graphs, target='in', source='in')  # if not considering 'in' for both s and t, yields constant


print(f'Initial Adjacency Matrix (rounded to {rounding} places): \n{np.round(G.A[0], rounding)} \n')
print(f'Final ({G.A.shape[0]-1}th) Adjacency Matrix (rounded to {rounding} places): \n{np.round(G.A[-1], rounding)} \n')
print(f'Difference of start and end adjacency matrices:\n{np.round(G.A[0] - G.A[-1], rounding)} \n')
print(f'Start node values:{np.round(G.nodes[0], rounding)}\n')
print(f'End node values: {np.round(G.nodes[-1], rounding)}\n')
print(f'Difference of start and end node values:\n{np.round(G.nodes[0] - G.nodes[-1], rounding)}\n')
print(f'G.A[:, :, {examined_node}].shape: {G.A[:, :, examined_node].shape}, G.nodes.shape: {G.nodes.shape} \n')

print(f'Initial x values: {np.round([(G.nodes[1][i] / G.nodes[1].sum()) for i in range(G.nodes.shape[1])], 3)}')
print(f'Final x values: {np.round([(G.nodes[-1][i] / G.nodes[-1].sum()) for i in range(G.nodes.shape[-1])], 3)}')



