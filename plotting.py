import graph
import plotter
import numpy as np

num_nodes = 10
num_runs = 200
outgoing_edges_per_node = int(num_nodes/3)
value_per_nugget = 0.05

examined_node = 2
rounding = 3

G = graph.SumEffDisGraph(num_nodes, exp_edge_weighting=0.5, enforcement_factor=0.01)
G.sparse_random_edge_init(outgoing_edges_per_node)
# G.uniform_random_edge_init()
G.run(num_runs, update_interval=1, exp_decay_param=0.1)
# plotter.plot_weight_histogram(G, num_bins=100, show=True)
# plotter.plot_effective_distance_histogram(G.get_eff_dist(multiple_path=True), num_bins=100)
# plotter.plot_adjacency_matrix_as_heatmap(G, show=True)

# plotter.plot_node_edges(G, examined_node, num_nodes, num_runs, value_per_nugget, show=True, save_fig=False)
# plotter.plot_node_values(G, value_per_nugget, node=1)
# plotter.plot_ave_node_values(G, value_per_nugget)
# plotter.plot_ave_node_values(G, value_per_nugget, as_efficiency=True)
# plotter.plot_node_value_over_time(G, examined_node, value_per_nugget, show=True, save_fig=False)
# plotter.plot_edge_stds(G, examined_node, num_nodes, value_per_nugget, show=True, all_nodes=True, save_fig=False)
plotter.plot_global_eff_dist(G)
plotter.plot_edge_sum(G)

# plotter.plot_network(G, value_per_nugget, show=True, save_fig=False)
# plotter.plot_single_network(G, num_runs, source_weighting=True)
plotter.plot_network(G, value_per_nugget, nodes_sized_by_eff_distance=True)

plotter.gif_of_network_evolution(G, num_runs_per_fig=10, gif_duration_in_sec=20, source_weighting=True, verbose=True)


# print(f'Initial Adjacency Matrix (rounded to {rounding} places): \n{np.round(G.A[0], rounding)}')
# print(f'Final ({G.A.shape[0]-1}th) Adjacency Matrix (rounded to {rounding} places): \n{np.round(G.A[-1], rounding)}')
# print(f'Difference of start and end adjacency matrices:\n{np.round(G.A[0] - G.A[-1], rounding)}')
# print(f'Start node values:{np.round(G.nodes[0], rounding)}')
# print(f'End node values: {np.round(G.nodes[-1], rounding)}')
# print(f'Difference of start and end node values:\n{np.round(G.nodes[0] - G.nodes[-1], rounding)}')
# print(f'G.A[:, :, {examined_node}].shape: {G.A[:, :, examined_node].shape}, G.nodes.shape: {G.nodes.shape}')
