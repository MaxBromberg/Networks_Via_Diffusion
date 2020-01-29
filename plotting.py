import graph
import plotter
import numpy as np


num_nodes = 10
num_runs = 100
outgoing_edges_per_node = 1
value_per_nugget = 0.005

examined_node = 2
rounding = 3

G = graph.SumEffDisGraph(num_nodes, alpha=1, gamma=None, q=0.2)
# G = graph.LogEffDisGraph(num_nodes, value_per_nugget=value_per_nugget, beta=None)
# G.sparse_random_edge_init(outgoing_edges_per_node)
G.uniform_random_edge_init()
G.run(num_runs)#, constant_source_node=0)

# plotter.plot_weight_histogram(G, num_bins=100, show=True)
# plotter.plot_effective_distance_histogram(G.get_eff_dist(multiple_path=True), num_bins=100)
# plotter.plot_adjacency_matrix_as_heatmap(G, show=True)
# plotter.plot_node_edges(G, examined_node, num_nodes, num_runs, value_per_nugget, show=True, save_fig=False)
# plotter.plot_node_values(G, value_per_nugget, node=1)
# plotter.plot_ave_node_values(G, value_per_nugget)
# plotter.plot_ave_node_values(G, value_per_nugget, as_efficiency=True)
# plotter.plot_node_value_over_time(G, examined_node, value_per_nugget, show=True, save_fig=False)
# plotter.plot_edge_stds(G, examined_node, num_nodes, value_per_nugget, show=True, all_nodes=True, save_fig=False)
# plotter.plot_network(G, value_per_nugget, show=True, save_fig=False)
plotter.plot_single_network(G, num_runs)
plotter.plot_network(G, value_per_nugget, nodes_sized_by_eff_distance=True)
"""
iterations = 5
for i in range(0, iterations):
    G = graph.LogEffDisGraph(num_nodes, value_per_nugget=value_per_nugget)
    G.sparse_random_edge_init(outgoing_edges_per_node)
    # The above init does not yield the same values each initialization, though as it's seeded, it should...
    # G.uniform_random_edge_init()
    G.run(num_runs, verbose=False)

    plotter.plot_node_edges(G, examined_node, num_nodes, num_runs, value_per_nugget, show=False, save_fig=True)
    # plotter.plot_node_value_over_time(G, examined_node, value_per_nugget, show=False, save_fig=True)
    plotter.plot_edge_stds(G, examined_node, num_nodes, value_per_nugget, show=False, all_nodes=False, save_fig=True)

    print(f'Run with {value_per_nugget} nugget value complete ({i+1} of {iterations})')
    value_per_nugget = np.round(value_per_nugget + 0.05, 2)
"""

print(f'Initial Adjacency Matrix (rounded to {rounding} places): \n{np.round(G.A[0], rounding)}')
print(f'Final ({G.A.shape[0]-1}th) Adjacency Matrix (rounded to {rounding} places): \n{np.round(G.A[-1], rounding)}')
print(f'Difference of start and end adjacency matrices:\n{np.round(G.A[0] - G.A[-1], rounding)}')
print(f'Start node values:{np.round(G.nodes[0], rounding)}')
print(f'End node values: {np.round(G.nodes[-1], rounding)}')
print(f'Difference of start and end node values:\n{np.round(G.nodes[0] - G.nodes[-1], rounding)}')
print(f'G.A[:, :, {examined_node}].shape: {G.A[:, :, examined_node].shape}, G.nodes.shape: {G.nodes.shape}')
