import numpy as np
import clustering_model
import plotter


num_nodes = 10
num_runs = 300
outgoing_edges_per_node = 5 
value_per_nugget = 0.05

examined_node = 1
rounding = 3

G = clustering_model.Graph(num_nodes, value_per_nugget=value_per_nugget)
# G.sparse_random_edge_init(outgoing_edges_per_node)
G.uniform_random_edge_init()
G.run(num_runs, verbose=True)

# plotter.plot_node_edges(G, examined_node, num_nodes, num_runs, value_per_nugget, show=False, save_fig=False)
# plotter.plot_node_value_over_time(G, examined_node, value_per_nugget, show=False, save_fig=True)
plotter.plot_edge_stds(G, examined_node, num_nodes, value_per_nugget, show=True, all_nodes=False, save_fig=False)
# plotter.plot_network(G, value_per_nugget, save_fig=False)

"""
iterations = 5
for i in range(0, iterations):
    G = clustering_model.Graph(num_nodes, value_per_nugget=value_per_nugget)
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

# print(f'Initial Adjacency Matrix (rounded to {rounding} places): \n{np.round(G.A[0], rounding)}')
# print(f'Difference of start and end adjacency matrices:\n{np.round(G.A[0] - G.A[-1], rounding)}')
# print(f'Difference of start and end node values:\n{np.round(G.nodes[0] - G.nodes[-1], rounding)}')
# print(f'G.A[:, :, {examined_node}].shape: {G.A[:, :, examined_node].shape}, G.nodes.shape: {G.nodes.shape}')
