import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import utility_funcs
import effective_distance as ed
from pathlib import Path
import plotter
import graph
import hierarchy_coordinates as hc
import multiprocessing as mp

num_nodes = 20
edge_conservation_val = 0.3
selectivity_val = 0.3
reinforcement_info_score_coupling = False
positive_eff_dist_and_reinforcement_correlation = False
eff_dist_is_towards_source = False
nodes_adapt_outgoing_edges = False
incoming_edges_conserved = True
N = 4
runs = 10

init_args = [num_nodes, edge_conservation_val, selectivity_val, reinforcement_info_score_coupling, positive_eff_dist_and_reinforcement_correlation, eff_dist_is_towards_source, nodes_adapt_outgoing_edges, incoming_edges_conserved]
G = graph.Graph(*init_args)
# G.uniform_random_edge_init()
# G.sparse_random_edge_init(3, connected=True)
G.edge_initialization_conditional(1.2)
print(f'G.A[0]: \n {np.round(G.A[0], 3)}')

# print(np.round(utility_funcs.exponentially_distribute(exponent=0.5, dist_max=1, dist_min=0, num_exp_distributed_values=N), 2))
G.simulate(num_runs=runs, constant_source_node=2)
print(f'G.A[{runs}]: \n {np.round(G.A[-1], 3)}')


plotter.plot_nx_network(nx.to_directed(nx.from_numpy_matrix(np.array(G.A[-1]), create_using=nx.DiGraph)), show=True)
# condensed_graphs, nx_graphs = G.node_weighted_condense(timestep=-1, num_thresholds=N, exp_threshold_distribution=0.5)
condensed_graphs, nx_graphs = G.node_weighted_condense(timestep=-1, num_thresholds=N, exp_threshold_distribution=None)

for condensed_graph in condensed_graphs:
    plotter.plot_nx_network(condensed_graph, node_size_scaling=500, show=True)
    print(nx.to_numpy_array(condensed_graph))

# plotter.general_3d_data_plot(data=np.random.rand(100, 3), xlabel="Treeness", ylabel="Feedforwardness", zlabel="Orderability", fig_title='Hierarchy Coordinates (Exponential Thresholds)', show=True)
#

# data = np.random.rand(10, 3)
# plotter.general_3d_data_plot(data, plot_projections=True, projections=False, show=True, title='/home/maqz/Desktop/test')


# plotter.twoD_grid_search_meta_plots(path_to_data_dir=Path(str(Path.home()), 'data/node_num_50/'), output_dir=Path(str(Path.home()), 'data/'),
#                                     selectivity_range=np.arange(0, 1.05, 0.05), efficiency_coords=True, edge_conservation_range=np.arange(0, 1.05, 0.05))
