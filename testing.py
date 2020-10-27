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

num_nodes = 10
edge_conservation_val = 0.3
selectivity_val = 0.3
reinforcement_info_score_coupling = False
positive_eff_dist_and_reinforcement_correlation = False
eff_dist_is_towards_source = False
nodes_adapt_outgoing_edges = False
incoming_edges_conserved = True

init_args = [num_nodes, edge_conservation_val, selectivity_val, reinforcement_info_score_coupling, positive_eff_dist_and_reinforcement_correlation, eff_dist_is_towards_source, nodes_adapt_outgoing_edges, incoming_edges_conserved]
G = graph.Graph(*init_args)
# G.uniform_random_edge_init()
# G.sparse_random_edge_init(3, connected=True)
G.edge_initialization_conditional(1.2)
print(f'G.A[0]: \n {np.round(G.A[0], 3)}')
G.simulate(100, constant_source_node=3)
print(f'G.A[0]: \n {np.round(G.A[-1], 3)}')
plotter.plot_network(G, show=True)
# plotter.general_3d_data_plot(data=np.random.rand(100, 3), xlabel="Treeness", ylabel="Feedforwardness", zlabel="Orderability", fig_title='Hierarchy Coordinates (Exponential Thresholds)', show=True)
#

# data = np.random.rand(10, 3)
# plotter.general_3d_data_plot(data, plot_projections=True, projections=False, show=True, title='/home/maqz/Desktop/test')

# plotter.plot_nx_network(nx.to_directed(nx.from_numpy_matrix(np.array(G.A[-1]), create_using=nx.DiGraph)), show=True)
