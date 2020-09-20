import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
sys.path.append('../')
import effective_distance as ed
from pathlib import Path
import utility_funcs
import plotter
import graph
import hierarchy_coordinates as hc


num_nodes = 10
edge_conservation_val = 0.3
selectivity_val = 0.95
reinforcement_info_score_coupling = True
positive_eff_dist_and_reinforcement_correlation = False
eff_dist_is_towards_source = False
nodes_adapt_outgoing_edges = False
incoming_edges_conserved = True

init_args = [num_nodes, edge_conservation_val, selectivity_val, reinforcement_info_score_coupling, positive_eff_dist_and_reinforcement_correlation, eff_dist_is_towards_source, nodes_adapt_outgoing_edges, incoming_edges_conserved]
G = graph.Graph(*init_args)
# G.uniform_random_edge_init()
# G.sparse_random_edge_init(3, connected=True)
G.nx_scale_free_edge_init(degree_exponent=1.2)
# print(f'G.A[0]: \n {np.round(G.A[0], 3)}')

G.simulate(100, constant_source_node=3)
nx_graph = G.convert_to_nx_graph(timestep=-1)
condensed, not_condensed = G.node_weighted_condense(timestep=-1, exp_threshold_distribution=False)
threshold_index = 1
# Plot Original graphs
plotter.plot_nx_network(not_condensed[threshold_index])
# plotter.plot_nx_network(not_branched[threshold_index])

# Individual Graph Values of Hierarchy Coordinates
start_time = time.time()
all_condensed_branches = hc.recursive_leaf_removal(condensed[threshold_index], from_top=False)
print(f'Orderability: {hc.orderability(nx_graph, condensed[threshold_index])}')
print(f'Feedforwardness: {hc.feedforwardness(condensed[threshold_index])}')
print(f'Treeness: {hc.treeness(condensed[threshold_index])}')
print(f'Time Elapsed: {utility_funcs.time_lapsed_h_m_s(time.time()-start_time)}')

# Plot Pruned Tree, associated weights:
# for index in range(len(all_condensed_branches)):
#     weights = nx.get_node_attributes(all_condensed_branches[index], 'weight')
#     print(weights)
#     plotter.plot_nx_network(all_condensed_branches[index])
