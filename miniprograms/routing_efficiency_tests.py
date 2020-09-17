import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import utility_funcs
import effective_distance as ed
from pathlib import Path
import plotter
import graph
import hierarchy_coordinates as hc

num_nodes = 10
edge_conservation_val = 0.3
selectivity_val = 0.3
reinforcement_info_score_coupling = True
positive_eff_dist_and_reinforcement_correlation = False
eff_dist_is_towards_source = False
nodes_adapt_outgoing_edges = False
incoming_edges_conserved = True

init_args = [num_nodes, edge_conservation_val, selectivity_val, reinforcement_info_score_coupling, positive_eff_dist_and_reinforcement_correlation, eff_dist_is_towards_source, nodes_adapt_outgoing_edges, incoming_edges_conserved]
G = graph.Graph(*init_args)
# G.uniform_random_edge_init()    # node_weights = [node_weight[1] for node_weight in nx_graph.nodes(data="weight")]
G.sparse_random_edge_init(3, connected=True)
# print(f'G.A[0]: \n {np.round(G.A[0], 3)}')

G.simulate(10, constant_source_node=3)
# shortest_paths = G.shortest_path()
# print(f'Shortest Paths: \n {np.round(shortest_paths, 3)}')
nx_graph = G.convert_to_nx_graph(timestep=-1)


# ave_shortest_paths = np.sum(shortest_paths)/(num_nodes*(num_nodes - 1))
# print(f'Average Shortest Paths: {ave_shortest_paths}')
# print(f'G.routing_efficiency: {G.routing_efficiency(nx_graph)}')
print(f'G.E_routing(timestep=-1): {G.E_routing(timestep=-1)}')
print(f'G.E_diff(timestep=-1): {G.E_diff(timestep=-1)} \n')
# plotter.plot_single_network(G, timestep=G.nodes.shape[0])
# plotter.plot_edge_histogram(G)
