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
reinforcement_info_score_coupling = True
positive_eff_dist_and_reinforcement_correlation = False
eff_dist_is_towards_source = False
nodes_adapt_outgoing_edges = False
incoming_edges_conserved = True

init_args = [num_nodes, edge_conservation_val, selectivity_val, reinforcement_info_score_coupling, positive_eff_dist_and_reinforcement_correlation, eff_dist_is_towards_source, nodes_adapt_outgoing_edges, incoming_edges_conserved]
G = graph.Graph(*init_args)
# G.uniform_random_edge_init()
G.sparse_random_edge_init(3, connected=True)
# print(f'G.A[0]: \n {np.round(G.A[0], 3)}')
# G.simulate(100, constant_source_node=3)
# plotter.plot_hierarchy_evolution(G, 3)


# data = np.random.rand(10, 3)
# plotter.general_3d_data_plot(data, plot_projections=True, projections=False, show=True, title='/home/maqz/Desktop/test')
def f(a):
    if a:
        print('True!')


p = mp.Pool(mp.cpu_count())
results = p.map(f, [bool(1), bool(0)])
p.close()
p.join()


