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
edge_conservation_val = 0.05
selectivity_val = 0.95
runs = 25

G = graph.Graph(num_nodes=num_nodes, edge_conservation_coefficient=edge_conservation_val, selectivity=selectivity_val)
# G.edge_initialization_conditional(10)
# G.simulate(num_runs=runs, constant_source_node=3)
# G.null_simulate(num_runs=runs)
G.simulate_ensemble(num_simulations=10, num_runs_per_sim=runs, null_simulate=False, edge_init=4, constant_source_node=2, null_normalize=True)
# G.null_normed_simulate(num_runs=runs, constant_source_node=3)
# print(f'G.A[0]: \n {np.round(G.A[0], 3)}')
# print(f'G.A[-1]: \n {np.round(G.A[-1], 3)}')
plotter.plot_adjacency_matrix_as_heatmap(G, show=True)
print(G.eff_dist_diff(all_to_all_eff_dist=True))

# plotter.parallelized_animate_network_evolution(G, parent_directory=Path('/home/maqz/data'))

# normalize = True
# print(G.E_diff(normalize=normalize), G.E_routing(timestep=-2, normalize=normalize))

