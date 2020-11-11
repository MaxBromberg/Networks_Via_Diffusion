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

num_nodes = 60
edge_conservation_val = 0.95
selectivity_val = 0.95
runs = 10

# G = graph.Graph(num_nodes=num_nodes, edge_conservation_coefficient=edge_conservation_val, selectivity=selectivity_val)
# G.edge_initialization_conditional(None)
# G.simulate(num_runs=runs, constant_source_node=3)
# print(f'G.A[0]: \n {np.round(G.A[0], 3)}')
#
# normalize = True
# print(G.E_diff(normalize=normalize), G.E_routing(timestep=-2, normalize=normalize))

import efficiency_coordinates as ef
import numpy as np

A = np.random.rand(10, 10)  # Declaring Adjacency Matrix
E_diff, E_rout = ef.network_efficiencies(A, normalize=True)

# Equivalent to:
E_diff, E_rout = ef.E_diff(A), ef.E_rout(A)
