import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.append('../')
import graph
import plotter
import utility_funcs


num_runs = 1000
num_nodes = 20
edge_conservation = 0.5
edge_selectivity = 0.85
title_intro = str('constant_seeding_default_mods')
video_directory: str = '/home/maqz/Desktop/data/videos/'

# G = graph.Graph(num_nodes=num_nodes, edge_conservation_coefficient=edge_conservation, selectivity=edge_selectivity,
#                 reinforcement_info_score_coupling=True, eff_dist_is_towards_source=False, nodes_adapt_outgoing_edges=False, incoming_edges_conserved=True)
G = graph.Graph(num_nodes=num_nodes, edge_conservation_coefficient=edge_conservation, selectivity=edge_selectivity,
                reinforcement_info_score_coupling=True, eff_dist_is_towards_source=False, nodes_adapt_outgoing_edges=False,
                incoming_edges_conserved=True, undirected=False)
# G.uniform_random_edge_init()
# G.barabasi_albert_edge_init(num_edges_per_new_node=int(num_nodes / 10))
G.sparse_random_edge_init(3, connected=True, even_distribution=True, undirectify=True)
# G.scale_free_edge_init(degree_exponent=0.4, min_k=1, equal_edge_weights=True, connected=True, even_distribution=False, undirectify=False)

print(f'G.A[-1]: \n {np.round(G.A[-1], 3)}')
utility_funcs.print_row_column_sums(G.A[-1])
utility_funcs.sum_matrix_signs(G.A[-1])
G.simulate(num_runs, eff_dist_delta_param=12, constant_source_node=False, num_shifts_of_source_node=False, seeding_power_law_exponent=8,
           seeding_sigma_coeff=False, beta=None, equilibrium_distance=1000, multiple_path=False, verbose=True)
print(f'np.round(G.A[-1], 2): \n {np.round(G.A[-1], 2)}')
utility_funcs.print_row_column_sums(G.A[-1])
utility_funcs.sum_matrix_signs(G.A[-1])
plotter.plot_network(G)
plotter.parallellized_animate_network_evolution(graph=G, parent_directory=video_directory, file_title=f'{title_intro}_{edge_conservation}_selectivity_{edge_selectivity}', verbose=True, source_weighting=True, num_runs_per_fig=2, gif_duration_in_sec=60, changing_layout=False, node_size_scaling=2)
plotter.plot_single_network(G, timestep=0)
plotter.plot_single_network(G, timestep=-1)
plotter.plot_source_distribution(G)
# plotter.plot_node_edges(G, node=3)
plotter.plot_edge_histogram(G)
