import graph
import numpy as np
import plotter
import pickle
from pathlib import Path

num_runs = 500
num_nodes = 15
edge_conservation = 0.1
edge_selectivity = 0.5

G = graph.Graph(num_nodes=num_nodes, edge_conservation_coefficient=edge_conservation,
                selectivity=edge_selectivity, reinforcement_info_score_coupling=False)
G.uniform_random_edge_init()
G.run(1000, exp_decay_param=12, constant_source_node=False, num_shifts_of_source_node=False, seeding_sigma_coeff=False, seeding_power_law_exponent=5, beta=None, equilibrium_distance=200, multiple_path=False, verbose=True)
# G.run(num_runs, exp_decay_param=12, constant_source_node=2, num_shifts_of_source_node=False, seeding_sigma_coeff=False, seeding_power_law_exponent=False, equilibrium_distance=200, multiple_path=False, verbose=True)
video_directory: str = '/home/maqz/Desktop/data/videos/'
plotter.parallellized_animate_network_evolution(graph=G, parent_directory=video_directory, file_title=f'pwr_law_exp_w_switch_at_100_edge_conservation_{edge_conservation}_selectivity_{edge_selectivity}', verbose=True, source_weighting=True, num_runs_per_fig=2, gif_duration_in_sec=60, changing_layout=False, node_size_scaling=2)
plotter.plot_single_network(G, -1)
# plotter.plot_source_distribution(G)
# plotter.plot_degree_histogram(G)
# plotter.plot_degree_distribution_var_over_time(G)
