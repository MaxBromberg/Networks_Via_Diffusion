from pathlib import Path
import numpy as np
import sys
sys.path.append('../')
import plotter


num_nodes = 50
data_directory = "/home/maqz/Desktop/data/random_seeding"
output_directory = Path(data_directory, f"node_num_{num_nodes}")
edge_conservation_range = np.arange(0, 1.05, 0.05)
selectivity_range = np.arange(0, 1.05, 0.05)
plotter.grid_search_plots(output_directory, edge_conservation_range=edge_conservation_range, selectivity_range=selectivity_range,
                          num_nodes=num_nodes, network_graphs=False, node_plots=False, ave_nbr=False, cluster_coeff=False, shortest_path=False, degree_dist=True, output_dir=data_directory)
