import plotter
from pathlib import Path
import numpy
import sys

"""
Parallelizes plot creation across all node_num_## (or source_reward_##) datasets.
Arguments via terminal to be given in the following order:
data_directory, edge_conservation_range, selectivity_range, output_directory
[Path,          list,           list,             Path]

To be run with python3.6 or later. (includes fstrings)
Required packages: numpy, random, matplotlib, networkx, [pickle, os, pathLib]
In the same directory: graph.py, plotter.py, effective_distance.py, utility_funcs.py
"""
node_num = 50
input_path = str(sys.argv[1])
edge_conservation_range = numpy.arange(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
selectivity_range = numpy.arange(float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]))
try:
    output_path = str(sys.argv[-1])
except IndexError:
    output_path = input_path
    print('No output directory (last argument) given. Creating plots in data (input) directory.')

if __name__ == '__main__':
    # node_num = int(str(str(input_path).split('/')[-1]).split('_')[-1])
    # plotter.twoD_grid_search_plots(input_path, edge_conservation_range=edge_conservation_range, selectivity_range=selectivity_range, num_nodes=node_nums, output_dir=output_path,
    #                                node_plots=False, ave_nbr=False, cluster_coeff=False, shortest_path=False, degree_dist=True)  # edit directly here for desired more computationally cumbersome observables
    source_reward = int(str(str(input_path).split('/')[-1]).split('_')[-1])  # assumes that the source reward is the last value after a _ in the directory name
    plotter.twoD_grid_search_plots(input_path, edge_conservation_range=edge_conservation_range, selectivity_range=selectivity_range,
                                   num_nodes=node_num, output_dir=Path(output_path, f'source_reward_{source_reward}_plots'),
                                   network_graphs=True, node_plots=False, ave_nbr=False, cluster_coeff=False, shortest_path=False, degree_dist=True)

