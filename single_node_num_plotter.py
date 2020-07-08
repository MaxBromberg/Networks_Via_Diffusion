import plotter
from pathlib import Path
import numpy
import sys

"""
Parallelizes plot creation across all node_num_## datasets.
Arguments via terminal to be given in the following order:
data_directory, edge_conservation_range, selectivity_range, output_directory
[Path,          list,           list,             Path]

To be run with python3.6 or later. (includes fstrings)
Required packages: numpy, random, matplotlib, networkx, [pickle, os, pathLib]
In the same directory: graph.py, plotter.py, effective_distance.py, utility_funcs.py
"""
input_path = str(sys.argv[1])
coupling_range = numpy.arange(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
adaptation_range = numpy.arange(float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]))
try:
    output_path = str(sys.argv[-1])
except IndexError:
    print('No output directory (last argument) given. Creating plots in data (input) directory.')

if __name__ == '__main__':
    node_nums = int(str(str(input_path).split('/')[-1]).split('_')[-1])
    plotter.twoD_grid_search_plots(input_path, edge_conservation_range=coupling_range, selectivity_range=adaptation_range, num_nodes=node_nums, output_dir=output_path,
                                   node_plots=False, ave_nbr=False, cluster_coeff=False, shortest_path=False, degree_dist=True)  # edit directly here for desired more computationally cumbersome observables

