import plotter
import graph

from pathlib import Path
import sys

"""
Arguments via terminal to be given in the following order:
output_directory, num_nodes, run_index, coupling_val, adaptation_val

To be run with python3.6 or later. (includes fstrings)
Required packages: numpy, random, matplotlib, networkx, [pickle, os, pathLib]
In the same directory: graph.py, plotter.py, effective_distance.py, utility_funcs.py
"""

output_path = str(sys.argv[1])
num_nodes, run_index = [int(arg) for arg in sys.argv[2:4]]  # eliminates name of file as initial input string
coupling_val, adaptation_val = [float(arg) for arg in sys.argv[4:]]
assert isinstance(run_index, int), "Run index records order of runs, and ought be an integer"
assert isinstance(num_nodes, int), "Number of nodes ought be an integer"

num_runs = 1000

if __name__ == '__main__':
    G = graph.EffDisGraph(num_nodes=num_nodes, eff_dist_and_edge_response=coupling_val, fraction_info_score_redistributed=adaptation_val)
    G.uniform_random_edge_init()
    G.run(num_runs, exp_decay_param=12, constant_source_node=1, equilibrium_distance=200, multiple_path=False)
    plotter.save_object(G, Path(output_path, f'{run_index:04}_graph_obj.pkl'))
    print(f'Run {run_index}, [coupling_val: {coupling_val}, edge_adaptation: {adaptation_val}] complete.')
# print(f'Raw graph data recorded at: {output_path}')

