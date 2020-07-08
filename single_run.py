import plotter
import graph

from pathlib import Path
import sys

"""
Arguments via terminal to be given in the following order:
output_directory, num_nodes, run_index, edge_conservation_val, selectivity_val

To be run with python3.6 or later. (includes fstrings)
Required packages: numpy, random, matplotlib, networkx, [pickle, os, pathLib]
In the same directory: graph.py, plotter.py, effective_distance.py, utility_funcs.py
"""

output_path = str(sys.argv[1])
num_nodes, run_index = [int(arg) for arg in sys.argv[2:4]]  # eliminates name of file as initial input string
edge_conservation_val, selectivity_val = [float(arg) for arg in sys.argv[4:6]]
assert isinstance(run_index, int), "Run index records order of runs, and ought be an integer"
assert isinstance(num_nodes, int), "Number of nodes ought be an integer"

num_runs = 1000
parameter = 1
equilibrium_distance = 200
constant_source_node = 3
num_shifts_of_source_node = False
seeding_sigma_coeff = False
seeding_power_law_exponent = False
beta = None
multiple_path = False
assert bool(constant_source_node) + bool(num_shifts_of_source_node) + bool(seeding_sigma_coeff) + bool(seeding_power_law_exponent) + bool(beta) < 2, 'Choose one seeding method, as all are mutually incompatible. If none, defaults to random seeding'

if __name__ == '__main__':
    G = graph.Graph(num_nodes=num_nodes, edge_conservation_coefficient=edge_conservation_val, selectivity=selectivity_val,
                    reinforcement_info_score_coupling=True, positive_eff_dist_and_reinforcement_correlation=False)
    G.uniform_random_edge_init()
    G.run(num_runs, exp_decay_param=parameter, constant_source_node=constant_source_node, num_shifts_of_source_node=num_shifts_of_source_node,
          equilibrium_distance=equilibrium_distance, seeding_sigma_coeff=seeding_sigma_coeff, seeding_power_law_exponent=seeding_power_law_exponent, beta=beta, multiple_path=multiple_path)
    plotter.save_object(G, Path(output_path, f'{run_index:04}_graph_obj.pkl'))
    print(f'Run {run_index}, [edge conservation: {edge_conservation_val}, selectivity: {selectivity_val}] complete.')
# print(f'Raw graph data recorded at: {output_path}')
