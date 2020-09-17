import os
from pathlib import Path

"""
Central program to order sequential grid-searches based on total parameter dictionaries
"""


def run_grid_search(param_dic):
    os.system('python grid_search.py {data_directory} {run_index} {num_nodes} {edge_conservation_range} {selectivity_range} {reinforcement_info_score_coupling} {positive_eff_dist_and_reinforcement_correlation} {eff_dist_is_towards_source} {nodes_adapt_outgoing_edges} {incoming_edges_conserved} {undirected} {edge_init} {ensemble_size} {num_runs} {delta} {equilibrium_distance} {constant_source_node} {num_shifts_of_source_node} {seeding_sigma_coeff} {seeding_power_law_exponent} {beta} {multiple_path} {update_interval} {source_reward} {undirectify_init}'.format(**param_dic))
    print(f'Simulation with the following parameters complete:')
    nice_dictionary_print(param_dic)


def nice_dictionary_print(dic):
    # Thanks sudo from https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python
    print("\n".join("{}\t{}".format(k, v) for k, v in dic.items()))


# Throughout, use False = 0 and True = 1 for binaries
parameter_dictionary = {
    'data_directory': "/home/maqz/Desktop/data/",
    'run_index': 1,
    'num_nodes': 50,
    'edge_conservation_range': '0_1.05_0.05',  # work with me here. (args to np.arange separated by _)
    'selectivity_range': '0_1.05_0.05'
}
search_wide_dic = {
    'reinforcement_info_score_coupling': 0,  # Default = False (0)
    'positive_eff_dist_and_reinforcement_correlation': 0,  # Default = False (0)
    'eff_dist_is_towards_source': 0,   # Default = False (0)
    'nodes_adapt_outgoing_edges': 0,  # Default = False (0)
    'incoming_edges_conserved': 1,  # Default = True (1)
    'undirected': 0,  # if true, averages reciprocal connections after every run, i.e. e_ji, e_ji --> (e_ij + e_ji)/2)
}
edge_init = {
    'edge_init': 'None',  # None is uniform rnd, int is num_edges/node in sparse init, float is degree exp in scale free init.
}
ensemble_params = {
    'ensemble_size': 0,  # num sims to average over. 0 if just one sim is desired (e.g. for graph pictures)
    'num_runs': 600,  # num runs, could be cut off if reaches equilibrium condition first
    'delta': 10,  # Delta parameter in (RW/MP)ED, recommended >= 1
    'equilibrium_distance': 200,
    'constant_source_node': 0,  # If no seeding mechanism is set, defaults to rnd. Activate below seeding by setting values != 0
    'num_shifts_of_source_node': 0,  # use 0 as False
    'seeding_sigma_coeff': 0,  # \in [0, \infty), the coefficient before the standard sigma to determine width of normal distribution
    'seeding_power_law_exponent': 0,  # \in [0, \infty), higher power => higher concentration about single (highest index) node
    'beta': 0,  # Extent seeding is correlated with diversity of connection
    'multiple_path': 0,  # RWED -> MPED, Likely has faults in present implementation.  (Also much slower)
    'update_interval': 1,  # num runs per edge update
    'source_reward': 2.6,  # how much more the source is rewarded than the next best. Only affects source
    'undirectify_init': 0  # Start edges with reciprocated (simple) edges? (Boolean)
}
eff_dist_to_source = {'eff_dist_is_towards_source': 1}
node_adapt_outgoing_edges = {'nodes_adapt_outgoing_edges': 1}
outgoing_edges_conserved = {'incoming_edges_conserved': 0}

# higgly_default_data_directory = '/home/maqz/Desktop/data/Mechanic_Mods/'
undirected_sparse_data_directory = '/home/maqz/Desktop/data/Mechanic_Mods/undirected_init_sparse_1.2'
default_sparse_init = {**parameter_dictionary, **search_wide_dic, 'edge_init': 1.2, **ensemble_params, 'num_nodes': 60, 'constant_source_node': 4}
undirected_sparse_init = {**default_sparse_init, 'undirectify_init': 1}

undirected_sparse_run_data_directory = '/home/maqz/Desktop/data/Mechanic_Mods/undirected_run_sparse_1.2'
undirected_sparse_run = {**undirected_sparse_init, 'undirected': 1}
undirected_run_sparse_base_case = {**undirected_sparse_run, 'data_directory': Path(undirected_sparse_run_data_directory, "base_case")}
undirected_run_sparse_reversed_source_edge_conservation = {**undirected_sparse_run, **eff_dist_to_source, **node_adapt_outgoing_edges, **outgoing_edges_conserved, 'data_directory': Path(undirected_sparse_run_data_directory, "switched_ED_Adaptation_Conservation")}
undirected_run_sparse_reversed_edge_conservation = {**undirected_sparse_run, **node_adapt_outgoing_edges, **outgoing_edges_conserved, 'data_directory': Path(undirected_sparse_run_data_directory, "switched_Adaptation_Conservation")}
undirected_run_sparse_reversed_source_conservation = {**undirected_sparse_run, **eff_dist_to_source, **outgoing_edges_conserved, 'data_directory': Path(undirected_sparse_run_data_directory, "switched_ED_Conservation")}
undirected_run_sparse_reversed_source_edge = {**undirected_sparse_run, **eff_dist_to_source, **node_adapt_outgoing_edges, 'data_directory': Path(undirected_sparse_run_data_directory, "switched_ED_Adaptation")}
undirected_run_sparse_reversed_source = {**undirected_sparse_run, **eff_dist_to_source, 'data_directory': Path(undirected_sparse_run_data_directory, "switched_ED")}
undirected_run_sparse_reversed_edge = {**undirected_sparse_run, **node_adapt_outgoing_edges, 'data_directory': Path(undirected_sparse_run_data_directory, "switched_Adaptation")}
undirected_run_sparse_reversed_conservation = {**undirected_sparse_run, **outgoing_edges_conserved, 'data_directory': Path(undirected_sparse_run_data_directory, "switched_Conservation")}

run_grid_search(undirected_run_sparse_base_case)
run_grid_search(undirected_run_sparse_reversed_source_edge_conservation)
run_grid_search(undirected_run_sparse_reversed_edge_conservation)
run_grid_search(undirected_run_sparse_reversed_source_conservation)
run_grid_search(undirected_run_sparse_reversed_source_edge)
run_grid_search(undirected_run_sparse_reversed_source)
run_grid_search(undirected_run_sparse_reversed_edge)
run_grid_search(undirected_run_sparse_reversed_conservation)
