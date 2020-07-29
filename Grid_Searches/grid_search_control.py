import os

"""
Central program to order sequential grid-searches based on total parameter dictionaries
"""


def run_grid_search(param_dic):
    os.system('python grid_search.py {data_directory} {run_index} {num_nodes} {edge_conservation_range} {selectivity_range} {reinforcement_info_score_coupling} {positive_eff_dist_and_reinforcement_correlation} {eff_dist_is_towards_source} {nodes_adapt_outgoing_edges} {incoming_edges_conserved} {edge_init} {ensemble_size} {num_runs} {delta} {equilibrium_distance} {constant_source_node} {num_shifts_of_source_node} {seeding_sigma_coeff} {seeding_power_law_exponent} {beta} {multiple_path} {update_interval} {source_reward}'.format(**param_dic))
    print(f'Simulation with the following parameters complete:')
    print("\n".join("{}\t{}".format(k, v) for k, v in param_dic.items()))  # Thanks sudo from https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python


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
}
edge_init = {
    'edge_init': 'None',  # None is uniform rnd, int is num_edges/node in sparse init, float is degree exp in scale free init.
}
ensemble_params = {
    'ensemble_size': 0,  # num sims to average over. 0 if just one sim is desired (e.g. for graph pictures)
    'num_runs': 500,  # num runs, could be cut off if reaches equilibrium condition first
    'delta': 1,  # Delta parameter in (RW/MP)ED, recommended >= 1
    'equilibrium_distance': 100,
    'constant_source_node': 4,  # If no seeding mechanism is set, defaults to rnd.
    'num_shifts_of_source_node': 0,  # use 0 as False
    'seeding_sigma_coeff': 0,
    'seeding_power_law_exponent': 0,
    'beta': 0,
    'multiple_path': 0,
    'update_interval': 1,
    'source_reward': 2.6
}
default_parameters = {**parameter_dictionary, **search_wide_dic, **edge_init, **ensemble_params}  # Order matters!

custom_params_1 = {**default_parameters, 'data_directory': "/home/maqz/Desktop/data/test"}
run_grid_search(custom_params_1)

