import os
import sys
import time
import shutil
import numpy as np
from pathlib import Path, PurePath
import multiprocessing as mp
import pprint
import graph
import plotter
import utility_funcs as uf

"""
Central program to order sequential grid-searches based on comprehensive parameter dictionaries
"""


def directionality_dic(data_directory):
    eff_dist_to_source = {'eff_dist_is_towards_source': 1, 'positive_eff_dist_and_reinforcement_correlation': 1}  # The effective distance flip should be accompanied by its info-score equivalent
    node_adapt_outgoing_edges = {'nodes_adapt_outgoing_edges': 1}
    outgoing_edges_conserved = {'incoming_edges_conserved': 0}
    directionality = {
        'base_case': {'data_directory': Path(data_directory, "base_case/")},
        'reversed_source_edge_conservation': {**eff_dist_to_source, **node_adapt_outgoing_edges, **outgoing_edges_conserved, 'data_directory': Path(data_directory, "switched_ED_Adaptation_Conservation/")},
        'reversed_edge_conservation': {**node_adapt_outgoing_edges, **outgoing_edges_conserved, 'data_directory': Path(data_directory, "switched_Adaptation_Conservation/")},
        'reversed_source_conservation': {**eff_dist_to_source, **outgoing_edges_conserved, 'data_directory': Path(data_directory, "switched_ED_Conservation/")},
        'reversed_source_edge': {**eff_dist_to_source, **node_adapt_outgoing_edges, 'data_directory': Path(data_directory, "switched_ED_Adaptation/")},
        'reversed_source': {**eff_dist_to_source, 'data_directory': Path(data_directory, "switched_ED/")},
        'reversed_edge': {**node_adapt_outgoing_edges, 'data_directory': Path(data_directory, "switched_Adaptation/")},
        'reversed_conservation': {**outgoing_edges_conserved, 'data_directory': Path(data_directory, "switched_Conservation/")}
    }
    return directionality


def initializations_dic(data_directory):
    inits = {
        'scale_free': {'edge_init': float(1.2), 'data_directory': Path(data_directory, 'scale_free_deg_exp_1.2_edges/')},
        'uniform_random': {'edge_init': None, 'data_directory': Path(data_directory, 'uniform_random_edges/')},
        'sparse': {'edge_init': int(3), 'data_directory': Path(data_directory, 'sparse_3_edges/')}
    }
    return inits


def seeding_dic(data_directory):
    seed_dic = {
        'constant': {'constant_source_node': 1, 'data_directory': Path(data_directory, 'constant_seeding/')},
        'random': {'constant_source_node': 0, 'num_shifts_of_source_node': 0, 'seeding_sigma_coeff': 0, 'seeding_power_law_exponent': 0, 'data_directory': Path(data_directory, 'random_seeding/')},
        '10_source_shifts': {'num_shifts_of_source_node': 10, 'data_directory': Path(data_directory, '10_shifts_seeding/')},
        'pwr_law_5': {'seeding_power_law_exponent': 5, 'data_directory': Path(data_directory, 'pwr_law_5_seeding/')},
    }
    return seed_dic


def simulate(param_list):
    output_path = str(param_list[0])
    run_index = int(param_list[1])
    num_nodes = int(param_list[2])
    edge_conservation_val = float(param_list[3])
    selectivity_val = float(param_list[4])
    r_i_s_c = bool(int(param_list[5]))
    p_ed_and_r_c = bool(int(param_list[6]))
    ed_to_s = bool(int(param_list[7]))
    n_a_o_e = bool(int(param_list[8]))
    i_e_c = bool(int(param_list[9]))
    undirected_run = bool(int(param_list[10]))
    if str(param_list[11]).__contains__('.'):
        edge_init = float(param_list[11])
    elif str(param_list[11]) == "None" or str(param_list[11]) == "False":
        edge_init = None
    else:
        edge_init = int(param_list[11])  # I don't see how to possibly pass a np.array through shell script, so that option's out
    ensemble_size = int(param_list[12])
    num_runs = int(param_list[13])
    delta = float(param_list[14])
    equilibrium_distance = int(param_list[15])
    constant_source_node = int(param_list[16])
    num_shifts_of_source_node = int(param_list[17])
    seeding_sigma_coeff = float(param_list[18])
    seeding_power_law_exponent = float(param_list[19])
    beta = float(param_list[20])
    multiple_path = bool(int(param_list[21]))
    update_interval = int(param_list[22])
    source_reward = float(param_list[23])
    undirectify_init = bool(param_list[24])

    graph_start_time = time.time()
    G = graph.Graph(num_nodes=num_nodes, edge_conservation_coefficient=edge_conservation_val, selectivity=selectivity_val,
                    reinforcement_info_score_coupling=r_i_s_c, positive_eff_dist_and_reinforcement_correlation=p_ed_and_r_c,
                    eff_dist_is_towards_source=ed_to_s, nodes_adapt_outgoing_edges=n_a_o_e, incoming_edges_conserved=i_e_c,
                    undirected=undirected_run)

    if not ensemble_size:
        G.edge_initialization_conditional(edge_init=edge_init, undirectify=undirectify_init)
        G.simulate(num_runs=num_runs, eff_dist_delta_param=delta, constant_source_node=constant_source_node,
                   num_shifts_of_source_node=num_shifts_of_source_node, seeding_sigma_coeff=seeding_sigma_coeff,
                   seeding_power_law_exponent=seeding_power_law_exponent, beta=beta, multiple_path=multiple_path,
                   equilibrium_distance=equilibrium_distance, update_interval=update_interval, source_reward=source_reward)
    else:
        G.simulate_ensemble(num_simulations=ensemble_size, num_runs_per_sim=num_runs, eff_dist_delta_param=delta, edge_init=edge_init,
                            constant_source_node=constant_source_node, num_shifts_of_source_node=num_shifts_of_source_node,
                            equilibrium_distance=equilibrium_distance, seeding_sigma_coeff=seeding_sigma_coeff,
                            seeding_power_law_exponent=seeding_power_law_exponent, beta=beta, multiple_path=multiple_path,
                            update_interval=update_interval, source_reward=source_reward, undirectify=undirectify_init, verbose=False)
    plotter.save_object(G, Path(output_path, f'{run_index:04}_graph_obj.pkl'))
    print(f'Run {run_index}, [edge conservation: {edge_conservation_val}, selectivity: {selectivity_val}] complete. {G.get_num_errors()} errors, ({uf.time_lapsed_h_m_s(time.time()-graph_start_time)})')


def run_over_all_directionality_combos(mods, data_directory, via_pool=True):
    """
    Runs grid searches for every directionality combination
    :param mods: dictionary of all mods unrelated to directionality (initialization, directedness, etc...)
    :param data_directory: where all directionality grid-searches will be output
    """
    modded_directionality_dic = directionality_dic(data_directory=data_directory)
    for v in modded_directionality_dic.values():
        run_grid_search({**mods, **v}, via_pool=via_pool)


def run_over_all_seeding_combos(mods, data_directory, via_pool=True):
    """
    Runs grid searches for every directionality combination
    :param mods: dictionary of all mods unrelated to seeding (initialization, directedness, etc...)
    :param data_directory: where all directionality grid-searches will be output
    """
    modded_seeding_dic = seeding_dic(data_directory=data_directory)
    for v in modded_seeding_dic.values():
        run_grid_search({**mods, **v}, via_pool=via_pool)


def run_over_all_initialization_combos(mods, data_directory, via_pool=True):
    """
    Runs grid searches for every directionality combination
    :param mods: dictionary of all mods unrelated to initialization
    :param data_directory: where all directionality grid-searches will be output
    """
    modded_initialization_dic = initializations_dic(data_directory=data_directory)
    for v in modded_initialization_dic.values():
        run_grid_search({**mods, **v}, via_pool=via_pool)


def grid_search(param_dic, num_cores_used=mp.cpu_count(), remove_data_post_plotting=True):
    unvarying_dic_values = list(param_dic.values())[5:]
    data_directory = str(param_dic['data_directory'])
    run_index, num_nodes = [int(arg) for arg in [param_dic['run_index'], param_dic['num_nodes']]]  # eliminates name of file as initial input string
    raw_data_directory = Path(data_directory, f"node_num_{num_nodes}")

    edge_conservation_range = np.arange(*[float(arg) for arg in param_dic['edge_conservation_range'].split('_')])
    selectivity_range = np.arange(*[float(arg) for arg in param_dic['selectivity_range'].split('_')])

    try:
        os.makedirs(data_directory)
    except OSError:
        print(f'{data_directory} already exists, adding or overwriting contents')
        pass

    try:
        os.mkdir(raw_data_directory)
    except OSError:
        print(f'{raw_data_directory} already exists, adding or overwriting contents')
        pass

    grid_search_start_time = time.time()
    pool = mp.Pool(num_cores_used)
    args = []
    for coupling_val in edge_conservation_range:
        for selectivity_val in selectivity_range:
            varying_param_dic = {
                'output_directory': raw_data_directory,
                'run_index': run_index,
                'num_nodes': num_nodes,
                'edge_conservation_val': np.round(coupling_val, 2),
                'selectivity_val': np.round(selectivity_val, 2),
            }
            run_index += 1
            args.append(list(varying_param_dic.values()) + unvarying_dic_values)
    pool.map(simulate, args)
    pool.close()
    pool.join()
    print(f"Time lapsed for {num_nodes} nodes, {edge_conservation_range.size * selectivity_range.size} parameter combinations: {uf.time_lapsed_h_m_s(time.time()-grid_search_start_time)}")
    network_graphs = bool(param_dic['network_graphs'])
    if param_dic['ensemble_size']:
        network_graphs = False
    plotter.twoD_grid_search_plots(raw_data_directory, edge_conservation_range=edge_conservation_range, selectivity_range=selectivity_range,
                                   num_nodes=num_nodes,
                                   network_graphs=network_graphs,
                                   node_plots=bool(param_dic['node_plots']),
                                   ave_nbr=bool(param_dic['ave_nbr']),
                                   cluster_coeff=bool(param_dic['cluster_coeff']),
                                   eff_dist=bool(param_dic['eff_dist']),
                                   global_eff_dist=bool(param_dic['global_eff_dist']),
                                   shortest_path=bool(param_dic['shortest_path']),
                                   degree_dist=bool(param_dic['degree_dist']),
                                   edge_dist=bool(param_dic['edge_dist']),
                                   meta_plots=bool(param_dic['meta_plots']),
                                   # output_dir=Path(data_directory, 'Plots'))
                                   output_dir=None)
    if remove_data_post_plotting:
        shutil.rmtree(raw_data_directory)


def run_grid_search(param_dic, via_pool=True):
    if via_pool:
        grid_search(param_dic)
    else:
        os.system('python3 grid_search.py {data_directory} {run_index} {num_nodes} {edge_conservation_range} {selectivity_range} {reinforcement_info_score_coupling} {positive_eff_dist_and_reinforcement_correlation} {eff_dist_is_towards_source} {nodes_adapt_outgoing_edges} {incoming_edges_conserved} {undirected} {edge_init} {ensemble_size} {num_runs} {delta} {equilibrium_distance} {constant_source_node} {num_shifts_of_source_node} {seeding_sigma_coeff} {seeding_power_law_exponent} {beta} {multiple_path} {update_interval} {source_reward} {undirectify_init} {network_graphs} {node_plots} {ave_nbr} {cluster_coeff} {eff_dist} {global_eff_dist} {shortest_path} {degree_dist} {edge_dist} {meta_plots}'.format(**param_dic))

    print(f'Simulation with the following parameters complete:')
    pprint.pprint(param_dic)


def list_of_dicts(base_dic, dic_1, dic_2=None, dic_3=None):
    base_dicts = []

    if dic_2 is not None:
        two_dicts_combined = []
        dic_2_directory_labels = []
        for v_2 in dic_2.values():
            dic_2_directory_labels.append(PurePath(v_2.pop('data_directory')).name)  # pop should delete the 'data_directory' item from v_2, so it doesn't overwrite the key
    if dic_3 is not None:
        three_dicts_combined = []
        dic_3_directory_labels = []
        for v_3 in dic_3.values():
            dic_3_directory_labels.append(PurePath(v_3.pop('data_directory')).name)  # pop should delete the 'data_directory' item from v_2, so it doesn't overwrite the key

    for v_1 in dic_1.values():
        base_dicts.append({**base_dic, **v_1})

    if dic_2 is not None:
        for value in base_dicts:
            i = 0
            for v_2 in dic_2.values():
                two_dicts_combined.append({**value, **v_2})
                two_dicts_combined[-1]['data_directory'] = Path(str(two_dicts_combined[-1]['data_directory']), dic_2_directory_labels[i])
                i += 1
    if dic_3 is not None:
        for value in two_dicts_combined:
            i = 0
            for v_3 in dic_3.values():
                three_dicts_combined.append({**value, **v_3})
                three_dicts_combined[-1]['data_directory'] = Path(str(three_dicts_combined[-1]['data_directory']), dic_3_directory_labels[i])
                i += 1

    if dic_3 is not None:
        return three_dicts_combined
    if dic_2 is not None:
        return two_dicts_combined
    return base_dicts


# Throughout, use False = 0 and True = 1 for binaries
directory = Path(str(Path.home()), 'data/')
parameter_dictionary = {
    'data_directory': directory,
    'run_index': 1,
    'num_nodes': 60,
    'edge_conservation_range': '0_1.05_0.05',  # work with me here. (args to np.arange separated by _)
    'selectivity_range': '0_1.05_0.05'
}
search_wide_dic = {
    'reinforcement_info_score_coupling': 1,  # Default = False (0)
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
plots = {
    'network_graphs': 1,  # graphs the networks
    'node_plots': 0,  # plots Evolution of node values over time
    'ave_nbr': 0,  # Plots average neighbor connections over time
    'cluster_coeff': 0,  # Plots evolution of cluster coefficient
    'eff_dist': 1,  # Plots evolution of average effective distance to source
    'global_eff_dist': 1,  # Plots evolution of average effective distance from every node to every other
    'shortest_path': 0,  # Plots the average shortest path over time. Very computationally expensive
    'degree_dist': 1,  # Yields the degree (total weight) distribution as a histogram
    'edge_dist': 1,  # Plots the edge distribution (individual edge counts) as a histogram
    'meta_plots': 1,  # Plots all the meta-plots, specifically: last_ave_nbr_deg, ed diffs, mean ed, ave_neighbor diffs,
    # global ed diffs, ave_nbr variance, log_deg_dist variance, hierarchy coordinates (with exponential and linear thresholds) and efficiency coordinates
}

default_dict = {**parameter_dictionary, **search_wide_dic, **edge_init, **ensemble_params, **plots}
# master_dict = list_of_dicts(default_dict, initializations_dic(directory), seeding_dic(directory), directionality_dic(directory))
master_dict = list_of_dicts(default_dict, initializations_dic(directory), seeding_dic(directory))

# num_shifts_of_source_node = 'num_shifts_of_source_node'
# print(f'index: {i} | directory: {list(master_dict[i].values())[0]} | num_shifts_of_source_node: {master_dict[i][num_shifts_of_source_node]}')

if __name__ == '__main__':
    for i in range(3):
        run_grid_search(param_dic=master_dict[i])
    run_grid_search(param_dic=master_dict[int(sys.argv[1])])
