import os
import sys
import time
import shutil
import numpy as np
from pathlib import Path
import multiprocessing as mp
import pprint

sys.path.append('../')
import graph
import plotter
import utility_funcs as uf

"""
Central program to order sequential grid-searches based on total parameter dictionaries
"""


def directionality_dic(data_directory):
    eff_dist_to_source = {'eff_dist_is_towards_source': 1}
    node_adapt_outgoing_edges = {'nodes_adapt_outgoing_edges': 1}
    outgoing_edges_conserved = {'incoming_edges_conserved': 0}
    directionality = {
        'base_case': {'data_directory': Path(data_directory, "base_case")},
        'reversed_source_edge_conservation': {**eff_dist_to_source, **node_adapt_outgoing_edges, **outgoing_edges_conserved, 'data_directory': Path(data_directory, "switched_ED_Adaptation_Conservation")},
        'reversed_edge_conservation': {**node_adapt_outgoing_edges, **outgoing_edges_conserved, 'data_directory': Path(data_directory, "switched_Adaptation_Conservation")},
        'reversed_source_conservation': {**eff_dist_to_source, **outgoing_edges_conserved, 'data_directory': Path(data_directory, "switched_ED_Conservation")},
        'reversed_source_edge': {**eff_dist_to_source, **node_adapt_outgoing_edges, 'data_directory': Path(data_directory, "switched_ED_Adaptation")},
        'reversed_source': {**eff_dist_to_source, 'data_directory': Path(data_directory, "switched_ED")},
        'reversed_edge': {**node_adapt_outgoing_edges, 'data_directory': Path(data_directory, "switched_Adaptation")},
        'reversed_conservation': {**outgoing_edges_conserved, 'data_directory': Path(data_directory, "switched_Conservation")}
    }
    return directionality


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


def grid_search(param_dic, num_cores_used=mp.cpu_count(), remove_data_post_plotting=True):
    unvarying_dic_values = list(param_dic.values())[5:]
    data_directory = str(param_dic['data_directory'])
    run_index, num_nodes = [int(arg) for arg in [param_dic['run_index'], param_dic['num_nodes']]]  # eliminates name of file as initial input string
    subdata_directory = Path(data_directory, f"node_num_{num_nodes}")

    edge_conservation_range = np.arange(*[float(arg) for arg in param_dic['edge_conservation_range'].split('_')])
    selectivity_range = np.arange(*[float(arg) for arg in param_dic['selectivity_range'].split('_')])

    try:
        super_data_dir = data_directory[:-len(data_directory.split('/')[-1])] if data_directory[-1] != '/' else data_directory[:-(len(data_directory.split('/')[-2]) + 1)]
        os.mkdir(super_data_dir)
    except OSError:
        print(f'{super_data_dir} already exists, adding to contents')
        pass

    try:
        os.mkdir(data_directory)
    except OSError:
        print(f'{data_directory} already exists, adding or overwriting contents')
        pass

    try:
        os.mkdir(subdata_directory)
    except OSError:
        print(f'{subdata_directory} already exists, adding or overwriting contents')
        pass

    grid_search_start_time = time.time()
    pool = mp.Pool(num_cores_used)
    args = []
    for coupling_val in edge_conservation_range:
        for selectivity_val in selectivity_range:
            varying_param_dic = {
                'output_directory': subdata_directory,
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

    plotter.twoD_grid_search_plots(subdata_directory, edge_conservation_range=edge_conservation_range, selectivity_range=selectivity_range,
                                   num_nodes=num_nodes,
                                   network_graphs=bool(plots['network_graphs']),
                                   node_plots=bool(plots['node_plots']),
                                   ave_nbr=bool(plots['ave_nbr']),
                                   cluster_coeff=bool(plots['cluster_coeff']),
                                   eff_dist=bool(plots['eff_dist']),
                                   global_eff_dist=bool(plots['global_eff_dist']),
                                   shortest_path=bool(plots['shortest_path']),
                                   degree_dist=bool(plots['degree_dist']),
                                   edge_dist=bool(plots['edge_dist']),
                                   meta_plots=bool(plots['meta_plots']),
                                   output_dir=Path(data_directory, 'Plots'))
    if remove_data_post_plotting:
        shutil.rmtree(subdata_directory)


def run_grid_search(param_dic, via_pool=True):
    if via_pool:
        grid_search(param_dic)
    else:
        os.system('python grid_search.py {data_directory} {run_index} {num_nodes} {edge_conservation_range} {selectivity_range} {reinforcement_info_score_coupling} {positive_eff_dist_and_reinforcement_correlation} {eff_dist_is_towards_source} {nodes_adapt_outgoing_edges} {incoming_edges_conserved} {undirected} {edge_init} {ensemble_size} {num_runs} {delta} {equilibrium_distance} {constant_source_node} {num_shifts_of_source_node} {seeding_sigma_coeff} {seeding_power_law_exponent} {beta} {multiple_path} {update_interval} {source_reward} {undirectify_init} {network_graphs} {node_plots} {ave_nbr} {cluster_coeff} {eff_dist} {global_eff_dist} {shortest_path} {degree_dist} {edge_dist} {meta_plots}'.format(**param_dic))
    print(f'Simulation with the following parameters complete:')
    pprint.pprint(param_dic)


def run_over_all_directionality_combos(mods, data_directory, via_pool=True):
    """
    Runs grid searches for every directionality combination
    :param mods: dictionary of all mods unrelated to directionality (initialization, directedness, etc...)
    :param data_directory: where all directionality grid-searches will be output
    """
    modded_directionality_dic = directionality_dic(data_directory=data_directory)
    for v in modded_directionality_dic.values():
        run_grid_search({**mods, **v}, via_pool=via_pool)


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
plots = {
    'network_graphs': 1,  # graphs the networks
    'node_plots': 0,  # plots Evolution of node values over time
    'ave_nbr': 0,  # Plots average neighbor connections over time
    'cluster_coeff': 0,  # Plots evolution of cluster coefficient
    'eff_dist': 1,  # Plots evolution of average effective distance to source
    'global_eff_dist': 1,  # Plots evolution of average effective distance from every node to every other
    'shortest_path': 0,  # PLots the average shortest path over time. Very computationally expensive
    'degree_dist': 0,  # Yields the degree (total weight) distribution as a histogram
    'edge_dist': 1,  # PLots the edge distribution (individual edge counts) as a histogram
    'meta_plots': 1,  # Plots all the meta-plots, specifically: last_ave_nbr_deg, ed diffs, mean ed, ave_neighbor diffs,
    # global ed diffs, ave_nbr variance, log_deg_dist variance, hierarchy coordinates (with exponential and linear thresholds) and efficiency coordinates
}

default_dic = {**parameter_dictionary, **search_wide_dic, **edge_init, **ensemble_params, **plots}


default_sparse_init = {**default_dic, 'edge_init': 1.2, 'num_nodes': 60, 'constant_source_node': 4, 'edge_conservation_range': '0.4_1.05_0.1'}
directory = '/home/maqz/Desktop/data/Mechanic_Mods/sparse_edge_init_1.2'
start_time = time.time()
run_over_all_directionality_combos(mods=default_sparse_init, data_directory=directory, via_pool=True)

print(f'Total Time Elapsed: {uf.time_lapsed_h_m_s(time.time()-start_time)}')

# undirected_sparse_run_data_directory = '/home/maqz/Desktop/data/Mechanic_Mods/undirected_run_sparse_1.2'
# undirected_sparse_init = {**default_sparse_init, 'undirectify_init': 1}
# undirected_sparse_run = {**undirected_sparse_init, 'undirected': 1}
