import numpy as np
import os
import shutil
import multiprocessing as mp
from pathlib import Path
import time
import sys
sys.path.append('../')
import plotter
import utility_funcs
import graph

data_directory = str(sys.argv[1])
run_index, num_nodes = [int(arg) for arg in sys.argv[2:4]]  # eliminates name of file as initial input string
subdata_directory = Path(data_directory, f"node_num_{num_nodes}")


edge_conservation_range = np.arange(*[float(arg) for arg in sys.argv[4].split('_')])
selectivity_range = np.arange(*[float(arg) for arg in sys.argv[5].split('_')])
# print(f'Edge_conservation_range: {edge_conservation_range}')
# print(f'Selectivity_range: {selectivity_range}')

num_cores_used = mp.cpu_count() - 3

search_wide_dic = {
    'reinforcement_info_score_coupling': sys.argv[6],
    'positive_eff_dist_and_reinforcement_correlation': sys.argv[7],
    'eff_dist_is_towards_source': sys.argv[8],
    'nodes_adapt_outgoing_edges': sys.argv[9],
    'incoming_edges_conserved': sys.argv[10],
    'undirected': sys.argv[11],
    'edge_init': sys.argv[12],
    'ensemble_size': sys.argv[13],
    'num_runs': sys.argv[14],
    'delta': sys.argv[15],
    'equilibrium_distance': sys.argv[16],
    'constant_source_node': sys.argv[17],
    'num_shifts_of_source_node': sys.argv[18],
    'seeding_sigma_coeff': sys.argv[19],
    'seeding_power_law_exponent': sys.argv[20],
    'beta': sys.argv[21],
    'multiple_path': sys.argv[22],
    'update_interval':  sys.argv[23],
    'source_reward': sys.argv[24],
    'undirectify_init': sys.argv[25]
}
# Plots Needn't be passed to simulate.py
plots = {
    'network_graphs': sys.argv[26],  # graphs the networks
    'node_plots': sys.argv[27],  # plots Evolution of node values over time
    'ave_nbr': sys.argv[28],  # Plots average neighbor connections over time
    'cluster_coeff': sys.argv[29],  # Plots evolution of cluster coefficient
    'eff_dist': sys.argv[30],  # Plots evolution of average effective distance to source
    'global_eff_dist': sys.argv[31],  # Plots evolution of average effective distance from every node to every other
    'shortest_path': sys.argv[32],  # PLots the average shortest path over time. Very computationally expensive
    'degree_dist': sys.argv[33],  # Yields the degree (total weight) distribution as a histogram
    'edge_dist': sys.argv[34],  # PLots the edge distribution (individual edge counts) as a histogram
    'meta_plots': sys.argv[35],  # Plots all the meta-plots, specifically: last_ave_nbr_deg, ed diffs, mean ed, ave_neighbor diffs,
    # global ed diffs, ave_nbr variance, log_deg_dist variance, hierarchy coordinates (with exponential and linear thresholds) and efficiency coordinates
}
print('Plots in grid Search')
utility_funcs.print_inventory(plots)


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


def process_wrapper(param_dic):
    os.system('python3 simulate.py {output_directory} {run_index} {num_nodes} {edge_conservation_val} {selectivity_val} {reinforcement_info_score_coupling} {positive_eff_dist_and_reinforcement_correlation} {eff_dist_is_towards_source} {nodes_adapt_outgoing_edges} {incoming_edges_conserved} {undirected} {edge_init} {ensemble_size} {num_runs} {delta} {equilibrium_distance} {constant_source_node} {num_shifts_of_source_node} {seeding_sigma_coeff} {seeding_power_law_exponent} {beta} {multiple_path} {update_interval} {source_reward} {undirectify_init}'.format(**param_dic))


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
    print(f'Run {run_index}, [edge conservation: {edge_conservation_val}, selectivity: {selectivity_val}] complete. {G.get_num_errors()} errors, ({utility_funcs.time_lapsed_h_m_s(time.time()-graph_start_time)})')


if __name__ == '__main__':
    start_time = time.time()
    pool = mp.Pool(num_cores_used)
    args = []
    for coupling_val in edge_conservation_range:
        for selectivity_val in selectivity_range:
            parameter_dictionary = {
                'output_directory': subdata_directory,
                'run_index': run_index,
                'num_nodes': num_nodes,
                'edge_conservation_val': np.round(coupling_val, 2),
                'selectivity_val': np.round(selectivity_val, 2),
            }
            run_index += 1
            parameter_dictionary.update(search_wide_dic)
            args.append(list(parameter_dictionary.values()))
    p = pool.map(simulate, args)
    pool.close()
    pool.join()
    print(f"Time lapsed for {num_nodes} nodes, {edge_conservation_range.size * selectivity_range.size} parameter combinations: {utility_funcs.time_lapsed_h_m_s(time.time()-start_time)}")
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
shutil.rmtree(subdata_directory)

"""
if __name__ == '__main__':
    start_time = time.time()
    for coupling_val in edge_conservation_range:
        selectivity_range_iter = iter(range(selectivity_range.size))
        for selectivity_val_index in selectivity_range_iter:
            used_cores = 0
            processes = []
            left_over_selectivity_values = selectivity_range.size - selectivity_val_index
            # print(f'selectivity_val_index: {selectivity_val_index} | mp.cpu_count(): {mp.cpu_count()} | selectivity_range.size: {selectivity_range.size}')
            if left_over_selectivity_values < num_cores_used:  # To ensure that parallelization persists when there are fewer tasks than cores
                while used_cores < left_over_selectivity_values:
                    # print(f'used_cores: {used_cores} | selectivity_val_index: {selectivity_val_index} | selectivity_range[selectivity_val_index + used_cores]: {np.round(selectivity_range[selectivity_val_index + used_cores], 2)}')
                    parameter_dictionary = {
                        'output_directory': subdata_directory,
                        'run_index': run_index,
                        'num_nodes': num_nodes,
                        'edge_conservation_val': np.round(coupling_val, 2),
                        'selectivity_val': np.round(selectivity_range[selectivity_val_index + used_cores], 2),
                    }
                    parameter_dictionary.update(search_wide_dic)
                    p = mp.Process(target=process_wrapper, args=(parameter_dictionary, ))
                    processes.append(p)
                    p.start()
                    used_cores += 1
                    run_index += 1
                utility_funcs.consume(selectivity_range_iter, left_over_selectivity_values - 1)  # -1 because the iteration forwards 1 step still proceeds directly after
            else:
                while used_cores < num_cores_used:
                    parameter_dictionary = {
                        'output_directory': subdata_directory,
                        'run_index': run_index,
                        'num_nodes': num_nodes,
                        'edge_conservation_val': np.round(coupling_val, 2),
                        'selectivity_val': np.round(selectivity_range[selectivity_val_index + used_cores], 2),
                    }
                    parameter_dictionary.update(search_wide_dic)
                    p = mp.Process(target=process_wrapper, args=(parameter_dictionary, ))
                    processes.append(p)
                    p.start()
                    used_cores += 1
                    run_index += 1
                utility_funcs.consume(selectivity_range_iter, num_cores_used - 1)  # Advances skew iter cpu count iterations

            for process in processes:
                process.join()  # join's created processes to run simultaneously.

    print(f"Time lapsed for {num_nodes} nodes, {edge_conservation_range.size * selectivity_range.size} parameter combinations: {utility_funcs.time_lapsed_h_m_s(time.time()-start_time)}")

"""