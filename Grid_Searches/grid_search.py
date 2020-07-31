import numpy as np
import os
import multiprocessing as mp
from pathlib import Path
import time
import sys
sys.path.append('../')
import plotter
import utility_funcs

data_directory = str(sys.argv[1])
run_index, num_nodes = [int(arg) for arg in sys.argv[2:4]]  # eliminates name of file as initial input string
subdata_directory = Path(data_directory, f"node_num_{num_nodes}")


edge_conservation_range = np.arange(*[float(arg) for arg in sys.argv[4].split('_')])
selectivity_range = np.arange(*[float(arg) for arg in sys.argv[5].split('_')])
print(f'Edge_conservation_range: {edge_conservation_range}')
print(f'Selectivity_range: {selectivity_range}')

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
    os.system('python simulate.py {output_directory} {run_index} {num_nodes} {edge_conservation_val} {selectivity_val} {reinforcement_info_score_coupling} {positive_eff_dist_and_reinforcement_correlation} {eff_dist_is_towards_source} {nodes_adapt_outgoing_edges} {incoming_edges_conserved} {undirected} {edge_init} {ensemble_size} {num_runs} {delta} {equilibrium_distance} {constant_source_node} {num_shifts_of_source_node} {seeding_sigma_coeff} {seeding_power_law_exponent} {beta} {multiple_path} {update_interval} {source_reward} {undirectify_init}'.format(**param_dic))


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
plotter.twoD_grid_search_plots(subdata_directory, edge_conservation_range=edge_conservation_range, selectivity_range=selectivity_range,
                               num_nodes=num_nodes, network_graphs=True, node_plots=False, ave_nbr=False, cluster_coeff=False,
                               eff_dist=True, global_eff_dist=True, shortest_path=False, degree_dist=True, edge_dist=True,
                               output_dir=Path(data_directory, 'Plots'))
# print(f"Time lapsed for all source reward values, {source_reward_range.size * edge_conservation_range.size * selectivity_range.size} total parameter combinations: {utility_funcs.time_lapsed_h_m_s(time.time() - total_start_time)}")

