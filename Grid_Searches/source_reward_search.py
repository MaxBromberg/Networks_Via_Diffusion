import numpy as np
import os
import utility_funcs
from pathlib import Path
import time
import sys
sys.path.append('../')
import multiprocessing as mp

import plotter

num_nodes = 50
num_cores_used = mp.cpu_count() - 3

source_reward_range = np.arange(2, 2.2, 0.2)
edge_conservation_range = np.arange(0, 1.05, 0.1)
selectivity_range = np.arange(0, 1.05, 0.1)


def process_wrapper(param_dic):
    os.system('python source_reward_run.py {output_directory} {source_reward} {run_index} {edge_conservation_val} {selectivity_val}'.format(**param_dic))


total_start_time = time.time()
for source_reward_val in source_reward_range:
    data_directory = "/home/maqz/Desktop/data/contant_seeding"
    output_directory = Path(data_directory, f"source_reward_{source_reward_val}")
    try:
        os.mkdir(output_directory)
    except OSError:
        print(f'{output_directory} already exists, adding or overwriting contents')
        pass

    print(f'grid search with source reward value of {source_reward_val} begun:')
    start_time = time.time()
    run_index = 1
    for coupling_val in edge_conservation_range:
        selectivity_range_iter = iter(range(selectivity_range.size))
        for selectivity_val_index in selectivity_range_iter:
            used_cores = 0
            processes = []
            left_over_selectivity_values = selectivity_range.size - selectivity_val_index
            if left_over_selectivity_values < num_cores_used:  # To ensure that parallelization persists when there are fewer tasks than cores
                while used_cores < left_over_selectivity_values:
                    parameter_dictionary = {
                        'subdata_directory': output_directory,
                        'source_reward': source_reward_val,
                        'run_index': run_index,
                        'edge_conservation_val': np.round(coupling_val, 2),
                        'selectivity_val': np.round(selectivity_range[selectivity_val_index + used_cores], 2),
                    }
                    p = mp.Process(target=process_wrapper, args=(parameter_dictionary, ))
                    processes.append(p)
                    p.start()
                    used_cores += 1
                    run_index += 1
                utility_funcs.consume(selectivity_range_iter, left_over_selectivity_values - 1)  # -1 because the iteration forwards 1 step still proceeds directly after
            else:
                while used_cores < num_cores_used:
                    parameter_dictionary = {
                        'subdata_directory': output_directory,
                        'source_reward': source_reward_val,
                        'run_index': run_index,
                        'edge_conservation_val': np.round(coupling_val, 2),
                        'selectivity_val': np.round(selectivity_range[selectivity_val_index + used_cores], 2),
                    }
                    p = mp.Process(target=process_wrapper, args=(parameter_dictionary, ))
                    processes.append(p)
                    p.start()
                    used_cores += 1
                    run_index += 1
                utility_funcs.consume(selectivity_range_iter, num_cores_used - 1)  # Advances skew iter cpu count iterations

            for process in processes:
                process.join()  # join's created processes to run simultaneously.

    print(f"Time lapsed for {source_reward_val} source reward grid search: {utility_funcs.time_lapsed_h_m_s(time.time()-start_time)}")
print(f"Time lapsed for all source reward values, {source_reward_range.size * edge_conservation_range.size * selectivity_range.size} total parameter combinations: {utility_funcs.time_lapsed_h_m_s(time.time()-total_start_time)}")
# plotter.twoD_grid_search_plots(subdata_directory, edge_conservation_range=edge_conservation_range, selectivity_range=selectivity_range, num_nodes=num_nodes, network_graphs=True, node_plots=False, ave_nbr=False, cluster_coeff=False, shortest_path=False, degree_dist=True, output_dir=data_directory)

