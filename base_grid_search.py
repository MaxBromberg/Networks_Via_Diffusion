import numpy as np
import os
import multiprocessing as mp
import utility_funcs
from pathlib import Path
import time

import plotter

num_nodes = 50
run_index = 1
data_directory = "/home/maqz/Desktop/data/random_seeding"
output_directory = Path(data_directory, f"node_num_{num_nodes}")

try: os.mkdir(output_directory)
except OSError:
    print(f'{output_directory} already exists, adding or overwriting contents')
    pass

edge_conservation_range = np.arange(0, 1.05, 0.05)
selectivity_range = np.arange(0, 1.05, 0.05)


def process_wrapper(param_dic):
    os.system('python single_run.py {output_directory} {num_nodes} {run_index} {edge_conservation_val} {selectivity_val}'.format(**param_dic))


if __name__ == '__main__':
    start_time = time.time()
    for coupling_val in edge_conservation_range:
        selectivity_range_iter = iter(range(selectivity_range.size))
        for selectivity_val_index in selectivity_range_iter:
            used_cores = 0
            processes = []
            left_over_selectivity_values = selectivity_range.size - selectivity_val_index
            # print(f'selectivity_val_index: {selectivity_val_index} | mp.cpu_count(): {mp.cpu_count()} | selectivity_range.size: {selectivity_range.size}')
            if left_over_selectivity_values < mp.cpu_count():  # To ensure that parallelization persists when there are fewer tasks than cores
                while used_cores < left_over_selectivity_values:
                    # print(f'used_cores: {used_cores} | selectivity_val_index: {selectivity_val_index} | selectivity_range[selectivity_val_index + used_cores]: {np.round(selectivity_range[selectivity_val_index + used_cores], 2)}')
                    parameter_dictionary = {
                        'output_directory': output_directory,
                        'num_nodes': num_nodes,
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
                while used_cores < mp.cpu_count():
                    parameter_dictionary = {
                        'output_directory': output_directory,
                        'num_nodes': num_nodes,
                        'run_index': run_index,
                        'edge_conservation_val': np.round(coupling_val, 2),
                        'selectivity_val': np.round(selectivity_range[selectivity_val_index + used_cores], 2),
                    }
                    p = mp.Process(target=process_wrapper, args=(parameter_dictionary, ))
                    processes.append(p)
                    p.start()
                    used_cores += 1
                    run_index += 1
                utility_funcs.consume(selectivity_range_iter, mp.cpu_count() - 1)  # Advances skew iter cpu count iterations

            for process in processes:
                process.join()  # join's created processes to run simultaneously.

    print(f"Time lapsed for {num_nodes} node, {edge_conservation_range.size * selectivity_range.size} parameter combinations: {utility_funcs.time_lapsed_h_m_s(time.time()-start_time)}")
    plotter.twoD_grid_search_plots(output_directory, edge_conservation_range=edge_conservation_range, selectivity_range=selectivity_range, num_nodes=num_nodes, network_graphs=True, node_plots=False, ave_nbr=False, cluster_coeff=False, shortest_path=False, degree_dist=True, output_dir=data_directory)

