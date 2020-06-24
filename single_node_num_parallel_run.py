import numpy as np
import os
import multiprocessing as mp
import utility_funcs
from pathlib import Path
import time

import plotter

num_nodes = 15
run_index = 1
data_directory = "/home/maqz/Desktop/data/power_law_w_exp_10"
output_directory = Path(data_directory, f"node_num_{num_nodes}")
try: os.mkdir(output_directory)
except OSError:
    print(f'{output_directory} already exists, adding or overwriting contents')
    pass

coupling_range = np.arange(0, 1.05, 0.05)
skew_range = np.arange(0, 1.05, 0.05)


def process_wrapper(param_dic):
    os.system('python single_run.py {output_directory} {num_nodes} {run_index} {coupling_val} {skew_val}'.format(**param_dic))


if __name__ == '__main__':
    start_time = time.time()
    for coupling_val in coupling_range:
        skew_range_iter = iter(range(skew_range.size))
        for skew_val_index in skew_range_iter:
            used_cores = 0
            processes = []
            left_over_skew_values = skew_range.size - skew_val_index
            # print(f'skew_val_index: {skew_val_index} | mp.cpu_count(): {mp.cpu_count()} | selectivity_range.size: {selectivity_range.size}')
            if left_over_skew_values < mp.cpu_count():  # To ensure that parallelization persists when there are fewer tasks than cores
                while used_cores < left_over_skew_values:
                    # print(f'used_cores: {used_cores} | skew_val_index: {skew_val_index} | selectivity_range[skew_val_index + used_cores]: {np.round(selectivity_range[skew_val_index + used_cores], 2)}')
                    parameter_dictionary = {
                        'output_directory': output_directory,
                        'num_nodes': num_nodes,
                        'run_index': run_index,
                        'coupling_val': np.round(coupling_val, 2),
                        'skew_val': np.round(skew_range[skew_val_index + used_cores], 2),
                    }
                    p = mp.Process(target=process_wrapper, args=(parameter_dictionary, ))
                    processes.append(p)
                    p.start()
                    used_cores += 1
                    run_index += 1
                utility_funcs.consume(skew_range_iter, left_over_skew_values - 1)  # -1 because the iteration forwards 1 step still proceeds directly after
            else:
                while used_cores < mp.cpu_count():
                    parameter_dictionary = {
                        'output_directory': output_directory,
                        'num_nodes': num_nodes,
                        'run_index': run_index,
                        'coupling_val': np.round(coupling_val, 2),
                        'skew_val': np.round(skew_range[skew_val_index + used_cores], 2),
                    }
                    p = mp.Process(target=process_wrapper, args=(parameter_dictionary, ))
                    processes.append(p)
                    p.start()
                    used_cores += 1
                    run_index += 1
                utility_funcs.consume(skew_range_iter, mp.cpu_count() - 1)  # Advances skew iter cpu count iterations

            for process in processes:
                process.join()  # join's created processes to run simultaneously.

    print(f"Time lapsed for {num_nodes} node, {coupling_range.size * skew_range.size} parameter combinations: {int((time.time()-start_time) / 60)} minutes, {np.round((time.time()-start_time) % 60, 2)} seconds")
    plotter.twoD_grid_search_w_plots(output_directory, edge_conservation_range=coupling_range, selectivity_range=skew_range, num_nodes=num_nodes, ave_nbr=False, cluster_coeff=False, shortest_path=False, output_dir=data_directory)

