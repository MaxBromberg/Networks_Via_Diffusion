from pathlib import Path
import time
import sys
sys.path.append('../')
import plotter
import graph
import utility_funcs


"""
Arguments via terminal to be given in the following order:
subdata_directory, run_index, [graph initialization args], edge_init, [simulate_ensemble args]
graph_init_args = [num_nodes, edge_conservation_val, selectivity_val, reinforcement_info_score_coupling, positive_eff_dist_and_reinforcement_correlation, eff_dist_is_towards_source, nodes_adapt_outgoing_edges, incoming_edges_conserved]

To be run with python3.6 or later. (includes fstrings)
Required packages: numpy, random, matplotlib, networkx, [pickle, os, pathLib, time]
In the same directory or below (thus sys.path.append('../')): 
graph.py, plotter.py, effective_distance.py, utility_funcs.py
"""

# Output path and run_index:
output_path = str(sys.argv[1])
run_index, num_nodes = [int(arg) for arg in sys.argv[2:4]]  # eliminates name of file as initial input string

# Graph initialization arguments:
edge_conservation_val, selectivity_val = [float(arg) for arg in sys.argv[4:6]]
assert isinstance(run_index, int), "Run index records order of runs, and ought be an integer"
assert isinstance(num_nodes, int), "Number of nodes ought be an integer"
print(f'[int(arg) for arg in sys.argv[6:12]]: {[arg for arg in sys.argv[6:12]]}')
assert set([int(arg) for arg in sys.argv[6:12]]).union({0, 1}) == {0, 1}, "All values should be integer castable booleans, i.e. (0, 1)"
r_i_s_c, p_ed_and_r_c, ed_to_s, n_a_o_e, i_e_c, undirected_run = [bool(int(arg)) for arg in sys.argv[6:12]]  # the bool cast is a bit much,

# Edge initialization arguments: Though the function allows passing a np.array directly, this cannot be done via shell script :(
if str(sys.argv[12]).__contains__('.'):
    edge_init = float(sys.argv[12])
elif str(sys.argv[12]) == "None" or str(sys.argv[12]) == "False":
    edge_init = None
else:
    edge_init = int(sys.argv[12])  # I don't see how to possibly pass a np.array through shell script, so that option's out

# Ensemble simulation arguments
ensemble_size, num_runs = [int(arg) for arg in sys.argv[13:15]]
delta = float(sys.argv[15])
equilibrium_distance, constant_source_node = [int(arg) for arg in sys.argv[16:18]]
num_shifts_of_source_node = int(sys.argv[18])
seeding_sigma_coeff, seeding_power_law_exponent, beta = [float(arg) for arg in sys.argv[19:22]]
multiple_path = bool(int(sys.argv[22]))
update_interval = int(sys.argv[23])
source_reward = float(sys.argv[24])
undirectify_init = bool(sys.argv[25])
# A Check(Bool)Sum ensuring no incompatible seeding command input
assert bool(constant_source_node) + bool(num_shifts_of_source_node) + bool(seeding_sigma_coeff) + bool(seeding_power_law_exponent) + bool(beta) < 2, 'Choose one seeding method, as all are mutually incompatible. If none, defaults to random seeding'


if __name__ == '__main__':
    start_time = time.time()
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
    print(f'Run {run_index}, [edge conservation: {edge_conservation_val}, selectivity: {selectivity_val}] complete ({utility_funcs.time_lapsed_h_m_s(time.time()-start_time)})')
