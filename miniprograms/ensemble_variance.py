import numpy as np
import sys

sys.path.append('../')
import graph
import plotter

num_nodes = 50
source_reward = 2.6
# edge_init_val = None  # leads to uniform random edge init
# edge_init_val = 3   # leads to 3 edges/node. (int())
edge_init_val = 1.2  # leads to scale free edge init, k_min=1, \gamma = 1.2 (float())
num_simulations_per_parameter_combo = 10
num_runs = 1000
delta = 1
equilibrium_distance = 200
constant_source_node = 3
num_shifts_of_source_node = False
seeding_sigma_coeff = False
seeding_power_law_exponent = False
beta = False
multiple_path = False

# ----------------------- Ensemble Graph ---------------------- #
ensemble_G = graph.Graph(num_nodes=num_nodes, edge_conservation_coefficient=0.5, selectivity=0.5,
                         reinforcement_info_score_coupling=True, positive_eff_dist_and_reinforcement_correlation=False)

ensemble_G.simulate_ensemble(num_simulations=num_simulations_per_parameter_combo, num_runs_per_sim=num_runs,
                             update_interval=1, eff_dist_delta_param=delta,
                             source_reward=source_reward, edge_init=edge_init_val,
                             constant_source_node=constant_source_node,
                             num_shifts_of_source_node=num_shifts_of_source_node,
                             equilibrium_distance=equilibrium_distance, seeding_sigma_coeff=seeding_sigma_coeff,
                             seeding_power_law_exponent=seeding_power_law_exponent, beta=beta,
                             multiple_path=multiple_path, verbose=True)

# -----------------------     Graph     ---------------------- #
G = graph.Graph(num_nodes=num_nodes, edge_conservation_coefficient=0.5, selectivity=0.5,
                reinforcement_info_score_coupling=True, positive_eff_dist_and_reinforcement_correlation=False)
# G.uniform_random_edge_init()
# G.sparse_random_edge_init(nonzero_edges_per_node=3, connected=True)
G.scale_free_edge_init(degree_exponent=1.2, min_k=1, connected=True)
G.simulate(num_runs=num_runs, eff_dist_delta_param=delta, constant_source_node=constant_source_node,
           source_reward=source_reward, num_shifts_of_source_node=num_shifts_of_source_node,
           equilibrium_distance=equilibrium_distance, seeding_sigma_coeff=seeding_sigma_coeff,
           seeding_power_law_exponent=seeding_power_law_exponent, beta=beta, multiple_path=multiple_path, verbose=True)

ensemble_A, ensemble_H, ensemble_global_eff_dist = ensemble_G.A[-1], ensemble_G.eff_dist_to_source_history, ensemble_G.evaluate_effective_distances(source_reward=source_reward, parameter=delta, multiple_path_eff_dist=False, timestep=-1)
A, H, global_eff_dist = G.A[-1], G.eff_dist_to_source_history, G.evaluate_effective_distances(source_reward=source_reward,
                                                                                              parameter=delta,
                                                                                              multiple_path_eff_dist=False,
                                                                                              timestep=-1)

diff_A = ensemble_A - A
diff_H = np.array(ensemble_H) - np.array(H)
diff_global_eff_dist = np.array(ensemble_global_eff_dist) - np.array(global_eff_dist)

np.set_printoptions(suppress=True)
print(f'Base A Value: \n {A}')
print(f'Base Effective Distance Value: \n {np.round(global_eff_dist, 2)}')
print(f'Final Adjacency Matrix Differences: \n {np.round(diff_A, 2)} \n with variance, min/max: {np.var(diff_A)}, {np.min(diff_A)}/{np.max(diff_A)}')
print(f'Effective Distance History Differences: \n {np.round(diff_H, 2)} \n with variance, min/max: {np.var(diff_H)}, {np.min(diff_H)}/{np.max(diff_H)}')
print(f'Global Effective Distance Differences: \n {np.round(diff_global_eff_dist, 2)} \n with variance, min/max: {np.var(diff_global_eff_dist)}, {np.min(diff_global_eff_dist)}/{np.max(diff_global_eff_dist)}')

plotter.plot_heatmap(diff_global_eff_dist, x_range=np.arange(num_nodes), y_range=np.arange(num_nodes), tick_scale=5,
                     fig_title='Final Effective Distance Differences', show=True)
plotter.plot_heatmap(diff_A, x_range=np.arange(num_nodes), y_range=np.arange(num_nodes), tick_scale=5,
                     fig_title='Ensemble A - A', show=True)
plotter.plot_eff_dist(ensemble_G, all_to_all=False)
plotter.plot_eff_dist(G, all_to_all=False)
plotter.plot_eff_dist(ensemble_G, all_to_all=True)
plotter.plot_eff_dist(G, all_to_all=True)
