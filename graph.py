import numpy as np
import random
import utility_funcs
import networkx as nx
import effective_distance as ed
import multiprocessing as mp
import time
np.random.seed(42)
random.seed(42)

# Global Variables for default run values
update_interval_val = 1
exp_decay_param_val = 12
source_reward_val = 2.6
equilibrium_distance_val = 100


class Graph:
    _starting_node = None   # holds the starting nodes for each run, reset after every run.

    def __init__(self, num_nodes, eff_dist_and_edge_response=None, fraction_info_score_redistributed=None, reinforcement_infoscore_coupling=True, beta=None):
        """
        Initialization; the 'extra' dimensionality (i.e. 1 for nodes, A) are there to dynamically store their history
        via use of vstack later on. To minimize variables, many features are activated by specifying their relevant
        macro parameter, e.g. diversity based seeding by setting beta != None
        :param num_nodes: Number of nodes in network. Remains constant
        :param eff_dist_and_edge_response: Tune between 0 (edge reinforcement is entirely based on eff_dist) and 1 (edge reinforcement is entirely based on extant edge values)
        :param fraction_info_score_redistributed: Determines if reweight_edges_via_info_score is used.
        Takes the lowest fraction \in (0,1) of edge values for each node and rather than reinforcing them, redistributes
         their reward to that node's oher edges proportional to their standing value.
        :param reinforcement_infoscore_coupling if False decouples the above, leading to the same constant reward for the remaining node's edges
        :param beta: Determines if info is seeded by diversity of connexions, how much (exponential factor, \in (0,1) )
        """
        self.num_nodes = num_nodes
        self.eff_dist_and_edge_coupling = eff_dist_and_edge_response  # tunes info score based on between 0 (pure eff. dist. dependence) and 1 (pure edge dependence)
        self.fraction_infoscore_redistributed = fraction_info_score_redistributed  # rescaling of all rewards in reweight_edges_via_info_score
        self.reinforcement_infoscore_coupling = reinforcement_infoscore_coupling
        self.beta = beta  # Determines how much the seeding is weighted towards diversely connected nodes (The None default leads to an explicitly random seeding run)

        self.nodes = np.zeros((1, num_nodes))  # node values (and history of via the first dimension)
        self.A = np.zeros((1, self.num_nodes, self.num_nodes))  # Adjacency matrix (and history of)
        self.source_node_history = []
        self.eff_dist_history = []

    # Edge Initialization: ------------------------------------------------------------------------------------------
    def sparse_random_edge_init(self, nonzero_edges_per_node=1):
        """
        Randomly seeds Adjacency matrix with 1 fully weighted edge (or 2 0.5 weighted for nonzero_edges_per_node=2,
        etc. Presently allows for overlap of random edge assignment in the case of nonzero_edges_per_node > 1.
        """
        for node in range(0, self.num_nodes):
            range_excluding_self = list(range(0, self.num_nodes))
            range_excluding_self.remove(node)  # eliminates A_ij diagonals (self-referential looping 'edges')
            for edge_assigned in range(0, nonzero_edges_per_node):
                self.A[-1][node][random.choice(range_excluding_self)] += (1/nonzero_edges_per_node)

    def uniform_random_edge_init(self):
        """Most general case, fills Adjacency matrix with uniform random values and normalizes them"""
        self.A = np.random.rand(1, self.num_nodes, self.num_nodes)  # creates 1 x n x n adjacency matrix filled with rand(0,1)
        for node in range(0, self.num_nodes):
            self.A[-1][node][node] = 0  # eliminates looping edges (i.e. no edge refers to itself)
            self.A[-1][node] /= self.A[-1][node].sum()  # normalizes each node's total weights to 1

    # Information Seeding: ------------------------------------------------------------------------------------------
    def seed_info_random(self):
        self._starting_node = None  # resets starting nodes such that new seed_info call will not conflict
        while not self._starting_node:
            seeded_node = int(np.random.rand(1)*self.num_nodes)
            self._starting_node = seeded_node
            self.source_node_history.append(seeded_node)

    def seed_info_constant_source(self, constant_source_node):
        assert isinstance(constant_source_node, int) & constant_source_node >= 0 & constant_source_node <= self.num_nodes, f'Please choose constant source node to be in range of num_nodes, i.e. in (0, {self.num_nodes})'
        self._starting_node = constant_source_node
        self.source_node_history.append(constant_source_node)

    def seed_info_by_diversity_of_connections(self):
        """
        Computes standard deviation as a (negatively correlated) metric for diversity of connection between nodes,
        then uses this to distribute 'nuggets' of information (via canonical ensemble of standard deviation).
        Potentially variance would be faster (no sqrt) and better, changing the effect of connectedness.
        """
        self._starting_node = None  # resets starting nodes such that new seed_info call will not conflict
        exp_stds = []
        for node_edges in self.A[-1][:]:
            exp_stds.append(np.exp(-self.beta * node_edges.std()))  # sum of e^(\beta \sigma_i) for i \in node[weights]
        std_partition = sum(exp_stds)
        while self._starting_node is None:
            seeded_node = np.random.randint(0, self.nodes[-1].size)
            if random.uniform(0, std_partition) < exp_stds[seeded_node]/std_partition:
                self._starting_node = seeded_node
                self.source_node_history.append(seeded_node)

    def seed_info_conditional(self, constant_source_node, num_shifts_of_source_node, num_runs, index):
        if self.beta:
            self.seed_info_by_diversity_of_connections()
        elif isinstance(constant_source_node, bool) & constant_source_node:
            self.seed_info_constant_source(0)   # Just to ensure seeding if set == True, it'll work without setting the constant seed to be a specific node
        elif not constant_source_node:
            self.seed_info_random()
        elif num_shifts_of_source_node:
            assert num_shifts_of_source_node < self.num_nodes, "More changes to constant source node than number of nodes. Set constant_source_node to false to activate continues random seeding"
            if (index % int((num_runs / num_shifts_of_source_node))) == 0:
                self.seed_info_constant_source(random.choice(list(set(range(len(self.nodes[-1]))) - set(self.source_node_history))))
        else:
            self.seed_info_constant_source(constant_source_node)

    # Info Score Evaluation: ----------------------------------------------------------------------------------------
    def evaluate_info_score(self, from_node_index, node_sum):
        """
         By design, x is the (scaled) proportion of connected node value to all node connections
         (scaled proportional to effective distance, thus the inverse relation via (1-x) while
         y is the edge's value (i.e. probability od transition, given normalization of A columns)
        To absolve the model of arbitrarity w.r.t. info score space geometry, we let the control parameter vary between two extremes (of uncoupled linear relation w.r.t. x, y)
        To make the info_score space have an inverse correlation between Eff_dist (node value) and edge value (\in A):
        Z = pow(x, (alpha - 1)) * pow(y, alpha) || alpha : Coupling parameter

        For general responsiveness to parameters rather than coupling between them:
        # return [pow((self.A[-1][node_index][from_node_index] / (self.nodes[-1][node_index] / node_sum)), self.eff_dist_and_edge_response) for node_index in range(self.nodes.shape[1])]
        """
        return [(pow((self.nodes[-1][node_index] / node_sum), (self.eff_dist_and_edge_coupling - 1)) * pow(self.A[-1][node_index][from_node_index], self.eff_dist_and_edge_coupling)) for node_index in range(self.nodes.shape[1])]

    def reweight_info_score(self, info_score, info_score_sum):
        """
        Reweights info_score by redistributing ____ fraction of info_score overall reward to the remaining (1-___)
        fraction of highest info_scores proportional to their respective percentage of total remaining info_score value
        By default, only redistributes the base info_score_sum value, but through coefficient could be more/less
        :param info_score_sum: simply passed to avoid recomputing sum, though only more efficient for larger node_numbers
        :param decoupled: if true, decouples edge value and reinforcement of edge value. (edge value only effects infoscore, and thus which edges are reinforced, not how much.)
        """
        if self.fraction_infoscore_redistributed == 1:
            cutoff_val = sorted(info_score)[-1]
        else:
            cutoff_val = sorted(info_score)[int(self.fraction_infoscore_redistributed * len(info_score))]

        if self.reinforcement_infoscore_coupling:
            reduced_info_score = [val if val >= cutoff_val else 0 for val in info_score]
        else:
            reduced_info_score = [1 if val >= cutoff_val else 0 for val in info_score]

        reduced_info_score_sum = sum(reduced_info_score)
        return [info_score_sum * (val / reduced_info_score_sum) for val in reduced_info_score]

    # Edge Reweighing: ----------------------------------------------------------------------------------------------
    def reweight_edges_via_info_score(self):
        info_scores = np.zeros(self.A[-1].shape)
        node_sum = self.nodes[-1].sum()
        for from_node in range(0, self.nodes[-1].size):
            info_score = self.evaluate_info_score(from_node, node_sum)
            info_score /= np.sum(info_score)  # Optional info_score normalization. **Should be incompatible with simple positive power in next step, but isn't...
            info_scores[:, from_node] = self.reweight_info_score(info_score, 1)  # replace 1 with sum(info_score) if not normalized [below]
            # info_scores[:, from_node] = self.reweight_info_score(info_score, np.sum(info_score))  # replace 1 with sum(info_score) if not normalized
        # print(np.round(info_scores, 3), '\n')
        self.A[-1] += info_scores

    def update_edges(self):
        """
        We may use the node values directly, as assigned by effective distance methods, to determine their effect on
         the node they were connected to. Even though this computational mechanic is the opposite of the
         conceptualization, it should yield the same results.
        The normalization (along incoming edges) is where the conservation of edge weight applies,
         negatively effecting those not reinforced.
        """
        if self.fraction_infoscore_redistributed:
            self.reweight_edges_via_info_score()
        
        # Normalization (columns of A, incoming edges)
        for node in range(0, self.A[-1][0].size):
            incoming_edge_sum = self.A[-1][:, node].sum()
            if incoming_edge_sum > 0:
                self.A[-1][:, node] /= incoming_edge_sum  # normalizes each node's total INCOMING weights to 1
            # For sparse networks, there will likely be some columns (outgoing edges) which sum to zero. (thus the conditional)

    # Effective Distance Evaluation: --------------------------------------------------------------------------------
    def get_eff_dist(self, adjacency_matrix=None, multiple_path=False, random_walk=False, source=None, target=None, parameter=1, saveto=""):
        """
        Returns effective distance based on the effective distance library built by Andreas Koher. Random walk
        estimations require row normalized adjacency matrix values.
        """
        if adjacency_matrix is None:
            adjacency_matrix = self.A[-1]
        if multiple_path:
            return ed.EffectiveDistances(np_array=adjacency_matrix).get_multiple_path_distance(source=source, target=target, parameter=parameter, saveto=saveto)
        if random_walk:
            return ed.EffectiveDistances(np_array=adjacency_matrix).get_random_walk_distance(source=source, target=target, parameter=parameter, saveto=saveto)
        else:
            print('No path type chosen in get_eff_dist call. Set multiple_path, shortest_path, dominant_path or random_walk=True')

    def evaluate_effective_distances(self, source, source_reward, parameter, multiple_path_eff_dist, timestep=-1, rounding=3):
        """
        Defaults to random walker effective distance metric unless given multiple_path_eff_dist=True
        returns array of effective distances to each node (from source) according to effective dist library´methods
        """
        if multiple_path_eff_dist:
            """
            Rounding of A to ensure that when the matrix is converted to a nx graph, near zero (O(^rounding)) paths 
            will be eliminated (by being set to zero before the nx conversion) from the nx all paths algorithm.
            The inversion is necessary to ensure that large probabilities of RW steps between nodes for the default A 
            are translated to small distances according to the MPED edge algorithm. 
            """
            inv_A = utility_funcs.return_one_over_2d_matrix(np.round(self.A[timestep], rounding))
            # inv_A = self.A[timestep] + 1  # alternate method for making el in A > 1 for MPED, doesn't nonlinearly penalize low probabilities.
            eff_dists = self.get_eff_dist(adjacency_matrix=inv_A, multiple_path=True, source=source, parameter=parameter)
            # should be negative of end result eff dist(as algorithm uses - log? Or not inverted, if so...)
        else:
            # pre-normalize rows (as both columns and rows must be normalized for RWED)
            row_sums = self.A[timestep].sum(axis=1)
            normalized_A = np.array([self.A[timestep][node, :]/row_sums[node] for node in range(self.A[timestep].shape[0])])
            # normalized_A = np.round(utility_funcs.matrix_normalize(self.A[timestep], row_normalize=True), 20)
            eff_dists = self.get_eff_dist(adjacency_matrix=normalized_A, random_walk=True, source=source, parameter=parameter)

        assert np.isclose(eff_dists[source], 0, rtol=1e-10), f'Source has nonzero effective distance of {eff_dists[source]}'
        eff_dists = np.delete(eff_dists, source)  # awkward deletion/insertion to ensure min search of remaining eff_distances
        eff_dists = np.insert(eff_dists, source, min(eff_dists)/source_reward)
        self.eff_dist_history.append(eff_dists)  # Must come before normalization otherwise sum will always be 1
        return eff_dists / np.sum(eff_dists)

    # Utility Functions: --------------------------------------------------------------------------------------------
    def write_graph_as_xml(self, timestep=-1, path=None):
        if path is None:
            path = f"graph_at_{timestep}.graphml"
        nx_G = nx.to_directed(nx.from_numpy_matrix(np.array(self.A[timestep]), create_using=nx.DiGraph))
        nx.write_graphml(nx_G, f'{path}.graphml')

    def convert_to_nx_graph(self, timestep=-1):
        return nx.to_directed(nx.from_numpy_matrix(np.array(self.A[timestep]), create_using=nx.DiGraph))

    def convert_history_to_list_of_nx_graphs(self, verbose=False):
        nx_graph_history = []
        if verbose:
            print(f'Beginning conversion of all graphs to nx_graphs...')
            start_time = time.time()
            for i in range(self.A.shape[0]):
                nx_graph_history.append(self.convert_to_nx_graph(i))
                utility_funcs.print_run_percentage(index=i, runs=self.A.shape[0])
            print(f'Conversion of all graphs to nx_graphs completed. It took {int((time.time()-start_time) / 60)} minutes, {np.round((time.time()-start_time) % 60, 2)}')
            return nx_graph_history
        else:
            return [self.convert_to_nx_graph(i) for i in range(self.A.shape[0])]

    # Observables: --------------------------------------------------------------------------------------------------
    def eff_dist_diff(self, eff_dist_to_all=False, multiple_path_eff_dist=False, source_reward=2.6, higher_order_paths_suppression=1):
        if eff_dist_to_all:
            return np.mean(self.evaluate_effective_distances(source=self.source_node_history[0], source_reward=source_reward,
                                                 multiple_path_eff_dist=multiple_path_eff_dist,
                                                 parameter=higher_order_paths_suppression, timestep=0)) \
                   - np.mean(self.evaluate_effective_distances(source=self.source_node_history[-1], source_reward=source_reward,
                                                 multiple_path_eff_dist=multiple_path_eff_dist,
                                                 parameter=higher_order_paths_suppression, timestep=-1))
        else:
            return np.mean(self.eff_dist_history[0]) - np.mean(self.eff_dist_history[-1])

    # Run Function: -------------------------------------------------------------------------------------------------
    def run(self, num_runs, update_interval=1, exp_decay_param=exp_decay_param_val, source_reward=source_reward_val, constant_source_node=False,
            num_shifts_of_source_node=False, multiple_path=False, equilibrium_distance=equilibrium_distance_val, verbose=False):
        """
        :param num_runs: Constant natural number, number of runs.
        :param update_interval: Number of seed steps per run (times information is seeded and diffused before reweighing edges)
        :param exp_decay_param: determines exponential suppression of higher order paths for both RWED and MPED
        :param source_reward: Determines how much the source node is scaled, recommended values \in (1,2)
        :param constant_source_node: Sets seed node to be a given node (the integer given).  True defaults to 0th node.
        :param multiple_path: if True, uses multiple path effective distance algorithm. Otherwise, defaults to random walker effective distance algorithm
        :param equilibrium_distance: length at which two nearly equal A matricies are constantly compared to break run loop
        :param verbose: if True, details approximate completion percentage and run parameters, methods.
        :return: Returns nothing, updates graph values. Use plotter library to evaluate and graph observables
        """
        # Verbose functions:
        def print_run_methods():
            print(f'Parameter and Method Details:\nRan {num_runs} runs with {update_interval} seed steps per run, and sources were rewarded by a {source_reward} scaling')
            if isinstance(constant_source_node, bool) & constant_source_node:
                print(f'All information was seeded to the 0th node. (Constant Source)')
            elif isinstance(constant_source_node, int):
                print(f'All information was seeded to the {constant_source_node}th node. (constant source)')
            elif self.beta:
                print(f'Information seeded proportional to diversity of connections, with partition exponent of beta = {self.beta}')
            elif not constant_source_node or self.beta:
                print(f'All information was seeded randomly.')

            if multiple_path:
                print(f'The Multiple Path Effective Distance (MPED) was used, with higher order paths suppressed by a exp_decay_param of {exp_decay_param}')
            else:
                print(f'The (analytic) Random Walker Effective Distance (RWED) was used, with an exp_decay_param of {exp_decay_param}.')

            print('Edge Update algorithm:')
            if self.fraction_infoscore_redistributed:
                print(f'Reweight_edges_via_info_score algorithm, with {self.fraction_infoscore_redistributed} as the exponential scaling/rate of edge adaptation')
            else:
                print(f'Edges were reweighed by simply adding the product of their edge value and origin node (from which they were directed) to their value')

        # Body of run function
        self.A = np.vstack((self.A, [self.A[-1]]))  # so that initial values (before initial update) are preserved
        equilibrium_span = 1  # if a greater range of values between equilibrium distance ought be compared
        for i in range(0, num_runs):
            self.seed_info_conditional(constant_source_node, num_shifts_of_source_node, num_runs=num_runs, index=i)
            self.nodes[-1] = np.array(self.evaluate_effective_distances(self._starting_node, source_reward, exp_decay_param, multiple_path))
            if i % update_interval == 0:
                self.update_edges()
                # so the next values may be overwritten, we start with 0 node values.
                self.nodes = np.vstack((self.nodes, np.zeros((1, self.num_nodes))))
                self.A = np.vstack((self.A, [self.A[-1]]))
            if verbose:
                utility_funcs.print_run_percentage(i, num_runs, 17)
            if self.A.shape[0] > equilibrium_distance+equilibrium_span:
                if np.all(np.array([np.allclose(self.A[-i], self.A[-(equilibrium_distance+i)], rtol=1e-5) for i in range(equilibrium_span)])):
                    print(f'Equilibrium conditions met after {i} runs, run halted.')
                    break  # Automatic break if equilibrium is reached. Lets run times be arb. large for MC parameter search
        self.A = np.delete(self.A, -1, axis=0)
        self.nodes = self.nodes[:-1]
        if verbose:
            print_run_methods()

