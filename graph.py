import numpy as np
import random
import utility_funcs
import networkx as nx
import effective_distance as ed
np.random.seed(42)
random.seed(42)


class Graph:
    starting_node = None

    def __init__(self, num_nodes, value_per_nugget=1, edge_weighting_exp_rate=None, take_the_best_reward_rate=None, rate_of_edge_adaptation=None, beta=None, gamma=None, q=None):
        """
        Initialization; the 'extra' dimensionality (i.e. 1 for nodes, A) are there to dynamically store their history
        via use of vstack later on. To minimize variables, many features are activated by specifying their relevant
        macro parameter, e.g. diversity based seeding by setting beta != None
        """
        self.beta = beta  # Determines how much the seeding is weighted towards diversely connected nodes
        # (The None default leads to an explicitly random seeding (and thus faster) run)
        self.num_nodes = num_nodes
        self.nugget_value = value_per_nugget
        self.gamma = gamma  # Determines punishment for connecting to the same node, as default strategy for both
        # random and diverse connection weighted seeding would otherwise be to connect all nodes to one other
        self.q = q  # exp{-gamma*[(sum_j w_ij)-q*N]} => for 0<q(<1) nodes are incentivized to strengthen outgoing edges
        self.edge_weighting_exp = edge_weighting_exp_rate  # determines exponential suppression of rate of adaptation in
        # info scores for both reweight_edges_via_info_score and reweight_edges_via_info_score, should be between 0 and 1
        self.take_the_best_reward_rate = take_the_best_reward_rate  # change in edge weight in take_the_best
        self.rate_of_edge_adaptation = rate_of_edge_adaptation  # rescaling of all rewards in reweight_edges_via_info_score
        self.nodes = np.zeros((1, num_nodes))  # node values (and history of via the first dimension)
        self.A = np.zeros((1, self.num_nodes, self.num_nodes))  # Adjacency matrix (and history of)
        self.starting_node = None  # holds the starting nodes for each run, reset after every run.
        self.source_node_history = []
        self.eff_dist_history = []

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

    def seed_info_random(self):
        self.starting_node = None  # resets starting nodes such that new seed_info call will not conflict
        while not self.starting_node:
            seeded_node = int(np.random.rand(1)*self.num_nodes)
            self.nodes[-1][seeded_node] += self.nugget_value
            self.starting_node = seeded_node
            self.source_node_history.append(seeded_node)

    def seed_info_constant_source(self, constant_source_node):
        assert isinstance(constant_source_node, int) & constant_source_node >= 0 & constant_source_node <= self.num_nodes, f'Please choose constant source node to be in range of num_nodes, i.e. in (0, {self.num_nodes})'
        self.starting_node = constant_source_node
        self.source_node_history.append(constant_source_node)

    def seed_info_by_diversity_of_connections(self):
        """
        Computes standard deviation as a (negatively correlated) metric for diversity of connection between nodes,
        then uses this to distribute 'nuggets' of information (via canonical ensemble of standard deviation).
        Potentially variance would be faster (no sqrt) and better, changing the effect of connectedness.
        """
        self.starting_node = None  # resets starting nodes such that new seed_info call will not conflict
        exp_stds = []
        for node_edges in self.A[-1][:]:
            exp_stds.append(np.exp(-self.beta * node_edges.std()))  # sum of e^(\beta \sigma_i) for i \in node[weights]
        std_partition = sum(exp_stds)
        while self.starting_node is None:
            seeded_node = np.random.randint(0, self.nodes[-1].size)
            if random.uniform(0, std_partition) < exp_stds[seeded_node]/std_partition:
                self.nodes[-1][seeded_node] += self.nugget_value
                self.starting_node = seeded_node
                self.source_node_history.append(seeded_node)

    def reweight_edges_with_clustering(self):
        outgoing_weight_sums = [weights.sum() for weights in self.A[-1]]  # sums adjacency matrix rows (outgoing edges)
        for node in range(0, self.nodes[-1].size):
            for edge in range(0, self.A[-1][node].size):
                if self.q:
                    self.A[-1][node][edge] += self.nodes[-1][node] * self.A[-1][node][edge] * np.exp(
                        -(self.gamma * outgoing_weight_sums[node] - (self.q * self.nodes.shape[0])))
                else:
                    self.A[-1][node][edge] += self.nodes[-1][node] * self.A[-1][node][edge] * np.exp(
                        -(self.gamma * outgoing_weight_sums[node]))

    def reweight_edges_without_clustering(self):
        for node in range(0, self.nodes[-1].size):
            for edge in range(0, self.A[-1][node].size):
                self.A[-1][node][edge] += self.nodes[-1][node] * self.A[-1][node][edge]
            # We normalize along the outgoing edges (columns) so that we do not simply reset the rows (as with rows)

    def evaluate_info_score(self, from_node_index, node_sum):
        """
         By design, x is the (scaled) proportion of connected node value to all node connections
         (scaled proportional to effective distance, thus the inverse relation via (1-x) while
         y is the edge's value (i.e. probability od transition, given normalization of A columns)
        """
        # returns all edges directed to from node reweighted by raising each edge to value proportional the node's value as a fraction of the total
        # rewards the edge which leads to the greatest info score with a constant (the take_the_best_reward_rate factor)
        # return [pow(self.A[-1][node_index][from_node_index], (self.edge_weighting_exp * (1 - (self.nodes[-1][node_index] / node_sum)))) for node_index in range(self.nodes.shape[1])]

        # z = pow((2 * x - 1), 2) * np.sqrt(y) where x is the (self.nodes[-1][node_index] / node_sum) and y the edge value
        # return [pow((2 * self.edge_weighting_exp * (self.nodes[-1][node_index] / node_sum) - 1), 2) * np.sqrt(self.A[-1][node_index][from_node_index]) for node_index in range(self.nodes.shape[1])]

        # To make the info_score space have an inverse correlation between Eff_dist (node value) and edge value (\in A)
        # return [pow((1 - (self.edge_weighting_exp * (self.nodes[-1][node_index] / node_sum))), 2) * np.sqrt(self.A[-1][node_index][from_node_index]) for node_index in range(self.nodes.shape[1])]
        return [np.sqrt(1 - (self.edge_weighting_exp * (self.nodes[-1][node_index] / node_sum))) * np.sqrt(self.A[-1][node_index][from_node_index]) for node_index in range(self.nodes.shape[1])]

    def reweight_edges_via_take_the_best(self):
        """
        with y being the edge value and x the connected node's fraction of total node value
        """
        max_info_score_indices = [self.nodes.shape[1] for i in range(self.nodes.shape[1])]
        node_sum = self.nodes[-1].sum()
        for from_node in range(0, self.nodes[-1].size):
            info_score = self.evaluate_info_score(from_node, node_sum)
            max_info_score_indices[from_node] = info_score.index(max(info_score))
        # Update must come after all info_scores are evaluated so as not to interfer with itself
        for i in range(0, len(max_info_score_indices)):
            self.A[-1][max_info_score_indices[i]][i] += self.take_the_best_reward_rate

    def reweight_edges_via_info_score(self):
        info_scores = np.zeros(self.A[-1].shape)
        node_sum = self.nodes[-1].sum()
        for from_node in range(0, self.nodes[-1].size):
            info_score = self.evaluate_info_score(from_node, node_sum)
            info_scores[:, from_node] = np.power(np.array(info_score), self.rate_of_edge_adaptation)
            # print(f'x values: {np.round([(self.edge_weighting_exp * (self.nodes[-1][i] / node_sum)) for i in range(self.nodes.shape[1])], 3)}')
            # print(f'y values: {np.round(self.A[-1][:][from_node], 3)}')
            # if from_node == self.nodes[-1].size - 1: print('')
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
        if self.take_the_best_reward_rate:
            self.reweight_edges_via_take_the_best()
        elif self.rate_of_edge_adaptation:
            self.reweight_edges_via_info_score()
        elif self.gamma:
            self.reweight_edges_with_clustering()
        else:
            self.reweight_edges_without_clustering()
        for node in range(0, self.A[-1][0].size):
            incoming_edge_sum = self.A[-1][:, node].sum()
            if incoming_edge_sum > 0:
                self.A[-1][:, node] /= incoming_edge_sum  # normalizes each node's total INCOMING weights to 1
            # For sparse networks, there will likely be some columns (outgoing edges) which sum to zero.

    def get_eff_dist(self, adjacency_matrix=None, multiple_path=False, shortest_path=False, dominant_path=False, random_walk=False, source=None, target=None, parameter=1, saveto=""):
        """
        Returns effective distance based on the effective distance library built by Andreas Koher. Random walk
        estimations require row normalized adjacency matrix values.
        """
        if adjacency_matrix is None:
            adjacency_matrix = self.A[-1]
        if multiple_path:
            return ed.EffectiveDistances(np_array=adjacency_matrix).get_multiple_path_distance(source=source, target=target, parameter=parameter, saveto=saveto)
        if shortest_path:
            return ed.EffectiveDistances(np_array=adjacency_matrix).get_shortest_path_distance(source=source, target=target, parameter=parameter, saveto=saveto)
        if dominant_path:
            return ed.EffectiveDistances(np_array=adjacency_matrix).get_dominant_path_distance(source=source, target=target, parameter=parameter, saveto=saveto)
        if random_walk:
            return ed.EffectiveDistances(np_array=adjacency_matrix).get_random_walk_distance(source=source, target=target, parameter=parameter, saveto=saveto)
        else:
            print('No path type chosen in get_eff_dist call. Set multiple_path, shortest_path, dominant_path or random_walk=True')

    def write_graph_as_xml(self, timestep=-1, path=None):
        if path is None:
            path = f"graph_at_{timestep}.graphml"
        nx_G = nx.to_directed(nx.from_numpy_matrix(np.array(self.A[timestep]), create_using=nx.DiGraph))
        nx.write_graphml(nx_G, path)

    def convert_to_nx_graph(self, timestep=-1):
        return nx.to_directed(nx.from_numpy_matrix(np.array(self.A[timestep]), create_using=nx.DiGraph))

    def convert_history_to_list_of_nx_graphs(self, verbose=False):
        nx_graph_history = []
        if verbose:
            print(f'Beginning conversion of all graphs to nx_graphs...')
            for i in range(self.A.shape[0]):
                nx_graph_history.append(self.convert_to_nx_graph(i))
                utility_funcs.print_run_percentage(index=i, runs=self.A.shape[0])
            print(f'Conversion of all graphs to nx_graphs completed.')
            return nx_graph_history
        else:
            return [self.convert_to_nx_graph(i) for i in range(self.A.shape[0])]


class EffDisGraph(Graph):

    def __init__(self, num_nodes, value_per_nugget=1, edge_weighting_exp_rate=1, take_the_best_reward_rate=None, rate_of_edge_adaptation=None, beta=None, gamma=None, q=None):
        """
        Here initialization seems unnecessary to be independent of the graph superclass, but may yet be useful in future applications.
        Presently defaults to take_the_best
        :param num_nodes: Number of Nodes in network. Remains constant
        :param value_per_nugget: floating point giving info per nugget. As all edge weights are normalized in
        reweighing process, this should not effect anything in present implementation (thus set to 1 by default)
        :param edge_weighting_exp_rate: Used in the exponential reweighing of edges via take_the_best and info_score reweighing
        Suggested range (0,1]
        :param take_the_best_reward_rate: Determines if take_the_best is used, and the amount added each node's best edge.
        Suggested range is around 0.01.
        :param rate_of_edge_adaptation: Determines if reweight_edges_via_info_score is used, rate of adaptation.
        Suggested range > 1.
        :param beta: Determines if info is seeded by diversity of connexions, how much (exponential factor, \in (0,1) )
        :param gamma: Determines if edges are reweighed with a tendency to cluster
        :param q: Determines the fraction of the nodes each edge is incentivized to connect to.
        """
        super().__init__(num_nodes, value_per_nugget, edge_weighting_exp_rate, take_the_best_reward_rate, rate_of_edge_adaptation, beta, gamma, q)

    def evaluate_effective_distances(self, source, source_reward, multiple_path_eff_dist, parameter=1, timestep=-1, rounding=3):
        """
        Defaults to random walker effective distance metric unless given multiple_path_eff_dist=True
        returns array of effective distances to each node (from source) according to effective dist library´methods
        TODO: eff_dis = ((# edges)^\alpha)/sum_weights
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
            normalized_A = utility_funcs.matrix_normalize(self.A[timestep], row_normalize=True)
            eff_dists = self.get_eff_dist(adjacency_matrix=normalized_A, random_walk=True, source=source, parameter=parameter)

        assert np.isclose(eff_dists[source], 0, rtol=1e-10), f'Source has nonzero effective distance of {eff_dists[source]}'
        eff_dists = np.delete(eff_dists, source)  # awkward deletion/insertion to ensure min search of remaining eff_distances
        eff_dists = np.insert(eff_dists, source, min(eff_dists)/source_reward)
        return eff_dists

    def set_node_values_as_eff_dists(self, exp_decay_param, source_reward, multiple_path):
        eff_dists = np.array(self.evaluate_effective_distances(self.starting_node, parameter=exp_decay_param, source_reward=source_reward, multiple_path_eff_dist=multiple_path))
        self.eff_dist_history.append(eff_dists)
        self.nodes[-1] = eff_dists

    def seed_info_conditional(self, constant_source_node):
        if self.beta:
            self.seed_info_by_diversity_of_connections()
        elif isinstance(constant_source_node, bool) & constant_source_node:
            self.seed_info_constant_source(0)   # Just to ensure seeding if set == True, it'll work without setting the constant seed to be a specific node
        elif not constant_source_node:
            self.seed_info_random()
        else:
            self.seed_info_constant_source(constant_source_node)

    def run(self, num_runs, update_interval=1, exp_decay_param=0.4, source_reward=1.6, constant_source_node=False, multiple_path=False, equilibrium_distance=50, verbose=False):
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
            if self.rate_of_edge_adaptation:
                print(f'Reweight_edges_via_info_score algorithm, with {self.rate_of_edge_adaptation} as the exponential scaling/rate of edge adaptation')
            elif self.take_the_best_reward_rate:
                print(f'Take the best edge reweighing algorithm, with {self.take_the_best_reward_rate} reward rate')
            elif self.gamma:
                print(f'Reweight_edges_with_clustering was used, with gamma of {self.gamma}')
            else:
                print(f'Edges were reweighed by simply adding the product of their edge value and origin node (from which they were directed) to their value')

        # Body of run function
        self.A = np.vstack((self.A, [self.A[-1]]))  # so that initial values (before initial update) are preserved
        for i in range(0, num_runs):
            self.seed_info_conditional(constant_source_node)
            self.set_node_values_as_eff_dists(exp_decay_param, source_reward, multiple_path)
            if i % update_interval == 0:
                self.update_edges()
                # so the next values may be overwritten, we start with 0 node values.
                self.nodes = np.vstack((self.nodes, np.zeros((1, self.num_nodes))))
                self.A = np.vstack((self.A, [self.A[-1]]))
            if verbose:
                utility_funcs.print_run_percentage(i, num_runs, 17)
            if self.A.shape[0] > equilibrium_distance:
                if np.all(np.isclose(self.A[-1], self.A[-equilibrium_distance], rtol=1e-5)):
                    print(f'Equilibrium conditions met after {i} runs, run halted.')
                    break  # Automatic break if equilibrium is reached. Lets run times be arb. large for MC parameter search
        self.A = np.delete(self.A, -1, axis=0)
        self.nodes = self.nodes[:-1]
        if verbose:
            print_run_methods()

    # Observables:
    def eff_dist_diff(self, eff_dist_to_all=False, eff_dist_to_distribution=False, multiple_path_eff_dist=False, source_reward=1.6, higher_order_paths_suppression=1):
        if eff_dist_to_all:
            return np.mean(self.evaluate_effective_distances(source=self.source_node_history[0], source_reward=source_reward,
                                                 multiple_path_eff_dist=multiple_path_eff_dist,
                                                 parameter=higher_order_paths_suppression, timestep=0)
                    - self.evaluate_effective_distances(source=self.source_node_history[-1], source_reward=source_reward,
                                                 multiple_path_eff_dist=multiple_path_eff_dist,
                                                 parameter=higher_order_paths_suppression, timestep=-1))
        # if eff_dist_to_distribution: #How is the distribution given? Presumably as a generator fct...
        #     eff_dist_to_all = (self.evaluate_effective_distances(source=None, source_reward=source_reward,
        #                                               multiple_path_eff_dist=multiple_path_eff_dist,
        #                                               parameter=higher_order_paths_suppression, timestep=0)
        #                        - self.evaluate_effective_distances(source=None, source_reward=source_reward,
        #                                                 multiple_path_eff_dist=multiple_path_eff_dist,
        #                                                 parameter=higher_order_paths_suppression, timestep=-1))
        else:
            return np.mean(self.eff_dist_history[0] - self.eff_dist_history[-1])

