import numpy as np
import random
import utility_funcs
import networkx as nx
import effective_distance as ed
import hierarchy_coordinates as hc
import efficiency_coordinates as ef

from scipy.sparse.linalg import inv
from scipy.sparse import diags, eye, csc_matrix
import time

import plotter

np.random.seed(42)
random.seed(42)

# Global Variables for default run values
update_interval_val = 1
source_reward_val = 2.6
equilibrium_distance_val = 100
equilibrium_span_val = 0  # virtually never triggers for val > 0


class Graph:
    _source_node = None  # holds the source node for each run, reset after every run.
    _singular_fundamental_matrix_errors = 0  # keeps running count of would be log zero errors (seen below) and replaces 0s with 1e-100:
    # RuntimeWarning: divide by zero encountered in log: RWED = -np.log(Z.dot(D).toarray()) Errors

    def __init__(self, num_nodes, edge_conservation_coefficient=None, selectivity=None,
                 reinforcement_info_score_coupling=True,
                 positive_eff_dist_and_reinforcement_correlation=False, eff_dist_is_towards_source=False,
                 nodes_adapt_outgoing_edges=False,
                 incoming_edges_conserved=True, undirected=False, global_eff_dist_diff=False, eff_dist_diff=False,
                 mean_eff_dist=False, ave_nbr=False, cluster_coeff=False, shortest_path=False,
                 degree_dist=False, hierarchy_coordinates=False, efficiency_coords=False):
        """
        Initialization; the 'extra' dimensionality (i.e. 1 for nodes, A) are there to dynamically store their history
        via use of vstack later on. To minimize variables, many features are activated by specifying their relevant
        macro delta, e.g. diversity based seeding by setting beta != None
        :param num_nodes: Number of nodes in network. Remains constant
        :param edge_conservation_coefficient: Tune between 0 (edge reinforcement is entirely based on eff_dist) and 1 (edge reinforcement is entirely based on extant edge values)
        :param selectivity: Determines if reweight_edges_via_info_score is used.
        Takes the lowest fraction \in (0,1) of edge values for each node and rather than reinforcing them, redistributes
         their reward to that node's oher edges proportional to their standing value.
        :param reinforcement_info_score_coupling if False decouples the above, leading to the same constant reward for the remaining node's edges
        """
        self.num_nodes = num_nodes
        self.edge_conservation_coefficient = edge_conservation_coefficient  # tunes info score between 0 (pure eff. dist. dependence) and 1 (pure edge dependence)
        self.fraction_infoscore_redistributed = selectivity  # rescaling of all rewards in reweight_edges_via_info_score
        self.reinforcement_infoscore_coupling = reinforcement_info_score_coupling  # if False, sparse matricies divolve into
        self.positive_eff_dist_and_reinforcement_correlation = positive_eff_dist_and_reinforcement_correlation
        self.eval_eff_dist_to_source = eff_dist_is_towards_source
        self.nodes_adapt_outgoing_edges = nodes_adapt_outgoing_edges
        self.conserve_incoming = incoming_edges_conserved
        self.undirected = undirected

        self.nodes = np.zeros((1, num_nodes))  # node values (and history of via the first dimension)
        self.A = np.zeros((1, self.num_nodes, self.num_nodes))  # Adjacency matrix (and history of)
        self.source_node_history = []
        self.eff_dist_to_source_history = []

        # Observables:
        self.observed_observables = {
            'ave_nbr': bool(ave_nbr),
            'global_eff_dist_diff': bool(global_eff_dist_diff),
            'mean_eff_dist': bool(mean_eff_dist),
            'eff_dist_diff': bool(eff_dist_diff),
            'cluster_coeff': bool(cluster_coeff),
            'shortest_path': bool(shortest_path),
            'degree_dist': bool(degree_dist),
            'efficiency_coordinates': bool(efficiency_coords),
            'hierarchy_coordinates': bool(hierarchy_coordinates),
        }
        if ave_nbr: self.ave_nbr_var = True
        if global_eff_dist_diff: self.global_eff_dist_diff = True
        if mean_eff_dist: self.mean_eff_dist = True
        if eff_dist_diff: self.source_eff_dist_diff = True
        if cluster_coeff: self.ave_cluster_coeff = True
        if shortest_path: self.ave_shortest_path = True
        if degree_dist: self.degree_dist = True
        if hierarchy_coordinates:
            self.linear_threshold_hierarchy_coordinates = True
            self.exp_threshold_hierarchy_coordinates = True
        if efficiency_coords:
            self.efficiency_coordinates = True

    # Edge Initialization: ------------------------------------------------------------------------------------------
    def sparse_random_edge_init(self, nonzero_edges_per_node=1, connected=True, even_distribution=True,
                                undirectify=False):
        """
        Randomly seeds Adjacency matrix with 1 fully weighted edge (or 2 0.5 weighted for nonzero_edges_per_node=2, etc)
        Presently allows for overlap of random edge assignment in the case of nonzero_edges_per_node > 1.
        :param nonzero_edges_per_node: int, number of edges per node
        """
        seed_amount = np.round((1 / nonzero_edges_per_node), 15)
        for node in range(0, self.num_nodes):
            range_excluding_self = list(range(0, self.num_nodes))
            range_excluding_self.remove(node)  # eliminates A_ij diagonals (self-referential looping 'edges')
            for edge_assigned in range(0, nonzero_edges_per_node):
                self.A[-1][node][random.choice(range_excluding_self)] += seed_amount

        if connected:
            zero_column_indices = np.where(self.A[-1].sum(axis=0) == 0)[0]
            for zero_column in zero_column_indices:
                max_column = np.argmax(self.A[-1].sum(axis=0))  # Necessarily re-evaluated for every replacement, as the row sums change
                row_chosen = random.choice([row for row in np.where(self.A[-1][:, max_column] > 0)[0] if row != zero_column])
                self.A[-1][row_chosen][zero_column] += seed_amount
                self.A[-1][row_chosen][max_column] -= seed_amount

        for node in range(0, self.num_nodes):
            self.A[-1][node][node] = 0  # eliminates looping edges (i.e. no edge refers to itself -> simple graph)
            self.A[-1][node] /= self.A[-1][node].sum()  # normalizes each node's total (incoming) weights to 1

        if even_distribution:
            self.A[-1] = utility_funcs.evenly_distribute_matrix_values(self.A[-1],
                                                                       by_row=True)  # To ensure outgoing edges all have the same initial weight
        if undirectify:
            self.A[-1] = utility_funcs.undirectify(self.A[-1])

    def uniform_random_edge_init(self, undirectify=False):
        """Most general case, fills Adjacency matrix with uniform random values and normalizes them"""
        self.A[-1] = np.random.rand(self.num_nodes,
                                    self.num_nodes)  # makes latest A a n x n adjacency matrix filled with rand(0,1)
        for node in range(0, self.num_nodes):
            self.A[-1][node][node] = 0  # eliminates looping edges (i.e. no edge refers to itself -> simple graph)
            self.A[-1][node] /= self.A[-1][node].sum()  # normalizes each node's total (incoming) weights to 1
        if undirectify:
            self.A[-1] = utility_funcs.undirectify(self.A[-1])

    def scale_free_edge_init(self, degree_exponent, min_k=1, equal_edge_weights=True, connected=True,
                             even_distribution=True, undirectify=False):
        """
        Generates degree sequence based on probability of connecting to k nodes: p(k) = k^-degree_exponent,
        spreads connections uniformly across connections, with the number of connections per node dependant on above
        :param degree_exponent: determines scale for edge initialization, often written as \gamma in the literature
        As implemented, the range of k values in p(k) are given in line with indicies.
        """
        num_nodes = self.A.shape[1]
        degree_sequence = [pow(k, -degree_exponent) for k in range(1, num_nodes + 1)]
        degree_sequence = [degree_sequence[i] / sum(degree_sequence) for i in
                           range(len(degree_sequence))]  # Normalization via Riemann Zeta function required?
        # print(f'deg-seq: {degree_sequence} sum: {sum(degree_sequence)}')
        if min_k > 0:
            degree_sequence = [int(np.round(degree_sequence[node] * num_nodes)) if int(
                np.round(degree_sequence[node] * num_nodes)) > min_k else min_k for node in range(num_nodes)]
        else:
            degree_sequence = [int(np.round(degree_sequence[node] * num_nodes)) for node in range(num_nodes)]
        # print(f'deg-seq: {degree_sequence} sum: {sum(degree_sequence)}')

        self.A = np.zeros((1, self.A[-1].shape[0], self.A[-1].shape[1]))  # Reinitialization to zeros for following:
        columns_visited = set()
        for from_node in range(num_nodes):
            if connected:
                if len(set(set(range(num_nodes)) - columns_visited.union({from_node}))) >= degree_sequence[from_node]:
                    to_nodes = random.choices(list(set(range(num_nodes)) - {from_node}.union(columns_visited)),
                                              k=degree_sequence[from_node])
                    columns_visited = columns_visited.union(to_nodes)
                else:
                    to_nodes = random.choices(list(set(range(num_nodes)) - {from_node}), k=degree_sequence[from_node])
            else:
                to_nodes = random.choices(list(set(range(num_nodes)) - {from_node}), k=degree_sequence[from_node])
            for to_node in to_nodes:
                if equal_edge_weights:
                    self.A[-1][from_node][to_node] = 1 / degree_sequence[from_node]
                else:
                    self.A[-1][from_node][to_node] = np.random.rand()

        # simply ensuring connectedness
        if connected and np.any(np.where(np.sum(self.A[-1], axis=1) == 0)):
            columns_with_multiple_edges = []
            columns_without_any_edges = []
            for column in range(self.num_nodes):
                _column = np.array(np.where(self.A[-1][:, column] > 0))[0]
                if _column.shape[0] > 1:
                    _column.append(
                        column)  # So that the first value indicates the index of the column, the next values its non-zero row indices
                    columns_with_multiple_edges.append(_column)
                elif _column.shape[0] == 0:
                    columns_without_any_edges.append(column)

            # Ensuring connectedness arises through shifting existing degree_sequence
            index = 0
            while len(columns_without_any_edges) > 0:
                if len(columns_with_multiple_edges[index]) > 2:
                    edge_column = columns_with_multiple_edges[index][0]
                    edge_row = columns_with_multiple_edges[index][1]
                    self.A[-1][edge_row][columns_without_any_edges[-index]] += self.A[-1][edge_row][edge_column]
                    self.A[-1][edge_row][edge_column] = 0
                    del columns_with_multiple_edges[index][1]
                    del columns_without_any_edges[-index]
                    index += 1
                else:
                    index += 1
                if index > self.num_nodes:
                    print(f'A too sparse to be connected, breaking out with A of {self.A[-1]} (try setting min_k >= 1)')
                    break

        for from_node in range(self.A[-1].shape[0]):
            if self.A[-1][from_node].sum() > 0:
                self.A[-1][from_node] /= self.A[-1][
                    from_node].sum()  # normalizes each node's total (incoming) weights to 1

        if even_distribution:
            self.A[-1] = utility_funcs.evenly_distribute_matrix_values(self.A[-1],
                                                                       by_row=True)  # To ensure outgoing edges all have the same initial weight
        if undirectify:
            self.A[-1] = utility_funcs.undirectify(self.A[-1])

    def nx_scale_free_edge_init(self, degree_exponent, min_k=1, undirectify=False):
        # in and out degrees are considered to be the same by default
        num_nodes = self.A.shape[1]
        degree_sequence = [pow(k, -degree_exponent) for k in range(1, num_nodes + 1)]
        degree_sequence = [degree_sequence[i] / sum(degree_sequence) for i in
                           range(len(degree_sequence))]  # Normalization via Riemann Zeta function required?
        # print(f'deg-seq: {degree_sequence} sum: {sum(degree_sequence)}')
        if min_k > 0:
            degree_sequence = [int(np.round(degree_sequence[node] * num_nodes)) if int(
                np.round(degree_sequence[node] * num_nodes)) > min_k else min_k for node in range(num_nodes)]
        else:
            degree_sequence = [int(np.round(degree_sequence[node] * num_nodes)) for node in range(num_nodes)]

        nx_graph = nx.directed_configuration_model(degree_sequence, degree_sequence)
        self.A[-1] = nx.to_numpy_array(nx_graph)

        for from_node in range(self.A[-1].shape[0]):
            if self.A[-1][from_node].sum() > 0:
                self.A[-1][from_node] /= self.A[-1][
                    from_node].sum()  # normalizes each node's total (incoming) weights to 1
        if undirectify:
            self.A[-1] = utility_funcs.undirectify(self.A[-1])

    def nx_scale_free_edge_init_unconnected(self, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0,
                                            create_using=None, seed=None):
        # in and out degrees are considered to be the same by default
        nx_graph = nx.scale_free_graph(self.A.shape[1], alpha=alpha, beta=beta, gamma=gamma, delta_in=delta_in, delta_out=delta_out,
                                       create_using=create_using, seed=seed)
        self.A[-1] = nx.to_numpy_array(nx_graph)

        for from_node in range(self.A[-1].shape[0]):
            if self.A[-1][from_node].sum() > 0:
                self.A[-1][from_node] /= self.A[-1][
                    from_node].sum()  # normalizes each node's total (incoming) weights to 1

    def barabasi_albert_edge_init(self, num_edges_per_new_node, seed=42, undirectify=False):
        nx_graph = nx.barabasi_albert_graph(n=self.A[-1].shape[0], m=num_edges_per_new_node, seed=seed)
        self.A[-1] = nx.to_numpy_array(nx_graph)
        for from_node in range(self.A[-1].shape[0]):
            if self.A[-1][from_node].sum() > 0:
                self.A[-1][from_node] /= self.A[-1][
                    from_node].sum()  # normalizes each node's total (incoming) weights to 1
        if undirectify:
            self.A[-1] = utility_funcs.undirectify(self.A[-1])

    def edge_initialization_conditional(self, edge_init, undirectify=False):
        if edge_init is None:
            self.uniform_random_edge_init(undirectify=undirectify)
        if isinstance(edge_init, int):
            self.sparse_random_edge_init(nonzero_edges_per_node=edge_init, undirectify=undirectify)
        if isinstance(edge_init, float):
            self.scale_free_edge_init(degree_exponent=edge_init, min_k=1, undirectify=undirectify)
        if isinstance(edge_init, np.ndarray):
            assert len(edge_init.shape) == 2 and edge_init.shape[0] == edge_init.shape[
                1], f'edge_init as np.array must be a square (2d) matrix. Now edge_init is: \n {edge_init}'
            self.A[-1] = edge_init

    def imported_A_edge_init(self, A, undirectify=False):
        assert len(A[0]) == self.num_nodes, "imported A must be of dim == num nodes"
        assert len(A.shape) == 2, "imported A must be square (dim==2)"
        assert len(A[0]) == len(A[:, 0]), "imported A must be square"
        assert [A[i][i] for i in range(len(A[0]))] == [0]*len(A[0]), "Imported A must be simple (i.e. no self loops, i.e. diagonal are all 0)"
        self.A[-1] = A
        if undirectify:
            self.A[-1] = utility_funcs.undirectify(self.A[-1])

    # Information Seeding: ------------------------------------------------------------------------------------------
    def seed_info_random(self):
        """
        Sets source to be a random node
        """
        self._source_node = None  # resets starting nodes such that new seed_info call will not conflict
        seeded_node = np.random.default_rng().integers(self.num_nodes)
        self._source_node = seeded_node
        self.source_node_history.append(seeded_node)

    def seed_info_constant_source(self, constant_source_node):
        """
        Seeds the source as the indexed value
        :param constant_source_node: int, index of constant source
        """
        assert isinstance(constant_source_node, int) & constant_source_node >= 0 & constant_source_node <= \
               self.nodes.shape[
                   1], f'Please choose constant source node to be in range of num_nodes, i.e. in (0, {self.num_nodes})'
        self._source_node = constant_source_node
        self.source_node_history.append(constant_source_node)

    def seed_info_by_diversity_of_connections(self, beta):
        """
        Computes standard deviation as a (negatively correlated) metric for diversity of connection between nodes,
        then uses this to distribute 'nuggets' of information (via canonical ensemble of standard deviation).
        Potentially variance would be faster (no sqrt) and better, changing the effect of connectedness.
        """
        self._source_node = None  # resets starting nodes such that new seed_info call will not conflict
        exp_stds = []
        for node_edges in self.A[-1][:]:
            exp_stds.append(np.exp(-beta * node_edges.std()))  # sum of e^(-\beta \sigma_i) for i \in node[weights]
        std_partition = sum(exp_stds)
        while self._source_node is None:
            seeded_node = np.random.randint(0, self.nodes[-1].size)
            if random.uniform(0, std_partition) < exp_stds[seeded_node] / std_partition:
                self._source_node = seeded_node
                self.source_node_history.append(seeded_node)

    def seed_info_normal_distribution(self, mean, normal_sigma_coefficient):
        """
        Normally distributes source, according to a distribution centered around the (input) mean
        :param mean: Mean of nodes; defaults to num_nodes/2 (center of distribution)
        :param normal_sigma_coefficient: coefficient of normal sigma; e.g. 1 yields std normal sigma, 0.5 a tighter sigma (increased kurtosis)
        """
        if mean is None: Mean = self.nodes.shape[1] / 2
        Sigma = normal_sigma_coefficient * np.std(np.arange(self.nodes.shape[1]))
        self._source_node = None

        seeded_node = np.int(np.round(np.random.default_rng().normal(Mean, Sigma)) - 1)
        while seeded_node < 0 or seeded_node >= self.nodes.shape[1]:
            seeded_node = np.int(np.round(np.random.default_rng().normal(Mean, Sigma)) - 1)
        self._source_node = seeded_node
        self.source_node_history.append(seeded_node)

    def seed_info_power_law_distribution(self, power_law_exponent):
        """
        Seeds via a power law distribution, weighted concentration about higher-indexed nodes.
        :param power_law_exponent: Determines degree of power law weight.
        """
        self._source_node = None
        seeded_node = int(np.round(np.random.power(power_law_exponent, 1) * self.nodes[-1].size) + 0.5) - 1
        self._source_node = seeded_node
        self.source_node_history.append(seeded_node)

    def seed_info_conditional(self, constant_source_node, num_shifts_of_source_node, num_runs, sigma,
                              power_law_exponent, beta, index):
        """
        Determines which info_seeding function is selected. Used to abbreviate run function.
        Arguments are all mutually incompatible, save num_runs and index.
        :param constant_source_node: bool or int, set index of constant seeded node.
        :param num_shifts_of_source_node: int: sets number of shifts over the course of the entire run time.
        :param num_runs: int: number of runs
        :param sigma: float, if non-zero leads to normally distributed source seeding
        :param power_law_exponent: float, if non-zero leads to power law distributed source seeding
        :param beta: float, determines beta.
        :param index: int, run index
        """
        assert bool(constant_source_node) + bool(num_shifts_of_source_node) + bool(sigma) + bool(
            power_law_exponent) + bool(
            beta) < 2, 'Incompatible arguments for info_seeding conditional, choose one method only per run call'
        if beta:
            self.seed_info_by_diversity_of_connections(beta)
        elif num_shifts_of_source_node:
            # assert num_shifts_of_source_node <= self.num_nodes, "more changes to constant source node than number of nodes. Set constant_source_node to false to activate continues random seeding"
            if (index % int((num_runs / num_shifts_of_source_node))) == 0:
                if num_shifts_of_source_node > self.num_nodes or list(set(range(len(self.nodes[-1]))) - set(self.source_node_history)) == []:
                    self.seed_info_random()
                else:
                    self.seed_info_constant_source(random.choice(list(set(range(len(self.nodes[-1]))) - set(self.source_node_history))))
            else:
                self.source_node_history.append(self._source_node)
        elif sigma:
            self.seed_info_normal_distribution(mean=None, normal_sigma_coefficient=sigma)
        elif power_law_exponent:
            self.seed_info_power_law_distribution(power_law_exponent=power_law_exponent)
        elif not constant_source_node or sigma or power_law_exponent:
            self.seed_info_random()
        elif isinstance(constant_source_node, bool) & constant_source_node:
            self.seed_info_constant_source(
                0)  # Just to ensure seeding if set == True, it'll work without setting the constant seed to be a specific node
        else:
            self.seed_info_constant_source(constant_source_node)

    # Info Score Evaluation: ----------------------------------------------------------------------------------------
    def evaluate_info_score(self, from_node_index, node_sum):
        """
         By design, x is the (scaled) proportion of connected node value to all node connections
         (scaled proportional to effective distance, thus the inverse relation via (1-x) while
         y is the edge's value (i.e. probability od transition, given normalization of A columns)
        To absolve the model of arbitrarity w.r.t. info score space geometry, we let the control delta vary between two extremes (of uncoupled linear relation w.r.t. x, y)
        To make the info_score space have an inverse correlation between Eff_dist (node value) and edge value (\in A):
        Z = pow(x, (alpha - 1)) * pow(y, alpha) || alpha : Coupling delta = edge_conservation_coefficient

        For general responsiveness to parameters rather than coupling between them:
        # return [pow((self.A[-1][node_index][from_node_index] / (self.nodes[-1][node_index] / node_sum)), self.eff_dist_and_edge_response) for node_index in range(self.nodes.shape[1])]
        """
        if self.positive_eff_dist_and_reinforcement_correlation:
            if not self.nodes_adapt_outgoing_edges:
                return [(pow((self.nodes[-1][node_index] / node_sum), (1 - self.edge_conservation_coefficient)) * pow(
                    self.A[-1][node_index][from_node_index], self.edge_conservation_coefficient)) for node_index in
                        range(self.nodes.shape[1])]
            else:
                return [(pow((self.nodes[-1][node_index] / node_sum), (1 - self.edge_conservation_coefficient)) * pow(
                    self.A[-1][from_node_index][node_index], self.edge_conservation_coefficient)) for node_index in
                        range(self.nodes.shape[1])]
        else:
            if not self.nodes_adapt_outgoing_edges:
                return [(pow((self.nodes[-1][node_index] / node_sum), (self.edge_conservation_coefficient - 1)) * pow(
                    self.A[-1][node_index][from_node_index], self.edge_conservation_coefficient)) for node_index in
                        range(self.nodes.shape[1])]
            else:
                return [(pow((self.nodes[-1][node_index] / node_sum), (self.edge_conservation_coefficient - 1)) * pow(
                    self.A[-1][from_node_index][node_index], self.edge_conservation_coefficient)) for node_index in
                        range(self.nodes.shape[1])]

    def reweight_info_score(self, info_score, info_score_sum):
        """
        Reweights info_score by redistributing ____ fraction of info_score overall reward to the remaining (1-___)
        fraction of highest info_scores proportional to their respective percentage of total remaining info_score value
        By default, only redistributes the base info_score_sum value, but through coefficient could be more/less
        :param info_score_sum: simply passed to avoid recomputing sum, though only more efficient for larger node_numbers
        """
        if not self.fraction_infoscore_redistributed == 1:
            cutoff_val = sorted(info_score)[int(self.fraction_infoscore_redistributed * len(info_score))]
        else:
            cutoff_val = sorted(info_score)[-1]

        if self.reinforcement_infoscore_coupling:
            reduced_info_score = [val if val >= cutoff_val else 0 for val in info_score]
        else:
            reduced_info_score = [1 if val > cutoff_val else 0 for val in info_score]

        reduced_info_score_sum = sum(reduced_info_score)
        if reduced_info_score_sum == 0:
            # print(f'reduced_info_score: {reduced_info_score} \n cutoff value: {cutoff_val}')
            reduced_info_score = [0] * len(info_score)
            reduced_info_score[utility_funcs.argmax(info_score)] = sorted(info_score)[-1]
            reduced_info_score_sum = sorted(info_score)[-1]
        return [info_score_sum * (val / reduced_info_score_sum) for val in reduced_info_score]

    # Edge Reweighing: ----------------------------------------------------------------------------------------------
    def reweight_edges_via_info_score(self, null_run=False):
        """
        Normalizes then adds re-weighted info_scores to edge values.
        """
        info_scores = np.zeros(self.A[-1].shape)
        node_sum = self.nodes[-1].sum()
        for from_node in range(0, self.nodes[-1].size):
            if not null_run:
                info_score = self.evaluate_info_score(from_node, node_sum)
            else:
                info_score = self.A[-1][:, from_node]*self.edge_conservation_coefficient + np.random.rand(self.A[-1].shape[1]) * (1 - self.edge_conservation_coefficient)
            info_score /= np.sum(info_score)  # Optional info_score normalization.
            if not self.nodes_adapt_outgoing_edges:
                info_scores[:, from_node] = self.reweight_info_score(info_score, 1)  # replace 1 with sum(info_score) if not normalized [below]
                # info_scores[:, from_node] = self.reweight_info_score(info_score, np.sum(info_score))  # replace 1 with sum(info_score) if not normalized
            else:
                info_scores[from_node, :] = self.reweight_info_score(info_score, 1)  # as updating outwardly directed nodes means updating rows.
        self.A[-1] += info_scores

    def update_edges(self, null_run=False):
        """
        We may use the node values directly, as assigned by effective distance methods, to determine their effect on
         the node they were connected to. Even though this computational mechanic is the opposite of the
         conceptualization, it yields the same results.
        The normalization (along incoming edges) is where the conservation of edge weight applies,
         negatively effecting those not reinforced.
        """
        self.reweight_edges_via_info_score(null_run=null_run)

        # Normalization (columns of A, incoming edges)
        for node in range(0, self.A.shape[1]):
            if self.conserve_incoming:
                incoming_edge_sum = self.A[-1][:, node].sum()
                if incoming_edge_sum > 0:
                    self.A[-1][:, node] /= incoming_edge_sum  # normalizes each node's total INCOMING weights to 1
                # For (unconnected) sparse networks, there will be some columns (outgoing edges) which sum to zero. (thus the conditional)
            else:
                outgoing_edge_sum = self.A[-1][node, :].sum()
                if outgoing_edge_sum > 0:
                    self.A[-1][node, :] /= outgoing_edge_sum  # normalizes each node's total OUTGOING weights to 1
        if self.undirected:
            self.A[-1] = utility_funcs.undirectify(self.A[-1], average_connections=True)  # averages reciprocal connections (e_ji, e_ji --> (e_ij + e_ji)/2))

    # Effective Distance Evaluation: --------------------------------------------------------------------------------
    def get_eff_dist(self, adjacency_matrix=None, multiple_path=False, source=None, target=None, parameter=1):
        """
        Returns effective distance based on the effective distance library built by Andreas Koher. Random walk
        estimations require row normalized adjacency matrix values.
        :param adjacency_matrix: 2d np.array defaults to latest adjacency matrix, but allows free choice of adjacency matrix
        :param multiple_path: bool, if True uses the multiple path effective distance (far slower due to lack of analytic solution, as with RWED)
        :param source: int, source index. If None, all to all effective distance is computed (thus yielding 2d, not 1d matrix)
        :param target: target, to which all source nodes calculate their distance
        TODO: So far, the simulation has reversed the intuitive distance to source, and instead been calculating distance from source to all other nodes.
        """
        if adjacency_matrix is None:
            adjacency_matrix = self.A[-1]
        if not multiple_path:
            RWED, errors = ef.Random_Walker_Effective_Distance(A=adjacency_matrix, source=source, target=target, parameter=parameter, via_numpy=False, return_errors=True)
            self._singular_fundamental_matrix_errors += errors
            return RWED
        else:
            return ed.EffectiveDistances(np_array=adjacency_matrix).get_multiple_path_distance(source=source,
                                                                                               target=target,
                                                                                               parameter=parameter)

    def evaluate_effective_distances(self, source_reward, parameter, multiple_path_eff_dist, source=None, timestep=-1,
                                     rounding=3, add_to_history=False):
        """
        Defaults to random walker effective distance metric unless given MPED=True
        returns array of effective distances to each node (from source) according to effective dist library´methods
        :param source_reward: float, info score bonus relative to the best info_score of the other nodes.
        :param parameter: exponential suppression of higher order paths, directly from effective_distance library: log[ (alpha-beta)/kappa - lambda ]
        :param multiple_path_eff_dist: bool, determines if the MPED algorithm is used.
        :param source: start point of efective distance search through directed network.
        :param timestep: int, timestep of effective distance calculation over A. Defaults to latest timestep.
        :param rounding: relevant only for MPED, cutoff in decimal digits for paths to be considered in MPED evaluation.
        """
        if np.all(np.isclose(self.A[timestep], 0, rtol=1e-15)):  # This is for null normed results which may return 0 \in \mathbb{R}^{NxN}
            if source is not None:
                eff_dist = np.array([np.nan] * self.A[timestep].shape[0])
            else:
                eff_dist = np.empty((self.A[timestep].shape[0], self.A[timestep].shape[1]))
                eff_dist[:] = np.nan
            return eff_dist

        if multiple_path_eff_dist:
            """
            Rounding of A to ensure that when the matrix is converted to a nx graph, near zero (O(^rounding)) paths 
            will be eliminated (by being set to zero before the nx conversion) from the nx all paths algorithm.
            The inversion is necessary to ensure that large probabilities of RW steps between nodes for the default A 
            are translated to small distances according to the MPED edge algorithm. 
            """
            inv_A = utility_funcs.return_one_over_2d_matrix(np.round(self.A[timestep], rounding))
            # inv_A = self.A[timestep] + 1  # alternate method for making el in A > 1 for MPED, doesn't nonlinearly penalize low probabilities.
            eff_dists = self.get_eff_dist(adjacency_matrix=inv_A, multiple_path=True, source=source,
                                          parameter=parameter)
            # should be negative of end result eff dist(as algorithm uses - log? Or not inverted, if so...)
        else:
            # pre-normalize rows (as both columns and rows must be normalized for RWED)
            row_sums = self.A[timestep].sum(axis=1)
            normalized_A = np.array([self.A[timestep][node, :] / row_sums[node] for node in range(self.A[timestep].shape[0])])
            # normalized_A = np.round(utility_funcs.matrix_normalize(self.A[timestep], row_normalize=True), 20)
            if not self.eval_eff_dist_to_source:
                eff_dists = self.get_eff_dist(adjacency_matrix=normalized_A, multiple_path=False, source=source,
                                              parameter=parameter)  # Evaluates eff. dist. FROM source
            else:
                eff_dists = self.get_eff_dist(adjacency_matrix=normalized_A, multiple_path=False, target=source,
                                              parameter=parameter)  # Evaluates eff. dist. TO source
        if source is None:
            # if source is none, the eff_dist library defaults to an all-to-all eff_dist measure
            for node_index in range(self.nodes[-1].size):
                min_val = eff_dists[node_index][
                    np.argpartition(eff_dists[node_index], 1)[1]]  # yields non-zero (non-source) min eff_dist
                eff_dists[node_index][node_index] = (min_val / source_reward)
            return eff_dists
        else:
            assert np.isclose(eff_dists[source], 0,
                              rtol=1e-10), f'Source has nonzero effective distance of {eff_dists[source]}'
            eff_dists = np.delete(eff_dists,
                                  source)  # awkward deletion/insertion to ensure min search of remaining eff_distances
            eff_dists = np.insert(eff_dists, source, min(eff_dists) / source_reward)
            if add_to_history: self.eff_dist_to_source_history.append(eff_dists)  # Must come before normalization otherwise sum will always be 1
            return eff_dists / np.sum(eff_dists)

    # Utility Functions: --------------------------------------------------------------------------------------------
    def write_graph_as_xml(self, timestep=-1, path=None):
        """
        saves Graph class object as xml for use in Gephi, similiar.
        :param timestep: int, point in graph's history in which it is be saved
        :param path: save path of graph, title.
        """
        if path is None:
            path = f"graph_at_{timestep}"
        if path is None and timestep == -1:
            path = f"final_graph"
        nx_G = nx.to_directed(nx.from_numpy_matrix(np.array(self.A[timestep]), create_using=nx.DiGraph))
        nx.write_graphml(nx_G, f'{path}.graphml')

    def convert_to_nx_graph(self, timestep=-1):
        """
        Converts to networkX graph, as used for networkX graph metrics.
        """
        return nx.to_directed(nx.from_numpy_matrix(np.array(self.A[timestep]), create_using=nx.DiGraph))

    def convert_history_to_list_of_nx_graphs(self, verbose=False):
        """
        Converts graph at every point in its evolution (for every timestep) into nx_graphs.
        :param verbose: bool, if True, displays progress
        """
        nx_graph_history = []
        if verbose:
            print(f'Beginning conversion of all graphs to nx_graphs...')
            start_time = time.time()
            for i in range(self.A.shape[0]):
                nx_graph_history.append(self.convert_to_nx_graph(i))
                utility_funcs.print_run_percentage(index=i, runs=self.A.shape[0])
            print(
                f'Conversion of all graphs to nx_graphs completed. It took {int((time.time() - start_time) / 60)} minutes, {np.round((time.time() - start_time) % 60, 2)}')
            return nx_graph_history
        else:
            return [self.convert_to_nx_graph(i) for i in range(self.A.shape[0])]

    def print_graph_properties(self):
        """
        Prints Graph properties to terminal, simply for clarity if needed.
        """
        print(f'Num_nodes: {self.num_nodes}')
        print(f'Edge conservation coefficient: {self.edge_conservation_coefficient}')
        print(
            f'Selectivity: {self.fraction_infoscore_redistributed}')  # rescaling of all rewards in reweight_edges_via_info_score
        if self.reinforcement_infoscore_coupling:
            print(f'Edge reinforcement is proportional to their info score')
        else:
            print(
                f'Edge reinforcement is NOT proportional to their info score values (they simply have to be in the remaining {1 - self.fraction_infoscore_redistributed} fraction of info scores to be reinforced)')
        if self.positive_eff_dist_and_reinforcement_correlation:
            print(
                f'Effective distance and edge reinforcement correlation is reversed (i.e. Effective distance is positively correlated with edge reinforcement, eschewing the source)')
        else:
            print('Effective distance and edge reinforcement correlation is NOT reversed')

    def get_num_errors(self):
        return self._singular_fundamental_matrix_errors

    # Observables: --------------------------------------------------------------------------------------------------
    def eff_dist_diff(self, all_to_all_eff_dist=False, overall_average=False, MPED=False, source_reward=2.6, delta=10):
        """
        Computes the difference in effective distance between two time steps of the same graph.
        :param all_to_all_eff_dist: Bool, determines if all to all effective distance is to be calculated.
        :param overall_average: Bool, if True averages the overall effective distance measure
        :param MPED: Bool, determines if the multiple path effective distance is used.
        :param source_reward: float, determines boost to effective distance score of source
        (as it cannot be 0, we divide the min of remaining nodes' effective distances by this value and assign it as the eff dist of the source)
        :param delta: float/int.
        """
        if all_to_all_eff_dist:
            initial = self.evaluate_effective_distances(source_reward=source_reward, parameter=delta,
                                                        multiple_path_eff_dist=MPED, source=None, timestep=0)
            final = self.evaluate_effective_distances(source_reward=source_reward, parameter=delta,
                                                      multiple_path_eff_dist=MPED, source=None, timestep=-1)
            return np.mean(final - initial)
        if overall_average:
            return np.mean(self.eff_dist_to_source_history)  # of course same as simply np.mean(eff_dist_history), but written so for clarity
        else:
            # return np.mean(self.eff_dist_to_source_history[0]) - np.mean(self.eff_dist_to_source_history[-1])
            return np.mean(self.eff_dist_to_source_history[0]) - np.mean(self.eff_dist_to_source_history[-1])

    def degree_distribution(self, timestep=-1):
        """
        Yields the OUTGOING degree distribution for individual nodes
        :param timestep: point at which to evaluate the degree distribution of the graph.
        """
        return np.round(np.sum(self.A[timestep], axis=1), 20)
        # return np.round(np.sum(self.A[timestep], axis=0), 20)

    def shortest_path(self, timestep=-1, source=None, target=None, reversed_directions=False, Adj_Matrix=None):

        def sum_weighted_path(A, path: list):
            return sum([A[path[i]][path[i + 1]] for i in range(len(path) - 1)])

        if Adj_Matrix is not None:
            Adj = np.array(1 - Adj_Matrix)
        else:
            Adj = np.array(1 - self.A[timestep])

        Adj = np.where(Adj == 0, 1e-100, Adj)

        inverted_weights_nx_graph = nx.to_directed(nx.from_numpy_matrix(Adj, create_using=nx.DiGraph))
        if reversed_directions:
            inverted_weights_nx_graph = inverted_weights_nx_graph.reverse(copy=False)

        # SPD_dic = dict(nx.shortest_path_length(inverted_weights_nx_graph, source=source, target=target, weight='weight'))
        shortest_paths = nx.shortest_path(inverted_weights_nx_graph, source=source, target=target, weight='weight')
        # if shortest_paths
        if source is None:
            if target is None:
                SPD = np.array([[sum_weighted_path(Adj, shortest_paths[s][t]) for s in range(self.nodes.shape[1])] for t in range(self.nodes.shape[1])])
            else:
                SPD = np.array([sum_weighted_path(Adj, shortest_paths[s][target]) for s in range(self.nodes.shape[1])])
        else:
            if target is None:
                SPD = np.array([sum_weighted_path(Adj, shortest_paths[source][t]) for t in range(self.nodes.shape[1])])
            else:
                SPD = sum_weighted_path(Adj, shortest_paths[source][target])
        return SPD

    def E_routing(self, timestep=-1, reversed_directions=False, normalize=True):  # Routing Efficiency
        return ef.E_rout(A=self.A[timestep], reversed_directions=reversed_directions, normalize=normalize)

    def E_diff(self, timestep=-2, normalize=True):  # Diffusion Efficiency
        return ef.E_diff(A=self.A[timestep], normalize=normalize)

    # Run Functions: -------------------------------------------------------------------------------------------------
    def add_observables(self, null=False, source_reward=2.6):
        # Stored Observables:  (redeclaration precludes necessity for re-initialization in ensemble run)
        if not null:
            # Source based observables
            if self.observed_observables['eff_dist_diff']:  # Compares first and last eff_dist values
                self.source_eff_dist_diff = float(self.eff_dist_diff(all_to_all_eff_dist=False, overall_average=False, source_reward=source_reward))
            if self.observed_observables['mean_eff_dist']:  # Averages all eff_dist_history
                self.mean_eff_dist = float(self.eff_dist_diff(all_to_all_eff_dist=False, overall_average=True, source_reward=source_reward))
        if self.observed_observables['global_eff_dist_diff']:  # Compares first and last all to all eff_dist values
            self.global_eff_dist_diff = float(self.eff_dist_diff(all_to_all_eff_dist=True, overall_average=False, source_reward=source_reward))
        if self.observed_observables['degree_dist']:
            self.degree_dist = np.var(self.degree_distribution(timestep=-1))
        if self.observed_observables['efficiency_coordinates']:
            self.efficiency_coordinates = np.array([ef.E_diff(A=self.A[-1], normalize=True), ef.E_rout(A=self.A[-1], normalize=True)])
        if self.observed_observables['hierarchy_coordinates']:
            self.linear_threshold_hierarchy_coordinates = np.array(hc.average_hierarchy_coordinates(A=self.A[-1], exp_threshold_distribution=None))
            self.exp_threshold_hierarchy_coordinates = np.array(hc.average_hierarchy_coordinates(A=self.A[-1], exp_threshold_distribution=0.5))
        # nx graph based observables:
        if self.observed_observables['ave_nbr'] or self.observed_observables['cluster_coeff'] or self.observed_observables['shortest_path']:
            nx_final_graph = self.convert_to_nx_graph(timestep=-1)
        if self.observed_observables['ave_nbr']:
            last_ave_nbr_deg = list(nx.average_neighbor_degree(nx_final_graph, source='in', target='in', weight='weight').values())
            self.ave_nbr_var = np.array(last_ave_nbr_deg).var()  # Variance of average neighbor degree over time
        if self.observed_observables['cluster_coeff']:
            self.ave_cluster_coeff = nx.average_clustering(nx_final_graph, weight='weight')
        if self.observed_observables['shortest_path']:
            self.ave_shortest_path = float(nx.average_shortest_path_length(nx_final_graph, weight='weight'))

    def simulate(self, num_runs, eff_dist_delta_param=1, constant_source_node=False, num_shifts_of_source_node=False,
                 seeding_sigma_coeff=False, seeding_power_law_exponent=False, beta=None, multiple_path=False,
                 equilibrium_distance=equilibrium_distance_val, update_interval=1, source_reward=source_reward_val,
                 verbose=False):
        """
        Run function. Sequentially seeds source node, sets node values to effective distance evaluation, updates edges and corresponding history.
        :param num_runs: Constant natural number, number of runs.
        :param update_interval: Number of seed steps per run (times information is seeded and diffused before reweighing edges)
        :param eff_dist_delta_param: determines exponential suppression of higher order paths for both RWED and MPED
        :param source_reward: Determines how much the source node is scaled, recommended values \in (1,2)
        :param constant_source_node: Sets seed node to be a given node (the integer given).  True defaults to 0th node.
        :param multiple_path: if True, uses multiple path effective distance algorithm. Otherwise, defaults to random walker effective distance algorithm
        :param equilibrium_distance: length at which two nearly equal A matricies are constantly compared to break run loop
        :param verbose: if True, details approximate completion percentage and run parameters, methods.
        :return: Returns nothing, updates graph values. Use plotter library to evaluate and graph observables
        """

        # Verbose functions:
        def print_run_methods():
            print(
                f'Parameter and Method Details:\nRan {num_runs} runs with {update_interval} seed steps per run, and sources were rewarded by a {source_reward} scaling')
            if isinstance(constant_source_node, bool) & constant_source_node is True:
                print(f'All information was seeded to the 0th node. (Constant Source)')
            elif isinstance(constant_source_node, int):
                print(f'All information was seeded to the {constant_source_node}th node. (constant source)')
            elif beta:
                print(
                    f'Information seeded proportional to diversity of connections, with partition exponent of beta = {self.beta}')
            elif not constant_source_node or self.beta:
                print(f'All information was seeded randomly.')

            if multiple_path:
                print(
                    f'The Multiple Path Effective Distance (MPED) was used, with higher order paths suppressed by a eff_dist_delta_param of {eff_dist_delta_param}')
            else:
                print(
                    f'The (analytic) Random Walker Effective Distance (RWED) was used, with an eff_dist_delta_param of {eff_dist_delta_param}.')

            print('Edge Update algorithm:')
            print(
                f'Edges were reweighted via Info score values, with edge conservation and selectivity values of {self.edge_conservation_coefficient}, {self.fraction_infoscore_redistributed}, respectively.')
            if self.reinforcement_infoscore_coupling:
                print(f'Reinforcement of edges was coupled to the edges\' info score values')
            if self.positive_eff_dist_and_reinforcement_correlation:
                print(
                    'There was a positive effective distance and edge reinforcement correlation, the opposite of the usual relation.')

        # Body of run function
        self.A = np.vstack((self.A, [self.A[-1]]))  # so that initial values (before initial update) are preserved
        equilibrium_span = equilibrium_span_val  # if a greater range of values between equilibrium distance ought be compared
        for i in range(0, num_runs):
            self.seed_info_conditional(constant_source_node, num_shifts_of_source_node, num_runs=num_runs,
                                       sigma=seeding_sigma_coeff, power_law_exponent=seeding_power_law_exponent,
                                       beta=beta, index=i)
            self.nodes[-1] += np.array(self.evaluate_effective_distances(source_reward, eff_dist_delta_param, multiple_path,
                                                  source=self._source_node, add_to_history=True))
            if i % update_interval == 0:
                self.update_edges()
                # so the next values may be overwritten, we start each run with 0 node values.
                self.nodes = np.vstack((self.nodes, np.zeros((1, self.num_nodes))))
                self.A = np.vstack((self.A, [self.A[-1]]))
            if verbose:
                utility_funcs.print_run_percentage(i, num_runs, 17)
            if equilibrium_distance and self.A.shape[0] > equilibrium_distance + equilibrium_span + 1:  # +1 for range starting at 0 offset
                # if np.all(np.array([np.allclose(self.A[-(ii+1)], self.A[-(equilibrium_distance+(ii+1))], rtol=1e-10) for ii in range(equilibrium_span)])):
                if np.allclose(self.A[-1], self.A[-(equilibrium_distance + 1)], rtol=1e-15):
                    print(f'Equilibrium conditions met after {i} runs, run halted.')
                    break  # Automatic break if equilibrium is reached. Lets run times be arbitrarily large for MC delta search
        self.A = np.delete(self.A, -1, axis=0)
        self.nodes = self.nodes[:-1]
        self.add_observables(null=False, source_reward=source_reward)

        if verbose:
            print_run_methods()

    def null_simulate(self, num_runs, equilibrium_distance=equilibrium_distance_val, source_reward=source_reward_val):
        """
        Run function. Sequentially seeds source node, sets node values to effective distance evaluation, updates edges and corresponding history.
        :param num_runs: Constant natural number, number of runs.
        :param update_interval: Number of seed steps per run (times information is seeded and diffused before reweighing edges)
        :param equilibrium_distance: length at which two nearly equal A matricies are constantly compared to break run loop
        :return: Returns nothing, updates graph values. Use plotter library to evaluate and graph observables
        """

        self.A = np.vstack((self.A, [self.A[-1]]))  # so that initial values (before initial update) are preserved
        equilibrium_span = equilibrium_span_val  # if a greater range of values between equilibrium distance ought be compared
        for i in range(0, num_runs):
            self.update_edges(null_run=True)
            self.nodes = np.vstack((self.nodes, np.zeros((1, self.num_nodes))))
            self.A = np.vstack((self.A, [self.A[-1]]))
            if equilibrium_distance and self.A.shape[0] > equilibrium_distance + equilibrium_span + 1:  # +1 for range starting at 0 offset
                if np.allclose(self.A[-1], self.A[-(equilibrium_distance + 1)], rtol=1e-15):
                    print(f'Equilibrium conditions met after {i} runs, run halted.')
                    break  # Automatic break if equilibrium is reached. Lets run times be arbitrarily large for MC delta search

        # Stored Observables:
        self.add_observables(null=True, source_reward=source_reward)

    def simulate_ensemble(self, num_simulations, num_runs_per_sim, eff_dist_delta_param=1, edge_init=None,
                          constant_source_node=False,
                          num_shifts_of_source_node=False, seeding_sigma_coeff=False, seeding_power_law_exponent=False,
                          beta=None,
                          multiple_path=False, equilibrium_distance=0, update_interval=1,
                          source_reward=source_reward_val, undirectify=False, null_simulate=False, null_normalize=False, verbose=False):
        """
        Evaluates requested observables (as per observed observables in init) independently for null and normal sims in ensembles,
        then averages and returns their differences.
        :param num_simulations: number of simulations over which the end result will be averaged
        :param num_runs_per_sim: Constant natural number, number of runs.
        :param update_interval: Number of seed steps per run (times information is seeded and diffused before reweighing edges)
        :param eff_dist_delta_param: determines exponential suppression of higher order paths for both RWED and MPED
        :param source_reward: Determines how much the source node is scaled, recommended values \in (1,2)
        :param edge_init: determines edge initialization. If None (default) uniform random edge initialization is used.
        If an integer, then sparse edge initialization, with (int) num of non-zero edges per node.
        If a float, then scale-free edge initialization, with the float determining the degree exponent.
        If a numpy array
        :param constant_source_node: Sets seed node to be a given node (the integer given).  True defaults to 0th node.
        :param multiple_path: if True, uses multiple path effective distance algorithm. Otherwise, defaults to random walker effective distance algorithm
        :param equilibrium_distance: length at which two nearly equal A matricies are constantly compared to break run loop
        :param null_simulate: If true, simulates without any reference to effective distances, purely using mechanics
        :param verbose: if True, details approximate completion percentage and run parameters, methods.
        :return: Returns nothing, updates graph values. Use plotter library to evaluate and graph observables
        """
        # Observable initialization:
        source_node_history = []
        eff_dist_history = [0]*num_runs_per_sim  # relies on consistent run distance (equilibrium val == 0 / > runtime)
        A, null_A = np.zeros((1, self.A.shape[1], self.A.shape[2])), np.zeros((1, self.A.shape[1], self.A.shape[2]))
        global_eff_dist_diff, eff_dist_to_source_diff, mean_eff_dist, degree_dist, ave_nbr_var, ave_cluster_coeff, ave_shortest_path = 0, 0, 0, 0, 0, 0, 0
        efficiency_coordinates = np.zeros(2)
        linear_threshold_hierarchy_coordinates, exp_threshold_hierarchy_coordinates = np.zeros(3), np.zeros(3)
        # Initial null Observable declaration
        null_global_eff_dist_diff, null_degree_dist, null_ave_nbr_var, null_ave_cluster_coeff, null_ave_shortest_path = 0, 0, 0, 0, 0
        null_efficiency_coordinates = np.zeros(2)
        null_linear_threshold_hierarchy_coordinates, null_exp_threshold_hierarchy_coordinates = np.zeros(3), np.zeros(3)

        if null_normalize:
            null_graph = Graph(num_nodes=self.nodes.shape[1], edge_conservation_coefficient=self.edge_conservation_coefficient,
                               selectivity=self.fraction_infoscore_redistributed, undirected=self.undirected,
                               global_eff_dist_diff=bool(self.global_eff_dist_diff), eff_dist_diff=bool(self.eff_dist_diff),
                               mean_eff_dist=bool(self.mean_eff_dist), ave_nbr=bool(self.ave_nbr_var), cluster_coeff=bool(self.ave_cluster_coeff), shortest_path=bool(self.shortest_path),
                               degree_dist=bool(self.degree_dist), hierarchy_coordinates=bool(self.linear_threshold_hierarchy_coordinates), efficiency_coords=bool(self.efficiency_coordinates))
            # There's nearly certainly a better way of initializing a class via another
        for i in range(num_simulations):
            np.random.seed(i)
            random.seed(i)
            self.edge_initialization_conditional(edge_init=edge_init, undirectify=undirectify)
            if not null_simulate:
                self.simulate(num_runs=num_runs_per_sim, update_interval=update_interval,
                              eff_dist_delta_param=eff_dist_delta_param,
                              source_reward=source_reward, constant_source_node=constant_source_node,
                              num_shifts_of_source_node=num_shifts_of_source_node, seeding_sigma_coeff=seeding_sigma_coeff,
                              seeding_power_law_exponent=seeding_power_law_exponent, beta=beta, multiple_path=multiple_path,
                              equilibrium_distance=False, verbose=False)
                eff_dist_history += np.mean(self.eff_dist_to_source_history, axis=1)
            else:
                self.null_simulate(num_runs=num_runs_per_sim, equilibrium_distance=0)
            if null_normalize:
                null_graph.edge_initialization_conditional(edge_init=self.A[0], undirectify=undirectify)
                null_graph.null_simulate(num_runs=num_runs_per_sim, equilibrium_distance=equilibrium_distance)

            # Running Sum of Observable calculations for future sum:
            A += self.A[-1]
            if self.observed_observables['eff_dist_diff']:  # Compares first and last eff_dist values
                eff_dist_to_source_diff += self.source_eff_dist_diff
            if self.observed_observables['mean_eff_dist']:  # Averages all eff_dist_history
                mean_eff_dist += self.mean_eff_dist
            if self.observed_observables['global_eff_dist_diff']:  # Compares first and last all to all eff_dist values
                global_eff_dist_diff += self.global_eff_dist_diff
            if self.observed_observables['degree_dist']:
                degree_dist += self.degree_dist
            if self.observed_observables['ave_nbr']:
                ave_nbr_var += self.ave_nbr_var
            if self.observed_observables['cluster_coeff']:
                ave_cluster_coeff += self.ave_cluster_coeff
            if self.observed_observables['shortest_path']:
                ave_shortest_path += self.ave_shortest_path
            if self.observed_observables['efficiency_coordinates']:
                efficiency_coordinates += self.efficiency_coordinates
            if self.observed_observables['hierarchy_coordinates']:
                linear_threshold_hierarchy_coordinates += self.linear_threshold_hierarchy_coordinates
                exp_threshold_hierarchy_coordinates += self.exp_threshold_hierarchy_coordinates
            if not null_simulate:
                source_node_history.append(self.source_node_history)

            # Running (Norm) Sum of Observable calculations for future sum:
            if null_normalize:
                null_A += null_graph.A[-1]
                if null_graph.observed_observables['degree_dist']:
                    null_degree_dist += null_graph.degree_dist
                if null_graph.observed_observables['global_eff_dist_diff']:  # Compares first and last all to all eff_dist values
                    null_ave_nbr_var += float(null_graph.ave_nbr_var)
                if null_graph.observed_observables['cluster_coeff']:
                    null_ave_cluster_coeff += null_graph.ave_cluster_coeff
                if null_graph.observed_observables['ave_nbr']:
                    null_ave_shortest_path += null_graph.ave_shortest_path
                if null_graph.observed_observables['efficiency_coordinates']:
                    null_efficiency_coordinates += null_graph.efficiency_coordinates
                if null_graph.observed_observables['hierarchy_coordinates']:
                    null_linear_threshold_hierarchy_coordinates += null_graph.linear_threshold_hierarchy_coordinates
                    null_exp_threshold_hierarchy_coordinates += null_graph.exp_threshold_hierarchy_coordinates

            # Re-initialization:
            if null_normalize:
                null_graph.A = np.zeros((1, self.num_nodes, self.num_nodes))
            self.A = np.zeros((1, self.num_nodes, self.num_nodes))  # Re-initializing
            self.eff_dist_to_source_history = []
            self.source_node_history = []
            if verbose: utility_funcs.print_run_percentage(i, num_simulations)

        # Setting class variable to ensemble average:
        self.A = A / float(num_simulations)
        if not null_simulate: self.eff_dist_to_source_history = np.array(eff_dist_history) / float(num_simulations)
        self.source_node_history = source_node_history  # Makes final source node history 2d, with columns being the individual run histories.
        if self.observed_observables['global_eff_dist_diff']:     self.global_eff_dist_diff = global_eff_dist_diff / float(num_simulations)
        if self.observed_observables['eff_dist_diff']:            self.source_eff_dist_diff = eff_dist_to_source_diff / float(num_simulations)
        if self.observed_observables['mean_eff_dist']:            self.mean_eff_dist = mean_eff_dist / float(num_simulations)
        if self.observed_observables['degree_dist']:              self.degree_dist = degree_dist / float(num_simulations)
        if self.observed_observables['ave_nbr']:                  self.ave_nbr_var = ave_nbr_var / float(num_simulations)
        if self.observed_observables['cluster_coeff']:            self.ave_cluster_coeff = ave_cluster_coeff / float(num_simulations)
        if self.observed_observables['shortest_path']:            self.ave_shortest_path = ave_shortest_path / float(num_simulations)
        if self.observed_observables['efficiency_coordinates']:   self.efficiency_coordinates = efficiency_coordinates / float(num_simulations)
        if self.observed_observables['hierarchy_coordinates']:
            self.linear_threshold_hierarchy_coordinates = linear_threshold_hierarchy_coordinates / float(num_simulations)
            self.exp_threshold_hierarchy_coordinates = exp_threshold_hierarchy_coordinates / float(num_simulations)

        if null_normalize:
            null_graph.A = null_A / float(num_simulations)
            null_graph.A = np.delete(null_graph.A, -1, axis=0)
            self.A[1:] = self.A[1:] - null_graph.A[1:]  # Subtracts null values of equivalent graphs simulations for ALL timesteps. Sliced to exclude initializations, which are identical
            self.A = np.array([self.A[i] - np.min(self.A[i]) for i in range(self.A.shape[0])])  # Renormalization
            self.A = np.array([self.A[i] / np.max(self.A[i]) if np.max(self.A[i]) != 0 else self.A[i] for i in range(self.A.shape[0])])
            # Effective distance history and hierarchy coordinates are not examined w.r.t. null values, as it doesn't make much sense.
            # Nor does examining the difference in the entire time series, but most observables re concerned only with the final A anyway
            if null_graph.observed_observables['global_eff_dist_diff']:     null_graph.global_eff_dist_diff = null_global_eff_dist_diff / float(num_simulations)
            if null_graph.observed_observables['degree_dist']:              null_graph.degree_dist = null_degree_dist / float(num_simulations)
            if null_graph.observed_observables['ave_nbr']:                  null_graph.ave_nbr_var = null_ave_nbr_var / float(num_simulations)
            if null_graph.observed_observables['cluster_coeff']:            null_graph.ave_cluster_coeff = null_ave_cluster_coeff / float(num_simulations)
            if null_graph.observed_observables['shortest_path']:            null_graph.ave_shortest_path = null_ave_shortest_path / float(num_simulations)
            if null_graph.observed_observables['efficiency_coordinates']:   null_graph.efficiency_coordinates = null_efficiency_coordinates / float(num_simulations)
            if null_graph.observed_observables['hierarchy_coordinates']:
                null_graph.linear_threshold_hierarchy_coordinates = null_linear_threshold_hierarchy_coordinates / float(num_simulations)
                null_graph.exp_threshold_hierarchy_coordinates = null_exp_threshold_hierarchy_coordinates / float(num_simulations)

            if self.observed_observables['global_eff_dist_diff']:       self.global_eff_dist_diff -= null_graph.global_eff_dist_diff
            if self.observed_observables['degree_dist']:                self.degree_dist -= null_graph.degree_dist
            if self.observed_observables['ave_nbr']:                    self.ave_nbr_var -= null_graph.ave_nbr_var
            if self.observed_observables['cluster_coeff']:              self.ave_cluster_coeff -= null_graph.ave_cluster_coeff
            if self.observed_observables['shortest_path']:              self.ave_shortest_path -= null_graph.ave_shortest_path
            if self.observed_observables['efficiency_coordinates']:     self.efficiency_coordinates -= null_graph.efficiency_coordinates
            if self.observed_observables['hierarchy_coordinates']:
                self.linear_threshold_hierarchy_coordinates -= null_graph.linear_threshold_hierarchy_coordinates
                self.exp_threshold_hierarchy_coordinates -= null_graph.exp_threshold_hierarchy_coordinates
                # self.linear_threshold_hierarchy_coordinates += np.min(linear_threshold_hierarchy_coordinates)
                # self.exp_threshold_hierarchy_coordinates += np.min(exp_threshold_hierarchy_coordinates)

        # Debug:
        # if self.edge_conservation_coefficient == 1.0 and self.fraction_infoscore_redistributed == 1.0:
        #     print(f'eff_dist_to_source_history              :{self.eff_dist_to_source_history}')
        #     print(f'len(eff_dist_to_source_history)         :{len(self.eff_dist_to_source_history)}')
        #     print(f'global_eff_dist_diff                    :{self.global_eff_dist_diff}')
        #     print(f'source_eff_dist_diff                    :{self.source_eff_dist_diff}')
        #     print(f'mean_eff_dist                           :{self.mean_eff_dist}')
        #     print(f'degree_dist                             :{self.degree_dist}')
        #     print(f'ave_nbr_var                             :{self.ave_nbr_var}')
        #     print(f'ave_cluster_coeff                       :{self.ave_cluster_coeff}')
        #     print(f'shortest_path                           :{self.shortest_path}')
        #     print(f'efficiency_coordinates                  :{self.efficiency_coordinates}')
        #     print(f'linear_threshold_hierarchy_coordinates  :{self.linear_threshold_hierarchy_coordinates}')
        #     print(f'exp_threshold_hierarchy_coordinates     :{self.exp_threshold_hierarchy_coordinates}')

########################################################################################################################

if __name__ == "__main__":
    # CHECK VERSIONS
    vers_python0 = '3.7.3'
    vers_numpy0 = '1.17.3'
    vers_matplotlib0 = '3.1.1'
    vers_netx0 = '2.4'

    from sys import version_info
    from matplotlib import __version__ as vers_matplotlib
    from networkx import __version__ as vers_netx

    vers_python = '%s.%s.%s' % version_info[:3]
    vers_numpy = np.__version__

    print('\n------------------- Network Diffusion Adaptation ----------------------------\n')
    print('Required modules:')
    print('Python:        tested for: %s.  Yours: %s' % (vers_python0, vers_python))
    print('numpy:         tested for: %s.  Yours: %s' % (vers_numpy0, vers_numpy))
    print('matplotlib:    tested for: %s.  Yours: %s' % (vers_matplotlib0, vers_matplotlib))
    print('networkx:      tested for: %s.   Yours: %s' % (vers_netx0, vers_netx))
    print('\n------------------------------------------------------------------------------\n')
