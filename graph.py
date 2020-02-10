import numpy as np
import random
import utility_funcs
import networkx as nx
import effective_distance as ed
np.random.seed(42)
random.seed(42)


class Graph:
    starting_node = None

    def __init__(self, num_nodes, value_per_nugget=1, edge_weighting_factor=None, enforcement_factor=0.1, beta=None, gamma=None, q=None):
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
        self.edge_weighting_exp = edge_weighting_factor
        self.enforcement_factor = enforcement_factor
        self.nodes = np.zeros((1, num_nodes))  # node values (and history of via the first dimension)
        self.A = np.zeros((1, self.num_nodes, self.num_nodes))  # Adjacency matrix (and history of)
        self.starting_node_history = []  # holds the starting nodes for each run, reset after every run.
        self.eff_dist_history = []
        self.global_eff_dist_history = []

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
            self.starting_node_history.append(seeded_node)

    def seed_info_constant_source(self, constant_source_node):
        self.starting_node = constant_source_node
        self.starting_node_history.append(constant_source_node)

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
                self.starting_node_history.append(seeded_node)

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

    def reweight_edges_via_take_the_best(self):
        for from_node in range(0, self.nodes[-1].size):
            node_sum = self.nodes[-1].sum()
            info_score = [self.A[-1][node_index][from_node]**(self.edge_weighting_exp*(self.nodes[-1][node_index]/node_sum)) for node_index in range(self.nodes.shape[1])]
            self.A[-1][info_score.index(max(info_score))][from_node] += self.enforcement_factor  # Not sure this shouldn't be proportional

    def update_edges(self):
        """
        Due to having exclusively forward propagation, we may use the node values directly to determine their effect on
         the node they were connected to. Even though this computational mechanic is the opposite of the
         conceptualization, it should yield the same results.
        The normalization is where the conservation of edge weight applies,
         negatively effecting those not reinforced.
        """
        if self.gamma:
            self.reweight_edges_with_clustering()
        elif self.edge_weighting_exp:
            self.reweight_edges_via_take_the_best()
        else:
            self.reweight_edges_without_clustering()
        for node in range(0, self.A[-1][0].size):
            incoming_edge_sum = self.A[-1][:, node].sum()
            if incoming_edge_sum > 0:
                self.A[-1][:, node] /= incoming_edge_sum  # normalizes each node's total INCOMING weights to 1
            # For sparse networks, there will likely be some columns (outgoing edges) which sum to zero.

    def get_eff_dist(self, adjacency_matrix=None, multiple_path=False, shortest_path=False, dominant_path=False, random_walk_distance=False, source=None, target=None, parameter=1, saveto=""):
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
        if random_walk_distance:
            return ed.EffectiveDistances(np_array=adjacency_matrix).get_random_walk_distance(source=source, target=target, parameter=parameter, saveto=saveto)
        else:
            print(f'No path type chosen in get_eff_dist call. Set multiple_path, shortest_path, dominant_path or random_walk_path=True')

    def write_graph_as_xml(self, timestep=-1, path=None):
        if path is None:
            path = f"graph_at_{timestep}.graphml"
        nx_G = nx.to_directed(nx.from_numpy_matrix(np.array(self.A[timestep]), create_using=nx.DiGraph))
        nx.write_graphml(nx_G, path)


class LogEffDisGraph(Graph):
    invalid_nodes = []  # will be overwritten upon network_prop call. Lists nodes that have already propagated
    next_nodes = []  # lists nodes which are sufficiently connected to be visited next propagation step
    # starting_nodes_with_info = []  # holds the starting nodes for each run, reset after every run

    def __init__(self, num_nodes, beta=None, value_per_nugget=1, gamma=None, q=None, nuggets_per_timestep=1, lower_info_cutoff=0.1,
                 upper_info_cutoff=None):
        """
        Initialization; the 'extra' dimensionality (i.e. 1 for nodes, A) are there to dynamically store their history
        via use of vstack later on.
        """
        self.num_nuggets = nuggets_per_timestep
        if upper_info_cutoff is None:
            upper_info_cutoff = num_nodes
        self.nugget_value = value_per_nugget
        self.upper_cutoff = (upper_info_cutoff * value_per_nugget) / num_nodes
        self.lower_cutoff = (lower_info_cutoff * value_per_nugget) / num_nodes
        # This kind of normalization is advisable so that parameters may be shifted independently
        assert self.upper_cutoff > self.lower_cutoff
        super().__init__(num_nodes, beta, value_per_nugget, gamma, q)

    def node_propagation(self, index):
        """
        The functions that is to be used to recursively distribute info through the network.
        Index references the node which is presently distributing information.
        """
        self.invalid_nodes.append(index)  # stops future propagation from this node
        for node in range(0, self.nodes[-1].size):
            info = self.A[-1][index][node] * self.nodes[-1][index]
            # could use ln(info) to examine effect of moderating learning rate, align with special algorithm in
            # Effective distances for epidemics spreading on complex networks [2017] Flavio Iannelli et al.
            if info >= self.lower_cutoff and node not in set(self.invalid_nodes):  # likely inefficient
                # not in statement ensures every contribution to the node's value is from the same propagation step
                self.nodes[-1][node] += info  # adds info to connected nodes
                self.next_nodes.append(node)
                if self.nodes[-1][node] > self.upper_cutoff:  # only needs to check if >= lower_cutoff
                    self.nodes[-1][node] = self.upper_cutoff  # limits info gain to upper cutoff

    def propagate(self):
        """
        Propagates info to all nodes which are free, i.e. the difference of nodes rewarded last time and those that
        have already propagated information themselves. (it's possible for a node to receive info after propagating)
        """
        for node in list(set(self.next_nodes) - set(self.invalid_nodes)):
            self.node_propagation(node)
        self.next_nodes = []

    def propagate_info_through_network(self):
        """
        Propagates the information according to node_prop until new nodes with strong enough connections are exhausted
        """
        self.next_nodes = []
        self.invalid_nodes = []
        assert self.starting_node is not None, 'No node seeded for propagation'
        self.node_propagation(self.starting_node)
        while list(set(self.next_nodes) - set(self.invalid_nodes)):
            self.propagate()

    def run(self, num_runs, verbose=False):
        self.A = np.vstack((self.A, [self.A[-1]]))  # so that initial values (before initial update) are preserved
        for i in range(0, num_runs):
            if self.beta:
                self.seed_info_by_diversity_of_connections()
            else:
                self.seed_info_random()
            self.propagate_info_through_network()
            self.update_edges()
            # so the next values may be overwritten, we start with 0 node values.
            self.nodes = np.vstack((self.nodes, np.zeros((1, self.num_nodes))))
            self.A = np.vstack((self.A, [self.A[-1]]))
            if i == num_runs-1:
                self.A = np.delete(self.A, -1, axis=0)
                self.nodes = self.nodes[:-1]
            if verbose:
                if int(i % num_runs) % int(num_runs/17) == 0:
                    print(f'{(i/num_runs)*100:.1f}%-ish done')


class SumEffDisGraph(Graph):

    def __init__(self, num_nodes,  value_per_nugget=1, exp_edge_weighting=None, enforcement_factor=0.01, beta=None, gamma=None, q=None, alpha=1):
        self.alpha = alpha  # Presently not used, ideally relevant via eff_dis = ((# edges)^\alpha)/sum_weights
        super().__init__(num_nodes, value_per_nugget, exp_edge_weighting, enforcement_factor, beta, gamma, q)

    def evaluate_effective_distances(self, source, source_reward_scalar, multiple_path_eff_dist=False, parameter=1, timestep=-1):
        """
        Defaults to random walker effective distance metric unless given multiple_path_eff_dist=True
        returns array of effective distances to each node (from source) according to effective dist library´methods
        TODO: eff_dis = ((# edges)^\alpha)/sum_weights
        """
        if multiple_path_eff_dist:
            eff_dists = self.get_eff_dist(adjacency_matrix=self.A[timestep], multiple_path=True, source=source,
                                          parameter=parameter)
        else:
            # pre-normalize
            # normed_A = [self.A[timestep][node, :]/self.A[timestep].sum(axis=1) for node in range(self.nodes.shape[1]) if node != 0]
            normalization_factors = self.A[timestep].sum(axis=1)
            normalized_A = np.where(normalization_factors != 0, self.A[timestep] / normalization_factors[:, np.newaxis], 0)
            # normalized_A = np.array([self.A[timestep][node, :]/normalization_factors[node] for node in range(self.A[timestep].shape[0])])
            # Here parameter is the exp_decay_parameter for the random walk effective distance
            eff_dists = self.get_eff_dist(adjacency_matrix=normalized_A, random_walk_distance=True,
                                                          source=source, parameter=parameter)
            # re-normalizing not needed, as self.A was not changed; only eff_distances can be scaled in desired
        assert np.isclose(eff_dists[source], 0, rtol=1e-10), "Source has nonzero effective distance"
        eff_dists = np.delete(eff_dists, source)  # awkward deletion/insertion to ensure min search of remaining eff_distances
        eff_dists = np.insert(eff_dists, source, min(eff_dists)/source_reward_scalar)
        self.global_eff_dist_history.append(eff_dists.mean())
        self.eff_dist_history.append(eff_dists)
        # print(f'Eff Distances: {np.round(eff_dists, 3)}  Source (Val): {source} ({np.round(eff_dists[source], 3)})')
        return eff_dists

    def add_inv_eff_distances_to_node_values(self, exp_decay_param, source_reward_scalar, multiple_path=False):
        # self.nodes[-1] = [1/val for val in np.array(self.evaluate_effective_distances(self.starting_node, parameter=exp_decay_param, source_reward_scalar=source_reward_scalar, multiple_path_eff_dist=multiple_path))]
        self.nodes[-1] = np.array(self.evaluate_effective_distances(self.starting_node, parameter=exp_decay_param, source_reward_scalar=source_reward_scalar, multiple_path_eff_dist=multiple_path))

    def seed_info_conditional(self, constant_source_node):
        if self.beta:
            self.seed_info_by_diversity_of_connections()
        elif constant_source_node or constant_source_node == 0:
            self.seed_info_constant_source(constant_source_node)
        else:
            self.seed_info_random()

    def run(self, num_runs, update_interval=1, exp_decay_param=0.4, source_reward_scalar=1.6, constant_source_node=None, verbose=False, multiple_path=False):
        # removed edge initialization, so it may be customized before call
        self.A = np.vstack((self.A, [self.A[-1]]))  # so that initial values (before initial update) are preserved
        for i in range(0, num_runs):
            self.seed_info_conditional(constant_source_node)
            self.add_inv_eff_distances_to_node_values(exp_decay_param, source_reward_scalar, multiple_path)
            if i % update_interval == 0:
                self.update_edges()
                # so the next values may be overwritten, we start with 0 node values.
                self.nodes = np.vstack((self.nodes, np.zeros((1, self.num_nodes))))
                self.A = np.vstack((self.A, [self.A[-1]]))
            if i == num_runs-1:
                self.A = np.delete(self.A, -1, axis=0)
                self.nodes = self.nodes[:-1]
            if verbose:
                assert num_runs > 17, "too few runs, set verbose to False"
                if int(i % num_runs) % int(num_runs / 17) == 0:
                    print(f'{(i / num_runs) * 100:.1f}%-ish done')
