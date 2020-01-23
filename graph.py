import numpy as np
import random
import utility_funcs
import networkx as nx
np.random.seed(42)
random.seed(42)


class Graph:
    starting_nodes_with_info = []  # holds the starting nodes for each run, reset after every run

    def __init__(self, num_nodes,  beta=None, value_per_nugget=1, gamma=None, q=None):
        """
        Initialization; the 'extra' dimensionality (i.e. 1 for nodes, A) are there to dynamically store their history
        via use of vstack later on.
        """
        self.beta = beta  # Determines how much the seeding is weighted towards diversely connected nodes
        # (The None default leads to an explicitly random seeding (and thus faster) run)
        self.num_nodes = num_nodes
        self.nugget_value = value_per_nugget
        self.gamma = gamma  # Determines punishment for connecting to the same node, as default strategy for both
        # random and diverse connection weighted seeding would otherwise be to connect all nodes to one other
        self.q = q  # exp{-gamma*[(sum_j w_ij)-q*N]} => for 0<q(<1) nodes are incentivized to strengthen outgoing edges
        self.nodes = np.zeros((1, num_nodes))  # node values (and history of via the first dimension)
        self.A = np.zeros((1, self.num_nodes, self.num_nodes))  # Adjacency matrix (and history of)

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
        self.starting_nodes_with_info = []  # resets starting nodes such that new seed_info call will not conflict
        while not self.starting_nodes_with_info:
            seeded_node = int(np.random.rand(1)*self.num_nodes)
            self.nodes[-1][seeded_node] += self.nugget_value
            self.starting_nodes_with_info.append(seeded_node)

    def seed_info_by_diversity_of_connections(self):
        """
        Computes standard deviation as a (negatively correlated) metric for diversity of connection between nodes,
        then uses this to distribute 'nuggets' of information (via canonical ensemble of standard deviation).
        Potentially variance would be faster (no sqrt) and better, changing the effect of connectedness.
        """
        self.starting_nodes_with_info = []  # resets starting nodes such that new seed_info call will not conflict
        exp_stds = []
        for node_edges in self.A[-1][:]:
            exp_stds.append(np.exp(-self.beta * node_edges.std()))  # sum of e^(\beta \sigma_i) for i \in node[weights]
        std_partition = sum(exp_stds)
        test_node = np.random.randint(0, self.nodes[-1].size)
        while not self.starting_nodes_with_info:
            if random.uniform(0, std_partition) < exp_stds[test_node]/std_partition:
                self.nodes[-1][test_node] += self.nugget_value
                self.starting_nodes_with_info.append(test_node)

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
        else:
            self.reweight_edges_without_clustering()
        for node in range(0, self.A[-1][0].size):
            incoming_edge_sum = self.A[-1][:, node].sum()
            if incoming_edge_sum > 0:
                self.A[-1][:, node] /= incoming_edge_sum  # normalizes each node's total INCOMING weights to 1
            # For sparse networks, there will likely be some columns (outgoing edges) which sum to zero.


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
        assert self.starting_nodes_with_info != []
        for starting_node in self.starting_nodes_with_info:
            self.node_propagation(starting_node)
            # potentially need edge renormalization here between nugget delivery
        while list(set(self.next_nodes) - set(self.invalid_nodes)):
            self.propagate()

    def run(self, num_runs, verbose=False):
        # removed edge initialization, so it may be customized before call
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

    def __init__(self, num_nodes,  beta=None, value_per_nugget=1, gamma=None, q=None, alpha=1):
        # here value_per_nugget acts to slow or speed the effective distance, to allow for more detailed investigation
        # of progression, as all reweighting is due to intermediary edge values. #DOESN'T RESCALE! POURQUE?
        self.alpha = alpha
        super().__init__(num_nodes,  beta, value_per_nugget, gamma, q)

    def evaluate_effective_distances(self, source, source_reward_scalar=1.2):
        """
        EDIT: This does not account for the shortest path according to the effective distance metric, only according
         to the weights! (i.e. the division by (# num connections)^alpha is not considered in shortest path)

        returns array of effective distances to each node (from source) according to
        eff_dis = ((# edges)^\alpha)/sum_weights
        """
        nx_G = nx.from_numpy_array(utility_funcs.return_one_over_2d_matrix(np.round(self.A[-1], 3)))
        # to use the networkx efficient shortest path algorithm, we first invert the edges so that
        # shortest path == most highly weighted, as we intend to depict via our network
        node_paths = [nx.algorithms.shortest_paths.weighted.dijkstra_path(nx_G, source, node, weight='weight') for node in list(nx_G.nodes) if node != source]
        # We eliminate the source term, and will add it back later with appropriate reward at the end.
        num_edges_in_path = [[i for i in range(len(path)-1)] for path in node_paths]
        # -1 as we measure edges via i and i + 1, and we don't want i+1 to go out of range of node_path(s).
        edge_paths = [[nx_G[node_paths[e][i]][node_paths[e][i + 1]]['weight'] for i in num_edges_in_path[e]] for e in range(len(num_edges_in_path))]
        # yields list of lists of edge values for each node's shortest path to the source
        eff_dists = [sum([1/edge for edge in path])/(pow((len(path)), self.alpha)) for path in edge_paths]
        eff_dists.insert(source, min(eff_dists) / source_reward_scalar)
        # source reward should be greatest of all, eff_dist ~ 0. Thus the variable scaling
        return eff_dists

    def weight_nodes_with_eff_distances(self):
        # eff_dists = np.array([1/el for el in self.evaluate_effective_distances(self.starting_nodes_with_info[-1])])
        # self.nodes[-1] = [eff_dist/sum(eff_dists) for eff_dist in eff_dists]  # normalize to allow for compatible efficiency metric
        self.nodes[-1] = np.array([1/el for el in self.evaluate_effective_distances(self.starting_nodes_with_info[-1])])
        # This inversion of every element could be prevented via initial calculation being inverted, but then eff_dist
        # is inverted. In this subclass's case, there should never be more than one node starting with info (per run)

    def run(self, num_runs, verbose=False):
        # removed edge initialization, so it may be customized before call
        self.A = np.vstack((self.A, [self.A[-1]]))  # so that initial values (before initial update) are preserved
        for i in range(0, num_runs):
            if self.beta:
                self.seed_info_by_diversity_of_connections()
            else:
                self.seed_info_random()
            self.weight_nodes_with_eff_distances()
            self.update_edges()
            # so the next values may be overwritten, we start with 0 node values.
            self.nodes = np.vstack((self.nodes, np.zeros((1, self.num_nodes))))
            self.A = np.vstack((self.A, [self.A[-1]]))
            if i == num_runs-1:
                self.A = np.delete(self.A, -1, axis=0)
                self.nodes = self.nodes[:-1]
            if verbose:
                if int(i % num_runs) % int(num_runs / 17) == 0:
                    print(f'{(i / num_runs) * 100:.1f}%-ish done')


