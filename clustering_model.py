import numpy as np
import random
np.random.seed(42)
random.seed(42)


class Graph:
    invalid_nodes = []  # will be overwritten upon network_prop call. Lists nodes that have already propagated
    next_nodes = []  # lists nodes which are sufficiently connected to be visited next propagation step
    starting_nodes_with_info = []  # holds the starting nodes for each run, reset after every run

    def __init__(self, num_nodes,  beta=1, nuggets_per_timestep=1, value_per_nugget=1, lower_info_cutoff=0.1, upper_info_cutoff=None):
        """
        Initialization; the 'extra' dimensionality (i.e. 1 for nodes, A) are there to dynamically store their history
        via use of vstack later on.
        """
        if upper_info_cutoff is None:
            upper_info_cutoff = num_nodes
        self.beta = beta
        self.num_nodes = num_nodes
        self.num_nuggets = nuggets_per_timestep
        self.nugget_value = value_per_nugget
        self.upper_cutoff = (upper_info_cutoff*value_per_nugget)/num_nodes
        self.lower_cutoff = (lower_info_cutoff*value_per_nugget)/num_nodes
        # This kind of normalization is advisable so that parameters may be shifted independently
        assert self.upper_cutoff > self.lower_cutoff
        self.nodes = np.zeros((1, num_nodes))
        self.A = np.zeros((1, self.num_nodes, self.num_nodes))

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
        self.A = np.random.rand(1, self.num_nodes, self.num_nodes)  # creates n x n adjacency matrix filled with rand(0,1)
        for node in range(0, self.num_nodes):
            self.A[-1][node][node] = 0  # eliminates looping edges (i.e. no edge refers to itself)
            self.A[-1][node] /= self.A[-1][node].sum()  # normalizes each node's total weights to 1

    def seed_info_random(self):
        self.starting_nodes_with_info = []  # resets starting nodes such that new seed_info call will not conflict
        total_nuggets = 0
        while total_nuggets < self.num_nuggets:
            seeded_node = int(np.random.rand(1)*self.num_nodes)
            self.nodes[-1][seeded_node] += self.nugget_value
            self.starting_nodes_with_info.append(seeded_node)
            total_nuggets += 1

    def seed_info_by_diversity_of_connections(self):
        """
        Computes standard deviation as a (negatively correlated) metric for diversity of connection between nodes,
        then uses this to distribute 'nuggets' of information (via canonical ensemble of standard deviation).
        Potentially variance would be faster (no sqrt) and better, changing the effect of connectedness.
        """
        self.starting_nodes_with_info = []  # resets starting nodes such that new seed_info call will not conflict
        total_nuggets = 0
        exp_stds = []
        for node_edges in self.A[-1][:]:
            exp_stds.append(np.exp(-self.beta * node_edges.std()))  # sum of e^(\beta \sigma_i) for i \in node[weights]
        std_partition = sum(exp_stds)
        while total_nuggets < self.num_nuggets:
            test_node = np.random.randint(0, self.nodes[-1].size)
            if random.uniform(0, std_partition) < exp_stds[test_node]/std_partition:
                self.nodes[-1][test_node] += self.nugget_value
                self.starting_nodes_with_info.append(test_node)
                total_nuggets += 1

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

    def update_edges(self):
        """
        Due to having exclusively forward propagation, we may use the node values directly to determine their effect on
         the node they were connected to. Even though this computational mechanic is the opposite of the
         conceptualization, it should yield the same results.
        The normalization is where the conservation of edge weight applies,
         negatively effecting those not reinforced.
        """
        for node in range(0, self.nodes[-1].size):
            for edge in range(0, self.A[-1][node].size):
                self.A[-1][node][edge] += self.nodes[-1][node] * self.A[-1][node][edge]
                # THIS IS A CONSTANT SCALING! OF COURSE THE RESULTING NORMALIZATION WILL YIELD THE SAME RESULTS!
                # We normalize along the outgoing edges (columns) so that we do not simply reset the rows
        for node in range(0, self.A[-1][0].size):
            incoming_edge_sum = self.A[-1][:, node].sum()
            if incoming_edge_sum > 0:
                self.A[-1][:, node] /= incoming_edge_sum  # normalizes each node's total INCOMING weights to 1
            # For sparse networks, there will likely be some columns (outgoing edges) which sum to zero.

    def run(self, num_runs, verbose=False):
        # removed edge initialization, so it may be customized before call
        self.A = np.vstack((self.A, [self.A[-1]]))  # so that initial values (before initial update) are preserved
        for i in range(0, num_runs):
            self.seed_info_by_diversity_of_connections()
            self.propagate_info_through_network()
            self.update_edges()
            # so the next values may be overwritten, we start with 0 node values.
            self.nodes = np.vstack((self.nodes, np.zeros((1, self.num_nodes))))
            self.A = np.vstack((self.A, [self.A[-1]]))
            if verbose:
                if int(i % num_runs) % int(num_runs/17) == 0:
                    print(f'{(i/num_runs)*100:.1f}%-ish done')
