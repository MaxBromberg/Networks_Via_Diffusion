from graph import *
import utility_funcs
import networkx as nx


class LogEffDisGraph(Graph):
    invalid_nodes = []  # will be overwritten upon network_prop call. Lists nodes that have already propagated
    next_nodes = []  # lists nodes which are sufficiently connected to be visited next propagation step
    # starting_nodes_with_info = []  # holds the starting nodes for each run, reset after every run

    def __init__(self, num_nodes, beta=1, nuggets_per_timestep=1, value_per_nugget=1, lower_info_cutoff=0.1,
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
        super().__init__(num_nodes, beta, value_per_nugget)

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
            self.seed_info_by_diversity_of_connections()
            self.propagate_info_through_network()
            self.update_edges()
            # so the next values may be overwritten, we start with 0 node values.
            self.nodes = np.vstack((self.nodes, np.zeros((1, self.num_nodes))))
            self.A = np.vstack((self.A, [self.A[-1]]))
            if verbose:
                if int(i % num_runs) % int(num_runs/17) == 0:
                    print(f'{(i/num_runs)*100:.1f}%-ish done')

