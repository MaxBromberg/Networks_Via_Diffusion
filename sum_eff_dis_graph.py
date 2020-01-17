from graph import *
import utility_funcs
import networkx as nx


class SumEffDisGraph(Graph):

    def __init__(self, num_nodes,  beta=1, value_per_nugget=1, alpha=1, eff_dist_scaler=1):
        # here value_per_nugget acts to slow or speed the effective distance, to allow for more detailed investigation
        # of progression, as all reweighting is due to intermediary edge values. #DOESN'T RESCALE! POURQUE?
        self.eff_dist_scaler = eff_dist_scaler
        self.alpha = alpha
        super().__init__(num_nodes,  beta, value_per_nugget)

    def evaluate_effective_distances(self, source, source_reward_scaler=1.2):
        """
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
        eff_dists = [sum([1/edge for edge in path])/(pow((len(num_edges_in_path)+1), self.alpha)) for path in edge_paths]
        eff_dists.insert(source, min(eff_dists)/source_reward_scaler)
        # source reward should be greatest of all, eff_dist ~ 0. Thus the variable scaling
        return [eff_dist*self.eff_dist_scaler for eff_dist in eff_dists]

    def weight_nodes_with_eff_distances(self):
        self.nodes[-1] = np.array([1/el for el in self.evaluate_effective_distances(self.starting_nodes_with_info[-1])])
        # This inversion of every element could be prevented via initial calculation being inverted, but then eff_dist
        # is inverted. In this subclass's case, there should never be more than one node starting with info (per run)

    def run(self, num_runs, verbose=False):
        # removed edge initialization, so it may be customized before call
        self.A = np.vstack((self.A, [self.A[-1]]))  # so that initial values (before initial update) are preserved
        for i in range(0, num_runs):
            self.seed_info_by_diversity_of_connections()
            self.weight_nodes_with_eff_distances()
            self.update_edges()
            # so the next values may be overwritten, we start with 0 node values.
            self.nodes = np.vstack((self.nodes, np.zeros((1, self.num_nodes))))
            self.A = np.vstack((self.A, [self.A[-1]]))
            if verbose:
                if int(i % num_runs) % int(num_runs / 17) == 0:
                    print(f'{(i / num_runs) * 100:.1f}%-ish done')
