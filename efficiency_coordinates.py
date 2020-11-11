import numpy as np
from networkx import to_directed, from_numpy_matrix, to_numpy_array, shortest_path, DiGraph, grid_graph, fast_gnp_random_graph
from scipy.sparse.linalg import inv
from scipy.sparse import diags, eye, csc_matrix
from utility_funcs import matrix_normalize


# Effective Distance Evaluation: --------------------------------------------------------------------------------
def Random_Walker_Effective_Distance(A, source=None, target=None, parameter=1, via_numpy=False, sub_zeros=False, return_errors=False):
    """
    Directly Adapted from Dr. Koher's version, though by default uses scipy.sparse rather than numpy.linalg (much faster)
    Compute the random walk effective distance:
    F. Iannelli, A. Koher, P. Hoevel, I.M. Sokolov (in preparation)

    Parameters
    ----------
         source : int or None
            If source is None, the distances from all nodes to the target is calculated
            Otherwise the integer has to correspond to a node index

        target : int or None
            If target is None, the distances from the source to all other nodes is calculated
            Otherwise the integer has to correspond to a node index

        parameter : float
            compound delta which includes the infection and recovery rate alpha and beta, respectively,
            the mobility rate kappa and the Euler-Mascheroni constant lambda:
                log[ (alpha-beta)/kappa - lambda ]

    Returns:
    --------
        random_walk_distance : ndarray or float
            If source and target are specified, a float value is returned that specifies the distance.

            If either source or target is None a numpy array is returned.
            The position corresponds to the node ID.
            shape = (Nnodes,)

            If both are None a numpy array is returned.
            Each row corresponds to the node ID.
            shape = (Nnodes,Nnodes)

        _singular_fundamental_matrix_errors : int
            Counts the number of times there are 0s present the in fundamental matrix Z arising from the original A.
            If instances of 0s occur in Z, they are in practice replaced by 1e-100 ~ 0
    """
    assert (isinstance(parameter, float) or isinstance(parameter, int)) and parameter > 0
    assert np.all(np.isclose(A.sum(axis=1), 1, rtol=1e-15)), f"The transition matrix has to be row normalized | A row sums: \n {A.sum(axis=1)}"
    # assert np.all(np.isclose(P.sum(axis=1), 1, rtol=1e-15, equal_nan=True)), "If there are dim incompatibility issues, as nan == nan is false."
    _singular_fundamental_matrix_errors = 0

    if not via_numpy:
        one = eye(A.shape[0], format="csc")
        Z = inv(csc_matrix(one - A * np.exp(-parameter)))
        D = diags(1. / Z.diagonal(), format="csc")
        ZdotD = Z.dot(D).toarray()
        if np.any(ZdotD == 0) or sub_zeros:
            ZdotD = np.where(ZdotD == 0, 1e-100, ZdotD)  # Substitute zero with low value (10^-100) for subsequent log evaluation
            RWED = -np.log(ZdotD)
            _singular_fundamental_matrix_errors += 1
        else:
            RWED = -np.log(Z.dot(D).toarray())
    else:
        one = np.identity(A.shape[0])
        Z = np.linalg.inv(one - A * np.exp(-parameter))
        D = np.diag(1. / Z.diagonal())
        ZdotD = Z.dot(D).toarray()
        if np.any(ZdotD == 0) or sub_zeros:
            ZdotD = np.where(ZdotD == 0, 1e-100, ZdotD)
            RWED = -np.log(ZdotD)
            _singular_fundamental_matrix_errors += 1
        else:
            RWED = -np.log(Z.dot(D))

    if source is not None:
        if target is not None:
            RWED = RWED[source, target]
        else:
            RWED = RWED[source, :]
    elif target is not None:
        RWED = RWED[:, target]
    if not return_errors:
        return RWED
    else:
        return RWED, _singular_fundamental_matrix_errors


# Supporting Functions: -----------------------------------------------------------------------------------------
def sum_weighted_path(A, path: list):
    return sum([A[path[i]][path[i + 1]] for i in range(len(path) - 1)])


def shortest_path_distances(A, source=None, target=None, reversed_directions=False, prenormalized_A=False):
    if not prenormalized_A:
        Adj = matrix_normalize(A, row_normalize=True)
    else:
        Adj = A
    Adj = np.array(1 - Adj)  # Preliminary effective distance, converting edge weights from indicating proximity to distances for evaluation by shortest path
    Adj = np.where(Adj == 0, 1e-100, Adj)  # Replacing 0s by ~0s, so network remains connected for shortest path algorithm

    inverted_weights_nx_graph = to_directed(from_numpy_matrix(Adj, create_using=DiGraph))
    if reversed_directions:
        inverted_weights_nx_graph = inverted_weights_nx_graph.reverse(copy=False)

    # SPD_dic = dict(nx.shortest_path_length(inverted_weights_nx_graph, source=source, target=target, weight='weight'))
    shortest_paths = shortest_path(inverted_weights_nx_graph, source=source, target=target, weight='weight')
    # if shortest_paths
    if source is None:
        if target is None:
            SPD = np.array([[sum_weighted_path(Adj, shortest_paths[s][t]) for s in range(Adj.shape[0])] for t in range(Adj.shape[0])])
        else:
            SPD = np.array([sum_weighted_path(Adj, shortest_paths[s][target]) for s in range(Adj.shape[0])])
    else:
        if target is None:
            SPD = np.array([sum_weighted_path(Adj, shortest_paths[source][t]) for t in range(Adj.shape[0])])
        else:
            SPD = sum_weighted_path(Adj, shortest_paths[source][target])
    return SPD


# Efficiency Coordinates: ---------------------------------------------------------------------------------------
def ave_network_efficiencies(n, ensemble_size: int, efficiency: str):
    if efficiency == "routing" or efficiency == "rout":
        routing_lattice_average = E_rout(A=matrix_normalize(to_numpy_array(grid_graph(dim=[2, int(n / 2)], periodic=True)), row_normalize=True), normalize=False)
        routing_rnd_graph_average = np.mean([E_rout(A=matrix_normalize(to_numpy_array(fast_gnp_random_graph(n=n, p=0.5, seed=i, directed=True)), row_normalize=True), normalize=False) for i in range(ensemble_size)])
        return routing_lattice_average, routing_rnd_graph_average
    if efficiency == "diffusive" or efficiency == "diff":
        diffusive_lattice_average = E_diff(A=matrix_normalize(to_numpy_array(grid_graph(dim=[2, int(n / 2)], periodic=True)), row_normalize=True), normalize=False)
        diffusive_rnd_graph_average = np.mean([E_diff(A=matrix_normalize(to_numpy_array(fast_gnp_random_graph(n=n, p=0.5, seed=i, directed=True)), row_normalize=True), normalize=False) for i in range(ensemble_size)])
        return diffusive_lattice_average, diffusive_rnd_graph_average


def E_rout(A, reversed_directions=False, normalize=True):
    # To avoid divide by 0 errors, we add the 'average' edge weight 1/self.nodes.shape[1] to the inverted weight shortest paths
    shortest_paths = shortest_path_distances(A=A, source=None, target=None, reversed_directions=reversed_directions)
    n = A.shape[0]

    if normalize:
        routing_lattice_average, routing_rnd_graph_average = ave_network_efficiencies(n=n, ensemble_size=10, efficiency="routing")
        E_routing_base = np.sum(1 / (np.array(shortest_paths) + (1 / n))) / (n * (n - 1))
        return (E_routing_base - routing_lattice_average) / (routing_rnd_graph_average - routing_lattice_average)

    return np.sum(1 / (np.array(shortest_paths) + (1 / n))) / (n * (n - 1))


def E_diff(A, normalize=True):  # Diffusion Efficiency
    Adj_Matrix, n = A, A.shape[0]
    row_sums = Adj_Matrix.sum(axis=1)
    normalized_A = np.array([Adj_Matrix[node, :] / row_sums[node] for node in range(Adj_Matrix.shape[0])])

    if normalize:
        diffusive_lattice_average, diffusive_rnd_graph_average = ave_network_efficiencies(n=n, ensemble_size=10, efficiency="diffusive")
        E_diff_base = np.sum(Random_Walker_Effective_Distance(A=normalized_A)) / (n * (n - 1))
        return (E_diff_base - diffusive_lattice_average) / (diffusive_rnd_graph_average - diffusive_lattice_average)

    return np.sum(Random_Walker_Effective_Distance(A=normalized_A)) / (n * (n - 1))


def network_efficiencies(A, normalize=True):
    return E_diff(A, normalize=normalize), E_rout(A, normalize=normalize)


if __name__ == "__main__":
    # CHECK VERSIONS
    vers_python0 = '3.7.3'
    vers_numpy0 = '1.17.3'
    vers_netx0 = '2.4'

    from sys import version_info
    from networkx import __version__ as vers_netx

    vers_python = '%s.%s.%s' % version_info[:3]
    vers_numpy = np.__version__

    print('\n------------------- Efficiency Coordinates ----------------------------\n')
    print('Required modules:')
    print('Python:        tested for: %s.  Yours: %s' % (vers_python0, vers_python))
    print('numpy:         tested for: %s.  Yours: %s' % (vers_numpy0, vers_numpy))
    print('networkx:      tested for: %s.   Yours: %s' % (vers_netx0, vers_netx))
    print('\n-----------------------------------------------------------------------\n')
