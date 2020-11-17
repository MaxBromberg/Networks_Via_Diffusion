import numpy as np
import utility_funcs
import networkx as nx
import copy

"""
Implementation of the hierarchy coordinates via networkX from 
Hierarchy in Complex Networks: The Possible and the Actual [B Corominas-Murtra - 2013]  [*] - Supporting Information

Though implemented for unweighted networkX graphs, in the context of its original application, 
these are applied to weighted graphs by averaging over the unweighted graphs resulting from 
applying thresholds to normalized weighted graphs. 
"""


def weakly_connected_component_subgraphs(G, copy=True):
    """Generate weakly connected components as subgraphs.

    Parameters
    ----------
    G : NetworkX Graph
       A directed graph.

    copy : bool
        If copy is True, graph, node, and edge attributes are copied to the
        subgraphs.
    """
    for comp in nx.weakly_connected_components(G):
        if copy:
            yield G.subgraph(comp).copy()
        else:
            yield G.subgraph(comp)


def node_weighted_condense(A, num_thresholds=8, exp_threshold_distribution=None):
    """
    returns a series of node_weighted condensed graphs (DAGs) [*] and their original nx_graphs.
    [*] Hierarchy in Complex Networks: the possible and the actual: Supporting information. Corominas-Murtra et al. [2013]
    :param timestep: timestep for conversion
    :param num_thresholds: Number of thresholds and resultant sets of node-weighted Directed Acyclic Graphs
    :param exp_threshold_distribution: if true or float, distributes the thresholds exponentially, with an exponent equal to the float input.
    :return condensed graphs, a list of node_weighted condensed nx_graphs, one for every threshold and corresponding binary A
    An exponent of 0 results in a linear distribution, otherwise the exp distribution is sampled from e^(exp_float)*2e - e^(exp_float)*e
    """
    # Establishing Thresholds
    if exp_threshold_distribution is None:
        if np.isclose(np.max(A)-np.min(A), 0, 1e-2): print(f"Breadth of A: {np.max(A)-np.min(A)}")
        try: thresholds = list(np.round(np.arange(np.min(A), np.max(A), (np.max(A - np.min(A))) / num_thresholds), 4))
        except: thresholds = [np.max(A)]*num_thresholds
    else:
        thresholds = utility_funcs.exponentially_distribute(exponent=exp_threshold_distribution,
                                                            dist_max=np.max(A),
                                                            dist_min=np.min(A),
                                                            num_exp_distributed_values=num_thresholds)
    # Converting to binary nx_graphs according to thresholds:
    nx_graphs = [nx.from_numpy_matrix(np.where(A > threshold, 1, 0), create_using=nx.DiGraph) for threshold in thresholds]
    # base_binary_graphs = [nx.to_numpy_array(nx_graphs[val]) for val in range(len(nx_graphs))]  # yes, it's silly to reconvert if this is actually needed.

    condensed_graphs = [nx.condensation(nx_graphs[index]) for index in range(len(nx_graphs))]
    largest_condensed_graphs = []
    for condensed_graph in condensed_graphs:
        largest_condensed_graphs.append(nx.convert_node_labels_to_integers(
            max(weakly_connected_component_subgraphs(condensed_graph, copy=True), key=len)))
        # networkx.weakly_connected_component_subgraphs comes from networkx 1.10 documentation, and has sense been discontinued.
        # For ease of access and future networkx compatibility, it was copied directly to this file before the class declaration.
        members = nx.get_node_attributes(largest_condensed_graphs[-1], 'members')
        node_weights = [len(w) for w in members.values()]
        for node_index in range(len(node_weights)):
            largest_condensed_graphs[-1].nodes[node_index]["weight"] = node_weights[node_index]

    return largest_condensed_graphs, nx_graphs


def weight_nodes_by_condensation(condensed_graph):
    node_weights = [len(w) for w in nx.get_node_attributes(condensed_graph, 'members').values()]
    for node_index in range(len(node_weights)):
        condensed_graph.nodes[node_index]["weight"] = node_weights[node_index]
    return condensed_graph  # Might not be necessary, as the graph itself is updated (not copied)


def max_min_layers(nx_graph, max_layer=True):
    """
    Returns the maximal (highest in hierarchy) layer (those nodes with in degree = 0) or the minimal layer (k_out = 0)
    :param nx_graph: networkX graph
    :param max_layer: Bool, if True, returns maximal layer (k_in = 0), else returns nodes for which k_out = 0, minimal layer
    :return: list of node indices as ints
    """
    if max_layer:
        return [node for node in nx_graph.nodes() if nx_graph.in_degree(node) == 0]
    else:
        return [node for node in nx_graph.nodes() if nx_graph.out_degree(node) == 0]


def leaf_removal(nx_graph, forward=True):
    """
    :param nx_graph: NetworkX graph to be pruned
    :param forward: Bool, if True, prunes from k_in=0 nodes
    :return: returns NetworkX graph with the first set of k_in = 0 nodes removed
    """
    layer = max_min_layers(nx_graph, max_layer=forward)
    peeled_graph = copy.deepcopy(nx_graph)
    for node in layer:
        peeled_graph.remove_node(node)
    return peeled_graph


def recursive_leaf_removal(nx_graph, from_top=True):
    """
    :param nx_graph: NetworkX graph
    :param from_top: Bool, if True, prunes from the top of the hierarchy (k_in = 0) nodes removed recursively. If False, from k_out = 0 nodes
    :return: list of networkX graphs, starting with the initial DAG and with successively pruned hierarchy layers
    """
    dissected_graphs = [copy.deepcopy(nx_graph)]
    while len(dissected_graphs[-1].nodes()) > 1:
        dissected_graphs.append(leaf_removal(dissected_graphs[-1], forward=from_top))
    return dissected_graphs


def orderability(nx_graph, condensed_nx_graph=None):
    """
    :param nx_graph: networkX graph
    :param condensed_nx_graph: Directed acyclic networkX graph, which is an optional input as in the context of this
    evaluation it's likely to already be evaluated
    :return: orderability (number of nodes which were not condensed over total number of nodes in original uncondensed graph)
    """
    if condensed_nx_graph is None:
        condensed_nx_graph = weight_nodes_by_condensation(nx.condensation(nx_graph))
    non_cyclic_nodes = [node[0] for node in nx.get_node_attributes(condensed_nx_graph, 'weight').items() if node[1] == 1]
    total_acyclic_node_weight = sum([weight for weight in nx.get_node_attributes(condensed_nx_graph, 'weight').values()])
    return len(non_cyclic_nodes) / total_acyclic_node_weight


def feedforwardness_iteration(nx_graph):
    """
    :return: g = sum of feedforwardness for a single DAG, num_paths = sum total paths considered
    """
    max_layer = max_min_layers(nx_graph, max_layer=True)
    min_layer = max_min_layers(nx_graph, max_layer=False)
    weights = nx.get_node_attributes(nx_graph, 'weight')
    g = 0
    num_paths = 0
    for max_node in max_layer:
        for min_node in min_layer:
            for path in nx.all_simple_paths(nx_graph, source=max_node, target=min_node):
                g += len(path) / sum([weights[node] for node in path])  # where each path calculation is F(path)
                num_paths += 1
    return g, num_paths


def feedforwardness(directed_acyclic_graph):
    """
    :param directed_acyclic_graph: Original graph which will be peeled and the ff will be averaged over all layered versions
    :return: Feedforwardness, as in [*]
    """
    # Must be fed the set of graphs with nodes lower on the hierarchy eliminate first
    successively_peeled_nx_graphs = recursive_leaf_removal(directed_acyclic_graph, from_top=False)
    if len(successively_peeled_nx_graphs) == 1 and len(successively_peeled_nx_graphs[0].nodes()) == 1:
        return 0
    #  Prunes the last, pathless, component of the decomposed graph set, as it won't count in feedforwardness
    if len(successively_peeled_nx_graphs[-1].nodes()) == 1:
        nx_graphs = successively_peeled_nx_graphs[:-1]
    else:
        nx_graphs = successively_peeled_nx_graphs

    f = 0
    total_num_paths = 0
    for nx_graph in nx_graphs:
        g, paths = feedforwardness_iteration(nx_graph)
        f += g
        total_num_paths += paths
    return f / total_num_paths


def graph_entropy(directed_acyclic_graphs, forward_entropy=False):
    """
    Graph Entropy, as defined in [*] via equation 14
    :param directed_acyclic_graphs: The pruned series of graphs as a list: output of recursive_leaf_removal performed on a DAG
    :param forward_entropy: Bool: if True, calculates entropy from k_in = 0 nodes to others.
    :return: float, Entropy
    """
    initial_dag = nx.convert_node_labels_to_integers(directed_acyclic_graphs[0])
    B = utility_funcs.matrix_normalize(nx.to_numpy_array(initial_dag), row_normalize=False)
    if not forward_entropy:
        P = sum([np.power(B.T, k) for k in range(len(directed_acyclic_graphs[0]))])
    else:
        P = sum([np.power(B, k) for k in range(len(directed_acyclic_graphs[0]))])

    boundary_layer = max_min_layers(initial_dag, max_layer=forward_entropy)
    non_maximal_nodes = set(initial_dag.nodes() - max_min_layers(initial_dag, max_layer=not forward_entropy))
    entropy = 0
    for layer_node in boundary_layer:
        for non_maximal_node in non_maximal_nodes:
            if forward_entropy:
                entropy += P[non_maximal_node][layer_node] * np.log(initial_dag.out_degree(non_maximal_node))
            else:
                entropy += P[non_maximal_node][layer_node] * np.log(initial_dag.in_degree(non_maximal_node))
    entropy /= len(boundary_layer)
    return entropy


def single_graph_treeness(directed_acyclic_graphs):
    """
    :param directed_acyclic_graphs: The pruned series of graphs as a list: output of recursive_leaf_removal performed on a DAG
    :return: treeness for a single DAG (equation 17 from [*])
    """
    if len(directed_acyclic_graphs) == 1 and len(directed_acyclic_graphs[-1].nodes()) == 1:
        return 0
    forward_entropy = graph_entropy(directed_acyclic_graphs, forward_entropy=True)
    backward_entropy = graph_entropy(directed_acyclic_graphs, forward_entropy=False)
    if forward_entropy == 0 and backward_entropy == 0:
        return 0
    return (forward_entropy - backward_entropy) / max(forward_entropy, backward_entropy)


def treeness(dag):
    """
    :param dag: Directed Acyclic Graph (networkX)
    :return: float, Treeness. Equation 18 from [*]
    """
    pruned_from_top = recursive_leaf_removal(nx_graph=dag, from_top=True)
    pruned_from_bottom = recursive_leaf_removal(nx_graph=dag, from_top=False)
    if len(pruned_from_top) > 1:
        pruned_from_top = pruned_from_top[:-1]
    if len(pruned_from_bottom) > 1:
        pruned_from_bottom = pruned_from_bottom[:-1]

    entropy_sum = 0
    for index in range(len(pruned_from_top)):
        entropy_sum += single_graph_treeness(pruned_from_top[index:])
    for index in range(len(pruned_from_bottom)):
        entropy_sum += single_graph_treeness(pruned_from_bottom[index:])

    return entropy_sum / (len(pruned_from_bottom) + len(pruned_from_top))


def average_hierarchy_coordinates(A, num_thresholds=8, exp_threshold_distribution=None):
    if utility_funcs.check_binary(A):
        num_thresholds = 1
    else:
        num_thresholds = num_thresholds

    o, f, t = 0, 0, 0
    condensed_graphs, original_graphs = node_weighted_condense(A=A, num_thresholds=num_thresholds, exp_threshold_distribution=exp_threshold_distribution)
    for index in range(len(condensed_graphs)):
        o += orderability(original_graphs[index], condensed_graphs[index])
        f += feedforwardness(condensed_graphs[index])
        t += treeness(condensed_graphs[index])
    o /= len(condensed_graphs); f /= len(condensed_graphs); t /= len(condensed_graphs)
    return o, f, t

########################################################################################################################


if __name__ == "__main__":
    # CHECK VERSIONS
    vers_python0 = '3.7.3'
    vers_numpy0 = '1.17.3'
    vers_netx0 = '2.4'

    from sys import version_info
    from networkx import __version__ as vers_netx

    vers_python = '%s.%s.%s' % version_info[:3]
    vers_numpy = np.__version__

    print('\n------------------- Hierarchy Coordinates ----------------------------\n')
    print('Required modules:')
    print('Python:        tested for: %s.  Yours: %s' % (vers_python0, vers_python))
    print('numpy:         tested for: %s.  Yours: %s' % (vers_numpy0, vers_numpy))
    print('networkx:      tested for: %s.   Yours: %s' % (vers_netx0, vers_netx))
    print('\n-----------------------------------------------------------------------\n')