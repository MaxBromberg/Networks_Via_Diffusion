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
    return len(non_cyclic_nodes) / len(nx_graph.nodes())


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
    if len(successively_peeled_nx_graphs[-1].nodes()) == 0:
        nx_graphs = successively_peeled_nx_graphs[:-1]
    else:
        nx_graphs = successively_peeled_nx_graphs

    f = 0
    total_length = 0
    for nx_graph in nx_graphs:
        g, num_paths = feedforwardness_iteration(nx_graph)
        f += g
        total_length += num_paths

    return f / total_length


def graph_entropy(directed_acyclic_graphs, forward_entropy=False):
    """
    Graph Entropy, as defined in [*] via equation 14
    :param directed_acyclic_graphs: The pruned series of graphs as a list: output of recursive_leaf_removal performed on a DAG
    :param forward_entropy: Bool: if True, calculates entropy from k_in = 0 nodes to others.
    :return: float, Entropy
    """
    largest_dag = nx.convert_node_labels_to_integers(directed_acyclic_graphs[0])
    B = utility_funcs.matrix_normalize(nx.to_numpy_array(largest_dag), row_normalize=False)
    if not forward_entropy:
        P = sum([np.power(B.T, k) for k in range(len(directed_acyclic_graphs[0]))])
    else:
        P = sum([np.power(B, k) for k in range(len(directed_acyclic_graphs[0]))])

    boundary_layer = max_min_layers(largest_dag, max_layer=forward_entropy)
    non_maximal_nodes = set(largest_dag.nodes() - max_min_layers(largest_dag, max_layer=not forward_entropy))
    entropy = 0
    for layer_node in boundary_layer:
        for non_maximal_node in non_maximal_nodes:
            if forward_entropy:
                entropy += P[non_maximal_node][layer_node] * np.log(largest_dag.out_degree(non_maximal_node))
            else:
                entropy += P[non_maximal_node][layer_node] * np.log(largest_dag.in_degree(non_maximal_node))
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