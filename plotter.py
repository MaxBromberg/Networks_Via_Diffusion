import numpy as np
from matplotlib import pyplot as plt
import networkx as nx


def plot_ave_node_values(graph, value_per_nugget, as_efficiency=None, show=True, save_fig=False):
    fig = plt.figure(figsize=(10, 4))
    if as_efficiency:
        if hasattr(graph, 'alpha'):  #
            print('No efficiency metric for shortest path (sum eff distance) methodology implemented')
            # plt.plot(graph.nodes[:].sum(axis=1) / (graph.nodes.shape[1] * value_per_nugget))
            # plt.title(f'Sum Node values as % of total, {value_per_nugget} nugget value')
            # plt.xlabel('Time step')
            # plt.ylabel(f'Information diffused')
        else:
            plt.plot(graph.nodes[:-2].sum(axis=1) / (graph.nodes.shape[1] * value_per_nugget))
            plt.title(f'Sum Node values as % of possible, {value_per_nugget} nugget value')
            plt.xlabel('Time step')
            plt.ylabel(f'Information diffused')
    else:
        plt.plot(graph.nodes[:-2].mean(axis=1))
        plt.title(f'Average node values, {value_per_nugget} nugget value')
        plt.xlabel('Time step')
        plt.ylabel(f'Average node values')
    if save_fig:
        plt.savefig(f'Ave_Node_Values {graph.nodes.shape[0]} runs.png')
    if show:
        plt.show()


def plot_node_edges(graph, node, num_nodes, num_runs, value_per_nugget, show=True, save_fig=False):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(graph.A[:, :, node])
    plt.title(f'{num_nodes} nodes, {num_runs} runs, {value_per_nugget} nugget value, default parameters')
    plt.xlabel('Time step')
    plt.ylabel(f'{node}th node\'s incoming edge values')
    if save_fig:
        plt.savefig(f'edge values of {node} node with {value_per_nugget} seed_val {graph.nodes.shape[0]} runs.png')
    if show:
        plt.show()


def plot_node_values(graph, value_per_nugget, node='all', show=True, save_fig=False):
    fig = plt.figure(figsize=(10, 4))
    if node == 'all':
        plt.plot(graph.nodes)
        plt.title(f'All nodes\' values, {value_per_nugget} nugget value, default parameters')
        plt.xlabel('Time step')
        plt.ylabel(f'Nodes values')  # reveals it generally gets all the information!
    else:
        plt.plot(graph.nodes[:-2, node])
        plt.title(f'{node}\'th Node\'s values, {value_per_nugget} nugget value, default parameters')
        plt.xlabel('Time step')
        plt.ylabel(f'{node}th node\'s values')  # reveals it generally gets all the information!
    if save_fig:
        plt.savefig(f'{node} node_values with {value_per_nugget} seed_val {graph.nodes.shape[0]} runs.png')
    if show:
        plt.show()


def plot_edge_stds(graph, node, num_nodes, value_per_nugget, show=True, all_nodes=False, save_fig=False):
    edge_std_for_all_nodes = np.zeros((num_nodes, graph.A[:, 0, 0].size))
    for Node in range(0, graph.A[0][-1].size):  # evaluates standard deviations, Node capitalized to distinguish scope
        edge_std_for_all_nodes[Node] = np.std(graph.A[:, Node], axis=1)
        # edge_std_for_all_nodes[Node] = [edge_values.std() for edge_values in graph.A[:, Node][:]]  # less efficient?
    if all_nodes:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(edge_std_for_all_nodes.T)
        plt.title(f'standard deviations, {graph.nodes.shape[0]} runs, {value_per_nugget} nugget value')
        plt.xlabel('Time step')
        plt.ylabel(f'std of all node edges')
        if save_fig:
            plt.savefig(f'std_of_all_node_edges with {value_per_nugget} seed_val {graph.nodes.shape[0]} runs.png')
        if show:
            plt.show()
    else:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(edge_std_for_all_nodes[node, :])
        plt.title(f'standard deviations, {graph.nodes.shape[0]} runs, {value_per_nugget} nugget value')
        plt.xlabel('Time steps')
        plt.ylabel(f'std of {node}th node\'s edges')
        if save_fig:
            plt.savefig(f'std_of_node_{node}_edges with {value_per_nugget} seed_val {graph.nodes.shape[0]} runs.png')
        if show:
            plt.show()


def plot_single_network(graph, timestep, directed=True, show=True, save_fig=False):
    fig = plt.figure(figsize=(10, 10))
    if directed:
        nx_G = nx.to_directed(nx.from_numpy_matrix(np.array(graph.A[timestep])))
        incoming_edge_sum = graph.A[timestep].sum(axis=1)
    else:
        nx_G = nx.from_numpy_matrix(np.array(graph.A[timestep]))
        incoming_edge_sum = graph.A[timestep].sum(axis=1)
    # nx_G = nx.convert_matrix.from_numpy_matrix(np.array(graph.A[timestep]), create_using=nx.Digraph)
    # This should be better, but nx.Digraph is not recognized?
    incoming_edge_sum = [200 * node / sum(incoming_edge_sum) for node in incoming_edge_sum]
    pos = nx.spring_layout(nx_G, k=0.5, scale=0.5, weight='weight')
    weights = [nx_G[u][v]['weight'] * 1.5 for u, v in nx_G.edges()]
    nx.draw_networkx_edges(nx_G, pos, nodelist=['0'], alpha=0.8, width=weights)
    edge_colors = range(2, nx_G.number_of_edges() + 2)
    node_colors = range(2, nx_G.number_of_nodes() + 2)
    print(f'weights: {np.round(np.array(weights), 2)}')
    nx.draw_networkx_nodes(nx_G, pos,
                           arrowstyle='->',
                           edge_color=edge_colors,
                           node_size=incoming_edge_sum,
                           node_color='grey',
                           # node_color=node_colors,
                           widths=weights,
                           cmap=plt.get_cmap('viridis'))
    plt.title(f"timestep: {timestep} Nodes size propto incoming edge weights")
    plt.show()
    if save_fig:
        plt.savefig(f'Network Structure(s) after {graph.nodes.shape[0]} runs.png')
    if show:
        plt.show()


def plot_network(graph, value_per_nugget, directed=True, show=True, save_fig=False):
    fig = plt.figure(figsize=(12, 6))
    count = 1
    for timestep in [0, int(graph.nodes.shape[0]/3), int(graph.nodes.shape[0]*(2/3)), (graph.nodes.shape[0])]:
        if directed:
            nx_G = nx.to_directed(nx.from_numpy_matrix(np.array(graph.A[timestep])))
            incoming_edge_sum = graph.A[timestep].sum(axis=1)
            if count == 1:
                pos = nx.spring_layout(nx_G, k=0.5, scale=0.5, weight='weight')
        else:
            nx_G = nx.from_numpy_matrix(np.array(graph.A[timestep]))
            incoming_edge_sum = graph.A[timestep].sum(axis=1)
            if count == 1:
                pos = nx.spring_layout(nx_G, k=0.5, scale=5.0, weight='weight')
        plt.subplot(1, 4, count)
        count += 1
        weights = [nx_G[u][v]['weight'] * 1.5 for u, v in nx_G.edges()]
        # colors=[G.node[n]['color'] for n in G.node()]
        incoming_edge_sum = [(200*node/sum(incoming_edge_sum)) for node in incoming_edge_sum]
        nx.draw_networkx_edges(nx_G, pos, nodelist=['0'], alpha=0.8, width=weights)
        edge_colors = range(2, nx_G.number_of_edges() + 2)
        nx.draw_networkx_nodes(nx_G, pos,
                               arrowstyle='->',
                               edge_color=edge_colors,
                               node_size=incoming_edge_sum,
                               node_color='grey',
                               widths=weights,
                               cmap=plt.get_cmap('viridis'))
        plt.title("timestep: {0}".format(timestep))
    if save_fig:
        plt.savefig(f'Network Structure(s) for {value_per_nugget} nugget value, {graph.nodes.shape[0]} runs.png')
    if show:
        plt.show()
