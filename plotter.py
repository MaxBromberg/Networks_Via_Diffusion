import numpy as np
from matplotlib import pyplot as plt
import networkx as nx


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


def plot_node_value_over_time(graph, node, value_per_nugget, show=True, save_fig=False):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(graph.nodes[:, node])
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


def plot_network(graph, value_per_nugget, show=True, save_fig=False):
    count = 1
    for timestep in [0, int(graph.nodes.shape[0]/3), int(graph.nodes.shape[0]*(2/3)), (graph.nodes.shape[0]-1)]:
        nx_G = nx.from_numpy_matrix(np.array(graph.A[timestep]))
        if count == 1:
            pos = nx.spring_layout(nx_G, k=0.5, scale=5.0, weight='weight')
        plt.subplot(1, 4, count)
        count += 1
        weights = [nx_G[u][v]['weight'] * 1.5 for u, v in nx_G.edges()]
        # colors=[G.node[n]['color'] for n in G.node()]
        nx.draw_networkx_edges(nx_G, pos, nodelist=['0'], alpha=0.8, width=weights)
        nx.draw_networkx_nodes(nx_G, pos,
                               node_size=20,
                               node_color='grey',
                               widths=weights,
                               cmap=plt.get_cmap('viridis'))
        plt.title("timestep: {0}".format(timestep))
    if save_fig:
        plt.savefig(f'Network Structure(s) for {value_per_nugget} seed_val {graph.nodes.shape[0]} runs.png')
    if show:
        plt.show()
