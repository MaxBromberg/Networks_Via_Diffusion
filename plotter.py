import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import multiprocessing as mp

import pickle  # used to save data (as class object)
import imageio  # Used for making gifs of the network plots
import os  # Used for putting the gifs somewhere
from pathlib import Path  # used for file path compatibility between operating systems
import time

import graph as g
import utility_funcs


def plot_ave_node_values(graph, value_per_nugget, as_efficiency=None, show=True, save_fig=False):
    assert show or save_fig, 'Graph will be neither shown nor saved'
    fig = plt.figure(figsize=(10, 4))
    if as_efficiency:
        if hasattr(graph, 'alpha'):  #
            print('No efficiency metric for shortest path (sum eff distance) methodology implemented')
            # plt.plot(graph.nodes[:].sum(axis=1) / (graph.nodes.shape[1] * value_per_nugget))
            # plt.title(f'Sum Node values as % of total, {value_per_nugget} nugget value')
            # plt.xlabel('Time step')
            # plt.ylabel(f'Information diffused')
        else:
            plt.plot(graph.nodes[:-1].sum(axis=1) / (graph.nodes.shape[1] * value_per_nugget))
            plt.title(f'Sum Node values as % of possible, {value_per_nugget} nugget value')
            plt.xlabel('Time step')
            plt.ylabel(f'Information diffused')
    else:
        plt.plot(graph.nodes[:-1].mean(axis=1))
        plt.title(f'Average node values, {value_per_nugget} nugget value')
        plt.xlabel('Time step')
        plt.ylabel(f'Average node values')
    if save_fig:
        plt.savefig(f'Ave_Node_Values {graph.nodes.shape[0]} runs.png')
    if show:
        plt.show()


def plot_node_edges(graph, node, num_nodes, num_runs, value_per_nugget, show=True, save_fig=False):
    assert show or save_fig, 'Graph will be neither shown nor saved'
    fig = plt.figure(figsize=(10, 4))
    plt.plot(graph.A[:, :, node])
    plt.title(f'{num_nodes} nodes, {num_runs} runs, {value_per_nugget} nugget value, default parameters')
    plt.xlabel('Time step')
    plt.ylabel(f'{node}th node\'s incoming edge values')
    if save_fig:
        plt.savefig(f'edge values of {node} node with {value_per_nugget} seed_val {graph.nodes.shape[0]} runs.png')
    if show:
        plt.show()


def plot_global_eff_dist(graph, fit=False, normalized=True, show=True, save_fig=False, title=None):
    fig = plt.figure(figsize=(12, 6))
    mean_eff_dist_history = np.mean(graph.eff_dist_history, axis=1)
    x = np.array(range(len(mean_eff_dist_history)))
    if normalized:
        y = np.array(mean_eff_dist_history) / np.amax(mean_eff_dist_history)
    else:
        y = mean_eff_dist_history
    plt.plot(x, y)
    plt.title(f'Total Effective Distance history')
    plt.xlabel('Time step')
    plt.ylabel(f'Total Effective distance')

    if fit:
        if fit == 'log':
            # log_fit = np.polyfit(np.log(x), y, 1, w=np.sqrt(y))
            # plt.plot(x, np.exp(log_fit[1])*np.exp(log_fit[0]*x))
            # a, b = optimize.curve_fit(lambda t, a, b: a * np.exp(b * t), x, y, p0=(1, 0.5))
            # plt.plot(x, a[0] * np.exp(a[1] * x))
            print("Logarithmic/exponential fitting encounters inf/NaN errors in regression :(")
        if fit == 'linear':
            linear_fit = np.polyfit(x, y, 1, w=np.sqrt(y))
            plt.plot(x, linear_fit[0]*x + linear_fit[1])
        if fit == 'average':
            ave_range = int(len(y)/20)
            assert ave_range % 2 == 0, 'Average range must be even (lest, for this algorithm). Default range is ave_range = int(len(y)/20)'
            half_range = int((ave_range/2))
            averaging_fit = [np.mean(y[index-half_range:index+half_range]) for index in x[half_range:-half_range]]
            # averaging_fit = np.insert(averaging_fit, 0, y[:half_range])
            # averaging_fit = np.insert(averaging_fit, averaging_fit.shape[0], y[-half_range:])
            plt.plot(averaging_fit)
    if show:
        plt.show()
    if title:
        plt.savefig(f'{title}.png')
        plt.close(fig)
    if save_fig:
        plt.savefig(f'Effective_Distance for edge_to_eff_dist_coupling of {graph.eff_dist_and_edge_coupling}.png')
        plt.close(fig)


def plot_node_values(graph, node='all', show=True, save_fig=False, title=None):
    assert show or save_fig or title, 'Graph will be neither shown nor saved'
    fig = plt.figure(figsize=(10, 4))
    if node == 'all':
        plt.plot(graph.nodes)
        plt.title(f'All nodes\' values')
        plt.xlabel('Time step')
        plt.ylabel(f'Nodes values')  # reveals it generally gets all the information!
    else:
        plt.plot(graph.nodes[:-2, node])
        plt.title(f'{node}\'th Node\'s values')
        plt.xlabel('Time step')
        plt.ylabel(f'{node}th node\'s values')  # reveals it generally gets all the information!
    if save_fig:
        plt.savefig(f'{node} node_values with edge_to_eff_dist_coupling of {np.round(graph.eff_dist_and_edge_coupling, 2)} and {graph.nodes.shape[0]} runs.png')
    if title:
        plt.savefig(f'{title}.png')
        plt.close(fig)
    if show:
        plt.show()


def plot_edge_sum(graph, node=None, incoming_edges=False, show=True, save_fig=False):
    # incoming edge sum only relevant if they are not normalized
    assert show or save_fig, 'Graph will be neither shown nor saved'
    edge_sums = graph.A.sum(axis=2)  # returns sums of rows for every timestep
    fig = plt.figure(figsize=(10, 4))
    if incoming_edges:
        edge_sums = graph.A.sum(axis=1)  # returns sums of columns for every timestep
    if node or node == 0:
        plt.plot(edge_sums[:, node])
        if incoming_edges:
            plt.plot(edge_sums[node, :])
        plt.title(f'sum of node {node} edges')
        plt.xlabel('Time steps')
        plt.ylabel(f'Sum of {node}th node\'s edges')
        if save_fig:
            plt.savefig(f'sum of node {node} edges.png')
        if show:
            plt.show()
    else:
        plt.plot(edge_sums)
        plt.title(f'Sum of every node edges')
        plt.xlabel('Time steps')
        plt.ylabel(f'Sum of every nodes\' edges')
        if save_fig:
            plt.savefig(f'sum of every node edges.png')
        if show:
            plt.show()


def plot_edge_stds(graph, node, num_nodes, value_per_nugget, show=True, all_nodes=False, save_fig=False):
    assert show or save_fig, 'Graph will be neither shown nor saved'
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


def plot_adjacency_matrix_as_heatmap(graph, timestep=-1, show=True, save_fig=False):
    """
    Returns adjacency matrix at timestep plotted as a heat map. Default timestep is the latest value.
    """
    assert show or save_fig, 'Graph will be neither shown nor saved'
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(graph.A[timestep], cmap='viridis')
    plt.colorbar()
    if timestep == -1:
        plt.title(f'Adjacency Matrix at final timestep as heat map')
    else:
        plt.title(f'Adjacency Matrix at timestep {timestep} as heat map')
    if save_fig:
        plt.savefig(f'Adjacency Matrix heat map at run {timestep}.png')
    if show:
        plt.show()


def plot_weight_histogram(graph, num_bins=False, timestep=-1, show=True, save_fig=False):
    edges = (graph.A[timestep]).flatten()
    fig = plt.figure(figsize=(10, 10))
    if num_bins:
        plt.hist(edges, bins=num_bins)
    else:
        plt.hist(edges)  # bins = auto, as per np.histogram
    if timestep == -1:
        plt.title(f"Weight histogram for all edges final timestep ({graph.A.shape[0]-1})")
    else:
        plt.title(f"Weight histogram for all edges timestep: {timestep} ")
    if save_fig:
        plt.savefig(f'Weight histogram with {num_bins} bins.png')
    if show:
        plt.show()


def plot_effective_distance_histogram(eff_dists, num_bins=False, timestep=-1, show=True, save_fig=False):
    eff_dists = eff_dists.flatten()
    fig = plt.figure(figsize=(10, 10))
    if num_bins:
        plt.hist(eff_dists, bins=num_bins)
    else:
        plt.hist(eff_dists)  # bins = auto, as per np.histogram
    if timestep == -1:
        plt.title(f"Effective distance histogram for all to all edges final timestep")
    else:
        plt.title(f"Effective distance histogram for all to all paths timestep: {timestep} ")
    if save_fig:
        plt.savefig(f'Effective distance histogram at step {timestep}.png')
    if show:
        plt.show()


def plot_single_network(graph, timestep, directed=True, node_size_scaling=200, source_weighting=False, position=None, show=True, save_fig=False, title=None):
    fig = plt.figure(figsize=(10, 10))
    if directed:
        nx_G = nx.to_directed(nx.from_numpy_matrix(np.array(graph.A[timestep]), create_using=nx.DiGraph))
    else:
        nx_G = nx.from_numpy_matrix(np.array(graph.A[timestep]))

    if position:  # allows for setting a constant layout
        pos = nx.spring_layout(nx_G, weight='weight', pos=position, fixed=list(nx_G.nodes))
    else:
        pos = nx.spring_layout(nx_G, k=0.5, scale=0.5, weight='weight', seed=42)
    # pos = nx.drawing.layout.spring_layout(nx_G, k=0.5, pos=pos, weight='weight', fixed=list(nx_G.nodes))
    weights = [nx_G[u][v]['weight'] * 1.5 for u, v in nx_G.edges()]
    nx.draw_networkx_edges(nx_G, pos, nodelist=['0'], alpha=0.8, width=weights, arrowsize=4,
                           connectionstyle='arc3, rad=0.2')
    edge_colors = range(2, nx_G.number_of_edges() + 2)
    node_colors = ['grey' for _ in nx_G]
    node_colors[graph.source_node_history[timestep - 1]] = 'red'
    incoming_edge_sum = graph.A[timestep].sum(axis=1)
    incoming_edge_sum = [node_size_scaling * node / sum(incoming_edge_sum) for node in incoming_edge_sum]
    if source_weighting:  # sizes nodes proportional to the number of times they've been a source
        source_weight = [graph.source_node_history.count(node) for node in range(graph.nodes.shape[1])]
        source_weight = [node_size_scaling*(weight/sum(source_weight)) for weight in source_weight]
        nx.draw_networkx_nodes(nx_G, pos,
                               arrowstyle='->',
                               edge_color=edge_colors,
                               node_size=source_weight,
                               node_color=node_colors,
                               widths=weights,
                               cmap=plt.get_cmap('viridis'))
        plt.title(f"Nodes size proportional to number of times they've been the source [timestep: {timestep}]")
    else:
        nx.draw_networkx_nodes(nx_G, pos,
                               arrowstyle='->',
                               edge_color=edge_colors,
                               node_size=incoming_edge_sum,
                               node_color=node_colors,
                               widths=weights,
                               cmap=plt.get_cmap('viridis'))
        plt.title(f"Nodes size proportional to incoming edge weights [timestep: {timestep}]")
    if title:
        plt.savefig(f'{title}.png')
        plt.close(fig)
    if show:
        plt.show()
    if save_fig and not title:
        plt.savefig(f'Network Structure(s) after {graph.nodes.shape[0]} runs.png')
        plt.close(fig)


def plot_network(graph, directed=True, node_size_scaling=200, nodes_sized_by_eff_distance=False,
                 show=True, save_fig=False, title=None):
    fig = plt.figure(figsize=(12, 6))
    assert show or save_fig or title, 'Graph will be neither shown nor saved'
    count = 1
    timesteps = [0, int(graph.nodes.shape[0] / 3), int(graph.nodes.shape[0] * (2 / 3)), (graph.nodes.shape[0])-1]
    for timestep in timesteps:
        if directed:
            nx_G = nx.to_directed(nx.from_numpy_matrix(np.array(graph.A[timestep]), create_using=nx.DiGraph))
            if count == 1:
                pos = nx.spring_layout(nx_G, k=0.5, scale=0.5, weight='weight')
                # pos = nx.spring_layout(nx_G.reverse(copy=True), k=0.5, scale=0.5, weight='weight')
                # Transposing is likely the intended effect, and more readily done
        else:
            nx_G = nx.from_numpy_matrix(np.array(graph.A[timestep]))
            if count == 1:
                pos = nx.spring_layout(nx_G, k=0.5, scale=5.0, weight='weight')
                # pos = nx.spring_layout(nx_G.reverse(copy=True), k=0.5, scale=0.5, weight='weight')
                # Transposing is likely the intended effect
        incoming_edge_sum = graph.A[timestep].sum(axis=1)
        plt.subplot(1, 4, count)
        count += 1
        weights = [nx_G[u][v]['weight'] * 1.5 for u, v in nx_G.edges()]
        # colors=[G.node[n]['color'] for n in G.node()]
        incoming_edge_sum = [(node_size_scaling * node / sum(incoming_edge_sum)) for node in incoming_edge_sum]
        edge_colors = range(2, nx_G.number_of_edges() + 2)
        node_colors = ['grey'] * graph.nodes.shape[1]
        node_colors[graph.source_node_history[timestep]] = 'red'
        nx.draw_networkx_edges(nx_G, pos, nodelist=['0'], alpha=0.8, width=weights, arrowsize=4, connectionstyle='arc3, rad=0.2')
        nx.draw_networkx_nodes(nx_G, pos,
                               arrowstyle='->',
                               edge_color=edge_colors,
                               node_size=incoming_edge_sum,
                               node_color=node_colors,
                               widths=weights,
                               cmap=plt.get_cmap('viridis'))
        plt.title("timestep: {0}".format(timestep))
        if nodes_sized_by_eff_distance:
            nx.draw_networkx_nodes(nx_G, pos,
                                   arrowstyle='->',
                                   edge_color=edge_colors,
                                   node_size=graph.nodes,
                                   node_color=node_colors,
                                   widths=weights,
                                   cmap=plt.get_cmap('viridis'))
        plt.title("timestep: {0}".format(timestep))
    if show:
        plt.show()
    if title:
        plt.savefig(f'{title}.png')
        plt.close(fig)
    if save_fig:
        plt.savefig(f'Network Structure(s) for edge_to_eff_dist_coupling of {np.round(graph.eff_dist_and_edge_coupling, 2)}, {graph.nodes.shape[0]} runs.png')
        plt.close(fig)


def gif_of_network_evolution(graph, node_size_scaling=200, source_weighting=False, directory_name='network_animation', file_title='network_evolution', parent_directory=None, gif_duration_in_sec=5,  num_runs_per_fig=None, verbose=False):
    """
    Creates a gif and mp4 of the network evolution and stores them in a folder with the individual frames.
    """

    assert num_runs_per_fig != 0, 'Number of runs per figure must be larger than 0, or else omitted for graph every run'

    if parent_directory is None:
        source_directory = os.path.dirname(__file__)
    else:
        source_directory = parent_directory
    vid_path = Path(source_directory, directory_name)
    fig_path = Path(vid_path, 'figures')
    try:
        os.mkdir(vid_path), f'Created folder for network structure gif at {vid_path}'
        os.mkdir(fig_path), f'Created folder for figures at {fig_path}'
    except OSError:
        print(f'{vid_path} already exists, adding or overwriting contents')
        pass

    nx_G = nx.to_directed(nx.from_numpy_matrix(np.array(graph.A[0]), create_using=nx.DiGraph))
    initial_position = nx.drawing.layout.spring_layout(nx_G, k=0.5, scale=0.5, weight='weight')
    for i in range(0, graph.A.shape[0]):
        files = Path(fig_path, f'{i:05}')
        if num_runs_per_fig:
            if i % num_runs_per_fig == 0:
                plot_single_network(graph, i, node_size_scaling=node_size_scaling, source_weighting=source_weighting,
                                    position=initial_position, show=False, save_fig=True, title=files)
        else:
            plot_single_network(graph, i, node_size_scaling=node_size_scaling, source_weighting=source_weighting,
                                position=initial_position, show=False, save_fig=True, title=files)
        if verbose:
            if int(i % graph.A.shape[0]) % int(graph.A.shape[0] / 10) == 0:
                print(f'{(i / graph.A.shape[0]) * 100:.1f}%-ish done')
            if i == graph.A.shape[0]-1:
                print('Now creating video from rendered images... (ignore resolution reformatting error)')

    if num_runs_per_fig:
        writer = imageio.get_writer(f'{Path(vid_path, file_title)}.mp4',
                                    fps=((graph.A.shape[0] / num_runs_per_fig) / gif_duration_in_sec))
    else:
        writer = imageio.get_writer(f'{Path(vid_path, file_title)}.mp4', fps=(graph.A.shape[0] / gif_duration_in_sec))

    images = []
    for filename in sorted(os.listdir(fig_path)):
        if filename.endswith(".png"):
            images.append(imageio.imread(Path(fig_path, filename)))
            writer.append_data(imageio.imread(Path(fig_path, filename)))
    imageio.mimsave(f'{Path(vid_path, file_title)}.gif', images)
    writer.close()
    if verbose:
        print(f'gif and mp4 of network evolution created in {vid_path} \n Stills stored in {fig_path}')


def plot_3d(function, x_range, y_range=None, piecewise=False, z_limits=None, spacing=0.05):
    """
    :param function: z(x,y)
    :param x_range: [lower bound, upper bound]
    :param y_range: defaults to x_range, otherwise list as [lower bound, upper bound]
    :param piecewise: set true if function is piecewise (i.e. contains conditional)
    :param z_limits: [lower bound, upper bound]
    :param spacing: interval between both x and y ranges.
    """
    if y_range is None:
        y_range = x_range
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca(projection='3d')
    X = np.arange(x_range[0], x_range[1], spacing)
    Y = np.arange(y_range[0], y_range[1], spacing)
    if piecewise:
        Z = np.zeros((len(X), len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                Z[i][j] = function(X[i], Y[j])
        X, Y = np.meshgrid(X, Y)
    else:
        X, Y = np.meshgrid(X, Y)
        Z = function(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.winter, linewidth=0, antialiased=False)
    if z_limits:
        ax.set_zlim(z_limits[0], z_limits[1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot_3d_data(three_d_data, x_range=None, y_range=None, z_range=None, show=True, raw_fig_title=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    data = np.swapaxes(np.swapaxes(three_d_data, 0, 1), 1, 2)  # sliding node_num axis to last
    xs = np.repeat(x_range, data.shape[1] * data.shape[2], axis=0).flatten()
    ys = np.repeat([y_range], data.shape[0] * data.shape[2], axis=0).flatten()
    zs = np.repeat([z_range], data.shape[0] * data.shape[1], axis=0).flatten()
    print(f'data.shape {data.shape}')
    data = np.round(data.flatten(), 6)
    # print(f'xs: ys: zs: \n {xs}, \n {ys}, \n {zs}')
    # print(f'xs.size, ys.size, zs.size, data.size: {xs.size}, {ys.size}, {zs.size}, {data.size}')
    logged_data = np.log(abs(data))
    # TODO: here the questionable assumption of the data being entirely positive/negative is used. FIX
    img = ax.scatter(xs, ys, zs, c=logged_data, cmap=plt.winter())
    fig.colorbar(img)
    ax.set_xlabel('Coupling')
    ax.set_ylabel('Adaptation')
    ax.set_zlabel('Nodes')
    if show:
        plt.show()
    if raw_fig_title:
        save_object(plt.figure(), str(raw_fig_title)+".pkl")


def plot_clustering_coefficients(nx_graphs, source=False, average_clustering=False, show=True, save_fig=False, title=None):
    """
    :param source: if not None, computes ave_clustering for the single (presumably source) node
    """
    if source:
        if average_clustering:
            clustering_coefficients = [nx.average_clustering(nx_graphs[i], weight='weight', nodes=[source]) for i in range(len(nx_graphs))]
        else:
            clustering_coefficients = [nx.clustering(nx_graphs[i], weight='weight', nodes=[source])[0] for i in range(len(nx_graphs))]
    else:
        if average_clustering:
            clustering_coefficients = [nx.average_clustering(nx_graphs[i], weight='weight') for i in range(len(nx_graphs))]
        else:
            clustering_coefficients = np.array([list(nx.clustering(nx_graphs[i], weight='weight').values()) for i in range(len(nx_graphs))])

    fig = plt.figure(figsize=(12, 6))
    plt.plot(clustering_coefficients)
    plt.xlabel('Time steps')
    plt.ylabel(f'Clustering Coefficient')

    if source and average_clustering or source is 0 and average_clustering:
        plt.title(f'Average Clustering Coefficients for node [{source}]')
    elif source or source is 0:
        plt.title(f'Clustering Coefficients for node [{source}]')
    elif average_clustering:
        plt.title(f'Average Clustering Coefficients')
    else:
        plt.title(f'Clustering Coefficients [for all nodes]')

    if title:
        plt.savefig(f'{title}.png')
        plt.close(fig)
    if save_fig:
        plt.savefig(f'Clustering Coefficients.png')
    if show:
        plt.show()


def plot_ave_neighbor_degree(nx_graphs, source='in', target='in', node=False, show=True, save_fig=False, title=None):
    """
    :param source: if not None, computes ave_neighborhood degree for the single (presumably source) node
    """
    if node:
        ave_neighbor_degree = [list(nx.average_neighbor_degree(nx_graphs[t], nodes=[node], source=source, target=target, weight='weight').values()) for t in range(len(nx_graphs))]
    else:
        ave_neighbor_degree = [list(nx.average_neighbor_degree(nx_graphs[t], source=source, target=target, weight='weight').values()) for t in range(len(nx_graphs))]

    fig = plt.figure(figsize=(12, 6))
    plt.plot(ave_neighbor_degree)
    plt.xlabel('Time steps')
    plt.ylabel(f'Average Neighbor Degree')

    if node or node is 0:
        plt.title(f'Neighbor Degree for node [{node}], target {target}, source {source}')
    else:
        plt.title(f'Neighbor Degree [for all nodes], target {target}, source {source}')

    if title:
        plt.savefig(f'{title}.png')
        plt.close(fig)
    if save_fig:
        plt.savefig(f'Neighbor_Degree.png')
    if show:
        plt.show()


def plot_shortest_path_length(nx_graphs, show=True, save_fig=False, title=None):
    """
    Requires fully connected graph (no islands0 though could be modified to allow for analysis of disprate components
    """
    ave_shortest_path_length = [nx.average_shortest_path_length(nx_graphs[t], weight='weight') for t in range(len(nx_graphs))]

    fig = plt.figure(figsize=(12, 6))
    plt.plot(ave_shortest_path_length)
    plt.xlabel('Time steps')
    plt.ylabel(f'Average Shortest Path Length')
    plt.title(f'Average Shortest Paths')

    if title:
        plt.savefig(f'{title}.png')
        plt.close(fig)
    if save_fig:
        plt.savefig(f'Average_Shortest_Path_Lengths.png')
    if show:
        plt.show()


def plot_heatmap(TwoD_data, x_range=None, y_range=None, normalize=False, title=None):
    if normalize:  # though data is automatically normalized in output heatmap.
        data = np.array(TwoD_data / np.amax(np.array(TwoD_data)))
    else:
        data = TwoD_data
    if x_range.any(): plt.xticks(x_range)
    if y_range.any(): plt.yticks(y_range)
    x_interval = (x_range[2]-x_range[1])
    y_interval = (y_range[2]-y_range[1])
    plt.imshow(data, cmap='viridis', extent=[x_range[0], x_range[-1]+x_interval, y_range[0], y_range[-1]+y_interval], aspect='auto')
    plt.colorbar()
    plt.xlabel('Skew')
    plt.ylabel('Coupling')
    if title:
        plt.savefig(f'{title}.png')
        plt.close()


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        del obj


def open_graph_obj(path, graph_id):
    """
    :param path: abs path to (.pkl) file
    :param graph_id: id as string (e.g. '0012')
    :return: [graph] class object
    """
    with open(Path(path, f'{graph_id}_graph_obj.pkl'), 'rb') as input:
        return pickle.load(input)


def open_figure(path, filename):
    # TODO: Evidently does not work with present implementation of matplotlib for interactive plots (e.g. 4d heatmap)
    if filename.endswith('.pkl'):
        with open(Path(path, filename), 'rb') as fig:
            ax = pickle.load(fig)
    else:
        with open(Path(path, (str(filename)+'.pkl')), 'rb') as fig:
            ax = pickle.load(fig)
    plt.show()


def three_d_plot_from_data(path_to_data_dir, coupling_range, skew_range, normalized=False, output_dir=None):
    if output_dir is None:
        output_dir = path_to_data_dir
    eff_dists_all_nodes = np.zeros((1, coupling_range.size, skew_range.size))
    ave_nbr_var_all_nodes = np.zeros((1, coupling_range.size, skew_range.size))
    eff_dist_diffs_flattened = []
    ave_nbr_var_flattened = []
    node_nums = []
    node_files = []

    sub_dirs = sorted([dirs[0] for dirs in os.walk(path_to_data_dir) if dirs[0] != str(path_to_data_dir)])
    for sub_dir in sub_dirs:
        node_nums.append(int(str(str(sub_dir).split('/')[-1]).split('_')[-1]))
        tmp_data_files = [files[2] for files in os.walk(Path(path_to_data_dir, sub_dir))]
        node_files.append(sorted(tmp_data_files[0]))
    node_files = np.array(node_files)

    for node_index in range(node_files.shape[0]):
        for file_index in range(node_files.shape[1]):
            with open(Path(sub_dirs[node_index], node_files[node_index][file_index]), 'rb') as input:
                G = pickle.load(input)
                input.close()
                last_ave_nbr_deg = list(nx.average_neighbor_degree(G.convert_to_nx_graph(timestep=-1), source='in', target='in', weight='weight').values())
                ave_nbr_var_flattened.append(np.array(last_ave_nbr_deg).var())
                eff_dist_diffs_flattened.append(G.eff_dist_diff(multiple_path_eff_dist=False))  # Compares first and last eff_dist values
        eff_dists_all_nodes = np.vstack((eff_dists_all_nodes, [np.array(eff_dist_diffs_flattened).reshape(coupling_range.size, skew_range.size)]))
        ave_nbr_var_all_nodes = np.vstack((ave_nbr_var_all_nodes, [np.array(ave_nbr_var_flattened).reshape(coupling_range.size, skew_range.size)]))
        eff_dist_diffs_flattened = []
        ave_nbr_var_flattened = []

    eff_dists_all_nodes = np.delete(eff_dists_all_nodes, 0, axis=0)
    ave_nbr_var_all_nodes = np.delete(ave_nbr_var_all_nodes, 0, axis=0)
    if normalized:
        eff_dists_all_nodes /= np.amax(eff_dists_all_nodes)
        ave_nbr_var_all_nodes /= np.amax(ave_nbr_var_all_nodes)

    plot_3d_data(eff_dists_all_nodes, x_range=coupling_range, y_range=skew_range, z_range=np.array(node_nums), raw_fig_title=Path(output_dir, "Eff_dist_diff_Grid_Search"))
    plot_3d_data(ave_nbr_var_all_nodes, x_range=coupling_range, y_range=skew_range, z_range=np.array(node_nums), raw_fig_title=Path(output_dir, "Ave_nbr_var_Grid_Search"))


def heatmaps_from_data(path_to_data_dir, coupling_range, skew_range, output_dir=None, normalize=False):
    if output_dir is None:
        output_dir = path_to_data_dir
    f = []
    eff_dist_diffs_flattened = []
    ave_neighbor_diffs_flattened = []
    ave_nbr_var_flattened = []
    for root, dirs, files in os.walk(path_to_data_dir):
        f = sorted(files)
    for file in f:  # Order preserved due to 0 padding.
        with open(Path(path_to_data_dir, file), 'rb') as input:
            G = pickle.load(input)
            input.close()
            last_ave_nbr_deg = list(nx.average_neighbor_degree(G.convert_to_nx_graph(timestep=-1), source='in', target='in', weight='weight').values())
            eff_dist_diffs_flattened.append(G.eff_dist_diff(multiple_path_eff_dist=False))  # Compares first and last eff_dist values
            ave_neighbor_diffs_flattened.append((lambda x: max(x) - min(x))(last_ave_nbr_deg))
            ave_nbr_var_flattened.append(np.array(last_ave_nbr_deg).var())
    plot_heatmap(np.array(eff_dist_diffs_flattened).reshape(coupling_range.size, skew_range.size), title=Path(output_dir, f'eff_dist_histogram'), x_range=skew_range, y_range=coupling_range, normalize=normalize)
    plot_heatmap(np.array(ave_neighbor_diffs_flattened).reshape(coupling_range.size, skew_range.size), title=Path(output_dir, f'ave_neighbor_diff_histogram'), x_range=skew_range, y_range=coupling_range, normalize=normalize)
    plot_heatmap(np.array(ave_nbr_var_flattened).reshape(coupling_range.size, skew_range.size), title=Path(output_dir, f'ave_neighbor_var_histogram'), x_range=skew_range, y_range=coupling_range, normalize=normalize)


def data_only_grid_search(num_nodes, num_runs, exp_decay_param, coupling_range, coupling_interval, skew_range, skew_interval, num_shifts_of_source_node=False, parent_directory=None, directory_name='grid_search', verbose=False):
    if parent_directory is None:
        source_directory = os.path.dirname(__file__)
    else:
        source_directory = parent_directory
    grid_search_dir = Path(source_directory, directory_name+f'_{num_nodes}_nodes')
    graphs_path = Path(grid_search_dir, 'graphs_path')
    try:
        os.mkdir(grid_search_dir), f'Created folder for grid search results at {grid_search_dir}'
        os.mkdir(graphs_path), f'Created folder for graph objects at {graphs_path}'
    except OSError:
        print(f'{grid_search_dir} already exists, adding or overwriting contents')
        pass

    if verbose:
        print(f'Beginning grid search, Num_nodes: {num_nodes}, num_rums: {num_runs}, responsiveness, adaptation_rate ranges {coupling_range, skew_range} and intervals {coupling_interval, skew_interval}')
    full_coupling_range = np.arange(coupling_range[0], coupling_range[1], coupling_interval)
    rate_of_skew_range = np.arange(skew_range[0], skew_range[1], skew_interval)
    run_counter = 0

    eff_dist_diffs_flattened = []
    ave_neighbor_diffs_flattened = []
    ave_nbr_var_flattened = []
    for coupling_val in full_coupling_range:
        for skew_value in rate_of_skew_range:
            numbered_graph_path = Path(graphs_path, f'{run_counter+1:04}_graph_obj.pkl')
            G = g.EffDisGraph(num_nodes=num_nodes, eff_dist_and_edge_response=coupling_val, fraction_info_score_redistributed=skew_value)
            G.uniform_random_edge_init()
            G.run(num_runs=num_runs, exp_decay_param=exp_decay_param, num_shifts_of_source_node=num_shifts_of_source_node, constant_source_node=2, equilibrium_distance=200, multiple_path=False)
            save_object(G, numbered_graph_path)
            last_ave_nbr_deg = list(nx.average_neighbor_degree(G.convert_to_nx_graph(timestep=-1), source='in', target='in', weight='weight').values())
            eff_dist_diffs_flattened.append(G.eff_dist_diff(multiple_path_eff_dist=False))  # Compares first and last eff_dist values
            ave_neighbor_diffs_flattened.append((lambda x: max(x) - min(x))(last_ave_nbr_deg))
            ave_nbr_var_flattened.append(np.array(last_ave_nbr_deg).var())
            run_counter += 1
            if verbose:
                print(f'Run with responsiveness value {np.round(coupling_val, 2)}, rate_of_edge_adaptation: {np.round(skew_value, 2)} complete. (#{run_counter} of {full_coupling_range.shape[0]*rate_of_skew_range.shape[0]})')

    plot_heatmap(np.array(eff_dist_diffs_flattened).reshape(full_coupling_range.shape[0], rate_of_skew_range.shape[0]), title=Path(grid_search_dir, f'eff_dist_histogram'), x_range=rate_of_skew_range, y_range=full_coupling_range, normalize=True)
    plot_heatmap(np.array(ave_neighbor_diffs_flattened).reshape(full_coupling_range.shape[0], rate_of_skew_range.shape[0]), title=Path(grid_search_dir, f'ave_neighbor_diff_histogram'), x_range=rate_of_skew_range, y_range=full_coupling_range, normalize=True)
    plot_heatmap(np.array(ave_nbr_var_flattened).reshape(full_coupling_range.shape[0], rate_of_skew_range.shape[0]), title=Path(grid_search_dir, f'ave_neighbor_var_histogram'), x_range=rate_of_skew_range, y_range=full_coupling_range, normalize=True)


def grid_search(num_nodes, num_runs, exp_decay_param, coupling_range, coupling_interval, skew_range, skew_interval, num_shifts_of_source_node=False, parent_directory=None, directory_name='grid_search',
                node_values=False,
                eff_dists=True,
                network_evolution_graphs=True,
                ave_neighbor_plots=False,
                cluster_coefficients=False,
                shortest_paths=False,
                verbose=False):

    if parent_directory is None:
        source_directory = os.path.dirname(__file__)
    else:
        source_directory = parent_directory
    grid_search_dir = Path(source_directory, directory_name+f'_{num_nodes}_nodes')
    graphs_path = Path(grid_search_dir, 'graphs_path')
    if node_values: node_path = Path(grid_search_dir, 'node_plots')
    if eff_dists: eff_dist_path = Path(grid_search_dir, 'eff_dist_plots')
    if network_evolution_graphs: graph_path = Path(grid_search_dir, 'network_graphs')
    if ave_neighbor_plots: neighbor_path = Path(grid_search_dir, 'ave_neighbor_plots')
    if cluster_coefficients: cluster_coeff_path = Path(grid_search_dir, 'cluster_coefficients_plots')
    if shortest_paths: shortest_paths_path = Path(grid_search_dir, 'shortest_paths_plots')
    try:
        os.mkdir(grid_search_dir), f'Created folder for grid search results at {grid_search_dir}'
        os.mkdir(graphs_path), f'Created folder for graph objects at {graphs_path}'
        if node_values: os.mkdir(node_path), f'Created folder for node plots at {node_path}'
        if eff_dists: os.mkdir(eff_dist_path), f'Created folder for eff dist plots at {eff_dist_path}'
        if network_evolution_graphs: os.mkdir(graph_path), f'Created folder for graphs at {graph_path}'
        if ave_neighbor_plots: os.mkdir(neighbor_path), f'Created folder for graphs at {neighbor_path}'
        if cluster_coefficients: os.mkdir(cluster_coeff_path), f'Created folder for graphs at {cluster_coeff_path}'
        if shortest_paths: os.mkdir(shortest_paths_path), f'Created folder for graphs at {shortest_paths_path}'
    except OSError:
        print(f'{grid_search_dir} already exists, adding or overwriting contents')
        pass

    if verbose:
        print(f'Beginning grid search, Num_nodes: {num_nodes}, num_rums: {num_runs}, coupling, skew ranges {coupling_range, skew_range} and intervals {coupling_interval, skew_interval}')
    full_coupling_range = np.arange(coupling_range[0], coupling_range[1], coupling_interval)
    rate_of_adaptation_range = np.arange(skew_range[0], skew_range[1], skew_interval)
    run_counter = 0

    eff_dist_diffs_flattened = []
    ave_neighbor_diffs_flattened = []
    ave_nbr_var_flattened = []
    for coupling_val in full_coupling_range:
        for adaption_value in rate_of_adaptation_range:
            numbered_graph_path = Path(graphs_path, f'{run_counter+1:04}_graph_obj.pkl')
            G = g.EffDisGraph(num_nodes=num_nodes, eff_dist_and_edge_response=coupling_val, fraction_info_score_redistributed=adaption_value)
            G.uniform_random_edge_init()
            G.run(num_runs=num_runs, exp_decay_param=exp_decay_param, num_shifts_of_source_node=num_shifts_of_source_node, constant_source_node=2, equilibrium_distance=200, multiple_path=False)
            save_object(G, numbered_graph_path)
            if node_values:
                plot_node_values(G, node='all', show=False, save_fig=False,
                                 title=Path(node_path, f'{run_counter:03}_node_values_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))
            if eff_dists:
                plot_global_eff_dist(G, show=False, save_fig=False,
                                 title=Path(eff_dist_path, f'{run_counter:03}_eff_dist_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))
            if network_evolution_graphs:
                plot_network(G, nodes_sized_by_eff_distance=True, show=False, save_fig=False,
                         title=Path(graph_path, f'{run_counter:03}_graph_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))
            if cluster_coefficients or ave_neighbor_plots:
                nx_graphs = G.convert_history_to_list_of_nx_graphs()
                last_ave_nbr_deg = list(
                    nx.average_neighbor_degree(G.convert_to_nx_graph(timestep=-1), source='in', target='in',
                                               weight='weight').values())
                eff_dist_diffs_flattened.append(
                    G.eff_dist_diff(multiple_path_eff_dist=False))  # Compares first and last eff_dist values
                ave_neighbor_diffs_flattened.append((lambda x: max(x) - min(x))(last_ave_nbr_deg))
                ave_nbr_var_flattened.append(np.array(last_ave_nbr_deg).var())
                if ave_neighbor_plots:
                    plot_ave_neighbor_degree(nx_graphs, target='in', source='in', show=False, save_fig=False,
                                             title=Path(neighbor_path, f'{run_counter:03}_neighbor_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))
                if cluster_coefficients:
                    plot_clustering_coefficients(nx_graphs, show=False, save_fig=False,
                                                 title=Path(cluster_coeff_path, f'{run_counter:03}_cluster_coeffs_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))
                if shortest_paths:
                    plot_shortest_path_length(nx_graphs, show=False, save_fig=False,
                             title=Path(shortest_paths_path, f'{run_counter:03}_shortest_path_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))
            run_counter += 1
            if verbose:
                print(f'Run with coupling value {np.round(coupling_val, 2)}, edge reinforcement skew: {np.round(adaption_value, 2)} complete. (#{run_counter} of {full_coupling_range.shape[0]*rate_of_adaptation_range.shape[0]})')

    plot_heatmap(np.array(eff_dist_diffs_flattened).reshape(full_coupling_range.shape[0], rate_of_adaptation_range.shape[0]), title=Path(grid_search_dir, f'eff_dist_histogram'), x_range=rate_of_adaptation_range, y_range=full_coupling_range, normalize=True)
    plot_heatmap(np.array(ave_neighbor_diffs_flattened).reshape(full_coupling_range.shape[0], rate_of_adaptation_range.shape[0]), title=Path(grid_search_dir, f'ave_neighbor_diff_histogram'), x_range=rate_of_adaptation_range, y_range=full_coupling_range, normalize=True)
    plot_heatmap(np.array(ave_nbr_var_flattened).reshape(full_coupling_range.shape[0], rate_of_adaptation_range.shape[0]), title=Path(grid_search_dir, f'ave_neighbor_var_histogram'), x_range=rate_of_adaptation_range, y_range=full_coupling_range, normalize=True)


def all_plots_from_super_data_dir(path_to_data_dir, coupling_range, skew_range, output_dir=None):
    if output_dir is None:
        output_dir = path_to_data_dir

    sub_dirs = sorted([dirs[0] for dirs in os.walk(path_to_data_dir) if dirs[0] != str(path_to_data_dir)])
    for sub_dir in sub_dirs:
        node_nums = int(str(str(sub_dir).split('/')[-1]).split('_')[-1])  # takes the node number as the end number of the latest subdirectory
        parallelized_twoD_grid_search_plots(Path(sub_dir), coupling_range, skew_range, num_nodes=node_nums, output_dir=output_dir)


def twoD_grid_search_plots(path_to_data_dir, coupling_range, skew_range, num_nodes, ave_nbr=False, cluster_coeff=False, shortest_path=False, output_dir=None):
    # Creates all plots for a given dataset (sub)directory, and puts the results in new appropriately named subdirectories
    start_time = time.time()
    if output_dir is None:
        output_dir = path_to_data_dir
    grid_search_plots_dir = Path(output_dir, f'plots_for_{num_nodes}_nodes')
    node_path = Path(grid_search_plots_dir, 'node_plots')
    eff_dist_path = Path(grid_search_plots_dir, 'eff_dist_plots')
    graph_path = Path(grid_search_plots_dir, 'network_graphs')
    if ave_nbr: neighbor_path = Path(grid_search_plots_dir, 'ave_neighbor_plots')
    if cluster_coeff: cluster_coeff_path = Path(grid_search_plots_dir, 'cluster_coefficients_plots')
    if shortest_path: shortest_paths_path = Path(grid_search_plots_dir, 'shortest_paths_plots')
    try:
        os.mkdir(grid_search_plots_dir), f'Created folder for grid search results at {grid_search_plots_dir}'
        os.mkdir(node_path), f'Created folder for node plots at {node_path}'
        os.mkdir(eff_dist_path), f'Created folder for eff dist plots at {eff_dist_path}'
        os.mkdir(graph_path), f'Created folder for graphs at {graph_path}'
        if ave_nbr: os.mkdir(neighbor_path), f'Created folder for graphs at {neighbor_path}'
        if cluster_coeff: os.mkdir(cluster_coeff_path), f'Created folder for graphs at {cluster_coeff_path}'
        if shortest_path: os.mkdir(shortest_paths_path), f'Created folder for graphs at {shortest_paths_path}'
    except OSError:
        print(f'{grid_search_plots_dir} already exists, adding or overwriting contents')
        pass
    run_counter = 0
    f = []
    eff_dist_diffs_flattened = []
    ave_neighbor_diffs_flattened = []
    ave_nbr_var_flattened = []
    for root, dirs, files in os.walk(path_to_data_dir):
        f = sorted(files)  # Order preserved due to 0 padding.
    assert len(f) == coupling_range.size * skew_range.size, f"Not as many files as parameter combinations: \n num_files: {len(f)}, coupling_range.size * adaptation_range.size {coupling_range.size * skew_range.size}"
    for coupling_val in coupling_range:
        for adaption_value in skew_range:
            with open(Path(path_to_data_dir, f[run_counter]), 'rb') as input:
                G = pickle.load(input)
                input.close()
            plot_node_values(G, node='all', show=False, save_fig=False,
                             title=Path(node_path, f'{run_counter:03}_node_values_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))
            plot_global_eff_dist(G, show=False, save_fig=False,
                                 title=Path(eff_dist_path, f'{run_counter:03}_eff_dist_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))
            if num_nodes > 20:  # prints end graphs alone for larger node values.
                plot_single_network(G, timestep=-1, show=False,
                                    title=Path(graph_path, f'{run_counter:03}_graph_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))
            else:
                plot_network(G, nodes_sized_by_eff_distance=True, show=False, save_fig=False,
                             title=Path(graph_path, f'{run_counter:03}_graph_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))

            nx_graphs = G.convert_history_to_list_of_nx_graphs()
            if ave_nbr:
                plot_ave_neighbor_degree(nx_graphs, target='in', source='in', show=False, save_fig=False,
                                        title=Path(neighbor_path, f'{run_counter:03}_neighbor_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))
            if cluster_coeff:
                plot_clustering_coefficients(nx_graphs, show=False, save_fig=False,
                                             title=Path(cluster_coeff_path, f'{run_counter:03}_cluster_coeffs_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))
            if shortest_path:
                plot_shortest_path_length(nx_graphs, show=False, save_fig=False,
                                          title=Path(shortest_paths_path, f'{run_counter:03}_shortest_path_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(adaption_value, 2)}'))

            last_ave_nbr_deg = list(nx.average_neighbor_degree(G.convert_to_nx_graph(timestep=-1), source='in', target='in', weight='weight').values())
            eff_dist_diffs_flattened.append(G.eff_dist_diff(multiple_path_eff_dist=False))  # Compares first and last eff_dist values
            ave_neighbor_diffs_flattened.append((lambda x: max(x) - min(x))(last_ave_nbr_deg))
            ave_nbr_var_flattened.append(np.array(last_ave_nbr_deg).var())
            run_counter += 1  # Also serves as file index

    plot_heatmap(np.array(eff_dist_diffs_flattened).reshape(coupling_range.size, skew_range.size), title=Path(grid_search_plots_dir, f'eff_dist_histogram'), x_range=skew_range, y_range=coupling_range, normalize=True)
    plot_heatmap(np.array(ave_neighbor_diffs_flattened).reshape(coupling_range.size, skew_range.size), title=Path(grid_search_plots_dir, f'ave_neighbor_diff_histogram'), x_range=skew_range, y_range=coupling_range, normalize=True)
    plot_heatmap(np.array(ave_nbr_var_flattened).reshape(coupling_range.size, skew_range.size), title=Path(grid_search_plots_dir, f'ave_neighbor_var_histogram'), x_range=skew_range, y_range=coupling_range, normalize=True)
    print(f"Time lapsed for {num_nodes} node, {run_counter} parameter combinations: {int((time.time()-start_time) / 60)} minutes, {(time.time()-start_time) % 60} seconds")


def parallelized_twoD_grid_search_plots(path_to_data_dir, coupling_range, skew_range, num_nodes, ave_nbr=False, cluster_coeff=False, shortest_path=False, output_dir=None):
    # Creates all plots for a given dataset (sub)directory, and puts the results in new appropriately named subdirectories.
    # Parallelization implementation ensures completion of all plots per dataset (or as many as supportable by the number of cpu cores) before continuing to the following set
    start_time = time.time()
    if output_dir is None:
        output_dir = path_to_data_dir
    grid_search_plots_dir = Path(output_dir, f'plots_for_{num_nodes}_nodes')
    node_path = Path(grid_search_plots_dir, 'node_plots')
    eff_dist_path = Path(grid_search_plots_dir, 'eff_dist_plots')
    graph_path = Path(grid_search_plots_dir, 'network_graphs')
    if ave_nbr: neighbor_path = Path(grid_search_plots_dir, 'ave_neighbor_plots')
    if cluster_coeff: cluster_coeff_path = Path(grid_search_plots_dir, 'cluster_coefficients_plots')
    if shortest_path: shortest_paths_path = Path(grid_search_plots_dir, 'shortest_paths_plots')
    try:
        os.mkdir(grid_search_plots_dir), f'Created folder for grid search results at {grid_search_plots_dir}'
        os.mkdir(node_path), f'Created folder for node plots at {node_path}'
        os.mkdir(eff_dist_path), f'Created folder for eff dist plots at {eff_dist_path}'
        os.mkdir(graph_path), f'Created folder for graphs at {graph_path}'
        if ave_nbr: os.mkdir(neighbor_path), f'Created folder for graphs at {neighbor_path}'
        if cluster_coeff: os.mkdir(cluster_coeff_path), f'Created folder for graphs at {cluster_coeff_path}'
        if shortest_path: os.mkdir(shortest_paths_path), f'Created folder for graphs at {shortest_paths_path}'
    except OSError:
        print(f'{grid_search_plots_dir} already exists, adding or overwriting contents')
        pass

    run_counter = 0
    f = []
    eff_dist_diffs_flattened = []
    ave_neighbor_diffs_flattened = []
    ave_nbr_var_flattened = []
    for root, dirs, files in os.walk(path_to_data_dir):
        f = sorted(files)  # Order preserved due to 0 padding.
    assert len(f) == coupling_range.size * skew_range.size, f"Not as many files as parameter combinations: \n num_files: {len(f)}, coupling_range.size * adaptation_range.size {coupling_range.size * skew_range.size}"
    for coupling_val in coupling_range:
        skew_range_iter = iter(range(skew_range.size))
        for skew_val_index in skew_range_iter:

            used_cores = 0
            skew_vals_per_full_cpu = 0
            processes = []
            left_over_skew_values = skew_range.size - skew_val_index
            # print(f'skew_val_index: {skew_val_index} | mp.cpu_count(): {mp.cpu_count()} | skew_range.size: {skew_range.size}')
            if left_over_skew_values < mp.cpu_count():  # To ensure that parallelization persists when there are fewer tasks than cores
                while skew_vals_per_full_cpu < left_over_skew_values:
                    with open(Path(path_to_data_dir, f[run_counter]), 'rb') as data:
                        G = pickle.load(data)
                        data.close()
                    # all args must be given for Process runs.
                    p_1 = mp.Process(target=plot_node_values, args=(G, 'all', False, False, Path(node_path,
                                                                                                 f'{run_counter:03}_node_values_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                    p_1.start()
                    processes.append(p_1)
                    used_cores += 1

                    p_2 = mp.Process(target=plot_global_eff_dist, args=(G, False, False, False, False,
                                                                        Path(eff_dist_path,
                                                                             f'{run_counter:03}_eff_dist_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                    p_2.start()
                    processes.append(p_2)
                    used_cores += 1

                    if num_nodes > 20:  # prints end graphs alone for larger node values.
                        p_3 = mp.Process(target=plot_single_network, args=(
                            G, -1, True, 200, False, None, False, False, Path(graph_path,
                                                                              f'{run_counter:03}_graph_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                        p_3.start()
                    else:
                        p_3 = mp.Process(target=plot_network, args=(G, True, 200, False, False, False,
                                                                    Path(graph_path,
                                                                         f'{run_counter:03}_graph_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                        p_3.start()
                    processes.append(p_3)
                    used_cores += 1

                    if ave_nbr or cluster_coeff or shortest_path:
                        nx_graphs = G.convert_history_to_list_of_nx_graphs()
                    if ave_nbr:
                        p_4 = mp.Process(target=plot_ave_neighbor_degree,
                                         args=(nx_graphs, 'in', 'in', False, False, False,
                                               Path(neighbor_path,
                                                    f'{run_counter:03}_neighbor_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                        p_4.start()
                        processes.append(p_4)
                        used_cores += 1
                    if cluster_coeff:
                        p_5 = mp.Process(target=plot_clustering_coefficients,
                                         args=(nx_graphs, False, False, False, False,
                                               Path(cluster_coeff_path,
                                                    f'{run_counter:03}_cluster_coeffs_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                        p_5.start()
                        processes.append(p_5)
                        used_cores += 1
                    if shortest_path:
                        p_6 = mp.Process(target=plot_shortest_path_length, args=(nx_graphs, False, False,
                                                                                 Path(shortest_paths_path,
                                                                                      f'{run_counter:03}_shortest_path_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                        p_6.start()
                        processes.append(p_6)
                        used_cores += 1

                    run_counter += 1  # Also serves as file index
                    skew_vals_per_full_cpu += 1

                    last_ave_nbr_deg = list(nx.average_neighbor_degree(G.convert_to_nx_graph(timestep=-1), source='in', target='in', weight='weight').values())
                    eff_dist_diffs_flattened.append(G.eff_dist_diff(multiple_path_eff_dist=False))  # Compares first and last eff_dist values
                    ave_neighbor_diffs_flattened.append((lambda x: max(x) - min(x))(last_ave_nbr_deg))
                    ave_nbr_var_flattened.append(np.array(last_ave_nbr_deg).var())

                utility_funcs.consume(skew_range_iter, left_over_skew_values - 1)  # -1 because the iteration forwards 1 step still proceeds directly after
            else:
                while skew_vals_per_full_cpu < mp.cpu_count():
                    with open(Path(path_to_data_dir, f[run_counter]), 'rb') as data:
                        G = pickle.load(data)
                        data.close()
                    p_1 = mp.Process(target=plot_node_values, args=(G, 'all', False, False, Path(node_path, f'{run_counter:03}_node_values_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                    p_1.start()
                    processes.append(p_1)
                    used_cores += 1

                    p_2 = mp.Process(target=plot_global_eff_dist, args=(G, False, False, False, False, Path(eff_dist_path, f'{run_counter:03}_eff_dist_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                    p_2.start()
                    processes.append(p_2)
                    used_cores += 1

                    if num_nodes > 20:  # prints end graphs alone for larger node values.
                        p_3 = mp.Process(target=plot_single_network, args=(G, -1, True, 200, False, None, False, False, Path(graph_path, f'{run_counter:03}_graph_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                        p_3.start()
                    else:
                        p_3 = mp.Process(target=plot_network, args=(G, True, 200, False, False, False, Path(graph_path, f'{run_counter:03}_graph_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                        p_3.start()
                    processes.append(p_3)
                    used_cores += 1

                    if ave_nbr or cluster_coeff or shortest_path:
                        nx_graphs = G.convert_history_to_list_of_nx_graphs()
                    if ave_nbr:
                        p_4 = mp.Process(target=plot_ave_neighbor_degree, args=(nx_graphs, 'in', 'in', False, False, False,
                                                                          Path(neighbor_path,
                                                                               f'{run_counter:03}_neighbor_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                        p_4.start()
                        processes.append(p_4)
                        used_cores += 1
                    if cluster_coeff:
                        p_5 = mp.Process(target=plot_clustering_coefficients, args=(nx_graphs, False, False, False, False,
                                                                              Path(cluster_coeff_path,
                                                                                   f'{run_counter:03}_cluster_coeffs_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                        p_5.start()
                        processes.append(p_5)
                        used_cores += 1
                    if shortest_path:
                        p_6 = mp.Process(target=plot_shortest_path_length, args=(nx_graphs, False, False,
                                                                           Path(shortest_paths_path,
                                                                                f'{run_counter:03}_shortest_path_plot_for_coupling_{np.round(coupling_val, 2)}_skew_{np.round(skew_range[skew_val_index + skew_vals_per_full_cpu], 2)}')))
                        p_6.start()
                        processes.append(p_6)
                        used_cores += 1

                    run_counter += 1  # Also serves as file index
                    skew_vals_per_full_cpu += 1

                    last_ave_nbr_deg = list(nx.average_neighbor_degree(G.convert_to_nx_graph(timestep=-1), source='in', target='in', weight='weight').values())
                    eff_dist_diffs_flattened.append(G.eff_dist_diff(multiple_path_eff_dist=False))  # Compares first and last eff_dist values
                    ave_neighbor_diffs_flattened.append((lambda x: np.log(max(x) - min(x)))(last_ave_nbr_deg))
                    ave_nbr_var_flattened.append(np.log(np.array(last_ave_nbr_deg).var()))

                utility_funcs.consume(skew_range_iter, mp.cpu_count() - 1)  # Advances skew iter cpu count iterations

            for process in processes:
                process.join()  # join's created processes to run simultaneously.

    plot_heatmap(np.array(eff_dist_diffs_flattened).reshape(coupling_range.size, skew_range.size), title=Path(grid_search_plots_dir, f'eff_dist_histogram'), x_range=skew_range, y_range=coupling_range, normalize=True)
    plot_heatmap(np.array(ave_neighbor_diffs_flattened).reshape(coupling_range.size, skew_range.size), title=Path(grid_search_plots_dir, f'log_ave_neighbor_diff_histogram'), x_range=skew_range, y_range=coupling_range, normalize=False)
    plot_heatmap(np.array(ave_nbr_var_flattened).reshape(coupling_range.size, skew_range.size), title=Path(grid_search_plots_dir, f'log_ave_neighbor_var_histogram'), x_range=skew_range, y_range=coupling_range, normalize=False)
    print(f"Time lapsed for {num_nodes} node, {run_counter} parameter combinations: {int((time.time()-start_time) / 60)} minutes, {np.round((time.time()-start_time) % 60, 2)} seconds")


# data_observables
def cluster_coeff_diff(data_dir, initial_graph=0, final_graph=-1, clustering_timestep=-1):
    """
    Loads and evaluates the clustering coefficients of the last (clustering_) timestep of the (initial, final) data graph
    and returns their difference.
    """
    for root, dirs, files in os.walk(data_dir):
        f = sorted(files)  # Order preserved due to 0 padding.
        with open(Path(data_dir, f[initial_graph]), 'rb') as initial_graph_loaded:
            G_initial = pickle.load(initial_graph_loaded)
            initial_graph_loaded.close()
        with open(Path(data_dir, f[final_graph]), 'rb') as final_graph_loaded:
            G_final = pickle.load(final_graph_loaded)
            final_graph_loaded.close()
        initial_clustering_coeff = nx.average_clustering(G_initial.convert_to_nx_graph(timestep=clustering_timestep), weight='weight')
        final_clustering_coeff = nx.average_clustering(G_final.convert_to_nx_graph(timestep=clustering_timestep), weight='weight')
        return final_clustering_coeff - initial_clustering_coeff


def shortest_path_diff(data_dir, initial_graph=0, final_graph=-1, shortest_path_at_timestep=-1):
    """
    Loads and evaluates the average shortest path length of the last (shortest_path_at) timestep of the (initial, final)
    data graph and takes their difference.
    """
    for root, dirs, files in os.walk(data_dir):
        f = sorted(files)  # Order preserved due to 0 padding.
        with open(Path(data_dir, f[initial_graph]), 'rb') as initial_graph_loaded:
            G_initial = pickle.load(initial_graph_loaded)
            initial_graph_loaded.close()
        with open(Path(data_dir, f[final_graph]), 'rb') as final_graph_loaded:
            G_final = pickle.load(final_graph_loaded)
            final_graph_loaded.close()
        initial_ave_shortest_path = nx.average_shortest_path_length(G_initial.convert_to_nx_graph(timestep=shortest_path_at_timestep), weight='weight')
        final_ave_shortest_path = nx.average_shortest_path_length(G_final.convert_to_nx_graph(timestep=shortest_path_at_timestep), weight='weight')
        return final_ave_shortest_path - initial_ave_shortest_path


def ave_degree_diff(data_dir, initial_graph=0, final_graph=-1, ave_degree_at_timestep=-1):
    """
    Loads and evaluates the average length of the last (shortest_path_at) timestep of the (initial, final)
    data graph and takes their difference. TODO: Will this always yield [0.]?
    Also known as ~k_nearest_neighbors algorithm
    """
    for root, dirs, files in os.walk(data_dir):
        f = sorted(files)  # Order preserved due to 0 padding.
        with open(Path(data_dir, f[initial_graph]), 'rb') as initial_graph_loaded:
            G_initial = pickle.load(initial_graph_loaded)
            initial_graph_loaded.close()
        with open(Path(data_dir, f[final_graph]), 'rb') as final_graph_loaded:
            G_final = pickle.load(final_graph_loaded)
            final_graph_loaded.close()
        initial_ave_degree = np.array(list(nx.average_degree_connectivity(G_initial.convert_to_nx_graph(timestep=ave_degree_at_timestep), weight='weight').values()))
        final_ave_degree = np.array(list(nx.average_degree_connectivity(G_final.convert_to_nx_graph(timestep=ave_degree_at_timestep), weight='weight').values()))
    return final_ave_degree - initial_ave_degree
