import numpy as np
from matplotlib import pyplot as plt
import networkx as nx  # Used for network plots

import imageio  # Used for making gifs of the network plots
import os  # Used for putting the gifs somewhere
from pathlib import Path  # used for file path compatibility between operating systems
from scipy import optimize


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


def plot_global_eff_dist(graph, fit=False):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(graph.global_eff_dist_history)
    plt.title(f'Total Effective Distance history')
    plt.xlabel('Time step')
    plt.ylabel(f'Total Effective distance')
    if fit:
        x = np.array(range(len(graph.global_eff_dist_history)))
        y = graph.global_eff_dist_history
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
            assert ave_range % 2 == 0, 'Average range must be even (lest, for this algorithm)'
            half_range = int((ave_range/2))
            averaging_fit = [np.mean(y[index-half_range:index+half_range]) for index in x[half_range:-half_range]]
            # averaging_fit = np.insert(averaging_fit, 0, y[:half_range])
            # averaging_fit = np.insert(averaging_fit, averaging_fit.shape[0], y[-half_range:])
            plt.plot(averaging_fit)
    plt.show()


def plot_node_values(graph, value_per_nugget, node='all', show=True, save_fig=False):
    assert show or save_fig, 'Graph will be neither shown nor saved'
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
    assert show or save_fig, 'Graph will be neither shown nor saved'
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
    node_colors[graph.starting_node_history[timestep-1]] = 'red'
    incoming_edge_sum = graph.A[timestep].sum(axis=1)
    incoming_edge_sum = [node_size_scaling * node / sum(incoming_edge_sum) for node in incoming_edge_sum]
    if source_weighting:  # sizes nodes proportional to the number of times they've been a source
        source_weight = [graph.starting_node_history.count(node) for node in range(graph.nodes.shape[1])]
        source_weight = [node_size_scaling*(weight/sum(source_weight)) for weight in source_weight]
        nx.draw_networkx_nodes(nx_G, pos,
                               arrowstyle='->',
                               edge_color=edge_colors,
                               node_size=source_weight,
                               node_color=node_colors,
                               # node_color=node_colors,
                               widths=weights,
                               cmap=plt.get_cmap('viridis'))
        plt.title(f"Nodes size proportional to number of times they've been the source [timestep: {timestep}]")
    else:
        nx.draw_networkx_nodes(nx_G, pos,
                               arrowstyle='->',
                               edge_color=edge_colors,
                               node_size=incoming_edge_sum,
                               node_color=node_colors,
                               # node_color=node_colors,
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


def plot_network(graph, value_per_nugget, directed=True, node_size_scaling=200, nodes_sized_by_eff_distance=False,
                 show=True, save_fig=False):
    fig = plt.figure(figsize=(12, 6))
    assert show or save_fig, 'Graph will be neither shown nor saved'
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
        node_colors[graph.starting_node_history[timestep]] = 'red'
        nx.draw_networkx_edges(nx_G, pos, nodelist=['0'], alpha=0.8, width=weights, arrowsize=4, connectionstyle='arc3, rad=0.2')
        # not sure reversal does anything in this context... (could just transpose?)
        # nx.draw_networkx_edges(nx_G.reverse(copy=True), pos, nodelist=['0'], alpha=0.8, width=weights, arrowsize=4,
        #                        connectionstyle='arc3, rad=0.2')
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
    if save_fig:
        plt.savefig(f'Network Structure(s) for {value_per_nugget} nugget value, {graph.nodes.shape[0]} runs.png')
        plt.close(fig)


def gif_of_network_evolution(graph, node_size_scaling=200, source_weighting=False, directory_name='network_animation', file_title='network_evolution', parent_directory=None, gif_duration_in_sec=5,  num_runs_per_fig=None, verbose=False):
    """
    Creates a gif and mp4 of the network evolution and stores them in a folder with the individual frames.
    TODO: when skipping files (e.g. showing only every 10 steps) and going over 100, the files are misordered, as
    TODO: the writer sees 100 as coming before 20, given its leading 1.
    """

    assert num_runs_per_fig != 0, 'Number of runs per figure must be larger than 0, or else omitted to graph every run'

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
        files = Path(fig_path, f'{i}')
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
    for filename in os.listdir(fig_path):
        if filename.endswith(".png"):
            images.append(imageio.imread(Path(fig_path, filename)))
            writer.append_data(imageio.imread(Path(fig_path, filename)))
    imageio.mimsave(f'{Path(vid_path, file_title)}.gif', images)
    writer.close()
    if verbose:
        print(f'gif and mp4 of network evolution created in {vid_path} \n Stills stored in {fig_path}')

