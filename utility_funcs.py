import collections
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

rounding = 3


def time_lapsed_h_m_s(time_in_seconds):
    hours, minutes, seconds = int(time_in_seconds / 3600), int((time_in_seconds % 3600) / 60), np.round(
        ((time_in_seconds % 3600) % 60), 2)
    if hours: return f'{hours} hours, {minutes} minutes, {seconds} seconds'
    if minutes: return f'{minutes} minutes, {seconds} seconds'
    if seconds: return f'{seconds} seconds'


def arr_dimen(a):
    # Finds list dimension, curtsy Archit Jain from https://stackoverflow.com/questions/17531796/find-the-dimensions-of-a-multidimensional-python-array
    return [len(a)] + arr_dimen(a[0]) if (type(a) == list) else []


def count(iterable, val):
    return sum(1 for _ in iterable if _ == val)


#################  Matrix Operations  #####################
def element_wise_array_average(list_of_np_arrays):
    """
    returns single array with every element the average of all input arrays respective element
    """
    assert np.all(np.array([list_of_np_arrays[0].shape == [list_of_np_arrays[i].shape for i in range(len(list_of_np_arrays))][ii] for ii in
         range(len(list_of_np_arrays))])), "Arrays to be averaged element-wise must be of equal dimension"
    return np.mean(np.array(list_of_np_arrays), axis=0)


def return_one_over_2d_matrix(matrix):
    quasi_inverted_matrix = np.array(matrix, float)
    for i in range(quasi_inverted_matrix.shape[0]):
        for j in range(quasi_inverted_matrix.shape[1]):
            if quasi_inverted_matrix[i][j] != 0:
                quasi_inverted_matrix[i][j] = 1 / matrix[i][j]
    return quasi_inverted_matrix


def matrix_normalize(matrix, row_normalize=False):
    if row_normalize:
        row_sums = matrix.sum(axis=1)
        return np.array([matrix[index, :] / row_sums[index] if row_sums[index] != 0 else [0] * row_sums.size for index in range(row_sums.size)])
    else:
        column_sums = matrix.sum(axis=0)
        return np.array([matrix[:, index] / column_sums[index] if column_sums[index] != 0 else [0]*column_sums.size for index in range(column_sums.size)]).T


def sum_matrix_signs(matrix, verbose=True):
    positive_vals = 0
    negative_vals = 0
    zeros = 0
    vals_greater_than_one = 0
    for row in matrix:
        for val in row:
            if val > 0:
                positive_vals += 1
            elif val == 0:
                zeros += 1
            else:
                negative_vals += 1
            if val > 1: vals_greater_than_one += 1
    if verbose:
        if vals_greater_than_one > matrix.shape[0]:
            print(f' There {vals_greater_than_one-matrix.shape[0]} are off diagonal elements greater than one.')
        if zeros:
            print(f'Matrix has {positive_vals} positive values, {zeros} zeros, and {negative_vals} negative values')
        else:
            print(f'Matrix has {positive_vals} positive values and {negative_vals} negative values')
    return vals_greater_than_one-matrix.shape[0], positive_vals, negative_vals


def print_row_column_sums(matrix):
    assert matrix.shape[0] == matrix.shape[1], 'Use square numpy matrix'
    row_sums = []
    column_sums = []
    for node in range(matrix.shape[0]):
        row_sums.append(sum(matrix[node, :]))
        column_sums.append(sum(matrix[:, node]))
        if np.isclose(sum(matrix[node, :]), 0): print(f'Row {node} sums to zero')
        if np.isclose(sum(matrix[:, node]), 0): print(f'Column {node} sums to zero')
    print(f'Row sums:\n {np.array(row_sums)}')
    print(f'Column sums:\n {np.round(np.array(column_sums), 2)}')


def evenly_distribute_matrix_values(matrix, by_row=False, by_column=False):
    # Defaults to evenly distributing row values (i.e. mean row value across the non-zero row values), can do columns
    M, i = np.zeros(matrix.shape), 0
    if by_column:
        for column in matrix.T:
            M[i] = np.array([np.sum(column) / len([1 for non_zero_val in column if non_zero_val != 0]) if non_zero_val != 0 else 0 for non_zero_val in column])
            i += 1
        return M.T
    elif by_row:
        for row in matrix:
            M[i] = np.array([np.sum(row)/len([1 for non_zero_val in row if non_zero_val != 0]) if non_zero_val != 0 else 0 for non_zero_val in row])
            i += 1
        return M
    else:
        return np.where(matrix == 0, 0, np.mean(matrix))


def undirectify(adjacency_matrix, average_connections=True):
    assert len(adjacency_matrix.shape) == 2 and adjacency_matrix.shape[0] == adjacency_matrix.shape[1], 'A must be square'
    # returns an undirected version of a given Adjacency matrix, mirroring values over the diagonal axis, and optionally averaging them
    A_top = np.zeros(adjacency_matrix.shape)
    A_bottom = np.zeros(adjacency_matrix.shape)
    if average_connections:
        A = np.zeros(adjacency_matrix.shape)
        for i in range(adjacency_matrix.shape[0]):
            A[i] = np.array([((adjacency_matrix[i][j] + adjacency_matrix[j][i]) / 2) if j > i else 0 for j in range(adjacency_matrix.shape[0])])  # >= to include diagonal
        return A + A.T
    else:
        for i in range(adjacency_matrix.shape[0]):
            A_top[i] = np.array([adjacency_matrix[i][j] if j > i else 0 for j in range(adjacency_matrix.shape[0])])  # >= to include diagonal
            A_bottom[i] = np.array([adjacency_matrix[i][j] if j < i else 0 for j in range(adjacency_matrix.shape[0])])
        A_bottom = np.where(A_bottom > A_top.T, A_bottom, 0)
        return A_top + A_bottom.T + (A_top + A_bottom.T).T


def A_to_csv(matrix, output_dir, csv_name, delimiter=None):
    Sources = [int(i/matrix.shape[0]) for i in range(matrix.shape[0]*matrix.shape[1])]
    Targets = list(np.repeat([np.arange(matrix.shape[0])], matrix.shape[0], axis=0).flatten())
    Fluxii = list(matrix.flatten())
    network_dict = {
        'SOURCE': Sources,
        'TARGET': Targets,
        'FLUX': Fluxii
    }
    print(f'SOURCE: {len(Sources)} {Sources}')
    print(f'TARGET: {len(Targets)} {Targets}')
    print(f'FLUX: {len(Fluxii)} {np.round(Fluxii, 2)}')
    df = pd.DataFrame(network_dict)
    column_order = ['SOURCE', 'TARGET', 'FLUX']
    if delimiter is not None:
        df[column_order].to_csv(Path(output_dir, csv_name), header=False, index=False, float_format='%.2f', sep=delimiter)
    else:
        df[column_order].to_csv(Path(output_dir, csv_name), header=False, index=False, float_format='%.2f')


# Other functions:
def print_run_percentage(index, runs, fraction_intervals=10):
    if runs < fraction_intervals:
        print(f"Too few runs, set verbose to False (num runs [{runs}] must be divisible by [{fraction_intervals}])")
    elif int(index % runs) % int(runs / fraction_intervals) == 0:
        print(f'{(index / runs) * 100:.1f}%-ish done')


def consume(iterator, n):
    collections.deque(itertools.islice(iterator, n))


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def exponentially_distribute(exponent, dist_max, dist_min, num_exp_distributed_values):
    """
    :param exponent: An exponent of 0 results in a linear distribution,
    otherwise the exp distribution is sampled from e^(exponent)*2e - e^(exponent)*e
    :param dist_max: Maximum of newly exponential distribution
    :param dist_min: Minimum of newly exponential distribution
    :param num_exp_distributed_values: Number of values in the eventual distribution
    """
    total_exp_range = np.exp(exponent * 2 * np.e) - np.exp(exponent * np.e)
    exponential_proportions = [(np.exp(exponent * (np.e + (np.e / (num_exp_distributed_values - threshold_index)))) - np.exp(exponent * np.e)) / total_exp_range for threshold_index in range(num_exp_distributed_values)]
    return np.array(exponential_proportions) * (dist_max - dist_min)


def crop_outliers(data, std_multiple_cutoff=2):
    data_std_cutoffs = [-np.std(data)*std_multiple_cutoff+np.mean(data), np.std(data)*std_multiple_cutoff+np.mean(data)]
    cropped_data = filter(lambda x: data_std_cutoffs[0] < x < data_std_cutoffs[1], data)

    filtered_data = []
    for el in cropped_data:
        filtered_data.append(el)
    #  Not understood why this cannot be done through list comprehension...
    return filtered_data


def check_binary(x):
    is_binary = True
    for v in np.nditer(x):
        if v.item() != 0 and v.item() != 1:
            is_binary = False
            break

    return is_binary


# just for debugging
def rounded_print(num):
    print(np.round(num, rounding))


def print_inventory(dct):
    for item, amount in dct.items():
        print("{} ({})".format(item, amount))

