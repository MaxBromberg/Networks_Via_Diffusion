import numpy as np
import itertools, collections

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
        return np.array([matrix[index, :] / row_sums[index] for index in range(row_sums.size) if
                         row_sums[index] is not np.isclose(row_sums[index], 0, 1e-15)])
    else:
        column_sums = matrix.sum(axis=0)
        return np.array([matrix[:, index] / column_sums[index] for index in range(column_sums.size) if
                         column_sums[index] is not np.isclose(column_sums[index], 0, 1e-15)]).T


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


## Other functions:

def print_run_percentage(index, runs, fraction_intervals=10):
    if runs < fraction_intervals:
        print(f"Too few runs, set verbose to False (num runs [{runs}] must be divisible by [{fraction_intervals}])")
    elif int(index % runs) % int(runs / fraction_intervals) == 0:
        print(f'{(index / runs) * 100:.1f}%-ish done')


def consume(iterator, n):
    collections.deque(itertools.islice(iterator, n))


# just for debugging
def rounded_print(num):
    print(np.round(num, rounding))
