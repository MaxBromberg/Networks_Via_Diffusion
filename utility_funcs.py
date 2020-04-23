import numpy as np

rounding = 3


def return_one_over_2d_matrix(matrix):
    quasi_inverted_matrix = np.array(matrix, float)
    for i in range(quasi_inverted_matrix.shape[0]):
        for j in range(quasi_inverted_matrix.shape[1]):
            if quasi_inverted_matrix[i][j] != 0:
                quasi_inverted_matrix[i][j] = 1/matrix[i][j]
    return quasi_inverted_matrix


def count(iterable, val):
    return sum(1 for _ in iterable if _ == val)


def matrix_normalize(matrix, row_normalize=False):
    if row_normalize:
        row_sums = matrix.sum(axis=1)
        return np.array([matrix[index, :] / row_sums[index] for index in range(row_sums.size) if row_sums[index] is not np.isclose(row_sums[index], 0, 1e-15)])
    else:
        column_sums = matrix.sum(axis=0)
        return np.array([matrix[:, index] / column_sums[index] for index in range(column_sums.size) if column_sums[index] is not np.isclose(column_sums[index], 0, 1e-15)]).T


def print_run_percentage(index, runs, fraction_intervals=10):
    if runs < fraction_intervals:
        print(f"Too few runs, set verbose to False (num runs [{runs}] must be divisible by [{fraction_intervals}])")
    elif int(index % runs) % int(runs / fraction_intervals) == 0:
        print(f'{(index / runs) * 100:.1f}%-ish done')


# just for debugging
def rounded_print(num):
    print(np.round(num, rounding))
