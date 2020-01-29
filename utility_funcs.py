import numpy as np


def return_one_over_2d_matrix(matrix):
    quasi_inverted_matrix = np.array(matrix, float)
    for i in range(quasi_inverted_matrix.shape[0]):
        for j in range(quasi_inverted_matrix.shape[1]):
            if quasi_inverted_matrix[i][j] != 0:
                quasi_inverted_matrix[i][j] = 1/matrix[i][j]
    return quasi_inverted_matrix


def count(iterable):
    return sum(1 for _ in iterable)
