import numpy as np
import sys
# sys.path.append('../')
import utility_funcs

size = 5

M = np.zeros((size, size))
# Testing Column based operations:
for column in range(0, size):
    a = np.round(np.random.random(size), 1)
    M[:, column] = a

M_row_sums = M.sum(axis=1)
print(f'M: \n {M}')
print(f'row sums (M.sum(axis=1)): \n {M_row_sums}')
# print(f'row 0 (M[0, :]) {M[0, :]}')
# print(np.array([np.array([5, 5, 5, 5, 5]) / (val+1) for val in range(size)]))
# print(np.array([np.array([5, 5, 5, 5, 5]) / M_row_sums[val] for val in range(size)]))
normed_M = np.array([M[node, :] / M_row_sums[node] for node in range(size)])
print(f'Row normalized M: \n {normed_M}')
utility_funcs.print_row_column_sums(normed_M)
utility_funcs.sum_matrix_signs(normed_M)