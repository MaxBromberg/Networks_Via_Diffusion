import numpy as np
import effective_distance as ed
from pathlib import Path
import pandas as pd

# In order to ensure compatibility with Python2.7(.11), a modified version of A_to_csv is used, and no other project files


def A_to_csv(matrix, output_dir, csv_name, delimiter=None):
    Sources = [int(i/matrix.shape[0]) for i in range(matrix.shape[0]*matrix.shape[1])]
    Targets = list(np.repeat([np.arange(matrix.shape[0])], matrix.shape[0], axis=0).flatten())
    Fluxii = list(matrix.flatten())
    network_dict = {
        'SOURCE': Sources,
        'TARGET': Targets,
        'FLUX': Fluxii
    }
    # print(f'SOURCE: {len(Sources)} {Sources}')
    # print(f'TARGET: {len(Targets)} {Targets}')
    # print(f'FLUX: {len(Fluxii)} {np.round(Fluxii, 2)}')
    column_order = ['SOURCE', 'TARGET', 'FLUX']
    df = pd.DataFrame(network_dict)
    if delimiter is not None:
        df[column_order].to_csv(Path(output_dir, csv_name), header=False, index=False, float_format='%.2f', sep=delimiter)
    else:
        df[column_order].to_csv(Path(output_dir, csv_name), header=False, index=False, float_format='%.2f')


output_dir = "/home/maqz/Desktop/"
# csv_name = 'text.csv'
csv_name = 'sparse.csv'
size = 10


normed_A = np.random.random((size, size))
for i in range(normed_A.shape[0]):
    normed_A[i][i] = 0
row_sums = normed_A.sum(axis=1)
normed_A = np.array([normed_A[node, :] / row_sums[node] for node in range(normed_A.shape[0])])
# print(np.round(normed_A, 3))


# A_to_csv(matrix=normed_A, csv_name=csv_name, output_dir=output_dir, delimiter=" ")
csv_file = str(Path(output_dir, csv_name))
eff_dists = ed.EffectiveDistances(csv_file, verbose=False).get_random_walk_distance(source=None, target=None, parameter=1, saveto="")
print('eff_dists:')
print(np.round(eff_dists, 3))
