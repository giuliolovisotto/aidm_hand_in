__author__ = 'giulio'

import numpy as np


def split_matrix(matrix, n_slices):
    if matrix.shape[0] % n_slices == 0:
        return np.split(matrix, n_slices)
    else:
        divisible_elements = matrix.shape[0] - matrix.shape[0] % n_slices
        firsts = np.split(matrix[0:divisible_elements], n_slices)
        last = np.split(matrix[divisible_elements:matrix.shape[0]], 1)[0]
        firsts[-1] = np.concatenate((firsts[-1], last), axis=0)
        return firsts

