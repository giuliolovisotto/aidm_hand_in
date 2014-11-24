__author__ = 'giulio'

import numpy as np


def compute_error(rec, test):
    """
    Given the recommender (UxM) matrix with predicted values
    and the test set, returns the rmse and the mae
    :param rec:
    :param test:
    :return:
    """
    # fix for values not in range [1,5]
    rec_fixed = np.array(rec, copy=True)
    rec_fixed[rec_fixed > 5] = 5
    rec_fixed[rec_fixed < 1] = 1
    predicted_values = np.zeros(test.shape[0])
    for i, el in enumerate(test):
        u, m, r = el[0], el[1], el[2]
        predicted_values[i] = rec[u, m]

    rmse = np.sqrt(np.mean((predicted_values-test[:, 2])**2))
    mae = np.mean(abs(predicted_values-test[:, 2]))

    return rmse, mae
