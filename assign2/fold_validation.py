__author__ = 'giulio'

import compute_error as ce


def fold_validation(users, movies, train, test, alg, **kwargs):
    """
    :param users:
    :param movies:
    :param ratings: sparse matrix with ratings (R)
    :param
    :return:
    """
    # build recommender
    results = alg(users, movies, train, test, **kwargs)
    recommender = results["R"]

    rmse, mae = ce.compute_error(recommender, test)

    return {
        "time": results["time"],
        "memory": results["memory"],
        "rmse": rmse,
        "mae": mae,
        "rmse_train": results["rmse_train"],
        "rmse_test": results["rmse_test"],  # this is an array-
        "mae_train": results["mae_train"],  # this is an array
        "mae_test": results["mae_test"]  # this is an array
    }
