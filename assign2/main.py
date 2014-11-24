__author__ = 'giulio'

import sys

import numpy as np

import algorithms
import load_data as cp
import fold_validation as mfv


_funcdict = {
    'naive_global': algorithms.naive_global,
    'naive_user': algorithms.naive_user,
    'naive_item': algorithms.naive_item,
    'naive_useritem': algorithms.naive_useritem,
    'cf_user_user': algorithms.cf_user_user,
    'cf_item_item': algorithms.cf_item_item,
    'mf_gradient_descent': algorithms.mf_gradient_descent,
    'mf_als': algorithms.mf_als,
}

if __name__ == "__main__":
    alg = _funcdict[sys.argv[1]]

    n_folds = 5
    RMSE = np.zeros(n_folds)
    MAE = np.zeros(n_folds)
    run_time = np.zeros(n_folds)
    mem_usage = np.zeros(n_folds)
    #notpresent = np.zeros(n_folds)
    ratings = cp.load_rat_fast()
    users = cp.load_user_fast()
    movies = cp.load_movies_fast()

    # generate a random permutation of ratings
    p = np.random.permutation(range(0, ratings.shape[0]))
    f = open("results/%s.txt" % sys.argv[1], "w")
    f.close()
    for k in range(n_folds):
        ind_train = np.mod(p, 5) != k
        ind_test = np.mod(p, 5) == k
        train = ratings[ind_train, :]
        test = ratings[ind_test, :]
        res = mfv.fold_validation(users, movies, train, test, alg, steps=30, K=4, lambd=0.03)
        RMSE[k] = res["rmse"]
        MAE[k] = res["mae"]
        run_time[k] = res["time"]
        mem_usage[k] = res["memory"]

        f = open("results/%s.txt" % sys.argv[1], "a")
        f.write("fold%s\ntime: %s\nmemory: %s\nrmse_last: %s\nmae_last: %s\nrmsetest: %s\nmaetest: %s\nrmsetrain: "
                "%s\nmaetrain: %s\n\n" % (str(k), str(res["time"]), str(res["memory"]), str(res["rmse"]),
                                          str(res["mae"]), str(res["rmse_test"]), str(res["mae_test"]),
                                          str(res["rmse_train"]), str(res["mae_train"])))
        f.close()
        print "step: %s" % str(k+1)

    print run_time, mem_usage, RMSE, MAE
