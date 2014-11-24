__author__ = 'giulio'
import time
import scipy.sparse as ss
import resource
import multiprocessing

import numpy as np
import joblib

import compute_error as ce
import split_matrix as gs
import similarities as pt


def naive_global(users, movies, train, test, **kwargs):
    just_train = ss.csr_matrix((train[:, 2], (train[:, 0], train[:, 1])), shape=(users.shape[0], movies.shape[0]))
    just_train = np.array(just_train.todense())
    just_train = just_train.astype(float)
    start = time.time()
    just_train[just_train == 0] = np.sum(just_train)/float(np.count_nonzero(just_train))
    elapsed = time.time() - start
    rmse_train, mae_train = ce.compute_error(just_train, train)
    rmse_test, mae_test = ce.compute_error(just_train, test)
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "R": just_train,
        "time": elapsed,
        "memory": mem_usage,
        "rmse_train": rmse_train,  # this is an array
        "rmse_test": rmse_test,  # this is an array
        "mae_train": mae_train,  # this is an array
        "mae_test": mae_test  # this is an array
    }


def naive_user(users, movies, train, test, **kwargs):
    just_train = ss.csr_matrix((train[:, 2], (train[:, 0], train[:, 1])), shape=(users.shape[0], movies.shape[0]))
    just_train = np.array(just_train.todense())
    just_train = just_train.astype(float)
    start = time.time()
    # for every user update the missing ratings
    for i, u in enumerate(just_train):
        if np.sum(u) > 0:
            u[u == 0] = (np.sum(u)/float(np.count_nonzero(u)))
    # print just_train
    elapsed = time.time() - start
    rmse_train, mae_train = ce.compute_error(just_train, train)
    rmse_test, mae_test = ce.compute_error(just_train, test)
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "R": just_train,
        "time": elapsed,
        "memory": mem_usage,
        "rmse_train": rmse_train,  # this is an array
        "rmse_test": rmse_test,  # this is an array
        "mae_train": mae_train,  # this is an array
        "mae_test": mae_test  # this is
    }


def naive_item(users, movies, train, test, **kwargs):
    just_train = ss.csr_matrix((train[:, 2], (train[:, 0], train[:, 1])), shape=(users.shape[0], movies.shape[0]))
    just_train = np.array(just_train.todense())
    just_train = just_train.astype(float)
    start = time.time()
    # for every user update the missing ratings
    just_train = just_train.T
    for i, itm in enumerate(just_train):
        itm[itm == 0] = (np.sum(itm)/float(np.count_nonzero(itm))) if np.sum(itm) > 0 else 2.5

    just_train = just_train.T

    elapsed = time.time() - start
    rmse_train, mae_train = ce.compute_error(just_train, train)
    rmse_test, mae_test = ce.compute_error(just_train, test)
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "R": just_train,
        "time": elapsed,
        "memory": mem_usage,
        "rmse_train": rmse_train,  # this is an array
        "rmse_test": rmse_test,  # this is an array
        "mae_train": mae_train,  # this is an array
        "mae_test": mae_test  # this is an array
    }


def naive_useritem(users, movies, train, test, **kwargs):
    just_train = ss.csr_matrix((train[:, 2], (train[:, 0], train[:, 1])), shape=(users.shape[0], movies.shape[0]))
    just_train = np.array(just_train.todense())
    just_train = just_train.astype(float)
    start = time.time()
    # ok we build A(nonzeros x 2), x(1 x nonzeros) and we solve to find y(1 x 2) = xA
    A = np.zeros((np.count_nonzero(just_train), 2))
    x = np.zeros(np.count_nonzero(just_train))
    r_user = np.zeros(users.shape[0])
    r_item = np.zeros(movies.shape[0])
    rows, cols = just_train.nonzero()
    for n, (i, j) in enumerate(zip(rows, cols)):
        x[n] = just_train[i, j]
        if r_user[i] == 0:
            r_user[i] = (np.sum(just_train[i, :])/float(np.count_nonzero(just_train[i, :]))) if np.sum(just_train[i,
                                                                                                       :]) > 0 else 2.5
        if r_item[j] == 0:
            r_item[j] = (np.sum(just_train[:, j])/float(np.count_nonzero(just_train[:, j]))) if np.sum(just_train[:,
                                                                                                       j]) > 0 else 2.5
        A[n] = np.array([r_user[i], r_item[j]])

    [alpha, beta] = np.linalg.lstsq(A, x)[0]

    for i in xrange(just_train.shape[0]):
        print i
        for j in xrange(just_train.shape[1]):
            if just_train[i, j] == 0:
                just_train[i, j] = alpha * r_user[i] + beta * r_item[j]

    elapsed = time.time() - start
    rmse_train, mae_train = ce.compute_error(just_train, train)
    rmse_test, mae_test = ce.compute_error(just_train, test)
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "R": just_train,
        "time": elapsed,
        "memory": mem_usage,
        "rmse_train": rmse_train,  # this is an array
        "rmse_test": rmse_test,  # this is an array
        "mae_train": mae_train,  # this is an array
        "mae_test": mae_test  # this is an array
    }


def cf_user_user(users, movies, train, test, **kwargs):
    sim = "pearson" if "sim" not in kwargs else kwargs["sim"]
    nbrs = 5 if "nbrs" not in kwargs else kwargs["nbrs"]

    just_train = ss.csr_matrix((train[:, 2], (train[:, 0], train[:, 1])), shape=(users.shape[0], movies.shape[0]))
    just_test = ss.csr_matrix((test[:, 2], (test[:, 0], test[:, 1])), shape=(users.shape[0], movies.shape[0]))
    just_train = np.array(just_train.todense())
    just_test = np.array(just_test.todense())
    just_train = just_train.astype(float)
    just_test = just_test.astype(float)

    full_sim = pt.full_pearson_sim(just_train) if sim == "pearson" else pt.full_cosine_sim(just_train)
    full_sim[np.isnan(full_sim)] = 0
    start = time.time()
    rows, cols = just_test.nonzero()
    neighbors_indexes = full_sim.argsort(axis=1)
    neighbors_indexes = np.fliplr(neighbors_indexes)

    for i, j in zip(rows, cols):
        i_neighb = np.array([])
        k = 0
        while len(i_neighb) < nbrs and k < len(neighbors_indexes[i]):
            if just_train[neighbors_indexes[i][k]][j] != 0:
                i_neighb = np.append(i_neighb, neighbors_indexes[i][k])
            k += 1

        weights = np.zeros(len(i_neighb))
        ratings = np.zeros(len(i_neighb))
        for c, nb in enumerate(i_neighb):
            weights[c] = full_sim[i, nb]
            ratings[c] = just_train[nb, j]

        just_train[i, j] = np.dot(weights, ratings)/(float(np.sum(weights)) if np.sum(weights) != 0 else 1)
    elapsed = time.time() - start
    rmse_train, mae_train = ce.compute_error(just_train, train)
    rmse_test, mae_test = ce.compute_error(just_train, test)
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "R": just_train,
        "time": elapsed,
        "memory": mem_usage,
        "rmse_train": rmse_train,  # this is an array
        "rmse_test": rmse_test,  # this is an array
        "mae_train": mae_train,  # this is an array
        "mae_test": mae_test  # this is an array'''
    }


def cf_item_item(users, movies, train, test, **kwargs):
    sim = "pearson" if "sim" not in kwargs else kwargs["sim"]
    nbrs = 5 if "nbrs" not in kwargs else kwargs["nbrs"]

    just_train = ss.csr_matrix((train[:, 2], (train[:, 0], train[:, 1])), shape=(users.shape[0], movies.shape[0]))
    just_test = ss.csr_matrix((test[:, 2], (test[:, 0], test[:, 1])), shape=(users.shape[0], movies.shape[0]))
    just_train = np.array(just_train.todense())
    just_test = np.array(just_test.todense())
    just_train = just_train.astype(float)
    just_test = just_test.astype(float)

    full_sim = pt.full_pearson_sim(just_train.T) if sim == "pearson" else pt.full_cosine_sim(just_train.T)
    full_sim[np.isnan(full_sim)] = 0
    start = time.time()
    rows, cols = just_test.nonzero()
    neighbors_indexes = full_sim.argsort(axis=1)
    neighbors_indexes = np.fliplr(neighbors_indexes)

    for i, j in zip(rows, cols):
        i_neighb = np.array([])
        k = 0
        while len(i_neighb) < nbrs and k < len(neighbors_indexes[j]):
            if just_train[i][neighbors_indexes[j][k]] != 0:
                i_neighb = np.append(i_neighb, neighbors_indexes[j][k])
            k += 1

        weights = np.zeros(len(i_neighb))
        ratings = np.zeros(len(i_neighb))
        for c, nb in enumerate(i_neighb):
            weights[c] = full_sim[j, nb]
            ratings[c] = just_train[i, nb]

        just_train[i, j] = np.dot(weights, ratings)/(float(np.sum(weights)) if np.sum(weights) != 0 else 1)
    elapsed = time.time() - start
    rmse_train, mae_train = ce.compute_error(just_train, train)
    rmse_test, mae_test = ce.compute_error(just_train, test)
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "R": just_train,
        "time": elapsed,
        "memory": mem_usage,
        "rmse_train": rmse_train,  # this is an array
        "rmse_test": rmse_test,  # this is an array
        "mae_train": mae_train,  # this is an array
        "mae_test": mae_test  # this is an array'''
    }


def _als_step_m(U, lambd, K, W, just_train, begin, last):
    result = np.zeros((K, last-begin+1))
    i = 0
    W = W.T
    #print "begin: %s, last: %s" % (begin, last)
    for m in range(begin, last+1):
        result[:, i] = np.linalg.solve(np.dot(U.T, np.dot(np.diag(W[m]), U)) + lambd * np.eye(K), np.dot(U.T, np.dot(np.diag(W[m]), just_train[:, m])))
        i += 1
        #print i
    #print "DONE"
    return result


def _als_step_u(M, lambd, K, W, just_train, begin, last):
    result = np.zeros((last-begin+1, K))
    i = 0
    #print "begin: %s, last: %s" % (begin, last)
    for u in range(begin, last+1):
        result[i] = np.linalg.solve(np.dot(M, np.dot(np.diag(W[u]), M.T)) + lambd * np.eye(K),np.dot(M,np.dot(np.diag(
                W[u]), just_train[u].T))).T
        i += 1
        #print i
    #print "DONE"
    return result


def mf_als(users, movies, train, test, **kwargs):
    K = 2 if "K" not in kwargs else kwargs["K"]
    steps = 30 if "steps" not in kwargs else kwargs["steps"]
    lambd = 0.065 if "lambda" not in kwargs else kwargs["lambd"]
    rmse_train = np.zeros(steps)
    rmse_test = np.zeros(steps)
    mae_train = np.zeros(steps)
    mae_test = np.zeros(steps)

    # TODO inizializza a media della minchia
    U = np.random.rand(users.shape[0], K)
    M = np.random.rand(K, movies.shape[0])

    just_train = ss.csr_matrix((train[:, 2], (train[:, 0], train[:, 1])), shape=(users.shape[0], movies.shape[0]))
    # just_test = ss.csr_matrix((test[:, 2], (test[:, 0], test[:, 1])), shape=(users.shape[0], movies.shape[0]))

    just_train = np.array(just_train.todense())

    elapsed = 0

    W = just_train > 0.5
    W[W == True] = 1
    W[W == False] = 0
    # To be consistent with our Q matrix
    W = W.astype(np.float64, copy=False)
    cores = multiprocessing.cpu_count()
    for step in range(steps):
        #print "step %s" % str(step+1)
        start = time.time()
        # slice matrix
        slices = gs.split_matrix(U, cores)
        ranges = np.zeros((len(slices), 2))
        first = 0
        for i, s in enumerate(slices):
            ranges[i, 0] = int(first)
            ranges[i, 1] = first + s.shape[0] - 1
            first += s.shape[0]
        ranges = ranges.astype(int)
        results = joblib.Parallel(n_jobs=cores, backend="threading")(joblib.delayed(_als_step_u)(M, lambd, K, W, just_train, b[0],
                                                                            b[1]) for b in ranges)
        for i, r in enumerate(results):
            U[ranges[i][0]:ranges[i][1]+1, :] = r

        # NOW WE DO IT FOR M
        slices = gs.split_matrix(M.T, cores)
        ranges = np.zeros((len(slices), 2))
        first = 0
        for i, s in enumerate(slices):
            ranges[i, 0] = int(first)
            ranges[i, 1] = first + s.shape[0] - 1
            first += s.shape[0]
        ranges = ranges.astype(int)
        results = joblib.Parallel(n_jobs=cores)(joblib.delayed(_als_step_m)(U, lambd, K, W, just_train, b[0],
                                                                            b[1]) for b in ranges)
        for i, r in enumerate(results):
            M[:, ranges[i][0]:ranges[i][1]+1] = r

        #here is the non parallel version
        #for u, Wu in enumerate(W):
        #    U[u] = np.linalg.solve(np.dot(M, np.dot(np.diag(Wu), M.T)) + lambd * np.eye(K),np.dot(M,np.dot(np.diag(
        # Wu), just_train[u].T))).T
        #for i, Wi in enumerate(W.T):
        #    M[:, i] = np.linalg.solve(np.dot(U.T, np.dot(np.diag(Wi), U)) + lambd * np.eye(K), np.dot(U.T,
        # np.dot(np.diag(Wi), just_train[:, i])))
        elapsed += time.time() - start
       
        #print "This step took %s" % str(elapsed)
        rmse_train[step], mae_train[step] = ce.compute_error(np.dot(U, M), train)
        rmse_test[step], mae_test[step] = ce.compute_error(np.dot(U, M), test)
        #print rmse_test[step], mae_test[step]
        f = open('als_status.txt', 'a')
        f.write("step (%s): time(%s)\n rmse: (%s), mae: (%s)\n" % (str(step),str(elapsed),str(rmse_test[step]),
                                                                 str(mae_test[step])))
        f.close()
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "R": np.dot(U, M),
        "U": U,
        "M": M,
        "time": elapsed,
        "memory": mem_usage,
        "rmse_train": rmse_train,  # this is an array
        "rmse_test": rmse_test,  # this is an array
        "mae_train": mae_train,  # this is an array
        "mae_test": mae_test  # this is an array
    }


def mf_gradient_descent(users, movies, train, test, **kwargs):
    """
    This function should return
     1) mae, rmse for every step and for separate sets test/train
     2) total execution time
     3) R matrix given by UxM
     4) U and M
     5) memory usage
    :param users:
    :param movies:
    :param train:
    :param test:
    :param kwargs:
    :return:
    """
    K = 40 if "K" not in kwargs else kwargs["K"]
    nu = 0.005 if "nu" not in kwargs else kwargs["nu"]
    steps = 30 if "steps" not in kwargs else kwargs["steps"]
    lambd = 0.02 if "lambda" not in kwargs else kwargs["lambd"]
    rmse_train = np.zeros(steps)
    rmse_test = np.zeros(steps)
    mae_train = np.zeros(steps)
    mae_test = np.zeros(steps)
    # initialize to random matrices
    # TODO maybe some values in range 1..5
    U = np.random.rand(users.shape[0], K)
    M = np.random.rand(K, movies.shape[0])

    # build the test and the train sparse matrix
    just_train = ss.csr_matrix((train[:, 2], (train[:, 0], train[:, 1])), shape=(users.shape[0], movies.shape[0]))
    # just_test = ss.csr_matrix((test[:, 2], (test[:, 0], test[:, 1])), shape=(users.shape[0], movies.shape[0]))

    rows, cols = just_train.nonzero()
    elapsed = 0

    for step in xrange(steps):
        start = time.time()
        for i, j in zip(rows, cols):
            eij = just_train[i, j] - np.dot(U[i, :], M[:, j])
            for k in xrange(K):
                U[i][k] += nu * (2 * eij * M[k][j] - lambd * U[i][k])
                M[k][j] += nu * (2 * eij * U[i][k] - lambd * M[k][j])
        elapsed += time.time() - start
        rmse_train[step], mae_train[step] = ce.compute_error(np.dot(U, M), train)
        rmse_test[step], mae_test[step] = ce.compute_error(np.dot(U, M), test)
        f = open('gd_status.txt', 'a')
        f.write("step (%s): time(%s)\n rmse: (%s), mae: (%s)\n" % (str(step), str(elapsed), str(rmse_test[step]),
                                                                   str(mae_test[step])))
        f.close()
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "R": np.dot(U, M),
        "U": U,
        "M": M,
        "time": elapsed,
        "memory": mem_usage,
        "rmse_train": rmse_train,  # this is an array
        "rmse_test": rmse_test,  # this is an array
        "mae_train": mae_train,  # this is an array
        "mae_test": mae_test  # this is an array
    }

