__author__ = 'giulio'

import load_data as cp
import numpy as np

"""
We want to clean the movies.dat and ratings.dat
to have consecutive ids
"""
if __name__ == "__main__":
    users = cp.loadusers()
    movies = cp.loadmovies()
    ratings = cp.loadratings()

    print users.shape, np.unique(ratings[:, 0]).shape
    print movies.shape, np.unique(ratings[:, 1]).shape
    i = 0
    for u in users:
        if i != u:
            users[i] = i
            # change all the elements in the second column from u to i
            ratings[:, 0][ratings[:, 0] == u] = i
        i += 1

    i = 0
    for m in movies:
        if i != m:
            movies[i] = i
            # change all the elements in the second column from m to i
            # print "cambio %s co %s" % (m, i)
            ratings[:, 1][ratings[:, 1] == m] = i
        i += 1

    print max(ratings[:,1])
    print np.unique(ratings[:, 1]).shape

    np.savetxt('datasets/movies_clean.dat', movies, fmt="%i", delimiter="::")
    np.savetxt('datasets/users_clean.dat', users, fmt="%i", delimiter="::")
    np.savetxt('datasets/ratings_clean.dat', ratings, fmt="%i", delimiter="::")

    print "done"

