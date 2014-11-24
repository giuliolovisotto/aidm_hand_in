__author__ = 'giulio'
import load_data as cp
import numpy as np
import sys

if __name__ == "__main__":
    fact = int(sys.argv[1])
    ratings = cp.my_load_ratings()
    users = cp.my_load_users()
    movies = cp.my_load_movies()
    half_ratings = np.array([]).reshape(0, 3)
    n_usrs = users.shape[0] / fact
    n_movies = movies.shape[0] / fact
    print np.unique(movies).shape, movies.shape, np.unique(ratings[:, 1]).shape
    half_ratings = ratings[ratings[:, 0] < n_usrs]
    half_ratings = half_ratings[half_ratings[:, 1] < n_movies]
    users = np.array([])
    movies = np.array([])
    users = np.arange(0, max(half_ratings[:, 0])+1)
    movies = np.arange(0, max(half_ratings[:, 1])+1)
    # print max(half_ratings[:, 1]), users.shape
    print max(half_ratings[:, 1]), movies.shape, np.unique(half_ratings[:, 1]).shape
    np.savetxt('datasets/movies_half.dat', movies, fmt="%i", delimiter="::")
    np.savetxt('datasets/users_half.dat', users, fmt="%i", delimiter="::")
    np.savetxt('datasets/ratings_half.dat', half_ratings, fmt="%i", delimiter="::")
    
    print "done"
