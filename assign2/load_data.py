
import numpy as np


def load_user_fast():
    return np.genfromtxt('datasets/users_half.dat', delimiter="::", dtype="int", usecols=0)


def load_movies_fast():
    return np.genfromtxt('datasets/movies_half.dat', delimiter="::", dtype="int", usecols=0)


def load_rat_fast():
    arr = np.genfromtxt('datasets/ratings_half.dat', usecols=(0, 1, 2), delimiter='::', dtype='int')
    return arr


def my_load_users():
    return np.genfromtxt('datasets/users_clean.dat', delimiter="::", dtype="int", usecols=0)


def my_load_movies():
    return np.genfromtxt("datasets/movies_clean.dat", delimiter="::", dtype="int", usecols=0)


def my_load_ratings():
    arr = np.genfromtxt('datasets/ratings_clean.dat', usecols=(0, 1, 2), delimiter='::', dtype='int')
    return arr


def loadratings():
    arr = np.genfromtxt('datasets/ratings.dat', usecols=(0, 1, 2), delimiter='::', dtype='int')
    return arr


def loadusers():
    arr = np.genfromtxt("datasets/users.dat", delimiter='::', dtype='int', usecols=0)
    return arr


def loadmovies():
    arr = np.genfromtxt("datasets/movies.dat", delimiter='::', dtype='int', usecols=0)
    return arr





