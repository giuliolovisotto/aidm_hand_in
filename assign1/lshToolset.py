__author__ = 'giulio & daniele'
import random
import itertools
import md5
import string

import numpy
import matplotlib.pyplot as plt
import sys


def findsubsets(S, m):
    """
    Finds all the subsets of dimension m for the set S
    Returns a set with the subsets
    """
    return set(itertools.combinations(S, m))


def falls_in_range(mean, diff, element):
    return True if (mean + diff) >= element >= (mean - diff) else False


def hash_md5(n):
    """
    Returns a hash function parametric on n
    """
    random.seed(n)
    rs = ''.join(random.choice(string.lowercase) for i in range(64))
    m = md5.new()

    def myhash(x):
        m.update(str(x)+rs)
        hs = m.hexdigest()
        return int(hs, 32)

    return myhash


def plot_s_curves():
    colours = ["r-", "b-", "g-", "y-", "c-"]
    plt.clf()
    plt.axis([0, 1, 0, 1])
    i = 0
    for jj in [(2, 50), (5, 20), (10, 10), (20, 5), (50, 2)]:
        xs = numpy.linspace(0.0, 1.0, 100)
        ys = [(1 - (1 - x ** jj[1]) ** jj[0]) for x in xs]
        plt.plot(xs, ys, colours[i], )
        i += 1
    plt.show()


def cossim(s1, s2):
    """
    Computes the cosine similarity of two vectors of doubles
    Returns the similarity, 
    """
    # print s1, s2
    return 1.0 if numpy.array_equal(s1, s2) else 1 - numpy.degrees(
        numpy.arccos(numpy.dot(s1, s2) / (numpy.linalg.norm(s1) * numpy.linalg.norm(s2)))) / 180


def sketch(M, k):
    """
    Generates a matrix of sketches 
    Parameters:
    M -- matrix containing the elements per columns
    k -- number of random directions
    Returns the matrix of sketches
    """
    useds = []
    # matrix = [[0 for x in range(len(M[0]))] for y in range(k)]

    matrix = numpy.zeros([k, M.shape[1]])

    m_t = M.T
    # Eseguo tante volte quanti i vettori random da generare
    for i in range(k):
        rdir = numpy.random.normal(size=(len(M)))
        useds.append(rdir)
        # print rdir
        # Ciclo sull'intero array. le COLONNE sono i numeri
        # we need to calculate dot products v_i * x_j for every entry in the matrix
        # x_j indicates the j-th input vector (one per column)
        # v_i indicates the i-th random vector (one per row)
        for j in range(len(m_t)):  # loop over the columns
            # print useds[i], m_t[j]
            matrix[i][j] = numpy.dot(useds[i], m_t[j])
        '''
        for j in range(len(M[0])):
            for h in range(len(M)):
                matrix[i][j] += rdir[h] * M[h][j]
        '''
    # print matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = 1 if matrix[i][j] >= 0 else -1
    # print matrix
    return [matrix, useds]


_memomask = {}


def hash_function(n):
    """
    Not used anymore
    :param n:
    :return:
    """
    mask = _memomask.get(n)
    if mask is None:
        random.seed(n)
        mask = _memomask[n] = random.getrandbits(64)

    def myhash(x):
        return hash(x) ^ mask

    return myhash


def jsim(s1, s2):
    """
    Calculates Jaccard's similarity between 2 sets.
    Parameters:
    s1 -- the first set
    s2 -- the other set
    Returns a float value. 
    """
    return len(s1 & s2) / float(len(s1 | s2))


def cossim_all(inputs):
    """
    Calculates cosine similarity for all the vectors in inputs, 
    Every column is a vector.
    Returns a square matrix of len(inputs)xlen(inputs) containing
    all the cosine similarity between vector pairs.
    
    """
    matrix = [[0 for x in range(len(inputs[0]))] for y in range(len(inputs[0]))]

    inputs = transpose(inputs)

    for i in range(len(inputs)):
        for j in range(i, len(inputs)):
            matrix[i][j] = cossim(inputs[i], inputs[j])
            matrix[j][i] = cossim(inputs[i], inputs[j])
    return matrix


def jsall(s):
    """
    Calculates jaccard similarity for every pair of sets in s
    """
    # matrix = [[0 for x in range(len(s))] for y in range(len(s))]

    matrix = numpy.zeros([len(s), len(s)])
    iters = 0
    tot = len(s)
    for i in range(len(s)):
        for j in range(i, len(s)):
            res = jsim(s[i], s[j])
            matrix[i][j] = res
            matrix[j][i] = res
        iters += 1
        sys.stdout.write("\r%s%%" % round(iters*100/float(tot), 2))
        sys.stdout.flush()
    sys.stdout.write("\n")
    return matrix


def minhash_p(S, k=8, seed=123):
    """
    Calculates the signatures matrix for the sets using permutations.
    Parameters:
    S -- a list of sets
    k -- the number of permutation to use (default 8)
    seed -- the seed for the random package, allows to repeat the experiment (default 123)
    Returns [words, signatures]
    words -- the sorted dictionary containing all the words in the sets
    signatures -- the signature matrix as a list of lists
    """

    random.seed(seed)

    words = set([])
    for s in S:
        words = words.union(s)
    words = sorted(words)
    # print(words)

    signatures = numpy.zeros([k, len(S)])
    # signatures = [[0 for x in range(len(S))] for y in range(k)]

    for i in range(k):
        permutation = range(0, len(words))
        random.shuffle(permutation)
        #print(permutation)

        texts = {}
        for j, item in enumerate(S):
            texts[j] = 0

        for wi in permutation:
            for t, s in texts.iteritems():
                if s == 0:
                    if words[wi] in S[t]:
                        #print(i, t)
                        signatures[i][t] = wi
                        texts[t] = 1
            if all(value == 1 for value in texts.values()):
                break

    #print(signatures)
    return [words, signatures]


def minhash_h(S, k=8):
    """
    Calculates the signatures matrix for the sets using hash functions.
    Parameters:
    S -- a list of sets
    k -- the number of hash functions to use (default 8)
    Returns [words, signatures]
    words -- the sorted dictionary containing all the words in the sets
    signatures -- the signature matrix as a list of lists 
    """
    words = set([])
    for s in S:
        words = words.union(s)
    words = sorted(words)

    # signatures = [[float("inf") for x in range(len(S))] for y in range(k)]
    signatures = numpy.zeros([k, len(S)])
    signatures[:] = numpy.inf

    tot = len(words)
    iters = 0
    for i in range(len(words)):
        for ik in range(k):
            sig = hash_md5(ik)(i)  # % len(words)

            for s in range(len(S)):
                if words[i] in S[s] and signatures[ik][s] > sig:
                    signatures[ik][s] = sig
        iters += 1
        sys.stdout.write("\r%s%%" % round(iters*100/float(tot), 2))
        sys.stdout.flush()

    sys.stdout.write("\n")
    return [words, signatures]


def sigsim(ss1, ss2):
    """
    Calculates the similarity between 2 signatures
    Parameters:
    ss1 -- the first signature as a list
    ss2 -- the second signature as a list
    Returns a value that express the similarity 
    """
    if len(ss1) != len(ss2):
        return False
    simcount = 0
    for i in xrange(len(ss1)):
        simcount += 1 if ss1[i] == ss2[i] else 0

    return simcount / float(len(ss1))


def transpose(M):
    """
    Transpose matrix M and returns the transposed one
    :param M:
    :return:
    """
    return [list(i) for i in zip(*M)]


def simmat(M):
    """
    Calculates the full similarity matrix for a list of signatures
    Parameters:
    M -- a numpy array containing a signatures matrix
    Returns a matrix containing the similarity between pair of signatures
    """
    # matrix = [[0 for x in range(len(M[0]))] for y in range(len(M[0]))]

    matrix = numpy.zeros([len(M[0]), len(M[0])])

    mt = M.T

    tot = matrix.shape[0]
    iters = 0
    for i in xrange(matrix.shape[0]):
        for j in xrange(matrix.shape[0]):
            matrix[i][j] = sigsim(mt[i], mt[j])
        iters += 1
        sys.stdout.write("\r%s%%" % round(iters*100/float(tot), 2))
        sys.stdout.flush()

    return matrix


def banding(c, b, r):
    """
    Returns the bands for the given colum, b*r should be equal to len(c)
    Parameters
    c -- the column as a list
    b -- band count
    r -- number of rows in one band
    Returns a list of lists.
    """
    # o = [[0 for x in range(r)] for y in range(b)]

    o = numpy.zeros([b, r])
    for k, v in enumerate(c):
        o[k / r][k % r] = v
    return o


def bandingsim(M, b, r, s=0):
    """
    Implements the banding similarity technique
    Parameters:
    M -- signature matrix
    b -- number of bands
    r -- number of rows per band
    s -- threshold
    Returns a square matrix with signatures on the rows and on the columns
    where an element e(i,j) has value 1 if the signatures S(i) and S(j)
    have at least one identical bands
    """
    # input is like this [[x1,x2,x3],[y1,y2,y3]]
    # where the columns represent the signatures so we transpose the matrix
    mt = M.T
    # and obtain [[x1,y1],[x2,y2],[x3,y3]] where signatures are in the rows

    # we check if the signatures have the correct length otherwise return error
    if len(mt[0]) != b * r:
        print "Can't apply banding for signatures of length=%s with b=%s, r=%s" % (len(mt[0]), b, r)
        return False

    # square matrix with the output
    # b_matrix = [[0 for x in range(len(mt))] for y in range(len(mt))]

    b_matrix = numpy.zeros([len(mt), len(mt)])
    # threshold = s if s != 0 else ((1 / float(b)) ** (1 / float(r)))

    iters = 0
    tot = b_matrix.shape[0]*(b_matrix.shape[0] - 1) / float(2) + b_matrix.shape[0]
    for i in xrange(b_matrix.shape[0]):
        for j in xrange(i, b_matrix.shape[0]):
            if i == j:
                b_matrix[i][j] = 1
            else:
                bands1 = banding(mt[i], b, r)
                bands2 = banding(mt[j], b, r)
                for ix in xrange(len(bands1)):
                    if (bands1[ix] == bands2[ix]).all():
                        b_matrix[i][j] = 1
                        b_matrix[j][i] = 1
                        break
            iters += 1
            sys.stdout.write("\r%s%%" % round(iters*100/float(tot), 2))
            sys.stdout.flush()

    return b_matrix
