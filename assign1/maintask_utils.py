__author__ = 'giulio'

import sys

import numpy as np

import lshToolset as ls


def get_track_genre(track, cur):
    q = "SELECT genre FROM lyrics WHERE track_id='%s'" % track
    cur.execute(q)
    return cur.fetchone()[0]


def get_whole_tracks_genres(cur):
    """
    :param cur:
    :return:
    """
    # load tracks into memory
    track_list = get_full_track_list(cur)
    tracks = {}
    # build the sets with the words
    sys.stdout.write("Loading tracks into memory\n")
    for idx, t in enumerate(track_list):
        sys.stdout.write("\r%s%%" % round(idx * 100 / float(len(track_list)), 2))
        sys.stdout.flush()
        tracks[t] = get_track_genre(t, cur)
    sys.stdout.write("\r100.00% Done\n")
    return tracks


def get_tracks_dictionary(cursor, howmany = 9999):
    """
    Returns a dictionary indexed on trackid which contains a set with the words for that track
    :param cursor:
    :return:
    """
    # load tracks into memory
    track_list = get_full_track_list(cursor)
    tracks = {}
    # build the sets with the words
    sys.stdout.write("Loading tracks into memory\n")
    for idx, t in enumerate(track_list):
        sys.stdout.write("\r%s%%" % round(idx * 100 / float(len(track_list)), 2))
        sys.stdout.flush()
        tracks[t] = get_words_for_track(t, cursor, howmany)
    sys.stdout.write("\r100.00% Done\n")
    return tracks


def get_words_for_track(track_id, cursor, howmany):
    """
    Returns a set containing the words for the given track
    :param track_id:
    :param cursor:
    :return:
    """
    q = "SELECT word FROM lyrics WHERE track_id = '%s' ORDER BY tfidf DESC LIMIT %s" % (str(track_id), str(howmany))
    cursor.execute(q)
    words = set([])
    for r in cursor.fetchall():
        words.add(r[0])
    # print words
    return words




def get_full_track_list(cursor):
    """
    Returns a numpy array containing the full list of tracks in the database
    :param cursor:
    :return:
    """
    q = "SELECT DISTINCT(track_id) FROM lyrics;"
    cursor.execute(q)
    full_list = np.array([])
    for r in cursor.fetchall():
        full_list = np.append(full_list, r[0])
    return full_list


def get_tracks_by_genre(count, genre, cursor):
    q = "SELECT DISTINCT(track_id) FROM lyrics WHERE genre = '%s' LIMIT %s" % (genre, count)
    ts = np.array([])
    for row in cursor.execute(q):
        ts = np.append(ts, row[0])
    return ts


def build_matrix(track_list, terms_count, cursor):
    """
    Given a list with the track ids to consider builds a matrix of sets,
    where they contain the words used in the given track
    :param track_list:
    :return:
    """
    # dictionary which keeps the bags of words for the dataset
    # indexed on the track_id, contains a set with the bag
    bags = {}
    for t in track_list:
        q = "SELECT word FROM lyrics WHERE track_id = '%s' ORDER BY tfidf DESC LIMIT %s;" % (str(t), terms_count)
        for row in cursor.execute(q):
            # print t, row
            if str(t) not in bags:
                bags[str(t)] = {row[0]}
            else:
                bags[str(t)].add(row[0])

    return np.array([v for k, v in bags.iteritems()])


def compute_banding(m, k, b, r):
    """
    Maybe transform minhash_h and bandingsim to work with numpy arrays?
    :param m: matrix with signatures
    :param k: number of hash functions
    :param b: number of bands
    :param r: number of rows per band
    :return: - a list which contains the number of 1s in the banding matrix
             - how many 1s over the maximum number of 1s
    """
    [words, minhashes] = ls.minhash_h(m, k)
    band = ls.bandingsim(minhashes, b, r)
    ones = sum([sum(rr) for rr in band])
    return [ones, (ones - len(m)) / float((len(m) * (len(m) - 1))), band]


