__author__ = 'giulio'

from maintask_utils import *
import numpy as np
import time
import json
import lshToolset as lsh


def compute_whole_minhash(conn):
    """
    Computes the jaccard similarity for every pair of track in the database.
    a) reate the db table to store the jaccard similarity
    b) drops it if it already exists,
    c) measure the time required for the only computation of the jsmin
    :param conn: the connection to the database
    :return: the time required for the only computation
    """
    # create tables and drop their content to allow multiple invocations
    q = ("CREATE TABLE IF NOT EXISTS minhashsim ("
         "tid1 TEXT, "
         "tid2 TEXT, "
         "sim REAL, "
         "FOREIGN KEY(tid1) REFERENCES lyrics(track_id), "
         "FOREIGN KEY(tid2) REFERENCES lyrics(track_id)"
         ");")
    cur = conn.cursor()
    cur.execute(q)
    conn.commit()
    q = "DELETE FROM minhashsim;"
    cur.execute(q)
    conn.commit()

    q = ("CREATE TABLE IF NOT EXISTS minhash ("
         "tid1 TEXT, "
         "minhash TEXT, "
         "FOREIGN KEY(tid1) REFERENCES lyrics(track_id) "
         ");")
    cur = conn.cursor()
    cur.execute(q)
    conn.commit()
    q = "DELETE FROM minhash;"
    cur.execute(q)
    conn.commit()

    # get the tracks
    t_dict = get_tracks_dictionary(cur)
    # build the input matrix
    t_matrix = np.array([])
    # build the map to keep track of the track ids
    idx = 0
    track_map = {}
    for k, v in t_dict.iteritems():
        t_matrix = np.append(t_matrix, v)
        track_map[idx] = k
        idx += 1
    # print t_matrix
    sys.stdout.write("Computing Minhashes...\n")
    # start the clock
    start_time = time.time()
    # actual function calls
    [whatever, signatures] = ls.minhash_h(t_matrix, 720)
    # stop the clock
    elapsed = time.time() - start_time
    sys.stdout.write("Computing dumb minhash similarity for all pairs...\n")
    simmat = ls.simmat(signatures)

    # now insert records to database
    sys.stdout.write("Filling database...\n")
    batch_size = 0
    for i in range(simmat.shape[0]):
        query = "INSERT INTO minhashsim (tid1, tid2, sim) VALUES"
        for j in range(i, simmat.shape[1]):
            query += "('%s', '%s', %s)," % (str(track_map[i]), str(track_map[j]), simmat[i, j])
            batch_size += 1
            # execute one big query is better for speed
            if batch_size > 499:
                # remove last comma
                query = query[:-1] + ";"
                cur.execute(query)
                query = "INSERT INTO minhashsim (tid1, tid2, sim) VALUES"
                batch_size = 0
        # if it's not empty, insert
        if batch_size > 0:
            query = query[:-1] + ";"
            cur.execute(query)

    conn.commit()

    batch_size = 0
    buffer_size = 0
    query = "INSERT INTO minhash (tid1, minhash) VALUES"
    for i in range(signatures.shape[1]):
        query += "('%s', '%s')," % (str(track_map[i]), json.dumps(signatures[:, i].tolist()))
        batch_size += 1
        # execute one big query is better for speed
        if batch_size > 499:
            # remove last comma
            query = query[:-1] + ";"
            cur.execute(query)
            query = "INSERT INTO minhash (tid1, minhash) VALUES"
            batch_size = 0
            buffer_size += 1
            if buffer_size > 499:
                conn.commit()
                buffer_size = 0
    # if it's not empty, insert
    if batch_size > 0:
        query = query[:-1] + ";"
        cur.execute(query)

    conn.commit()

    return elapsed


def banding_all(cur):
    q = "SELECT COUNT() FROM minhash;"
    cur.execute(q)
    s = 0
    for row in cur.fetchall():
        print row
        s = row[0]
    all_rows = np.zeros([s, 720])
    print s

    q = "SELECT tid1, minhash FROM minhash;"
    cur.execute(q)
    i = 0
    map_stuff = {}
    for row in cur.fetchall():
        map_stuff[i] = row[0]
        signature = json.loads(row[1])
        for j in range(0, len(signature)):
            all_rows[i, j] = signature[j]
        i += 1

    result = []
    for params in [(720, 1), (360, 2), (240, 3), (180, 4), (144, 5), (120, 6), (90, 8)]:
        matrix = lsh.bandingsim(all_rows.T, params[0], params[1])
        result.append(matrix)

    return result


def check_genre_matches(matrix, conn, t_list=None, t_genre=None):
    """
    Apply our beautiful algorithm to check if song matches by genre.
    :param conn: connection to db
    :return: returns a tuple containing (fp, fn, tp, tn)
    """
    cur = conn.cursor()
    if t_list is None or t_genre is None is None:
        t_list = get_full_track_list(cur)
        t_genre = get_whole_tracks_genres(cur)

    fp, fn, tp, tn = 0, 0, 0, 0
    # threshold = 0.001
    tot = (t_list.shape[0] * (t_list.shape[0] - 1)) / 2 + t_list.shape[0]
    done = 0
    sys.stdout.write("Computing results..\n")
    for i in range(t_list.shape[0]):
        for j in range(i, t_list.shape[0]):
            done += 1
            # print done
            sys.stdout.write("\r%s%% " % round(done * 100 / float(tot), 2))
            sys.stdout.flush()
            g1 = t_genre[t_list[i]]
            g2 = t_genre[t_list[j]]
            same_genre = g1 == g2
            if matrix[i, j] == 1:
                # positive
                tp += 1 if same_genre else 0
                fp += 0 if same_genre else 1
            else:
                # negative
                tn += 0 if same_genre else 1
                fn += 1 if same_genre else 0
    sys.stdout.write("Done\n")
    return fp, fn, tp, tn


def get_whole_similarities(cur):
    """
    Returns a dictionary indexed on tuples containing the 2 ids of the tracks
    for example
    {
    ('trackid1', 'trackid2'): jsim_1_2,
    ('trackid2', 'trackid3'): jsim_2_3,
    ...
    }
    :param cur:
    :return:
    """
    q = "SELECT tid1, tid2, sim FROM minhashsim;"
    cur.execute(q)
    d_js = {}
    for row in cur.fetchall():
        d_js[(row[0], row[1])] = row[2]

    return d_js