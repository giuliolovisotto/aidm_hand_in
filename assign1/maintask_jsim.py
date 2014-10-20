__author__ = 'giulio'

import time

from maintask_utils import *


_TABLES = []
_TABLES_MAP = {}
_MAP_TO_BUCKETS = {}
_TABLES_SIZES = []
for cnt, a in enumerate(range(10, 55, 4)):  # 12 of these
    _TABLES.append("jsim" + str(cnt + 1))
    _TABLES_MAP["jsim" + str(cnt + 1)] = "jacsim" + str(a)
    _MAP_TO_BUCKETS[int(a)] = cnt + 1
    _TABLES_SIZES.append(a)
# tables = ['jsim1', 'jsim2', 'jsim3', 'jsim4']  # every table keeps 3 values for jsim
# TABLES_MAP = {'jsim1': ('jacsim10', 'jacsim14', 'jacsim18'), 'jsim2': ('jacsim22', 'jacsim26', 'jacsim30'),
# 'jsim3': ('jacsim34', 'jacsim38', 'jacsim42'), 'jsim4': ('jacsim46', 'jacsim50', 'jacsim54')}


def compute_whole_jsim(conn):
    """
    Computes the jaccard similarity for every pair of track in the database.
    a) reate the db table to store the jaccard similarity
    b) drops it if it already exists,
    c) measure the time required for the only computation of the jsmin
    :param conn: the connection to the database
    :return: the time required for the only computation
    """

    # create table and drop its content to allow multiple invocations
    q = ("CREATE TABLE IF NOT EXISTS jsim ("
         "tid1 TEXT, "
         "tid2 TEXT, "
         "jacsim REAL, "
         "FOREIGN KEY(tid1) REFERENCES lyrics(track_id), "
         "FOREIGN KEY(tid2) REFERENCES lyrics(track_id)"
         ");")
    cur = conn.cursor()
    cur.execute(q)
    conn.commit()
    q = "DELETE FROM jsim;"
    cur.execute(q)

    for t in _TABLES:
        q = ("CREATE TABLE IF NOT EXISTS %s (tid1 TEXT, tid2 TEXT, %s REAL,"
             "FOREIGN KEY(tid1) REFERENCES lyrics(track_id), "
             "FOREIGN KEY(tid2) REFERENCES lyrics(track_id));" % (t, _TABLES_MAP[t]))
        cur.execute(q)
        conn.commit()
        q = "DELETE FROM %s" % t
        cur.execute(q)
        conn.commit()
        q = "CREATE INDEX IF NOT EXISTS %s_1 ON %s ('tid1');" % (t, t)
        cur.execute(q)
        q = "CREATE INDEX IF NOT EXISTS %s_2 ON %s ('tid2');" % (t, t)
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
    sys.stdout.write("Computing Jaccard similarity for all pairs...\n")
    # start the clock
    start_time = time.time()
    # actual function call
    jsim_output = ls.jsall(t_matrix)
    # stop the clock
    elapsed = time.time() - start_time
    result = dict()
    result['all'] = elapsed

    # now insert records to database
    sys.stdout.write("Filling database...\n")
    batch_size = 0
    iters = 0

    tot = jsim_output.shape[0] * (jsim_output.shape[0] - 1) / 2 + jsim_output.shape[0]
    for i in range(jsim_output.shape[0]):
        query = "INSERT INTO jsim (tid1, tid2, jacsim) VALUES"
        for j in range(i, jsim_output.shape[1]):
            query += ("('%s', '%s', %s),"
                      % (str(track_map[i]), str(track_map[j]), str(jsim_output[i, j])))
            batch_size += 1
            # execute one big query is better for speed
            if batch_size > 499:
                iters += 500
                sys.stdout.write("\r%s%%" % round(iters * 100 / float(tot), 2))
                sys.stdout.flush()
                # remove last comma
                query = query[:-1] + ";"
                cur.execute(query)
                conn.commit()
                query = "INSERT INTO jsim (tid1, tid2, jacsim) VALUES"
                batch_size = 0

        # if it's not empty, insert
        if batch_size > 0:
            iters += batch_size
            # remove last comma
            query = query[:-1] + ";"
            cur.execute(query)
            conn.commit()
            batch_size = 0

    q = "CREATE INDEX IF NOT EXISTS jsim_1 ON jsim ('tid1');"
    cur.execute(q)
    q = "CREATE INDEX IF NOT EXISTS jsim_2 ON jsim ('tid2');"
    cur.execute(q)
    conn.commit()

    sys.stdout.write("\r100.00%\n")
    # map_to_buckets = {10: 1, 14: 1, 18: 1, 22: 2, 26: 2, 30: 2, 34: 3, 38: 3, 42: 3, 46: 4, 50: 4, 54: 4}
    # map_to_buckets = {10: 1, 12: 1, 14: 1, 16: 2, 18: 2, 20: 2, 22: 3, 24: 3, 26: 3, 28: 4, 30: 4, 32: 4}

    for cn, word in enumerate(range(10, 55, 4)):

        # get dict with fixed terms set
        t_dict = get_tracks_dictionary(cur, word)
        t_matrix = np.array([])
        # rebuild matrix with fixed size sets
        for i, (k, v) in enumerate(t_dict.iteritems()):
            t_matrix = np.append(t_matrix, v)

        sys.stdout.write("Computing Jaccard similarity for all pairs with %s words...\n" % word)
        start_time = time.time()
        # actual function call
        jsim_output = ls.jsall(t_matrix)
        # stop the clock
        elapsed = time.time() - start_time
        result[word] = elapsed

        # now insert records to database
        sys.stdout.write("Filling database for jacsim%s...\n" % str(word))
        batch_size = 0
        iters = 0
        tot = jsim_output.shape[0] * (jsim_output.shape[0] - 1) / 2 + jsim_output.shape[0]
        for i in range(jsim_output.shape[0]):
            q = "INSERT INTO jsim%s (tid1, tid2, jacsim%s) VALUES" % (str(cn + 1), str(word))
            for j in range(i, jsim_output.shape[1]):
                q += ("('%s', '%s', %s),"
                      % (str(track_map[i]), str(track_map[j]),
                         str(jsim_output[i, j])))
                batch_size += 1
                # execute one big query is better for speed
                if batch_size > 499:
                    iters += 500
                    sys.stdout.write("\r%s%%" % round(iters * 100 / float(tot), 2))
                    sys.stdout.flush()
                    # remove last comma
                    q = q[:-1] + ";"
                    cur.execute(q)
                    conn.commit()
                    q = "INSERT INTO jsim%s (tid1, tid2, jacsim%s) VALUES" % (str(cn + 1), str(word))
                    batch_size = 0
            # if it's not empty, insert
            if batch_size > 0:
                iters += batch_size
                q = q[:-1] + ";"
                cur.execute(q)
                conn.commit()
                batch_size = 0
        sys.stdout.write("\r100.00%%, done for this size.\n")

    return result


def get_whole_similarities(cur, n_words=-1):
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
    if n_words != -1:
        q = "SELECT tid1, tid2, jacsim%s FROM jsim%s;" % (str(n_words), _MAP_TO_BUCKETS[n_words])
    else:
        q = "SELECT tid1, tid2, jacsim FROM jsim;"
    cur.execute(q)
    d_js = {}
    for row in cur.fetchall():
        d_js[(row[0], row[1])] = row[2]

    return d_js


def check_genre_matches(threshold, conn, t_list=None, t_genre=None, t_jsimilarities=None):
    """
    Apply our beautiful algorithm to check if song matches by genre.
    :param conn: connection to db
    :return: returns a tuple containing (fp, fn, tp, tn)
    """
    cur = conn.cursor()
    if t_list is None or t_genre is None or t_jsimilarities is None:
        t_list = get_full_track_list(cur)
        t_genre = get_whole_tracks_genres(cur)
        t_jsimilarities = get_whole_similarities(cur)

    fp, fn, tp, tn = 0, 0, 0, 0
    # threshold = 0.001
    tot = (t_list.shape[0] * (t_list.shape[0] - 1)) / 2 + t_list.shape[0]
    done = 0
    sys.stdout.write("Computing results..\n")
    for i in range(t_list.shape[0]):
        for j in range(i, t_list.shape[0]):
            done += 1
            sys.stdout.write("\r%s%% " % round(done * 100 / float(tot), 2))
            sys.stdout.flush()
            if (t_list[i], t_list[j]) in t_jsimilarities:
                this_js = t_jsimilarities[(t_list[i], t_list[j])]
            else:  # not very safe
                this_js = t_jsimilarities[(t_list[j], t_list[i])]
            g1 = t_genre[t_list[i]]
            g2 = t_genre[t_list[j]]
            same_genre = g1 == g2
            if this_js >= threshold:
                # positive
                tp += 1 if same_genre else 0
                fp += 0 if same_genre else 1
            else:
                # negative
                tn += 0 if same_genre else 1
                fn += 1 if same_genre else 0
    sys.stdout.write("Done\n")
    return fp, fn, tp, tn
