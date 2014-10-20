__author__ = 'giulio'

import sys
import sqlite3
import math

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "Use this like this\n python tfidf.py <db_file>"
        sys.exit()

    db_name = sys.argv[1]

    # open output SQLite file
    conn = sqlite3.connect(db_name)

    q = "ALTER TABLE lyrics ADD COLUMN tfidf REAL"
    conn.execute(q)
    conn.commit()
    cur = conn.cursor()
    #cur.execute(q)

    words = []  # contains the full list of words
    q = "SELECT DISTINCT(track_id) FROM lyrics;"
    tracks = []  # contains the full list of tracks
    for r in cur.execute(q):
        tracks.append(r[0])

    q = "SELECT DISTINCT(word) FROM lyrics;"
    cur.execute(q)
    for row in cur.fetchall():
        words.append(row[0])

    print "tracks and words loaded\ncomputing itf..."

    # compute inverse term frequency
    i_term_frequency = {}
    # we need to know how many documents contains word i -> n_i
    q = "SELECT word, COUNT(DISTINCT(track_id)) FROM lyrics GROUP BY word;"
    total_occurrences = {}
    for rw in cur.execute(q):
        total_occurrences[rw[0]] = rw[1]

    for i, word in enumerate(words):
        sys.stdout.write("\r%s%%" % str(round(i*100/float(len(words)), 2)))
        sys.stdout.flush()
        # q = "SELECT COUNT(DISTINCT(track_id)) FROM lyrics WHERE word = '%s'" % word
        # cur.execute(q)
        # occurrences = cur.fetchone()[0]  # n_i
        i_term_frequency[word] = math.log(len(tracks)/float(total_occurrences[word]), 2)
    print "\r100.00%"
    print "computing tf..."

    # compute term frequency
    for i, t in enumerate(tracks):
        sys.stdout.write("\r%s%%" % str(round(i*100/float(len(tracks)), 2)))
        sys.stdout.flush()
        q = "SELECT word, count FROM lyrics WHERE track_id = '%s'" % t
        term_frequency = {}
        max_frequency = 0
        for word in cur.execute(q):
            term_frequency[word[0]] = word[1]
            if word[1] > max_frequency:
                max_frequency = word[1]
        for k, v in term_frequency.iteritems():
            q = "UPDATE lyrics SET tfidf = %s WHERE track_id = '%s' AND word = '%s';" \
                % (v/float(max_frequency) * i_term_frequency[k], t, k)
            cur.execute(q)
        conn.commit()
    print "\r100.00%"
    print "Computed tf.idf."
    conn.close()
