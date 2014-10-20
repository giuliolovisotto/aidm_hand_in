__author__ = 'giulio'

import sys
import sqlite3

import stop_words


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "Use this like this\n python cleanup.py <db_file>"
        sys.exit()

    db_name = sys.argv[1]

    # open output SQLite file
    conn = sqlite3.connect(db_name)

    track_to_save_count = 250
    genres = ['classic pop and rock', 'folk', 'dance and electronica', 'jazz and blues', 'soul and reggae', 'punk',
              'metal', 'classical', 'pop', 'hip-hop']

    cur = conn.cursor()

    # drop table words since we don't need it
    q = "DROP TABLE IF EXISTS words"
    cur.execute(q)
    conn.commit()

    # delete not classified lyrics
    q = "DELETE FROM lyrics WHERE genre = '';"
    cur.execute(q)
    conn.commit()

    # delete stop words
    lis = ""
    for word in stop_words.get_stop_words("english"):
        lis += "\"%s\"," % str(word)
    lis = lis[:-1]  # remove last comma
    q = "DELETE FROM lyrics WHERE word IN (%s);" % lis
    # print q
    cur.execute(q)
    conn.commit()

    # delete some songs for faster computing time
    for g in genres:
        q = "SELECT COUNT(DISTINCT(track_id)) FROM lyrics WHERE genre = '%s'" % g
        cur.execute(q)
        g_count = int(cur.fetchone()[0])
        q = "SELECT DISTINCT(track_id) FROM lyrics WHERE genre = '%s'" % g
        cur.execute(q)
        tracks = [row[0] for row in cur.fetchall()]
        while g_count > track_to_save_count:
            p = "DELETE FROM lyrics WHERE track_id = '%s'" % str(tracks.pop())
            cur.execute(p)
            g_count -= 1
        conn.commit()

    conn.close()
