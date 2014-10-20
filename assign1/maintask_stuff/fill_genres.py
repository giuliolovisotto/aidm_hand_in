__author__ = 'giulio'


import os
import sys
import sqlite3

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "Use this file like this\npython fill_genres.py <genres_txt_file> <db_file>"
        exit()
    genres_file = sys.argv[1]
    db_name = sys.argv[2]

    # open output SQLite file
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    f = open(genres_file, 'r')

    q = "UPDATE lyrics SET genre = '%s' WHERE track_id = %s"

    for line in f.xreadlines():
        if line == '' or line[0] == '#' or line[0] == '%':
            continue
        else:
            values = line.strip().split(',')
            # track_id, genre = values[1], values[2]
            q = "UPDATE lyrics SET genre = '%s' WHERE track_id = '%s'" % (values[0], values[1])
            # print q
            cur.execute(q)

    f.close()
    conn.commit()
    conn.close()

    print("Added genres")
