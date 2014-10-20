__author__ = 'giulio'

"""
Load

* Average bag size per genre (# of words used per song) *
SELECT genre, avg(cunt) FROM (
    SELECT track_id, genre,  count(word) as cunt from lyrics group by track_id, genre
)
GROUP BY genre

"classic pop and rock","50.880434782608695"
"classical","44.25"
"dance and electronica","47.56521739130435"
"folk","53.95652173913044"
"hip-hop","135.70652173913044"
"jazz and blues","44.84782608695652"
"metal","55.53260869565217"
"pop","47.68478260869565"
"punk","44.46739130434783"
"soul and reggae","66.23913043478261"
"""

import sqlite3

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from maintask_utils import *
import maintask_jsim as mt_js
import maintask_minhash as mt_mh


def chart_effe(conn):
    """

    :param conn:
    :return:
    """
    cur = conn.cursor()
    # get stuff
    genres = ['classic pop and rock', 'soul and reggae', 'jazz and blues', 'dance and electronica', 'hip-hop', 'punk',
              'folk', 'pop', 'metal', 'classical']
    y_values = np.array([])
    for i, g in enumerate(genres):
        q = "SELECT tid1, tid2, jacsim FROM jsim WHERE tid1 IN (SELECT track_id FROM lyrics WHERE genre='%s') " \
            "AND tid2 IN (SELECT track_id FROM lyrics WHERE genre='%s');" % (g, g)
        cur.execute(q)
        print "genre %s" % str(i + 1)
        for r in cur.fetchall():
            y_values = np.append(y_values, r[2])

    plt.xlabel("jaccard similarity")
    plt.ylabel("number of tracks")
    plt.hist(y_values, 50)
    plt.show()


def chart_three(conn, threshold):
    cur = conn.cursor()
    t_list = get_full_track_list(cur)
    t_genre = get_whole_tracks_genres(cur)
    precisions, recalls, specificities, accuracies = np.array([]), np.array([]), np.array([]), np.array([])

    for s in mt_js._TABLES_SIZES:
        t_jsimilarities = mt_js.get_whole_similarities(cur, s)
        fp, fn, tp, tn = mt_js.check_genre_matches(threshold, conn, t_list, t_genre, t_jsimilarities)
        # fill result with values for precision, recall, sensitivity, accuracy
        precisions = np.append(precisions, tp / float(tp + fp))
        recalls = np.append(recalls, tp / float(tp + fn))
        specificities = np.append(specificities, tn / float(tn + fp))
        accuracies = np.append(accuracies, (tp + tn) / float(tp + tn + fp + fn))

    print accuracies, precisions
    N = len(mt_js._TABLES_SIZES)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects_prec = ax.bar(ind, precisions, width, color='r')
    rects_rec = ax.bar(ind + width, recalls, width, color='b')
    stackd = precisions + recalls
    rects_stckd = ax.bar(ind + width * 2, stackd, width, color='g')
    # rects_spec = ax.bar(ind+width*2, specificities, width, color='g')
    #rects_acc = ax.bar(ind+width*3, accuracies, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('value')
    ax.set_title('Measures with varying bag size')
    ax.set_xticks(ind + width * 3)
    ax.set_xticklabels(tuple(str(i) for i in mt_js._TABLES_SIZES))

    ax.legend((rects_prec[0], rects_rec[0], rects_stckd[0] ), ('precision', 'recall', 'both'))

    def autolabel(rects):
        pass
        # attach some text labels
        for rect in rects:
            height = float(rect.get_height())
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '',
                    ha='center', va='bottom')

    autolabel(rects_prec)
    autolabel(rects_rec)
    autolabel(rects_stckd)
    # autolabel(rects_acc)

    plt.show()


def chart_one(conn, n_words=-1):
    """
    Plots at varying thresholds the values for precision, recall, accuracy, sensitivity
    http://en.wikipedia.org/wiki/Precision_and_recall
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    specificity or true negative rate = tn/(tn+fp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    :return:
    """
    slices = 30
    r_min = 0.001
    r_max = 0.4
    ranges = np.linspace(r_min, r_max, slices)
    precisions, recalls, specificities, accuracies = np.array([]), np.array([]), np.array([]), np.array([])

    cur = conn.cursor()
    t_list = get_full_track_list(cur)
    t_genre = get_whole_tracks_genres(cur)
    t_jsimilarities = mt_js.get_whole_similarities(cur, n_words)

    for x in ranges:
        fp, fn, tp, tn = mt_js.check_genre_matches(x, conn, t_list, t_genre, t_jsimilarities)
        # fill result with values for precision, recall, sensitivity, accuracy
        precisions = np.append(precisions, tp / float(tp + fp))
        recalls = np.append(recalls, tp / float(tp + fn))
        specificities = np.append(specificities, tn / float(tn + fp))
        accuracies = np.append(accuracies, (tp + tn) / float(tp + tn + fp + fn))

    colours = ["r-", "b-", "g-", "y-", "c-"]
    red_line = mlines.Line2D([], [], color='red', label='precision')
    blue_line = mlines.Line2D([], [], color='blue', label='recall')
    green_line = mlines.Line2D([], [], color='green', label='specificities')
    yellow_line = mlines.Line2D([], [], color='yellow', label='accuracy')
    plt.title("Measures for JaccardSimilarity")
    plt.ylabel('measure value')
    plt.xlabel('threshold value')
    # results = np.array([precisions, recalls, sensitivities, accuracies])
    a = plt.plot(ranges, precisions, colours[0], label='precision')
    b = plt.plot(ranges, recalls, colours[1], label='recall')
    c = plt.plot(ranges, specificities, colours[2], label='specificities')
    d = plt.plot(ranges, accuracies, colours[3], label='accuracy')
    plt.legend(handles=[red_line, blue_line, green_line, yellow_line])
    plt.show()


def chart_two(conn):
    """
    Plots stuff about the banding etc
    :return:
    """
    ranges = [1, 2, 3, 4, 5, 6, 7]
    labels = ["(720,1)", "(360,2)", "(240,3)", "(180,4)", "(144,5)", "(120,6)", "(90,8)"]
    precisions, recalls, specificities, accuracies = np.array([]), np.array([]), np.array([]), np.array([])
    q = ("CREATE TABLE IF NOT EXISTS bandings ("
         "params TEXT, "
         "fp TEXT, "
         "fn TEXT, "
         "tp TEXT, "
         "tn TEXT, "
         "tid2 TEXT, "
         "sim REAL "
         ");")
    cur = conn.cursor()
    cur.execute(q)
    conn.commit()
    q = "DELETE FROM bandings;"
    cur.execute(q)
    conn.commit()

    t_list = get_full_track_list(cur)
    t_genre = get_whole_tracks_genres(cur)

    for x in mt_mh.banding_all(cur):
        fp, fn, tp, tn = mt_mh.check_genre_matches(x, conn, t_list, t_genre)
        print fp, fn, tp, tn
        # fill result with values for precision, recall, sensitivity, accuracy
        precisions = np.append(precisions, tp / float(tp + fp))
        recalls = np.append(recalls, tp / float(tp + fn))
        specificities = np.append(specificities, tn / float(tn + fp))
        accuracies = np.append(accuracies, (tp + tn) / float(tp + tn + fp + fn))
        print precisions, recalls, specificities, accuracies
        query = "INSERT INTO bandings (params, fp, fn, tp, tn) VALUES ('%s', '%s', %s, %s, %s);" % (
            str(x), fp, fn, tp, tn)
        cur.execute(query)
        conn.commit()

    colours = ["r-", "b-", "g-", "y-", "c-"]
    red_line = mlines.Line2D([], [], color='red', label='precision')
    blue_line = mlines.Line2D([], [], color='blue', label='recall')
    green_line = mlines.Line2D([], [], color='green', label='specificities')
    yellow_line = mlines.Line2D([], [], color='yellow', label='accuracy')
    plt.ylabel('measure value')
    plt.xlabel('threshold value')
    # results = np.array([precisions, recalls, sensitivities, accuracies])
    a = plt.plot(ranges, precisions, colours[0], label='precision')
    b = plt.plot(ranges, recalls, colours[1], label='recall')
    c = plt.plot(ranges, specificities, colours[2], label='specificities')
    d = plt.plot(ranges, accuracies, colours[3], label='accuracy')
    plt.legend(handles=[red_line, blue_line, green_line, yellow_line])
    plt.xticks(ranges, labels)
    plt.show()


def plot_mh_sim(conn):
    """
    Plots at varying thresholds the values for precision, recall, accuracy, sensitivity
    http://en.wikipedia.org/wiki/Precision_and_recall
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    specificity or true negative rate = tn/(tn+fp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    :return:
    """
    slices = 20
    r_min = 0.01
    r_max = 0.5
    ranges = np.linspace(r_min, r_max, slices)
    precisions, recalls, specificities, accuracies = np.array([]), np.array([]), np.array([]), np.array([])

    cur = conn.cursor()
    t_list = get_full_track_list(cur)
    t_genre = get_whole_tracks_genres(cur)
    t_jsimilarities = mt_mh.get_whole_similarities(cur)

    for x in ranges:
        fp, fn, tp, tn = mt_js.check_genre_matches(x, conn, t_list, t_genre, t_jsimilarities)
        # fill result with values for precision, recall, sensitivity, accuracy
        precisions = np.append(precisions, tp / float(tp + fp))
        recalls = np.append(recalls, tp / float(tp + fn))
        specificities = np.append(specificities, tn / float(tn + fp))
        accuracies = np.append(accuracies, (tp + tn) / float(tp + tn + fp + fn))

    colours = ["r-", "b-", "g-", "y-", "c-"]
    red_line = mlines.Line2D([], [], color='red', label='precision')
    blue_line = mlines.Line2D([], [], color='blue', label='recall')
    green_line = mlines.Line2D([], [], color='green', label='specificities')
    yellow_line = mlines.Line2D([], [], color='yellow', label='accuracy')
    plt.ylabel('measure value')
    plt.xlabel('threshold value')
    # results = np.array([precisions, recalls, sensitivities, accuracies])
    a = plt.plot(ranges, precisions, colours[0], label='precision')
    b = plt.plot(ranges, recalls, colours[1], label='recall')
    c = plt.plot(ranges, specificities, colours[2], label='specificities')
    d = plt.plot(ranges, accuracies, colours[3], label='accuracy')
    plt.legend(handles=[red_line, blue_line, green_line, yellow_line])
    plt.show()


def plot_sig_histogram(conn):
    cur = conn.cursor()
    # get stuff
    genres = ['classic pop and rock', 'soul and reggae', 'jazz and blues', 'dance and electronica', 'hip-hop', 'punk',
              'folk', 'pop', 'metal', 'classical']
    y_values = np.array([])
    for i, g in enumerate(genres):
        q = "SELECT tid1, tid2, sim FROM minhashsim WHERE tid1 IN (SELECT track_id FROM lyrics WHERE genre='%s') " \
            "AND tid2 IN (SELECT track_id FROM lyrics WHERE genre='%s');" % (g, g)
        cur.execute(q)
        print "genre %s" % str(i + 1)
        for r in cur.fetchall():
            y_values = np.append(y_values, r[2])
    plt.hist(y_values, 50)
    plt.xlabel("Signature similarity")
    plt.ylabel("# of songs")
    plt.show()

def plot_all_histograms(conn):
    cur = conn.cursor()
    q = "SELECT jacsim FROM jsim;"
    cur.execute(q)
    y_values = np.array([])
    for r in cur.fetchall():
            y_values = np.append(y_values, r[0])
    plt.hist(y_values, 50)
    plt.xlabel("Signature similarity")
    plt.ylabel("# of songs")
    plt.show()

    q = "SELECT sim FROM minhashsim;"
    cur.execute(q)
    y_values = np.array([])
    for r in cur.fetchall():
            y_values = np.append(y_values, r[0])
    plt.hist(y_values, 50)
    plt.xlabel("Signature similarity")
    plt.ylabel("# of songs")
    plt.show()



if __name__ == '__main__':

    if len(sys.argv) != 2:
        print "Use this file like this\npython maintask.py <db_file>"
        exit()

    dbname = sys.argv[1]

    conn = sqlite3.connect(dbname)
    cur = conn.cursor()
    # print mt_js.compute_whole_jsim(conn)
    # print check_genre_matches(0.01, conn)
    # chart_one(conn)
    # chart_effe(conn)
    #print mt_mh.compute_whole_minhash(conn)
    #chart_two(conn)
    #plot_mh_sim(conn)
    #plot_sig_histogram(conn)
    plot_all_histograms(conn)

