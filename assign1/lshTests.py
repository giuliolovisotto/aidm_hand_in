import numpy as np

from lshToolset import *


def jsim_test():
    s1 = {1, 2, 3, 4, 5}
    s2 = {1, 2, 3, 4, 5}
    assert jsim(s1, s2) == 1
    s1 = {1, 2, 3, 4, 5, 6}
    s2 = {1, 2, 3, 4, 5}
    assert jsim(s1, s2) == 5/float(6)
    s1 = {1, 0, 3, 4, 5}
    s2 = {1, 2}
    assert jsim(s1, s2) == 1/float(6)
    s1 = {1, 2}
    s2 = {0, 5, 4, 1, 2, 4, 3}
    assert jsim(s1, s2) == 2/float(6)


def jsall_test():
    a = [{1, 2, 3}, {1, 2, 3, 4}, {5, 0, 3}]
    print jsall(a)


def sigsim_test():
    s1 = np.array([2, 3, 4, 5, 4, 2, 3, 4, 5])
    s2 = np.array([2, 3, 4, 5, 4, 2, 3, 4, 5])
    assert sigsim(s1, s2) == 1
    s1 = np.array([2, 3, 4, 5, 4, 2, 3, 4, 5])
    s2 = np.array([2, 3, 4, 6, 4, 2, 3, 4, 5])
    assert sigsim(s1, s2) == 8/float(9)
    s1 = np.array([2, 3, 4, 5, 4, 2, 3, 4, 5])
    s2 = np.array([1, 0, 0, 0, 4, 2, 1, 1, 1])
    assert sigsim(s1, s2) == 2/float(9)
    s1 = np.array([2, 3, 4, 5, 4, 2, 3, 4, 5])
    s2 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1])
    assert sigsim(s1, s2) == 0/float(9)


def sketch_test():
    # 2 arrays of 4 elements: [0.5, 0.55, 0.45, 0.1] and [0.4, 0.65, 0.05, 0.8]
    M = [[0.5, 0.4], [0.55, 0.65], [0.45, 0.05], [0.1, 0.8]]
    m_t = transpose(M)
    np.random.seed(100)
    # m_t is [[0.5, 0.55, 0.45, 0.1], [0.4, 0.65, 0.05, 0.8]]
    [res, r_vecs] = sketch(M, 3)
    # res should have 3 rows and 2 columns
    assert len(res) == 3
    assert len(res[0]) == 2
    proof = [[0 for x in range(2)] for y in range(3)]
    for i in range(len(res)):
        for j in range(len(res[0])):
            proof[i][j] = -1 if np.dot(r_vecs[i], m_t[j]) < 0 else 1

    assert proof == res


def sketch_test2():
    M = np.array([[0.8, -0.4], [0.2, -0.3]])
    print M.T[0], M.T[1]
    print cossim(M.T[0], M.T[1])
    # M = np.random.rand(3, 2)  # 2 vectors of 3 elements
    randomss = [numpy.random.normal(size=(2, 1)) for i in range(100)]
    agree = 0
    sk1, sk2 = [], []

    for i, vec in enumerate(randomss):
        #print randomss[i]
        sk1.append(numpy.dot(M.T[0], vec))
        sk2.append(numpy.dot(M.T[1], vec))
        if (sk1[i] >= 0 and sk2[i] >= 0) or (sk1[i] < 0 and sk2[i] < 0):
            agree += 1

    #for i in range(len(sk1)):
    #    print sk1[i], sk2[i]

    print agree
    print str(agree/float(len(randomss)))


def banding_test():
    signature1 = [1, 0, 1, 2, 3, 2, 1, 3, 4, 4, 6, 76, 43, 3, 5, 1, 12, 2]  # len 18
    signature2 = [1, 0, 1, 2, 3, 2, 1, 3, 4, 4, 6, 76, 43, 3, 5, 1, 12, 2]  # len 18
    signature3 = [5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 55, 55, 5, 6, 1, 12, 2]  # len 18
    signature4 = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 77, 77, 7, 7, 7, 77, 7]  # len 18
    #s1, s2, s3, s4, s5, s6, s7, s8 = [1, 0], [1, 0], []

    assert banding(signature1, 6, 3) == [[1, 0, 1], [2, 3, 2], [1, 3, 4], [4, 6, 76], [43, 3, 5], [1, 12, 2]]
    assert sigsim(signature1, signature2) == 1
    assert sigsim(signature1, signature3) == 3/float(18)
    assert sigsim(signature1, signature4) == 0
    sig_matrix = transpose([signature1, signature2, signature3, signature4])
    assert bandingsim(sig_matrix, 6, 3).all() == np.array(
        [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]]).all()


def unittest():
    """
    s1 = {"The", "lazy", "brown", "fox"}
    s2 = {"My", "awesome", "pink", "fox"}
    s3 = {"Miguel", "is", "lazy", "ass"}
    s = [s1, s2, s3]
    mh = minhash(s, 50, 42)

    v1 = [1,4,6]
    v2 = [1,5,8]
    v3 = [4,1,6]
    v4 = [4,1,6]
    print (v1,v2)
    print sigsim(v1,v2)
    print (v3,v4)
    print sigsim(v3,v4)

    print(simmat(mh[1]))

    S = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7], [9, 8, 7, 6, 5, 4]]
    print bandingsim(transpose(S), 3, 2)

    test4 = [(1),(2),(3),(4),(5),(6),(7),(8),(9)]
    perms = [1,2,3,4,5,6,7,8,9]
    for i in range(2,10):
        for k in findsubsets(perms, i):
            test4.append(k)

    m = [[1,4],[2,5],[3,6]]
    print m
    print sketch(m,1)

    a = numpy.random.randn(2, 1000)
    cs = [[0 for x in len(a)] for y in len(a)]
    for i in range(len(a)):
        for j in range(len(a)):
            cs[i][j] = cossim(a[i], a[j])
    sketches = sketch(a, 100)
    bands = []
    for j in [(2, 50), (5, 20), (10, 10), (20, 5), (50, 2)]:
        bands.append([j, banding(sketches, j[0], j[1])])
    """
    s1 = {"The", "lazy", "brown", "fox"}
    s2 = {"My", "awesome", "pink", "fox"}
    s3 = {"Miguel", "is", "lazy", "ass"}
    s4 = {"The", "lazy", "brown", "fox"}
    s = [s1, s2, s3, s4]
    #print minhashh(s, 2, 42)

    return


def exercise4():
    testbed = []
    perms = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(1, 10):
        for k in findsubsets(perms, i):
            testbed.append(set([i for i in k]))

    # print len(testbed)*(len(testbed)-1)
    colours = ["r-", "b-", "g-", "y-", "c-"]
    plt.clf()
    plt.axis([0, 1, 0, 1])

    iiindex = 0
    for jjj in [(2, 50), (5, 20), (10, 10), (20, 5), (50, 2)]:
        xs = numpy.linspace(0.0, 1.0, 100)
        ys = [(1 - (1 - x ** jjj[1]) ** jjj[0]) for x in xs]
        plt.plot(xs, ys, colours[iiindex], )
        iiindex += 1

    js = jsall(testbed)
    minhash = minhash_h(testbed, 100)[1]
    bands = [0 for x in range(5)]
    counter = 0
    slices = 7
    for jj in [(2, 50), (5, 20), (10, 10), (20, 5), (50, 2)]:

        mean = {key: 0 for key in numpy.linspace(0.2, 0.8, slices)}
        print jj, " results are:"
        band = bandingsim(minhash, jj[0], jj[1])
        for k in range(1):
            for prec in numpy.linspace(0.2, 0.8, slices):
                hits = 0
                tots = 0
                for i in range(len(js)):
                    for j in range(i + 1, len(js)):
                        if js[i][j] > prec:
                            tots += 1
                            if band[i][j] != 0:
                                hits += 1
                mean[prec] = (mean[prec] * (k) + hits / float(tots)) / (k + 1)
        print mean
        plt.plot(numpy.linspace(0.2, 0.8, slices), sorted([itm[1] for itm in mean.iteritems()]), colours[counter])
        counter += 1

    plt.show()
    return

