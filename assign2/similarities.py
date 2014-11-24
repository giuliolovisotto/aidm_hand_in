__author__ = 'giulio'


import numpy as np
import scipy.stats.stats as sss
import time
import scipy.sparse as ss


def fastpearsoncorr(U):
    U = U.T
    U = ss.csr_matrix(U)
    U = np.asarray(U.todense(),dtype=float)
    U[np.where(U == 0)] = np.NaN

    V = np.nanmean(U,axis=0)

    U[np.where(np.isnan(U))] = 0

    U.astype(int)

    A = np.asarray(np.tile(V,(U.shape[0],1)))
    A[np.where(U == 0)] = 0

    U = ss.csr_matrix(U)
    A = ss.csr_matrix(A)

    U = np.abs(U-A);

    Utransp = U.T
    DotProd = Utransp.dot(U)

    IND = U.T > 0
    Usqr = U.multiply(U)
    M=IND*(Usqr)

    ProdM = M.multiply(M.T)
    NormProd = ProdM.sqrt()

    return np.array(DotProd/NormProd)



def full_cosine_sim(a):
    a = a.T
    a = ss.csr_matrix(a)
    a_t = a.T
    dot_prod = a_t.dot(a)
    ind = a.T > 0
    a_sqrd = a.multiply(a)
    m = ind * a_sqrd
    prod_m = m.multiply(m.T)
    norm = prod_m.sqrt()
    cossim = dot_prod / norm
    return np.array(cossim)


def full_pearson_sim(a):
    mat = np.zeros((a.shape[0], a.shape[0]))
    for i, el in enumerate(a):
        start = time.time()
        for j in range(i, len(a)):
            t = np.vstack((a[i], a[j])).T
            # WHAT THE HELL IS THIS I DONT EVEN
            t = t[~(t == 0).any(1), :].T  # U WOT M8
            mat[i, j] = np.corrcoef(t[0], t[1])[0, 1]
        print i, str(time.time() - start)
    mat = np.triu(mat).T + mat
    np.fill_diagonal(mat, 0)
    return mat


def psim(mat):
    mat = mat.astype(float)
    mat[np.where(mat == 0)] = np.nan
    means = np.nanmean(mat, axis=1)
    mat[np.isnan(mat)] = 0
    mat.astype(int)
    print means
    b = np.tile(means, (mat.shape[1], 1))
    ind = mat > 0
    c = ind * (mat - b.T)

    return full_cosine_sim(c)
    '''
    print c
    num = np.dot(c, c.T)  # users x usrs matrix numerators
    c2 = c ** 2
    sm = np.sum(c2, axis=1)
    d = np.tile(sm, (mat.shape[1], 1)).T
    d = np.dot(d, d.T)/mat.shape[1]
    d = np.sqrt(d)
    return np.divide(num, d)'''


# ITS NOT COMPLETE
def m_pearson(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = n*(np.add.reduce(xm*ym))
    #print r_num/float(n)
    r_den = n*np.sqrt(sss.ss(xm)*sss.ss(ym))
    r = (r_num / r_den)

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)
    df = n-2
    if abs(r) == 1.0:
        prob = 0.0
    else:
        t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
        prob = sss.betai(0.5*df, 0.5, df / (df + t_squared))
    return r, prob