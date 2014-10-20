
# coding: utf-8

# In[1]:

import lshToolset as ls
import numpy as np
print "====================================TASK 5============================================="
print "Generate 2 vectors of doubles"
v1, v2 = np.random.randn(5), np.random.randn(5)
print v1
print v2
print "The cosine similarity is: " + str(ls.cossim(v1, v2))


# In[2]:

print "====================================TASK 6============================================="
M = np.array([v1, v2])
M = M.T  # transpose the matrix since sketch wants the vectors on the columns
[sk, r_vecs] = ls.sketch(M,3)
print "Sketches for matrix composed by [v1, v2], sketch(M,3)"
print sk


# In[ ]:

print "====================================TASK 7============================================="
import matplotlib.pyplot as plt

print "Computing empirical chance of being detected.. "

testbed = np.random.randn(2, 1000)
cos_sims = ls.cossim_all(testbed)
[sketches, r_vecs] = ls.sketch(testbed, 100)

# matplotlib stuff
colours = ["r-", "b-", "g-", "y-", "c-"]
plt.clf()
plt.axis([0, 1, 0, 1])
# draw theoretical s-curves
iiindex=0
for jjj in [(2, 50), (5, 20), (10, 10), (20, 5), (50, 2)]:
    xs = np.linspace(0.0, 1.0, 100)
    ys = [(1-(1-x**jjj[1])**jjj[0]) for x in xs]
    plt.plot(xs, ys, colours[iiindex],)
    iiindex += 1

counter = 0
slices = 7

plot_ranges = np.linspace(0.2, 0.8, slices)

for jj in [(2, 50), (5, 20), (10, 10), (20, 5), (50, 2)]:
    mean = {round(key, 2): 0 for key in plot_ranges}
    print jj, " results are:"
    band = ls.bandingsim(sketches, jj[0], jj[1])
    for k in range(10):  # repeat 10 times and average results
        for prec in plot_ranges:
            prec = round(prec, 2)
            hits = 0
            tots = 0
            fn = 0
            fp = 0
            for i in range(len(cos_sims)):
                for j in range(i+1, len(cos_sims)):
                    if ls.falls_in_range(prec, 0.05, cos_sims[i][j]) and band[i][j] != 0:
                        tots += 1
                        hits += 1
                    elif ls.falls_in_range(prec, 0.05, cos_sims[i][j]) and band[i][j] == 0:
                        tots += 1
                        fn += 1
                    elif not ls.falls_in_range(prec, 0.05, cos_sims[i][j]) and band[i][j] != 0:
                        fp += 1
                        # tots += 1
            mean[prec] = (mean[prec]*k + hits/float(1 if tots == 0 else tots))/(k+1)
    print "hits: %s, fn: %s, fp: %s" % (hits, fn, fp)
    print mean
    # plot the empirical curves
    plt.plot(plot_ranges, sorted([itm[1] for itm in mean.iteritems()]), colours[counter])
    counter += 1

plt.show()


# In[ ]:



