
# coding: utf-8

# In[1]:

import numpy as np

import lshToolset as ls

print "====================================TASK 1============================================="
s1 = {"The", "quick", "brown", "fox"}
s2 = {"Jumped", "over", "the", "dog"}
s3 = {"Over", "the", "lazy", "dog"}
s4 = {"Over", "the", "lazy", "dog"}
print 'Initialize some sets'
print "s1: " + str(s1)
print "s2: " + str(s2)
print "s3: " + str(s3)
print "s4: " + str(s4)
print "Use jsim() to find similarity between sets"
print "jsim(s1,s2): " + str(ls.jsim(s1, s2))
print "jsim(s1,s3): " + str(ls.jsim(s1, s3))
print "jsim(s3,s4): " + str(ls.jsim(s3, s4))


# In[2]:

print "========================================TASK 2========================================"
print ("We defined 2 functions: \n1) minhash_p() calculates the signatures matrix with the permutations, \n2) "
       "minhash_h() calculates the matrix using hash functions \nBoth of them takes as input 2 arguments: \n  "
       "S -- the list of sets \n  k -- number of permutations/hash functions;\n"
       "(minhash_p also takes as input the seed for the random package)\n"
       "\nand they return in output a dictionary and the signatures matrix [Words, Signatures]\n")
print "Use sets from previous task to create the list"
sets = np.array([s1, s2, s3, s4])
print sets
print "\nNow compute minhash_p(sets, 8, 123) (signatures are in the columns)"
[words, signatures] = ls.minhash_p(sets, 8, 123)
print "Words: " + str(words)
print "Signatures: " + str(signatures)

print "\nNow compute minhash_h(sets, 8) (signatures are in the columns)"
[words, signatures] = ls.minhash_h(sets, 8)
print "Words: " + str(words)
print "Signatures: " + str(signatures)

print "\nExperiment is repeatable if we use permutations so we can see that\nminhash_p(sets, 8, 123) == minhash_p(sets, 8, 123) returns True"
print ls.minhash_p(sets, 8, 123)[1] == ls.minhash_p(sets, 8, 123)[1]



# In[3]:

print "========================================TASK 3========================================"
print ("sigsim(ss1,ss2) calculates the similarity between 2 signatures")
ss1 = np.array([0, 1, 4, 6, 3])
ss2 = np.array([0, 1, 2, 4, 9])
ss3 = np.array([5, 6, 3, 7, 8])
print ss1, ss2, ss3
print "sigsim(ss1,ss2) is " + str(ls.sigsim(ss1, ss2))
print "\nsimmat(M) calculates the full similarity matrix for the signatures"
mt = np.array([ss1, ss2, ss3]).T
# mt = ls.transpose([ss1, ss2, ss3])
print "simmat(mt) is"
print ls.simmat(mt)


# In[4]:

print "========================================TASK 4========================================"

sets = np.array([s1, s2, s3, s4])
print sets
[words, signatures] = ls.minhash_h(sets, 8)
print ("The function bandingsim implements the banding technique\n"
       "The result of the banding applied to the signature matrix is")
bsim = ls.bandingsim(signatures, 2, 4)
print bsim
print ("which contains ones where the 2 signatures were identical in at least one band")


print "\n\n"

import matplotlib.pyplot as plt
# generate all non empty subsets and calculate jaccard's similarity
testbed = np.array([])
perms = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
for i in range(1, 11):
    for k in ls.findsubsets(perms, i):
        testbed = np.append(testbed, set([i for i in k]))

# matplotlib stuff
colours = ["r-", "b-", "g-", "y-", "c-"]
plt.clf()
plt.axis([0, 1, 0, 1])
# draw theoretical s-curves
iiindex = 0
for jjj in [(2, 50), (5, 20), (10, 10), (20, 5), (50, 2)]:
    xs = np.linspace(0.0, 1.0, 100)
    ys = [(1-(1-x**jjj[1])**jjj[0]) for x in xs]
    plt.plot(xs, ys, colours[iiindex],)
    iiindex += 1
# compute jaccard similarity
js = ls.jsall(testbed)
# compute minhash
minhash = ls.minhash_h(testbed, 100)[1]
counter = 0
slices = 7
print "Computing empirical chance of being detected.. "
for jj in [(2, 50), (5, 20), (10, 10), (20, 5), (50, 2)]:

    mean = {round(key, 2): 0 for key in np.linspace(0.2, 0.8, slices)}
    print jj, " results are:"
    band = ls.bandingsim(minhash, jj[0], jj[1])
    for k in range(1):  # repeat 10 times and average results
        for prec in np.linspace(0.2, 0.8, slices):
            prec = round(prec, 2)
            hits = 0
            tots, fn, fp = 0, 0, 0
            for i in range(len(js)):
                for j in range(i + 1, len(js)):
                    if ls.falls_in_range(prec, 0.02, js[i][j]) and band[i][j] != 0:
                        tots += 1
                        hits += 1
                    elif ls.falls_in_range(prec, 0.02, js[i][j]) and band[i][j] == 0:
                        tots += 1
                        fn += 1
                    elif not ls.falls_in_range(prec, 0.02, js[i][j]) and band[i][j] != 0:
                        fp += 1
                        # hits += 1
            mean[prec] = (mean[prec]*k + hits/float(1 if tots == 0 else tots))/(k+1)
    print "hits: %s, fn: %s, fp: %s" % (hits, fn, fp)
    print mean
    # plot the empirical curves
    plt.plot(np.linspace(0.2, 0.8, slices), sorted([itm[1] for itm in mean.iteritems()]), colours[counter])
    counter += 1

plt.show()

