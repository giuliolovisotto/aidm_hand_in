#Assignment 2 - Recommender Systems#
Contribution. In this paper, we use the MovieLens 1M dataset to benchmark different recommender system algorithms. For every algorithm, we measure the execution time, the accuracy, and the memory requirements. Moreover, we give some insights on the bounds on the required time and memory of the implemented algorithms in general. We analyze how the time and memory requirements changes as a function of the size of the dataset: either users, movies, or ratings.

#Structure#
In this repository you find the following files:  
1.  algorithms.py -- the file with the algorithms implementations 
2.  cleanup.py -- scripts used to normalize data
3.  compute_error.py -- contains functions to compute errors (rmse, mae)
4.  fold_validation.py -- implements one of the k steps of the fold validation
5.  load_data.py -- utilities to load datas 
6.  main.py — contains the main 
7.  shrinkup.py -- randomly sample a fraction of the dataset, used to test correctness fast 
8.  similarities.py -- contains different similarity measures
9.  split_matrix — used for ALS algorithm parallelisation


#Notes#
We didn't include datasets (neither the original version, nor the normalized one) in this repository because they are too large and since they are available online. 
Look in the report references to find them.