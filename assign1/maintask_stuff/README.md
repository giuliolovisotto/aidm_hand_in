#SOURCES#
http://labrosa.ee.columbia.edu/millionsong/blog/11-2-28-deriving-genre-dataset
http://labrosa.ee.columbia.edu/millionsong/musixmatch

#DESCRIPTION#
I have created the database using the python script provided by the authors of the dataset which is
mxm_dataset_to_db see the file for more info.
This creates a db with both the train and the test records.
Now use another script to fill the genres using the information contained in the file
"msd_genre_dataset.txt". And at the end call the cleanup script to remove all the records
that have no genre in the db, since they are of no interest.


mxm_dataset_to_db -- takes 3 arguments:  
1. dataset train file  
2. dataset test file  
3. output sqlite file  
Takes the records contained in 1 & 2 and builds the .db file in 3  

fill_genres -- takes 2 arguments:
1. dataset genre file  
2. output sqlite file  
Iterates over the genre file and fills the genre field in the sqlite db  

cleanup.py -- does the following:  
a. removes all non classified tracks  
b. removes all the stop words  
c. eventually removes some of the songs for a speedup in the computation  

tfidf.py -- takes 1 argument:  
1. input sqlite file  
For every record in the database, fills the tfidf field  
    

The commands to generate and clean the db are (in this order):  
1) python mxm_dataset_to_db.py mxm_dataset_train.txt mxm_dataset_test.txt file.db  
2) python fill_genres.py msd_genre_dataset.txt file.db  
3) python cleanup.py file.db  
4) python tfidf.py file.db  

