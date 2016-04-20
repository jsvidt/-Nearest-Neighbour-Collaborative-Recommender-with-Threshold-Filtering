# Nearest Neighbour Collaborative Recommender with Threshold Filtering
This is a repository of the code used in a study of 'Nearest Neighbour Collaborative Recommender with Threshold Filtering' the implementation is done in python.

## Datasets ##
The testing have mainly been done on two data sets. Movielens 100k and Amazon music instruments. 

## Testing ##
If you want to try the code there are a couple of different options described below. 

### Matrix Factorization ###
To run matrix factorization with the same configuration and folds, as in the paper, on the movielens100k data set run:

```
python -i fact.py
```

and in the interactive python console call the function 

```
kFold()
```

To run matrix factorization on the amazon music instruments data set

```
python -i fact.py
kFoldMinstru()
```


### Nearest Neighbour ###
To run the nearest neighbour with threshold filtering. All you need to do is run the go() function in testing.py 

The current configuration of the method is a threshold of 0.2, significance minimum of 50, and it's being run on the movielens 100k data set.

If you want to change the threshold or the dataset you can do so at the end of the testing.py file. You will also need to create a new folder in the dataset folder. For a threshold of 0.2 you need to create the folder mLens/TH0.2 aswell as the folder mLens/TH0.2/data

If you want to change the significance minimum it can be done in the naboItem.py file.
