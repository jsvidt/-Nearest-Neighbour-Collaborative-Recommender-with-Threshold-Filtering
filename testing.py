import naboItem as ni
#import naboFact as nf
import knn
import math
import numpy as np
import scipy
from math import sqrt
from scipy.io import mmread, mmwrite, mminfo
from scipy.sparse import lil_matrix, coo_matrix
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from random import randint
import baseline

"""
Below we find the baseline predictor, 
which is used when the chosen algorithm, doesn't find any noNeiboughs. 

The Baseline that is used is baselineWithAvg,
which takes a Matrix, transposed Matrix, User, Item and the Avreage rating as parameters.
"""

def baseRating(matrix, user):
	if len(matrix.rows[user]) == 0:
		return 0
	else:
		ratings = 0 
		for x in matrix.rows[user]:
			ratings += matrix[user,x]
		return ratings/len(matrix.rows[user])

def simpelBaseline(matrix,tp, user, item):
	#tp = matrix.transpose()
	baseUser = baseRating(matrix, user)
	baseItem = baseRating(tp, item)
	return (baseUser + baseItem)/2

def baselineWithAvg(matrix,tp, user, item, avgRati):
	if avgRating(matrix,user) == 0:
		return avgRati
	else:
		deltaRati = avgRati/avgRating(matrix,user)
		#print deltaRati 
		baseUser = baseRating(matrix, user) * deltaRati
		baseItem = baseRating(tp, item)
		return (baseUser + baseItem)/2

def baselineFun(matrix,tp, user, item, avgRati):
	baseUser = baseRating(matrix, user) - avgRati
	baseItem = baseRating(tp, item) - avgRati
	return baseUser + baseItem + avgRati

def baselineParam(matrix,tp, user, item, avgRati):
	baseUser = baseRating(matrix, user)
	baseItem = baseRating(tp, item)
	return (baseUser*5 + baseItem * 2)/7
	

def randomBase():
	return randint(1,5)

"""
Here we have the different validation methods, 
Where validationMean is the RMSE (could have chosen a better name)
and the MAE is the MAE.
"""

# def RMSE(R,guess):
# 	return sqrt(((R-guess)**2).mean())

def validationMean(R,guess):
	return sqrt(mean_squared_error(R,guess))

def MAE(R,guess):
	fejl = np.mean(np.sqrt((R - guess)**2))
	#fejl /= len(R)
	return fejl

"""
Here we have the kfold Cross Validation for the kNN algorothm. 

The algorithm takes an array of folds of the format mtx, a k which is the number of folds
and C which is either the Threshold or the k neiboughs, depending on which algorithm is chosen.
"""

def kCrossValidation(matrices, k=5,C=100, dataset='mLens'):
	#fnamePre, fnameRati, fnameNN = "predredictions.{i}.txt", "ratings.{i}.txt", "rest.{i}.txt"	
	val, nn = [0]*k, [0]*k
	snit = [0]*3
	mae = 0
	for x in xrange(k):
		testMatrix = matrices[(0+x)%5]
		traningMatrix = matrices[(1+x)%5] + matrices[(2+x)%5] + matrices[(3+x)%5] + matrices[(4+x)%5]
		testMatrix = testMatrix.tolil()
		traningMatrix = traningMatrix.tolil()
		testMatrix = testMatrix.transpose()
		traningMatrix = traningMatrix.transpose()

		pred, rati, nn[x], cnsa, pre, val[x], nab, nei = testing(testMatrix, traningMatrix, C)

		arr = np.array([nn[x], cnsa, pre, val[x], nab, nei])
		np.savetxt(dataset+'/TH'+str(C)+'/data/fnamePre{}'.format(x),pred , fmt='%.2f')
		np.savetxt(dataset+'/TH'+str(C)+'/data/fnameRati{}'.format(x),rati , fmt='%.2f')
		np.savetxt(dataset+'/TH'+str(C)+'/data/fnameNN{}'.format(x),arr, fmt='%.7f')
		#np.savetxt(dataset+'/TH'+str(C)+'/data/nabo{}'.format(x),nab, fmt='%.7f')
		#np.savetxt(dataset+'/TH'+str(C)+'/data/nei{}'.format(x),nei, fmt='%.7f')
		mae += MAE(pred, rati)
	mae = mae/k
	print 'RMSE:', sum(val)/k, 'MAE:', mae, 'NN:', sum(nn)/k
	snit[0], snit[1], snit[2] = sum(val)/k, mae , sum(nn)/k
	np.savetxt(dataset+'/TH'+str(C)+'/snit.txt', snit, fmt='%.7f')

"""
This function reads the 5 folds and returns an array of the folds. 
"""

def readData(k=5,C=100, dataset='mLens'):
	matrices = [coo_matrix([])] * k
	for x in xrange(k):
		fil = mmread(dataset+"/A"+str(x)+".mtx")
		matrices[x] = fil
	return matrices

def avgRating(matrix, user):
	usersRatings = matrix.getrow(user).data[0]
	#print usersRatings
	if len(usersRatings) == 0:
		return 0
	else: 
		return sum(usersRatings)/len(usersRatings)

def avgUserRating(matrix):
	rating = 0
	numOfUsersWithRatings = matrix.shape[0]
	for user in xrange(numOfUsersWithRatings):
		userAvgRating = avgRating(matrix, user)
		#print 'U', user, 'R', userAvgRating
		if userAvgRating == 0:
			numOfUsersWithRatings -= 1
		else:
			rating += userAvgRating
	return rating/numOfUsersWithRatings


"""
This function tests a fold, given a testMatrix and a traningMatrix
"""

def testing(testMatrix, traningMatrix, TH):
	ratings, predictions = np.array([]), np.array([])
	coundNotSayAnything, noNeiboughs, pre = 0, 0, 0
	row, col, data = np.array([]), np.array([]), np.array([])
	tp = traningMatrix.transpose()
	tm = testMatrix.transpose()
	avgURati = avgUserRating(traningMatrix)
	naboer, neighborList = np.array([], dtype=object), np.array([], dtype=object)
	nab, nei = np.array([]), np.array([])
	UB,IB,mean = baseline.baselineReg(tp)
	for user in xrange(testMatrix.shape[0]):#testMatrix.shape[0]
		User = testMatrix.getrow(user)
		for item in xrange(len(testMatrix.rows[user])):
			Item = testMatrix.rows[user][item]
			
			#This line i for prediction with threshold filtering. 
			prediction, nab, nei = ni.predictionWithThresholdFiltering(User,Item,traningMatrix,TH)

			#naboer = np.concatenate((naboer, [nab]))
			#neighborList = np.concatenate((neighborList, [nei]))
			#print prediction, naboer, neighborList
			#prediction
			
			#Thise 3 lines is for knn prediction.
			#basePre = baseline.predict(Item,user,UB,IB,mean) 
			#i = traningMatrix.getcol(Item)
			#prediction = knn.ItemKnn(User,Item, basePre, traningMatrix,TH, user, i)
			
			rating = testMatrix[user,Item]
			if prediction > 0:
				predictions = np.append(predictions,[prediction])
				ratings = np.append(ratings,[rating])
				pre += 1
				row = np.append(row,[user])
				col = np.append(col,[Item])
				data = np.append(data,[testMatrix[user,Item]])
			if prediction < 0:
				noNeiboughs += 1
				print user, Item
				basePre = baseline.predict(Item,user,UB,IB,mean)
				predictions = np.append(predictions,[basePre])
				ratings = np.append(ratings,[rating])
			if prediction == 0:
				coundNotSayAnything += 1
				counPre = baseline.predict(Item,User,UB,IB,mean)
				predictions = np.append(predictions,[counPre])
				ratings = np.append(ratings,[rating])

			if user % 1 ==0:
				print 'User:', user, 'Item:', Item
				print 'P:', pre
				print 'NN:', noNeiboughs
				print 'CNSA:', coundNotSayAnything
				print 'RMSE:', validationMean(ratings,predictions)
				print 'MAE:' , MAE(ratings,predictions)
				print 'prediction', prediction#, prediction1
				print 'Real', rating
				#print '# Naboer:', naboer[user+item], 'Naboer of', neighborList[user+item]
				print (pre + noNeiboughs + coundNotSayAnything), 'of', testMatrix.nnz, 'done!'
				print '\n'
	predAt = coo_matrix((data, (row,col)), shape=(testMatrix.shape[0], testMatrix.shape[1]))
	return predictions, ratings, noNeiboughs, coundNotSayAnything, pre, validationMean(ratings,predictions), len(naboer), len(neighborList)	


def go():
	dataset = 'mLens'
	th = 0.2
	matri = readData(C=th, dataset=dataset)
	kCrossValidation(matri,C=th, dataset=dataset)


go()

