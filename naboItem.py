import math
import scipy
#import time
import numpy as np

def averageRating(item):
	return (float(item.sum())/float(item.getnnz()))+0.01


def commonRated(item, neighborItem):
	return list(set(item.rows[0]).intersection(neighborItem.rows[0]))

 #good rate is 50
def significanceMin(commonItems, rate = 50):
	return (min(float(commonItems), float(rate)))/float(rate)


 #good rate is 100
def significanceShrink(commonItems, rate = 100):
	return commonItems/(commonItems + rate)
 
def PC(item,neighborItem, commonRatedUsers):
	top = 0
	bottom1 = 0
	bottom2 = 0
	averageItem = averageRating(item)
	averageNeighbor = averageRating(neighborItem)

	for rated in commonRatedUsers:
		top1 = (item[0,rated]-averageItem)
		top2 = (neighborItem[0,rated]-averageNeighbor)
		top += top1*top2
		bottom1 += pow(top1, 2)
		bottom2 += pow(top2, 2)
	try:
		return top/math.sqrt(bottom1 * bottom2) * significanceMin(len(commonRatedUsers))
	except ZeroDivisionError:
		return 0


def predictionWithThresholdFiltering(item, rating, matrix, threshold = 0.1):
	#print 'hej'
	top = 0
	bot = 0  
	currentNeighbor = 0
	naboer = np.array([])
	neighborList = matrix.getcol(rating).nonzero()[0]
	#print 'Naboer:', neighborList
	#print 'hej'
	for neighbor in neighborList:
		#print 'hej'
		#time1s = time.time()
		commonRatedUsers = commonRated(item, matrix[neighbor])
		#time1e = time.time()
		#print time1e - time1s
		if len(commonRatedUsers) > 0:
			pc = PC(item, matrix[neighbor], commonRatedUsers)
			if pc > threshold:
				pc = pc # * significanceMin(len(commonRatedUsers))
				top += pc * matrix[neighbor, rating]
				bot += abs(pc)
				currentNeighbor += 1
				naboer = np.append(naboer,[neighbor])
				#print 'hej'
				try:
					d = top/bot
				except ZeroDivisionError:
					d = 0
				#print "Neighbor ", currentNeighbor, "rating: ", matrix[neighbor, rating], "pc: ", pc, "Top: ", top, "Bot: ", bot, "current: ", d
	#print 'Naboer:', naboer
	if currentNeighbor == 0:
		return -1, len(naboer), len(neighborList)
	try:
		#print 'hej	'
		return top / bot, len(naboer), len(neighborList)
	except ZeroDivisionError:
		return 0





