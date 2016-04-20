
import baseline
import math 
import scipy


#https://github.com/zenogantner/MyMediaLite/blob/master/src/MyMediaLite/Correlation/Pearson.cs

def Pearson(ratings, item1, item2):
	#if (item1 == item2):
	#	return 1
	#test det her en gang
	#print item1.rows[0]
	#print item2.rows[0]    
	e = list(set(item1.rows[0]).intersection(item2.rows[0]))
	#print set(item1.rows[0]).intersection(item2.rows[0])
	n = len(e)
	#print 'POUL:', n
	if (n < 2):
		return 0
		
	#single pass variant
	
	i_sum = 0
	j_sum = 0
	ij_sum = 0
	ii_sum = 0
	jj_sum = 0
	
	#foreach TODO
	for other_entity in e:
		
		#get ratings from ratings.
		#we only consider itembased
		r1 = item1[0, other_entity] #for item1 from user "other_entity"
		r2 = item2[0, other_entity] #For item2 from user "other_entity"
		#print 'r1', r1, 'r2', r2
		i_sum += r1
		j_sum += r2
		ij_sum += r1 * r2
		ii_sum += r1 * r1
		jj_sum += r2 * r2
		
	return ComputeCorrelation(i_sum, j_sum, ii_sum, jj_sum, ij_sum, n)
	
# ComputeCorrelation + GetDenominator + GetNumerator in one
def ComputeCorrelation (i_sum, j_sum, ii_sum, jj_sum, ij_sum, n):
	denominator = math.sqrt((n * ii_sum - i_sum * i_sum) * (n * jj_sum - j_sum * j_sum))
	#print denominator
	if (denominator == 0):
		return 0.0
		
	pmcc = (n * ij_sum - i_sum * j_sum) / denominator
	top = float(n - 1)
	bot = float(n - 1 + 2500)
	sig = top / bot
	#print 'sig',sig
	result = float(pmcc * sig)
	#print 'res', result
	#print 'top', top, 'bot', bot
	return result
	
	
def ItemKnn(user, item, base, matrix, k, User, Item):
	print User, item
	result = base
	MinRating = 1
	MaxRating = 5
	
	NeighborList = matrix.getcol(item).nonzero()[0]
	#print 'NeighborList:', NeighborList
	numberOfUsers = matrix.shape[0]
	numberOfUsersWithItem = len(NeighborList)
	listOfNeighbor = {}
	
	for neighbor in NeighborList:
		#print neighbor, matrix.shape, len(NeighborList)
		commonRatedItems = commondRated(user,matrix[neighbor])
		if len(commonRatedItems) > 1:
			#print matrix[neighbor].nnz, matrix[item].nnz, matrix[neighbor].rows[0], matrix[item].rows[0]
			pc=Pearson(matrix, matrix[neighbor], matrix[item])
			if pc > 0:
				listOfNeighbor[neighbor] = pc
	sortedListOfNeighbor = sorted(listOfNeighbor.items(), key=lambda x: x[1],reverse=True)
	#print 'pcs', listOfNeighbor
	#print 'sortedpcs', sortedListOfNeighbor
	Sum = 0
	weight_sum = 0
	x = 0
	for i, value in sortedListOfNeighbor:
		#print x, i, value
		#print len(sortedListOfNeighbor)
		weight = value#sortedListOfNeighbor[i][1]
		#print 'weight', weight
		rating = matrix[i, item]
		#print 'pos', i, item
		#print 'rating', rating
		weight_sum += weight
		Sum += weight * (rating - base)
		if x == 39:
			break
		x += 1

	if (weight_sum != 0):
		result += float(Sum/weight_sum)
		
	if(result > MaxRating):
		result = MaxRating
	
	if(result < MinRating):
		result = MinRating

	return result

def commondRated(user, neighbor):
	return list(set(user.rows[0]).intersection(neighbor.rows[0]))






"""
R = [ 
	[0, 1, 5, 5 ],
	[1, 0, 5, 1 ],
	[0, 2, 4, 3 ],
	[1, 1, 1, 0 ],
	]
	
"""