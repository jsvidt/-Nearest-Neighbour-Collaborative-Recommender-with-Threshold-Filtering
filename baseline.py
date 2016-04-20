from fact import load_movielens_kfold
import numpy as np


def calc_user_biases(UB,IB,R,mean,reg):
    UB[:] = 0
    for i in xrange(R.shape[0]):
        for j in xrange(R.shape[1]):
            if R[i,j]:
                UB[i] += R[i,j] - mean - IB[j]
    print UB.shape,(R > 0).sum(axis=1).flatten().shape
    UB /= reg + (R > 0).sum(axis=1).flatten()

def calc_item_biases(UB,IB,R,mean,reg):
    IB[:] = 0
    for i in xrange(R.shape[0]):
        for j in xrange(R.shape[1]):
            if R[i,j]:
                IB[j] += R[i,j] - mean - UB[i]

    IB /= reg + (R > 0).sum(axis=0).flatten()

def calc_base_line(R,reg_u=5,reg_i=2,iterations=10):
    UB = np.zeros(R.shape[0])
    IB = np.zeros(R.shape[1])
    mean = R[R > 0].mean()
    for i in xrange(iterations):
        calc_user_biases(UB,IB,R,mean,reg_u)
        calc_item_biases(UB,IB,R,mean,reg_i)
    return UB,IB,mean


def predict(i,j,UB,IB,mean):
    return mean + UB[i] + IB[j]

def validate(V,UB,IB,mean):
    sum = 0
    count = 0
    for i in xrange(V.shape[0]):
        for j in xrange(V.shape[1]):
            if V[i,j]:
                guess = predict(i,j,UB,IB,mean)
                val = V[i,j]
                print guess,val
                sum += (guess - val) ** 2
                count += 1
    return np.sqrt(sum / count)

def go():
    folds = load_movielens_kfold()
    rmses = []
    for i in xrange(len(folds)):
        R,V = folds[i]
        UB,IB,mean = calc_base_line(R.toarray())
        rmses.append(validate(V.toarray(),UB,IB,mean))

    print "avg rmse", sum(rmses)/len(rmses)

#folds = load_movielens_kfold()
#R,V = folds[0]

def baselineReg(R):
    return calc_base_line(R.toarray())

#print baselineReg(R,1,1)

#go()




