try:
	import numpy as np
	import numpy
except ImportError:
	import numpypy as np
	import numpypy as numpy
import math
import sys
import numexpr as ne
from sklearn.utils import shuffle

# try:
# 	import matplotlib.pyplot as plt
# except ImportError:
# 	pass

from scipy import sparse

def err(r,beta,betaO,P,Q,bP,bQ):
    return np.mean(r ** 2 + (beta * (np.sum(P**2,axis=1))[:,np.newaxis]) + (beta * np.sum(Q**2,axis=0)) + betaO * (bP[:,np.newaxis] ** 2 + bQ ** 2))
    #return math.sqrt(ne.evaluate("sum(r ** 2)") / r.size)


def lineSearch(startErr,fun,beta,betaO,P,Q,bP,bQ):
    th = .05
    alpha = alphaEst = 1
    optit = 0
    
    lastChng = -1 * float("inf")
    lastErr = 0
    while True:
        alphaEst *= th 
        r = fun(alphaEst)
        #r[(R == 0)] = 0
        newErr = err(r,beta,betaO,P,Q,bP,bQ)
        chng = startErr - newErr
        #armijo = fejlSum + alphaPEst * dGP / rP.size
        print "optit",optit,alpha,alphaEst,chng,startErr

        if chng <= lastChng or (chng > 0 and 0.99 < lastChng / chng < 1): #and lastErr <= armijo:
            print "breaking",chng,lastChng
            break
        
        
        lastChng = chng
        lastErr = newErr
        optit += 1
        alpha = alphaEst
    
    return alpha




def tanh(r):
    #e2x = np.exp(2 * r)
    #return (e2x + 1) / (e2x - 1)
    return np.tanh(r)

def deriv_tanh(P,Q,bP,bQ,u):
    return 1 - tanh(((P).dot(Q) + bP[:,np.newaxis] + bQ))

def sigmoid(r):
    return 1/(1 + np.exp(-1 * r))

def deriv_sigmoid(P,Q,bP,bQ,u):
    #print "NUMPY MAX",np.max(P.dot(Q)),np.min(P.dot(Q))
    #er = np.exp(-1 * (u + (P).dot(Q) + bP[:,np.newaxis] + bQ))
    #return er/((1+er)**2)
    s = sigmoid((P).dot(Q) + bP[:,np.newaxis] + bQ)
    return s * (1 - s)

def linear(r):
    return r

def deriv_linear(P,Q,bP,bQ,u):
    return 1

def sigresidual(R,P,Q,bP,bQ,discard=None,beta=0,betaO=0,trunc=False,u=0,activation=sigmoid):
    #prod = np.dot(P,Q)
    #u = R[R > 0].mean() #doesn't work: R.sum() / np.count_nonzero(R)
    print R.shape,u,bP.shape,bQ.shape
    r = R - activation(u + np.dot(P,Q) + bP[:,np.newaxis] + bQ) #ne.evaluate("R - prod") #- bP[:,np.newaxis] - bQ - 
    #if beta or betaO:
    #    r += beta * np.dot(P**2,Q**2) + betaO * (bP[:,np.newaxis] ** 2 + bQ ** 2)
    #    r += beta * ((np.sum(P**2,axis=1))[:,np.newaxis] + np.sum(Q**2,axis=0)) + betaO * (bP[:,np.newaxis] ** 2 + bQ ** 2)

    if discard != None:
        r[R == discard] = 0

    if trunc:
        r[r > 5.0] = 5.0
        r[r < 1] = 1.0
    return r


def calc_beta(new_rA,rA):
    #return np.multiply(new_rA,new_rA - rA).sum(axis=0) / np.multiply(rA,rA).sum(axis=0)
    new_rA_flat = new_rA.flatten()
    rA_flat = rA.flatten()
    return max(np.dot(new_rA_flat.T,new_rA_flat - rA_flat) / np.dot(rA_flat.T,rA_flat),0)

def sigFactO(R,P,Q,K,steps=500,discard=None,beta=None,bP=None,bQ=None,betaO=None,doBias=True,mean_center=False,activation=sigmoid,deriv=deriv_sigmoid,validate=None,every=1,doFactors=True,biasSteps=10):
    if beta == None:
        beta = 0
    if betaO == None:
        betaO = 0

    print "Starting sigFactO, K:{},steps:{},beta:{},betaO:{}".format(K,steps,beta,betaO)

    mu = np.finfo(float).eps

    if not doFactors:
        P = np.zeros((R.shape[0],K))
        Q = np.zeros((K,R.shape[1]))

    if doBias:
        if bP == None:
            bP = normalizer(np.random.rand(R.shape[0]),.1,0)
        else:
            print "got bP"
        if bQ == None:
            bQ = normalizer(np.random.rand(R.shape[1]),.1,0)
        else:
            print "got bQ"
    else:
        bP = np.zeros(R.shape[0])
        bQ = np.zeros(R.shape[1])

    u = R[R > 0].mean() #doesn't work: R.sum() / np.count_nonzero(R)

    if mean_center:
        mean_indicies = R > 0
        R[mean_indicies] -= u


    r = sigresidual(R,P,Q,bP,bQ,discard,beta,u=u)
    sigd = deriv(P,Q,bP,bQ,u)
    
    rP = np.dot(np.multiply(r,sigd),Q.T) - beta*P
    dP = rP.copy()
    
    rQ = np.dot(np.multiply(r,sigd).T,P).T - beta*Q
    dQ = rQ.copy()


    rbP = (np.multiply(r,sigd) - betaO * bP[:,np.newaxis]).sum(axis=1)
    #rbP = r.sum(axis=1) - betaO * bP
    dbP = rbP.copy()

    rbQ = (np.multiply(r,sigd) - betaO * bQ).sum(axis=0)
    #rbQ = r.sum(axis=0) - betaO * bQ
    dbQ = rbQ.copy()


    rmses = []
    alphaP = alphaQ = 0.00001
    lastRmse = 0
    for i in xrange(steps):
        fejlSum = err(r,beta,betaO,P,Q,bP,bQ)
        print i,"RMSE: ",fejlSum, "RMSEdiff: ", lastRmse - fejlSum
        rmses.append(np.sqrt(np.mean(r**2))) #(fejlSum)
        if validate and i % every == 0:
            validate(P.copy(),Q.copy(),bP.copy(),bQ.copy())

        if fejlSum == lastRmse:
            break
        lastRmse = fejlSum

        if doBias:
            alphabP = lineSearch(fejlSum,lambda alpha: sigresidual(R,P,Q,bP + alpha * dbP,bQ,discard,beta,betaO,u=u),beta,betaO,P,Q,bP,bQ)
            
            bP += alphabP * dbP
            
            r = sigresidual(R,P,Q,bP,bQ,discard,beta,betaO,u=u)

            fejlSum = err(r,beta,betaO,P,Q,bP,bQ)
            
            sigd = deriv(P,Q,bP,bQ,u)
            #r[(R == 0)] = 0

            #new_rbP = np.multiply(r,sigd).sum(axis=1) - betaO * bP# r.sum(axis=1) - betaO * bP#
            new_rbP = 2 * (np.multiply(r,sigd) - betaO * bP[:,np.newaxis]).sum(axis=1)
            
            rbPFlat = rbP.flatten()
            new_rbPFlat = new_rbP.flatten()
            betabP = calc_beta(new_rbP,rbP)#np.dot(new_rbPFlat.T,new_rbPFlat - rbPFlat) / np.dot(rbPFlat.T,rbPFlat)
            
            rbP = new_rbP
            dbP = rbP + betabP * dbP
            print "bP, alpha:{}, beta:{}".format(alphabP,np.mean(betabP))

            alphabQ = lineSearch(fejlSum,lambda alpha: sigresidual(R,P,Q,bP,bQ + alpha * dbQ,discard,beta,betaO,u=u),beta,betaO,P,Q,bP,bQ)
            
            bQ += alphabQ * dbQ 
            
            r = sigresidual(R,P,Q,bP,bQ,discard,beta,betaO,u=u)

            fejlSum = err(r,beta,betaO,P,Q,bP,bQ)

            sigd = deriv(P,Q,bP,bQ,u)
            #r[(R == 0)] = 0

            new_rbQ = 2 * (np.multiply(r,sigd) - betaO * bQ).sum(axis=0) #np.multiply(r,sigd).sum(axis=0) - betaO * bQ # r.sum(axis=0) - betaO * bQ#
            
            rbQFlat = rbQ.flatten()
            new_rbQFlat = new_rbQ.flatten()
            betabQ = calc_beta(new_rbQ,rbQ)#np.dot(new_rbQFlat.T,new_rbQFlat - rbQFlat) / np.dot(rbQFlat.T,rbQFlat)
            
            rbQ = new_rbQ
            dbQ = rbQ + betabQ * dbQ
            print "bQ, alpha:{}, beta:{}".format(alphabQ,np.mean(betabQ))

        r = sigresidual(R,P,Q,bP,bQ,discard,beta,betaO,u=u)


        if doFactors:
            # Update P

            alphaP = lineSearch(fejlSum,lambda alpha: sigresidual(R,P + alpha * dP,Q,bP,bQ,discard,beta,betaO,u=u),beta,betaO,P,Q,bP,bQ)

            P += alphaP * dP

            r = sigresidual(R,P,Q,bP,bQ,discard,beta,betaO,u=u)
            fejlSum = err(r,beta,betaO,P,Q,bP,bQ)

            sigd = deriv(P,Q,bP,bQ,u)

            new_rP = 2 * (np.dot(np.multiply(r,sigd),Q.T) - beta*P)

            rPFlat = rP.flatten()
            new_rPFlat = new_rP.flatten()
            betaP = calc_beta(new_rP,rP)#np.dot(new_rPFlat.T,new_rPFlat - rPFlat) / np.dot(rPFlat.T,rPFlat)

            rP = new_rP
            dP = rP + betaP * dP
            print "P, alpha:{}, beta:{}".format(alphaP,np.mean(betaP))
            
            # Update Q

            alphaQ = lineSearch(fejlSum,lambda alpha: sigresidual(R,P,Q + alpha * dQ,bP,bQ,discard,beta,betaO,u=u),beta,betaO,P,Q,bP,bQ)
            
            Q += alphaQ * dQ 
            
            r = sigresidual(R,P,Q,bP,bQ,discard,beta,betaO,u=u)

            fejlSum = err(r,beta,betaO,P,Q,bP,bQ)
            
            sigd = deriv(P,Q,bP,bQ,u)

            new_rQ = 2 * (np.dot((np.multiply(r,sigd)).T,P).T - beta * Q)
            
            rQFlat = rQ.flatten()
            new_rQFlat = new_rQ.flatten()
            betaQ = calc_beta(new_rQ,rQ)#np.dot(new_rQFlat.T,new_rQFlat - rQFlat) / np.dot(rQFlat.T,rQFlat)
            
            rQ = new_rQ
            dQ = rQ + betaQ * dQ
            print "Q, alpha:{}, beta:{}".format(alphaQ,np.mean(betaQ))
        r = sigresidual(R,P,Q,bP,bQ,discard,beta,betaO,u=u)

    
    
    return R,P,Q,bP,bQ,rmses



from scipy.io import mmread, mmwrite
from scipy import sparse


import random

def makeSampled(R=None,k=3,c=2,mean_center=False,doPCore=False,path="movielens.mtx",fraction=0.05,save=False,seed=None):
    if R == None:
        R = mmread(path).T.tocsr()

    if doPCore:
        R = pCore(R,k,c)

    R = shuffle(R,random_state=seed).toarray()#.todok()
    
    if mean_center:
        u = np.array(R.values()).mean()

        for k,v in R.items():
            R[k[0],k[1]] = v - u

    Rem = sparse.dok_matrix(R.shape,dtype=np.float64)

    total = (R > 0).sum() * fraction
    samples = 0

    rows = np.nonzero(R.sum(axis=1) > 1)
    cols = np.nonzero(R.sum(axis=0) > 1)

    print (R.sum(axis=1) > 0).all(),(R.sum(axis=1) > 0).all()

    for k,v in sparse.dok_matrix(R).items(): #,int(R.nnz * 0.2):
        row,col = k
        if samples > total:
            break

        if R[row,:].sum() > 2 and R[:,col].sum() > 2:
            Rem[row,col] = v
            R[row,col] = 0
            samples += 1
            print samples
        else:
            print "nogo"

    R = sparse.dok_matrix(R)
    if save:
        mmwrite("training.mtx",R)
        mmwrite("validation.mtx",Rem)

    return R,Rem

def validate(trunc = False,T = None,V = None,doRound=False,activation=sigmoid,P=None,Q=None,bP=None,bQ=None):
    if T == None:
        Rtraining = mmread('training.mtx').tocsr()
    else:
        Rtraining = T

    if V == None:
        R = mmread('validation.mtx').todok()
    else:
        R = V.todok()
    mean = (Rtraining.sum()) / (Rtraining > 0).sum()
    if not (P != None or Q != None or bP != None or bQ != None):
        P,Q,bP,bQ = np.loadtxt("P.txt"),np.loadtxt("Q.txt"),np.loadtxt("bP.txt"),np.loadtxt("bQ.txt")

    print R.shape,P.shape,Q.shape
    i = 0
    sum = 0
    sumAbs = 0
    lte1 = 0
    sumlte1 = 0
    errors = []
    for k,v in R.items():
        g = bP[k[0]] + bQ[k[1]] + np.dot(P[k[0],:],Q[:,k[1]]) 
        #if trunc:
        #    g = min(1,max(5,g))
        #for i in xrange(P.shape[1]):
        #    g += (P[k[0],i]) * (Q[i,k[1]])
        #    
        #    if trunc:
        #        g = max(1,min(g,5))
        g = activation(mean + g)
        g = renormalizefloat(g,1,0,5,0)

        
        if doRound:
            g = round(g)
        e = (v - g)**2
        sumAbs += math.sqrt((v - g)**2)
        errors.append(e)
        if e < 1.00001:
            lte1 += 1
            sumlte1 += e
        sum += e
        #if e > 5:
        #print i,v,g,e
        i+=1
    rmse = math.sqrt(sum/R.nnz)
    mae = sumAbs / R.nnz
    print "rmse",rmse
    print "mae",sumAbs / R.nnz
    print "lte1",lte1,len(R.items()), lte1/float(len(R.items()))
    print "lte1 rmse",math.sqrt((sumlte1 +1) / (lte1+1))
    print "validation mean",mean
    return rmse,mae,np.array(errors)



def pCore(R,k=3,c=3):
    R = R[np.array((R > 0).sum(axis=1) > k).flatten(),:]
    R = R[:,(np.array((R > 0).sum(axis=0)) > c).flatten()]
    R = R[np.array((R > 0).sum(axis=1) > 0).flatten(),:]
    R = R[:,np.array((R > 0).sum(axis=0) > 0).flatten()]
    return R


def makeAvgBaseline(R):
    mu = np.finfo(float).eps
    bP = R.sum(axis=1) / ((R > 0).sum(axis=1) + mu) / 2
    bQ = R.sum(axis=0) / ((R > 0).sum(axis=0) + mu) / 2
    return bP,bQ

def normalizer(R,max,min):
    Rmin = float(R.min())
    Rmax = float(R.max())
    N = R.copy()
    N[N > 0] -= Rmin
    N *= (max - min)
    N = ((N) / (Rmax - Rmin))
    N[N != 0] += min
    return N

def renormalizer(N,Nmax,Nmin,Rmax,Rmin):
    R = N.copy()


    print "scale",float(Nmax - Nmin), "origscale", float(Rmax - Rmin)
    zi = R == 0
    R[R != 0] -= Nmin
    R /= float(Nmax - Nmin)
    R *= float(Rmax - Rmin)
    R += Rmin
    return R

def renormalizefloat(f,Nmax,Nmin,Rmax,Rmin):
    return (f - float(Nmin)) / float(Nmax - (Nmin)) * float(Rmax - Rmin) + Rmin



def goMusic(K=80,steps=200,resume=False,normalize=True,R=None,V=None,mean_center=False,beta=0.0,betaO=0.0,normalizer=normalizer,doBias=True,every=1,doFactors=True,biasSteps=10):
    #R = mmread("reviews_Musical_Instruments.mtx").tocsr()
    if R == None:
        R = mmread("training.mtx").tocsr().toarray()
    else:
        R = R.toarray()

    if V == None:
        V = mmread("validation.mtx").todok()
    
    mu = np.finfo(float).eps



    if normalize:
        R = normalizer(R,1,0)
        print "normalizing, min/max", R.min(),R.max()

    
    #R = R[0:424,:]
    if not resume:
        P = normalizer(np.random.rand(R.shape[0],K),.1,0)
        Q = normalizer(np.asfortranarray(np.random.rand(K,R.shape[1])),.1,0)

        #bP,bQ = makeAvgBaseline(R)
        #print bP,bQ
        bP = None # np.zeros(R.shape[0])#None
        bQ = None #np.zeros(R.shape[1])#None#(R > 0).mean(axis=0)
        #bP,bQ = makeAvgBaseline(R)
    else:
        P = np.loadtxt("P.txt")
        Q = np.loadtxt("Q.txt")
        bP = np.loadtxt("bP.txt")
        bQ = np.loadtxt("bQ.txt")

    print R.shape,P.shape,Q.shape
    print "starting doFactO"
    #chunkFactO(R,P,Q,K,steps=steps,chunks=1,discard=0)#chunks=800,discard=0)

    #R,P,Q,bP,bQ = factO(R,P,Q,K,steps=steps,discard=0,bP=bP,bQ=bQ,beta=beta,betaO=betaO)
    rmses,maes,errs = [],[],[]

    def validation(P,Q,bP,bQ):
        rmse,mae,err = validate(T=R,V=V,P=P,Q=Q,bP=bP,bQ=bQ)
        rmses.append(rmse)
        maes.append(mae)
        errs.append(err)

    R,P,Q,bP,bQ,t_rmses = sigFactO(R,P,Q,K,bP=bP,bQ=bQ,steps=steps,discard=0.0,beta=beta,betaO=betaO,mean_center=mean_center,doBias=doBias,validate=validation,every=every,doFactors=doFactors,biasSteps=biasSteps)    

    if normalize:
        R = renormalizer(R,1,0,5,0)

    dumparrays(R,P,Q,bP,bQ)


    return t_rmses,rmses,maes,errs


def testBetas():
    arrs = load_movielens_kfold()
    T,V = arrs[0]
    results = []
    for i in xrange(1,10):
        beta = float(1) / float(10**i)
        betaO = 0.1
        #for j in xrange(9,10):
        #    betaO = 1 / (10^j)
        rs,es = testMusic(R=T,V=V,beta=beta,betaO=betaO)
        results.append(dict(rmse=rs,beta=beta,betaO=betaO))
    return results

import os
import time
import pickle

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/factresults/"

def kFold(iterations=5,beta=0.5,betaO=0.00001,K=80,biasSteps=30,steps=150):
    vals = []
    results = []
    arrs = load_movielens_kfold()

    for i in xrange(iterations):
        #makeSampled(path="movielens.mtx",doPCore=False)
        T,V = arrs[i]
        #T = T.T
        #V = V.T
        trs,rs,maes,es = goMusic(R=T,V=V,beta=beta,betaO=betaO,K=K,steps=steps,biasSteps=biasSteps)
        results.append((list(trs),list(rs),list(maes)))
        rs.sort()
        vals.append(rs[0])

    avgrmse = sum(vals) / 5
    try:
        path = RESULTS_DIR + "mvlens-avgrmse_{}-K_{}-beta_{}-betaO_{}-steps_{}-stamp_{}".format(avgrmse,K,beta,betaO,steps,time.time())
        #if os.path.exists(path):
        #os.makedirs(path)

        with open(path,'w') as f:
            pickle.dump(results,f)
    except Exception as e:
        print e

    print "avg rmse", avgrmse
    return vals,results

def kFoldSample(iterations=5,beta=0,betaO=0,K=80,steps=100):
    vals = []
    results = []
    #arrs = load_movielens_kfold()

    for i in xrange(iterations):
        T,V = makeSampled(path="movielens.mtx",doPCore=False,fraction=1.0/iterations,save=False,seed=i)
        #T,V = arrs[i]
        #T = T.T
        #V = V.T
        #mmwrite("{}.training.mtx".format(i),T)
        #mmwrite("{}.validation.mtx".format(i),T)
        t_rmses,rs,maes,es = goMusic(beta=beta,betaO=betaO,K=K,R=T, V=V,biasSteps=30,steps=steps)
        result.append((t_rmses,rs,maes,es))
        rs.sort()
        vals.append(rs[0])
    return vals,results

def kFoldMinstru(iterations=5,beta=0.0,betaO=0.0,K=80,biasSteps=60, steps=60):
    vals, vals2 = [], []
    arrs = []
    results = []

    folds = [mmread("mInstru/A{}.mtx".format(i)) for i in xrange(iterations)]

    for i in xrange(iterations):
        #makeSampled(path="movielens.mtx",doPCore=False)
        tid = list(range(5))
        tid.remove(i)
        print tid
        T = sum([folds[i] for i in tid]).tocsr()
        V = folds[i].tocsr()
        #T = T.T
        #V = V.T
        t_rmses,rs,maes,es = goMusic(R=T,V=V,beta=beta,betaO=betaO,K=K,steps=steps)
        results.append((t_rmses,rs,maes))
        rs.sort()
        vals.append(rs[0])
        #vals2.append(rs[1])

    avgrmse = sum(vals) / iterations
    avgmae = sum(vals2) / iterations
    try:
        path = RESULTS_DIR + "amazon-avgrmse_{}-K_{}-beta_{}-betaO_{}-steps_{}-stamp_{}".format(avgrmse,K,beta,betaO,steps,time.time())
        #if os.path.exists(path):
        #os.makedirs(path)

        with open(path,'w') as f:
            pickle.dump(results,f)
    except Exception as e:
        print e

    print "avg rmse", avgrmse, "avg mae", avgmae

    return vals

def result_values():
    for f_name in os.listdir(RESULTS_DIR):
        path = os.path.join(RESULTS_DIR,f_name)
        if os.path.isfile(path) and "rmse" in path:
            print f_name
            with open(path) as f:
                r = pickle.load(f)
                s = 0
                steps = []
                for sr in r:
                    s += min(sr[2])
                    steps.append(np.argmin(np.array(sr[1])))
                s /= 5.0
                print "mae",s
                print "step",steps





def dumparrays(R,P,Q,bP=None,bQ=None):
    #np.savez_compressed("test.npz",R=R,P=P,Q=Q)
    if R != None:
        np.savetxt("R.txt",R)
    np.savetxt("P.txt",P)
    np.savetxt("Q.txt",Q)
    if bP != None:
        np.savetxt("bP.txt",bP)

    if bQ != None:
        np.savetxt("bQ.txt",bQ)

def loadarrays():
    #d = np.load('test.npz')
    #return d["R"],d["P"],d["Q"]
    return np.loadtxt("R.txt"),np.loadtxt("P.txt"),np.loadtxt("Q.txt"),np.loadtxt("bP.txt"),np.loadtxt("bQ.txt")


from scipy.sparse import lil_matrix, coo_matrix

def convert_movielens(path="u.data",delimiter="\t"):
    users = set()
    items = set()
    ratings = []

    with open(path,"r") as f:
        for l in f.readlines():
            vals = l.strip().split(delimiter)
            print vals
            user,item,rating,timestamp = [int(val) for val in vals]
            users.add(user)
            items.add(item)
            ratings.append((user,item,rating))

    user_map = dict(list(enumerate(users)))
    
    item_map = dict(list(enumerate(items)))

    user_map = {v: k for k, v in user_map.items()}

    item_map = {v: k for k, v in item_map.items()}
    
    shape = (len(users),len(items))

    mat = sparse.dok_matrix(shape)

    for user,item,rating in ratings:
        try:
            uid = user_map[user]
        except KeyError:
            print "couldn't find uid"
            raise
        
        try:
            iid = item_map[item]
        except KeyError:
            print "couldn't find iid"
            raise
        
        mat[uid,iid] = rating

    return mat

def read_mvlens(path):
    users = set()
    items = set()
    ratings = []
    
    with open(path,"r") as f:
        for l in f.readlines():
            vals = l.strip().split("\t")
            print vals
            user,item,rating,timestamp = [int(val) for val in vals]
            users.add(user)
            items.add(item)
            ratings.append((user,item,rating))

    return users,items,ratings

def make_matrix(ratings,user_map,item_map,shape):
    mat = sparse.dok_matrix(shape)

    for user,item,rating in ratings:
        try:
            uid = user_map[user]
        except KeyError:
            print "couldn't find uid"
            raise
        
        try:
            iid = item_map[item]
        except KeyError:
            print "couldn't find iid"
            raise
        
        mat[uid,iid] = rating

    return mat


def convert_movielens_fold(iteration):
    
    users,items,ratings_t = read_mvlens("movielens/u{}.base".format(iteration))

    uv,iv,ratings_v = read_mvlens("movielens/u{}.test".format(iteration))

    users.update(uv)

    items.update(iv)

    user_map = dict(list(enumerate(users)))
    
    item_map = dict(list(enumerate(items)))

    user_map = {v: k for k, v in user_map.items()}

    item_map = {v: k for k, v in item_map.items()}
    
    shape = (len(users),len(items))

    T = make_matrix(ratings_t,user_map,item_map,shape).tocsr()
    V = make_matrix(ratings_v,user_map,item_map,shape).tocsr()

    return T,V


def load_movielens_kfold():
    arrs = []
    for i in xrange(5):
        T,V = convert_movielens_fold(i+1)
        arrs.append((T,V))

    return arrs



#kFoldMinstru()

