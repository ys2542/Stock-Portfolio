import sys
import numpy as np
from itertools import takewhile
import logging
import cvxpy as cp
from multiprocessing import Pool
from functools import partial
from numba import jit,int64,float64

def searchOpt(C):
    """
    @C: historical prices with corr high than rho
    """
    numAssets = C.shape[0]
    logger = logging.getLogger('CORN') 
    
    b = cp.Variable(numAssets, pos = True)
    prob = cp.Problem(cp.Minimize(-1 * cp.sum(cp.log(C.T * b))), [cp.sum(b) == 1])
    logger.debug('Solving Problem: ' + str(prob.solve(solver = 'SCS')))
    return np.array(b.value)

@jit("(float64[:,:], int64, float64)",nopython = True)
def computeIndex5(Xh, timeWindow, rho):
    numAssets, h = Xh.shape
    vec = np.ascontiguousarray(Xh.T).reshape(-1)
    fixed = vec[-timeWindow * numAssets:]
    fixed_residual =fixed- np.mean(fixed)
    fixed_std = np.std(fixed)
    indexSet =[]
    rho_fake = rho * fixed_std * timeWindow * numAssets
    for i in range(timeWindow * numAssets, h * numAssets, numAssets):
        if np.std(vec[i-timeWindow* numAssets: i]) > 0 and np.sum(fixed_residual * (vec[i-timeWindow* numAssets: i] - np.mean(vec[i-timeWindow* numAssets: i])))/np.std(vec[i-timeWindow* numAssets: i]) >= rho_fake:
            indexSet.append(i//numAssets - timeWindow)
    return indexSet

def combine(portfolioSet, wealthSet, q):
    """
    Combine the experts' portfolios
    @portfolioSet: array of portfolios, W * m(numAssets)
    @wealthSet: array of experts' current wealth, W
    @q: array of probability distribution function, m
    """
    nome = np.sum(portfolioSet.T * wealthSet * q, axis = 1)
    deno = np.sum(wealthSet * q)
    return nome/deno

def learning(timeWindow,Xh,rho):
    """
    @Xh: historical prices until time t, m*(t-1)
    Return:
    @portfolio: array of weights put on each assets
    """
    # Use h instead of t here since h = t-1
    numAssets, h = Xh.shape
    portfolio = np.ones(numAssets)/numAssets
    # if have reached w(timeWindow) + 1 days
    if h > timeWindow:
        
        indexSet = computeIndex5(Xh, timeWindow, rho)
        
        if indexSet:
            portfolio = searchOpt(Xh[:,timeWindow:][:,indexSet])
    return portfolio

def CORNU(rho, W, X):
    """
    main algorithm
    @rho: correlation threshold
    @W: maximum time window length
    @X: historical prices matrix, m(numAssets) * T
    """
    logger = logging.getLogger('CORN')
    # Initialize weights q and wealth
    q = np.ones(W)/W
    numAssets, T = X.shape
    wealth = 1
    wealthRecord = np.ones(T+1)
    portfolioRecord = np.zeros((numAssets,T))
    timeWindow = [w for w in range(1,W+1)]
    numAssets, T = X.shape
    #optimization
    num_process = W
    wealthSet = np.ones(W)
    for t in range(T):
        logger.info('Start Day ' + str(t))
        portfolioSet = []
        with Pool(num_process) as p:
             portfolioSet = p.map(partial(learning, Xh = X[:,:t], rho = rho),timeWindow)
        
        #wealthSet = [expertI.wealth for expertI in expertPort]
        #for i in range(len(tmpSet)):
            #portfolioSet[int(tmpSet[i][0]-1)] = tmpSet[i][1]
        #print(wealthSet)
        portfolioOverall = combine(np.array(portfolioSet), wealthSet, q)
        wealth *= np.sum(portfolioOverall * X[:,t])
        #print(wealth)
        update_wealth = np.dot(portfolioSet, X[:,t])
        wealthSet = np.multiply(wealthSet,update_wealth.T)
        #for expertI in expertPort:
            #expertI.update(X[:,t])
        logger.info('End Day with wealth '+ str(wealth))
        wealthRecord[t] = wealth
        portfolioRecord[:,t] = portfolioOverall
    p.close()
    return wealthRecord, portfolioRecord