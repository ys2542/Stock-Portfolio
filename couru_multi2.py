import sys
import numpy as np
from itertools import takewhile
import logging
import cvxpy as cp
from multiprocessing import Pool
from functools import partial
from numba import jit,int64,float64

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


class expert():
    def __init__(self, rho, timeWindow, initialWealth = 1):
        """
        initialize:
        @rho: correlation threshold
        @timeWindow: specific time windows
        @initialWealth: wealth
        """
        self.rho =rho
        self.timeWindow = timeWindow
        self.wealth = initialWealth
    #@staticmethod


    def learning(self, Xh):
        """
        @Xh: historical prices until time t, m*(t-1)
        Return:
        @portfolio: array of weights put on each assets
        """
        # Use h instead of t here since h = t-1
        numAssets, h = Xh.shape
        self.portfolio = np.ones(numAssets)/numAssets
        # if have reached w(timeWindow) + 1 days
        if h > self.timeWindow:
            indexSet = computeIndex5(Xh, self.timeWindow, self.rho)
            if indexSet:
                self.portfolio = searchOpt(Xh[:,self.timeWindow:][:,indexSet])
        return self.portfolio
    def update(self, xt):
        """
        Update at the end of the day
        @xt: prices of the current trading day
        """
        self.wealth *= np.sum(self.portfolio * xt)


def preProcess(data_path):
    dataWithWeekend = np.loadtxt(data_path, delimiter=' ')
    # Remove days when market closed
    dataWithoutWeekend = dataWithWeekend[:,dataWithWeekend[0]!= 0]
    # Start Date: the number of zeros before the first non-zero data
    startDate = [len(list(takewhile(i))) for i in dataWithoutWeekend]
    # Transform the data into relative prices compared to the day before
    # Seems that a Numba function @nb.jit could help fill nulls, not used here


def combine(portfolioSet, wealthSet, q):
    """
    Combine the experts' portfolios
    @portfolioSet: array of portfolios, W * m(numAssets)
    @wealthSet: array of experts' current wealth, W
    @q: array of probability distribution function, W
    """
    wealth_q = wealthSet  * q
    nome = np.sum(np.swapaxes(portfolioSet.T,0,1) * wealth_q, axis = 2)
    deno = np.sum(wealth_q,axis = 1)
    return nome/deno

def expert_learning(expertI,X, T):
    portfolioSet = []
    wealthSet = []
    for t in range(T):
        portfolioSet.append(expertI.learning(X[:,:t]))
        wealthSet.append(expertI.wealth)
        expertI.update(X[:,t])
    return np.array(portfolioSet).T,np.array(wealthSet)
        
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
    expertPort = [expert(rho,w) for w in range(1,W+1)]
    numAssets, T = X.shape
    num_process = W
    
    with Pool(num_process) as p:
        portfolio_wealthSet = p.map(partial(expert_learning, X = X, T = T),expertPort)
    portfolioSet = []
    wealthSet = []
    for expert_p,expert_w in portfolio_wealthSet:
        portfolioSet.append(expert_p)
        wealthSet.append(expert_w)
    #portfolioSet = np.vstack(np.array(portfolioSet))
    #wealthSet = np.hstack(np.array(wealthSet).reshape(W,-1,1))
    portfolioSet = np.array(portfolioSet)
    wealthSet = np.swapaxes(np.array(wealthSet),0,1)
    
    #print(np.multiply(wealthSet,q).shape)
    #print(portfolioSet.shape,wealthSet.shape,q.shape)
    portfolioRecord = combine(portfolioSet, wealthSet, q)
    #print(portfolioRecord)
    #print(portfolioRecord.shape)

    update_wealth = np.sum(portfolioRecord*X, axis = 0)
    
    #wealth *= np.sum(portfolioOverall * X[:,t])
    for i in range(T):
        wealth *= update_wealth[i]
        wealthRecord[i] = wealth
    '''
        logger.info('Start Day ' + str(t))
        print(t)
        portfolioSet = [expertI.learning(X[:,:t]) for expertI in expertPort]
        
        wealthSet = [expertI.wealth for expertI in expertPort]
        #print(wealthSet)
        portfolioOverall = combine(np.array(portfolioSet), np.array(wealthSet), q)
        #print(portfolioOverall)
        wealth *= np.sum(portfolioOverall * X[:,t])
        print(wealth)
        for expertI in expertPort:
            expertI.update(X[:,t])
        logger.info('End Day with wealth '+ str(wealth))
        wealthRecord[t] = wealth
        portfolioRecord[:,t] = portfolioOverall
    '''
    return wealthRecord, portfolioRecord