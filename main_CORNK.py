import sys
import numpy as np
from itertools import takewhile
import logging
from logbuilder import buildLogger
import cvxpy as cp
from pandas_datareader.data import DataReader

def searchOpt(C):
    """
    @C: historical prices with corr high than rho
    """
    numAssets = C.shape[0]
    logger = logging.getLogger('CORN') 
    
    b = cp.Variable(numAssets, pos = True)
    prob = cp.Problem(cp.Minimize(-1 * cp.sum(cp.log(C.T * b))), [cp.sum(b) == 1])
    logger.debug('Solving Problem: ' + str(prob.solve(solver = 'SCS', warm_start = True)))
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

    def __eq__(self, other):
        """
        Used for wealth ranking
        """
        return self.wealth == other.wealth

    def __lt__(self, other):
        """
        Used for wealth ranking
        """
        return self.wealth < other.wealth

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
            indexSet = [np.corrcoef(Xh[:,-self.timeWindow:].reshape(-1), Xh[:,i-self.timeWindow: i].reshape(-1))[1][0] >= self.rho\
                        for i in range(self.timeWindow ,h)]
            if sum(indexSet) > 0:
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
    @q: array of probability distribution function, m
    """
    nome = np.sum(portfolioSet.T * wealthSet * q, axis = 1)
    deno = np.sum(wealthSet * q)
    return nome/deno


def checkParams(**kwargs):
    if "rho" in kwargs:
        return False
    elif not "P" in kwargs and "K" in kwargs:
        raise ValueError("Definition for both P and K are required for CORNK")
    return True


def CORN(X, W, **kwargs):
    """
    main algorithm
    @W: maximum time window length
    @X: historical prices matrix, m(numAssets) * T
    There are two set of possible optional parameters:
    CORNU:
    @rho: correlation threshold
    CORNK:
    @P: max number of correlation thresholds
    @K: the value of K for top-K
    """
    logger = logging.getLogger('CORN')
    IS_CORNK = checkParams(**kwargs)
    # Initialize weights q and wealth
    wealth = 1
    wealthRecord = [1]
    numAssets, T = X.shape
    if IS_CORNK:
        K = kwargs["K"]
        P = kwargs["P"]
        q = np.ones(W*P)/(W*P)
        rhoSet = np.arange(P)/P
        expertPort = [expert(rho,w) for w in range(1,W+1) for rho in rhoSet]
    else:
        q = np.ones(W)/W
        expertPort = [expert(rho,w) for w in range(1,W+1)]

    for t in range(T):
        logger.info('Start Day ' + str(t))
        portfolioSet = [expertI.learning(X[:,:t]) for expertI in expertPort]
        wealthSet = [expertI.wealth for expertI in expertPort]
        portfolioOverall = combine(np.array(portfolioSet), np.array(wealthSet), q)
        wealth *= np.sum(portfolioOverall * X[:,t])
        print(wealth)
        for expertI in expertPort:
            expertI.update(X[:,t])
        logger.info('End Day with wealth '+ str(wealth))
        wealthRecord.append(wealth)
        if IS_CORNK:
            topK = np.argsort(expertPort)[:K]
            logger.debug("The top K experts are: " +  str(topK))
            q.fill(0)
            q[topK] = 1/K
    return wealthRecord


if __name__ == "__main__":
"""
input params:
    @m: # of stocks
    @T: length of trading
"""
    logger_file = 'C:/Users/Resurgam/Desktop/AdvancedPython/project'
    logger = buildLogger(logger_file, 'CORN')

    
    

    wealthRecord = CORNU(rho, W, data)
