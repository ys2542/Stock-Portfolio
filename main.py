import sys
import numpy as np
from itertools import takewhile
import logging
from logbuilder import buildLogger
import cvxpy as cp

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
        """[]
        initialize:
        @rho: correlation threshold
        @timeWindow: specific time windows
        @initialWealth: wealth
        """
        self.rho =rho
        self.timeWindow = timeWindow
        self.wealth = initialWealth
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
    wealth = 1
    wealthRecord = [1]
    expertPort = [expert(rho,w) for w in range(1,W+1)]
    numAssets, T = X.shape
    # initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()

    fig.show()
    fig.canvas.draw()
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

        # plot in real time
        ax.clear()
        ax.plot(wealthRecord)
        fig.canvas.draw()
        
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
