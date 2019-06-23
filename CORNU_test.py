from mpi4py import MPI
import numpy as np
from itertools import takewhile
import logging
import pickle as pkl
import time
from numba import jit,int64,float64
from logbuilder import buildLogger
import cvxpy as cp

import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if not sys.warnoptions:
    import os,warnings
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

def searchOpt(C):
    """
    @C: historical prices with corr high than rho
    """
    numAssets = C.shape[0]
    b = cp.Variable(numAssets, pos = True)
    prob = cp.Problem(cp.Minimize(-1 * cp.sum(cp.log(C.T * b))), [cp.sum(b) == 1])
    prob.solve(solver = 'SCS', normalize= False)
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


def combine_comm(wealth_portfolio, q):
    """
    Combine the experts' portfolios, specially designed for mpi
    @wealth_portfolio: first column is wealth, the rest are portfolios.numExperts * T * (numAssets + 1) 
    @q: array of probability distribution function, m
    """
    combined_p = []
    for t in range(T):
        wealthSet = wealth_portfolio[:,t,0].reshape(-1)
        portfolioSet = wealth_portfolio[:,t,1:]
        nome = np.sum(portfolioSet.T * wealthSet * q, axis = 1)
        deno = np.sum(wealthSet * q)
        combined_p.append(nome/deno)  
    return combined_p



numAssets = 36
T = 500
W = 5
rho = 0.4
assert W == size

logger = logging.getLogger('CORN')

recvbuf = np.empty((size, T,numAssets + 1))
sendbuf = np.empty((T, numAssets + 1))

#rank 0 is designed to deal with ocmbining
if rank == 0:
    X = np.ascontiguousarray(pkl.load(open("nyse_o_np.pkl", "rb"))[:numAssets,:T])
    q = np.ones(W)/W
    wealth = 1
    wealthRecord = [1]
    start = time.time()
else: 
    X = np.empty((numAssets, T))
w = rank + 1
expertI = expert(rho,w)

#broadcasting
comm.Bcast(X, root = 0)

for t in range(T):
    if t%100 == 0:
        print("rank {} has reached {} days".format(rank,t))
    sendbuf[t,1:] = expertI.learning(X[:,:t])
    sendbuf[t,0] = expertI.wealth
    expertI.update(X[:,t])
    sendbuf = np.ascontiguousarray(sendbuf)
    
#gather portfolio and wealth
comm.Gather(sendbuf, recvbuf, root = 0)    
if rank == 0:
    #print(recvbuf[:,0,:])
    #pkl.dump(recvbuf,open('recvbuf', 'wb'))
    portfolioOverall = combine_comm(recvbuf, q)
    for t in range(T):
        wealth *= np.sum(portfolioOverall[t] * X[:,t]) 
        wealthRecord.append(wealth)
        
    print("total time: ", time.time() - start)
