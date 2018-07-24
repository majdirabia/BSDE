from numpy import linalg, array
from termstructure import TermStructure
from math import log
class Process(object):
    """docstring for Process"""
    def __init__(self, iv):
        self.iv = iv
    def getiv(self):
        return self.iv

class VProcess(object):
    def __init__(self, listProcess, corr_=[1.]):
        self.listProcess = listProcess
        self.pnum = len(listProcess)
        if len(corr_) != 1:
            self.corr = linalg.cholesky(corr_)
        else:
            self.corr = array([corr_])

class BaseGBM(Process):
    """docstring for Stock"""
    def __init__(self,params, ts):
        ## ts : termstructure
        ## The class ts should not be changed onwards
        Process.__init__(self, params['initialvalue'])
        self.vol = params['volatility']
        self.ts = ts
    def drift(self, x, t):
        return self.ts.shortrate(t) * x
    def volatility(self, x, t):
        return self.vol *x


