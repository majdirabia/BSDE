from math import sqrt, exp, log
from numpy import zeros
from scipy.stats import norm

# Here, collateral follows a popular rule that is \gamma * clean price
class Collateral:
    def __init__(self, gamma):
        self.gamma = gamma

class CallForwardColl(Collateral):
    def __init__(self, gamma):
        Collateral.__init__(self, gamma)
    def collateral(self, t, time, product, listProcess, path):
        dsf = listProcess[0].ts.discountfactor(time[t]) 
        mat =product.maturity - time[t]
        k = product.xprice  
        r = listProcess[0].ts.shortrate(t)
        # 
        snum = len(path)
        ret = zeros(snum)
         
        for s in range(0, snum):
            s0 = path[s][0]
            ret[s]  = self.gamma *dsf* (s0  - exp(-r*mat)*k)
        return ret

class CallOptionColl(Collateral):
    def __init__(self, gamma):
        Collateral.__init__(self, gamma)
    def collateral(self, t, time, product, listProcess, path):
        dsf = listProcess[0].ts.discountfactor(time[t]) 
        mat = product.maturity - time[t]
        k = product.xprice  
        vol = listProcess[0].vol
        r = listProcess[0].ts.shortrate(t)
        # 
        snum = len(path)
        ret = zeros(snum)
        for s in range(0, snum):
            s0 = path[s][0]
            d1 = (log(s0/k) + ( r + vol*vol*0.5  )*mat  )\
                            / ( vol * sqrt(mat) )
            d2 = d1 - vol*sqrt(mat)
            Nd1 = norm.cdf(d1)
            Nd2 = norm.cdf(d2)
            ret[s]  = self.gamma *dsf* (s0 * Nd1 - exp(-r*mat)*k*Nd2)
        return ret
