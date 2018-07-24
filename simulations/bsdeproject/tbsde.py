import engine
import termstructure
import product 
import process
import collateral
from regression import polyregression
from numpy import zeros, arange, linalg, average, std, dot, array
import copy
import functools
from scipy import optimize
from scipy.stats import norm
from math import sqrt, log, exp
# import graphlab as gl
from sklearn.ensemble import RandomForestRegressor

def blackscholescalloption(s0,k,vol,r,mat):
    dsf = exp(-r*mat)
    d1 = ( log(s0/k) + ( r + vol*vol*0.5  )*mat\
            ) / ( vol * sqrt(mat))
    d2 = d1 - vol*sqrt(mat)
    Nd1 = norm.cdf(d1)  
    Nd2 = norm.cdf(d2)
    return s0 * Nd1 - dsf*k*Nd2


class SelfFinancingBSDE(engine.SimulationEngine):
    def __init__(self, vProcess, product, time, snum, deg):
        engine.SimulationEngine.__init__(self, vProcess,time, snum)
        _pnum = self.pnum
        self.pd = product
        self.Z = engine.zeros([snum, _pnum])
        self.Y = engine.zeros(snum)
        self.deg=deg
        self.count2 = 0 
    # This is the discounted process of
    # self financing strategy
    def generator(self, t, ct,y):
        return zeros(self.snum)
    def counting(self, ct):
        pass
    # If it is plain vanila self-financing
    # this function is describing only the terminal
    # payoff. If it is XVA bSDE,
    # this is cashflow + collateral
    def addedvalue(self, t):
        dsf = self.vProcess.listProcess[0].ts.discountfactor(\
                self.time[t]) 
        ret = zeros(self.snum)
        for s in range(0, self.snum):
            ret[s] = self.pd.payoff(self.time[t],self.path[t][s])
        return ret*dsf
    def sqsum(self, t, dt, yt1, ct,yt):
        return sum((yt - self.generator(t, yt, ct)*dt - yt1)**2)
    def sqsumprime(self, t, dt, yt1, ct,yt):
        return zeros(len(yt1))
    # Backward iteration below, as the name is
    def biteration(self):    
        _tnum = self.tnum
        _snum = self.snum
        _pnum = self.pnum
        # term structure 
        ts = self.vProcess.listProcess[0].ts 

        # discountfactor 
        tdata=zeros(_snum)
        ppath=zeros(_snum)
        ct = zeros(self.snum)
        # The last iteration is not covered.
        for t in range(_tnum-1, 0, -1):
            # dsf: discountfactor
            dt = self.time[t]-self.time[t-1]
            # Last path is referred here
            self.Y =self.Y +self.addedvalue(t)
            # input data for the regrassion in the following step
            # Previous Path
            ppath=self.path[t-1]
            regr = polyregression(ppath, ct, self.deg)
            idata = regr.bearray(ppath)
            rf = RandomForestRegressor(oob_score=False,
                                           n_jobs=-1)
            for p in range(0, _pnum):
                # tdata := y(t+1)*dw(t+1)
                # Be careful about the index t here
                tdata= self.Y * self.randomarray[t,:,p]/dt
                # sf = gl.SFrame({'tg':tdata, 'path':idata})
                #model = gl.linear_regression.create(\
                    #sf, target = 'tg',verbose=False)
                
                self.Z[:,p] = rf.fit(ppath, tdata)
                              #array(model.predict(sf))
                              #regr.setbasis()
                              #self.Z[:,p]=regr.evalarray(ppath)

                # Here, self.Z = z(t)
                # Note the path index
            ct = self.Collateral.collateral(
                            t-1, self.time, self.pd,\
                            self.vProcess.listProcess,\
                            ppath)
            tdata = self.Y + self.generator(t-1, ct, self.Y)*dt
            sf['tg'] = tdata
            #sf = gl.SFrame({'tg':copy.deepcopy(tdata), 'path':ppath })
            model = gl.linear_regression.create(sf, target = 'tg'\
                        , verbose = False)
            self.Y = array(model.predict(sf))
                                            #regr = polyregression(ppath, tdata, self.deg)
                                            #regr.setbasis()
                                            #self.Y = regr.evalarray(ppath)
        ###############
        # y(0) and z(0)
        ###############
        # z = vol*xi*s0
        self.z = zeros(_pnum)
        self.Y = self.Y+self.addedvalue(0)
        self.y = average(self.Y)
        for p in range(0, _pnum):
            self.z[p] = average(self.Y *
                    self.randomarray[0,:,p])/self.time[0]

##################################################################
class XVA(SelfFinancingBSDE):
    def __init__(self, vProcess, product,time, snum,deg,\
                    rates, Collateral):
        # rates and collateral information
        SelfFinancingBSDE.__init__(self, vProcess,product,time,\
                                snum, deg)
        # they are TermStruct classes
        self.rlt = rates['rl']
        self.rbt = rates['rb']
        self.rclt = rates['rcl']
        self.rcbt = rates['rcb']
        self.Collateral = Collateral
        self.count = 0
    def counting(self, ct):
        tco = copy.deepcopy(self.vProcess.corr)
        for p in range(0, self.pnum):
            tco[p][p] = tco[p][p]*self.vProcess.listProcess[p].vol
           
        for s in range(0, self.snum):
            z = sum(dot(linalg.inv(tco.T),self.Z[s]))
            amt = self.Y[s] - ct[s] - z
            if amt >= 0.00000000001:
                self.count2 = self.count2 +1
                def sqsumprime(self, t, dt, yt1, ct, yt):
        ret = 2.*(yt - self.generator(t, yt, ct)*dt-yt1)
        rfrate = self.vProcess.listProcess[0].ts.shortrate(self.time[t])
        bl  = self.rlt.shortrate(t)  - rfrate
        bb  = self.rbt.shortrate(t)  - rfrate
        bcl = self.rclt.shortrate(t) - rfrate
        bcb = self.rcbt.shortrate(t) - rfrate
        # return valuey
        # discounted
        tco = copy.deepcopy(self.vProcess.corr)
        for p in range(0, self.pnum):
            tco[p][p] = tco[p][p]*self.vProcess.listProcess[p].vol
           
        for s in range(0, self.snum):
            z = sum(dot(linalg.inv(tco.T),self.Z[s]))
            amt = yt[s] - ct[s] - z
            if amt >= 0.00000000001:
                ret[s] = ret[s]*(1.+bl)
            else:
                ret[s] = ret[s]*(1.+bb )
        return ret

    # Note that ct = collateral * dsf
    # xva generator 
    def generator(self, t, ct, yt):
        rfrate = self.vProcess.listProcess[0].ts.shortrate(self.time[t])
        bl  = self.rlt.shortrate(t)  - rfrate
        bb  = self.rbt.shortrate(t)  - rfrate
        bcl = self.rclt.shortrate(t) - rfrate
        bcb = self.rcbt.shortrate(t) - rfrate
        # return value
        ret = zeros(self.snum)
        # discounted
        tco = copy.deepcopy(self.vProcess.corr)
        for p in range(0, self.pnum):
            tco[p][p] = tco[p][p]*self.vProcess.listProcess[p].vol
           
        for s in range(0, self.snum):
            z = sum(dot(linalg.inv(tco.T),self.Z[s]))
            amt = yt[s] - ct[s] - z
            if amt >= 0.00000000001:
                self.count = self.count+1
                ret[s] = -bl * amt 
            else:
                ret[s] = -bb * amt

            #if ct[s] >= 0.:
                #ret[s] = ret[s] - bcl * ct[s]
            #else:
                #ret[s] = ret[s] - bcb * ct[s]
        return ret  
################################################################
################################################################
f = open('./numertest/10000k12d2.dat', 'w')
print >>f, "#k=1.2, snum=10000"
for tn in range(1, 31):
    s0 = 0.8+ 0.02*float(10)
    xprice = 0.8 + 0.02*float(20)
    #  setting poduct information
    maturity = 0.5
    timegrid = tn
    # simulation information
    dt = maturity / timegrid
    time = arange(dt, maturity+dt, dt)
    snum = 10000
    # To set parameters, corr and make the instance
    # of vector process
    vol = 0.2
    parameters1 ={'initialvalue':s0 ,'volatility': vol}  
    # xva set up, rates and collateral partial amount
    riskfreerate = 0.005
    rl  = 0.001
    rb  = 0.3
    rcl = 0.01
    rcb = 0.03
    ts   = termstructure.TermStructure(riskfreerate)
    tsl  = termstructure.TermStructure(rl)
    tsb  = termstructure.TermStructure(rb)
    tscl = termstructure.TermStructure(rcl)
    tscb = termstructure.TermStructure(rcb)
    rates = {'rl'  : tsl,\
             'rb'  : tsb,\
             'rcl' : tscl,\
             'rcb' : tscb}
    s1 = process.BaseGBM(parameters1, ts)
    vp = process.VProcess([s1])
    # Product and collateral informatoin
    gamma = 0.
    deg=2
    #pd  = product.CallForward(maturity, xprice)
    #coll = collateral.CallForwardColl(gamma)
    pd = product.CallOption(maturity, xprice)
    coll = collateral.CallOptionColl(gamma)

    bsde = XVA(vp, pd , time, snum, deg, rates, coll)
    bsde.biteration()
    coprice = blackscholescalloption(s0,xprice,vol,rb,maturity)
    print s0, "    ",xprice, "   ",bsde.y-coprice, "   ", bsde.z[0]/(vol*s0), "   ",\
            float(bsde.count)/float(snum*timegrid)
    print "-------------------"
    print >> f, timegrid,"    ", \
                abs(bsde.y-coprice)/coprice, "    ",\
                abs(bsde.y-coprice), "    ",\
                bsde.z[0]/vol, "    ",\
            float(bsde.count)/float(snum*timegrid)
#
f.close()
