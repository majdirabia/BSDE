from numpy import random,  zeros, dot
from math import log, sqrt
from copy import deepcopy
import process
###############################


class Engine(object):
    def __init__(self, vProcess, time, snum ):
        self.vProcess = vProcess
        self.time = time # time is an array indicating time grid
                         # Note that the first element in time is 
                         # always nonzero.
        self.snum = snum # snum is a real number indicating
                         # simulation number.
        self.pnum = vProcess.pnum
        self.tnum = len(time)
            ### pnum is the number of processes
            ### Therefore, pnum* tnum * snum = computational cost
        self.viv = zeros(self.pnum)
        for p in range(0, self.pnum):
            self.viv[p] = self.vProcess.listProcess[p].iv

        self.randomarray = zeros([self.tnum,snum,self.pnum])
        self.path = zeros([self.tnum,snum,self.pnum])
        # To fix seed
        random.seed(0)
    
    def RandNumGenerator(self):
        _pnum = self.pnum
        _snum = self.snum
        _tnum = self.tnum
        _corr = self.vProcess.corr
        for t in range(0, _tnum):
            for s in range(0, _snum):
                for p in range(0, _pnum):
                    self.randomarray[t][s][p]=random.normal(0., 1.)

class SimulationEngine(Engine):     
    def __init__(self,  vProcess, time, snum):
        Engine.__init__(self, vProcess,time, snum)
        self.MakePath()
        ## randomarray := corr * dW ##
        ## Therefore, randomarray here is browninan motion
    def RandNumGenerator(self):
        _pnum = self.pnum 
        _snum = self.snum
        _tnum = self.tnum
        _corr = self.vProcess.corr
        trand = zeros(_pnum)
        # randomarray[i] = w[i+1][i]

        for t in range(0, _tnum):
            if t == 0:
                sdt = sqrt(self.time[0])
            else: 
                sdt = sqrt(self.time[t] - self.time[t-1])
            # 
            for s in range(0, _snum):
                self.randomarray[t][s]=dot(_corr,\
                                    random.normal(0.,1.,_pnum)*sdt)
  

    def MakePath(self):## Temporariliy we do not assume
                       ## jump processes
        self.RandNumGenerator()
        _pnum = self.pnum
        _snum = self.snum
        _tnum = self.tnum
        listProcess = self.vProcess.listProcess
        dt=0.
        sdt=0.
        t_=0.
        x=0.
        for t in range(0, _tnum):
            for s in range(0, _snum):
                if t == 0:
                    dt = self.time[t]
                    vval = self.viv
                else:
                    dt = self.time[t]-self.time[t-1]
                    vval = self.path[t-1][s]
                for p in range(0, _pnum):
                    x = vval[p]
                    t_= self.time[t]

                    self.path[t][s][p]=x+\
                            listProcess[p].drift(x, t_)*dt+\
                            +listProcess[p].volatility(x, t_)*self.randomarray[t][s][p]
                            # randomarray = dw 
