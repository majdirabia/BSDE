from numpy import array, append, zeros, dot, linalg, vstack, empty
import operator as op
import copy
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

class regression:
    def __init__(self, idata, tdata):
	    #### idata and tdata means input data and target
	    #### data. When the input data is inserted the
	    #### row size should be same with that of tdata
        self.basis=0 #coefficient
        # basis is the desired result
        if len(idata)!=len(tdata):
            print ("data size is wrong")
            self.idata = idata
            self.tdata = tdata

    def basisexpand(self, x): 
        ##### x^2,  xy , etc
        #### This function should be overwrriten
        #### at offspring classes
        return array([])
    # vector by evaluation function
    def evalarray(self, xv):
        d= len(xv)
        ret = zeros(d)
        for i in range(0, d):
            ret[i] = self.evaluation(xv[i])

        return ret
    def evaluation(self, xs): # x, a are list
	# The purpose of this function is to make us
	# not to care of the order of basis
        bs = self.basisexpand(xs)
        if len(bs) != len(self.basis)-1:
            print ("index isn't matched")
            l=len(self.basis)
        ret =self.basis[0]
        for i in range(1, l):
            ret = ret + bs[i-1]*self.basis[i]
        return ret
    def bearray(self, x):
        rows = len(x)
        v = self.basisexpand(x[0])
        ret = zeros([rows, 1+len(v)])
        for r in range(0, rows):
            ret[r] = append(array([1.]),self.basisexpand(x[r]))
        return ret
    def setbasis(self):
        #### (X^T X)^{-1}X y
        rows = len(self.tdata)
        cols = len(self.idata[0])
        firstrow = self.basisexpand(self.idata[0].tolist())
        basisrow = len(firstrow)
        dbasis = zeros([rows, basisrow+1])
        tmp = array([1.])
        tmp = append(tmp, firstrow)
        dbasis[0] = tmp
        for i in range(1, rows):
            tmp = array([1.])
            tmp = append(tmp, self.basisexpand(self.idata[i]))
            dbasis[i] = tmp
        #self.basis = dot(
            #linalg.inv(dot(dbasis.T, dbasis)),
            #dot(dbasis.T, self.tdata))
            self.basis = linalg.lstsq(dbasis, self.tdata)[0]

class polyregression(regression):
    def __init__(self, idata, tdata, deg): #inputdata and target data
        regression.__init__(self, idata, tdata)
        self.deg = deg
        def mkbasis(self, x,deg_,i):
            d = len(x)

            ret=empty(0)
	if deg_-1 == 0:
            ret=[x[i]]
	else:
	    for j in range(i, d):
		tmp = self.mkbasis(x,deg_-1, j)
		ret = append(ret, \
                        array([y * x[i] for y in tmp] ))
	return ret

    def basisexpand(self, x): ## x^2 xy , etc
        ret =empty(0)
        d = len(x)
        for i in range(0, d):
                for dg in range(1, self.deg+1):
                    tmp = self.mkbasis(x, dg, i)
                    ret = append(ret, tmp)
            return ret

### Here is an example of writing offspring
### You only need to overwrite the 'basisexpand'
class offspringexample(regression):
    def __inti__(self, idata, tdata):
        regression.__init__(self, idata, tdata)
        def basisexpand(self, x):
        return x

x = array([0,2,3,-1])
x = vstack([x, [-1,3,4,1]])
y = array([1,2,1,3])
regr = polyregression(x.T, y, 2)
regr = offspringexample(x.T, y)
regr.setbasis()
print regr.basis
print "expand of 0, -1", regr.basisexpand([0,-1])
print regr.evaluation([0,-1])
print regr.evaluation([2, 3])
print regr.evaluation([3, 4])





#########  error message handling should be added

