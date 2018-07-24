from math import exp

class TermStructure:
    def __init__(self, quote):
        self.r = quote # considering constant interest rate
                       # for the time being
    def discountfactor(self, t):
        return self.pdiscountfactor(0., t)
    
    def pdiscountfactor(self, s, t):
        return exp( -self.r * (t - s) ) # should be remedied

    def forwardrate(self, s, t):
        return self.r # should be remedied

    def shortrate(self, t):
        return self.forwardrate(t, t + 0.000001)
