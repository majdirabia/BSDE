from math import fabs, exp

class Product(object):
    def __init__(self, maturity):
        self.maturity = maturity

class CallOption(Product):
    def __init__(self, maturity, xprice):
        Product.__init__(self, maturity)
        self.xprice = xprice

    def payoff(self, t, x):
        if t > self.maturity+0.000000001:
            print ("error call option payoff")
        elif fabs(t- self.maturity) < 0.0000001:
            return max(x[0]-self.xprice, 0.)
        else:
            return 0.;
class CallOptionMulti(Product):
    def __init__(self, maturity, xprice):
        Product.__init__(self, maturity)
        self.xprice = xprice
    def payoff(self, t, x):
        if t > self.maturity+0.000000001:
            print ("error call option payoff")
        elif fabs(t- self.maturity) < 0.0000001:
            return max(x[0]*x[0]-self.xprice, 0.)
        else:
            return 0.;
class CallForward(Product):
    def __init__(self, maturity, xprice):
        Product.__init__(self, maturity)
        self.xprice = xprice

    def payoff(self, t, x):
        if t > self.maturity+0.000000001:
            print ("error call forward payoff")
        elif fabs(t- self.maturity) < 0.0000001:
            return x[0]-self.xprice
        else:
            return 0.;
