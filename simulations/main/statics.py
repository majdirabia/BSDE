import os


class State(object):
    def __init__(self, maturity, strike, correlation_matrix, drift, dividend, vol, spot, bank_rate, dealer_rate,
                 ):
        self.maturity = maturity
        self.strike = strike
        self.correlation_matrix = correlation_matrix
        self.drift = drift
        self.dividend = dividend
        self.vol = vol
        self.spot = spot
        self.bank_rate = bank_rate
        self.dealer_rate = dealer_rate

    @classmethod
    def get_from_args(cls, args):
        pass
    
    def update_from_csv(self):
        path = os.getcwd()
        
