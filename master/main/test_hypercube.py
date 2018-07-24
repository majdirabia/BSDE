from BSDE import *
import unittest

class TestBSDE(unittest.TestCase):

    def test_hypercube(self):
        n_hc = 2
        x_data = [1,2,3,4,5,6]
        y_data = [3,7,8,2,1,0]
        x = [1.5, 3.5, 5.5]
        sol = [6, 1, 1]
        self.assertTrue((hypercube(x_data,y_data,x,n_hc) == sol).all())
