from pydogpack.utils import x_functions
from apps.generalizedshallowwater import generalized_shallow_water
from apps import problem

import numpy as np


class TorrilhonExample(problem.Problem):
    def __init__(self, num_moments=0):
        pass


class InitialCondition(x_functions.XFunction):
    # h =
    # u = constant
    # s =
    # k =
    def __init__(self, num_moments=0, ):
        self.num_moments = num_moments

    def function(self, x):
        result = np.zeros(self.num_moments + 2)

        return result

    def do_x_derivative(self, x, order=1):
        return super().do_x_derivative(x, order=order)

