from pydogpack.tests.utils import functions
from pydogpack.tests.utils import flux_functions
import numpy as np


class Advection:
    def __init__(self, wavespeed=1.0, initial_condition=None):
        if initial_condition is None:
            self.initial_condition = functions.Sine()
        else:
            self.initial_condition = initial_condition
        self.wavespeed = wavespeed
        self.max_wavespeed = wavespeed
        # TODO: could switch to using tests.utils.flux_functions
        # self.flux_function = flux_functions.Polynomial([0, self.wavespeed])

    def exact_solution(self, x, t):
        return self.initial_condition(x - self.wavespeed * t)

    # rewrite as q_t = L(q)
    # express L(q) as a function of x, t
    def exact_operator(self, x, t):
        return -1.0 * self.initial_condition.derivative(x - self.wavespeed * t)

    def flux_function(self, u, position):
        return self.wavespeed * u

    def flux_function_derivative(self, u, position):
        return self.wavespeed * np.ones(u.shape)

    def wavespeed_function(self, u, position):
        return self.wavespeed

    def flux_function_min(self, lower_bound, upper_bound, position):
        return np.min(
            [
                self.flux_function(lower_bound, position),
                self.flux_function(upper_bound, position),
            ]
        )

    def flux_function_max(self, lower_bound, upper_bound, position):
        return np.max(
            [
                self.flux_function(lower_bound, position),
                self.flux_function(upper_bound, position),
            ]
        )

    def quadrature_function(self):
        pass
