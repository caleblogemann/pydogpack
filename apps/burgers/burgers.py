from pydogpack.utils import functions
from pydogpack.utils import flux_functions
import numpy as np

# TODO: think about using utils.flux_functions to represent burgers.flux_function


class Burgers:
    def __init__(self, max_wavespeed, initial_condition=None):
        if initial_condition is None:
            self.initial_condition = functions.Sine()
        else:
            self.initial_condition = initial_condition
        self.max_wavespeed = max_wavespeed
        self.is_linearized = False
        # solution about which we are linearizing
        self.linearized_solution = None

    def linearize(self, dg_solution):
        self.is_linearized = True
        # TODO: maybe need to make a copy of dg_solution
        self.linearized_solution = dg_solution

    def exact_solution(self, x, t):
        # solve characteristics
        pass

    # TODO add time dependence
    def exact_operator(self, x, t):
        return -1.0 * self.initial_condition(x) * self.initial_condition.derivative(x)
