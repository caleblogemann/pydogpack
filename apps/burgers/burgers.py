from pydogpack.tests.utils import functions
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

    def flux_function(self, u, position):
        if self.is_linearized:
            return 0.5 * self.linearized_solution.evaluate(position) * u
        else:
            return 0.5 * np.power(u, 2.0)

    def flux_function_derivative(self, u, position):
        if self.is_linearized:
            return 0.5 * self.linearized_solution.evaluate(position)
        else:
            return u

    def wavespeed_function(self, u, position):
        return self.flux_function_derivative(u, position)

    def flux_function_min(self, lower_bound, upper_bound, position):
        if self.is_linearized:
            return np.min(
                [
                    self.flux_function(lower_bound, position),
                    self.flux_function(upper_bound, position),
                ]
            )
        else:
            if lower_bound <= 0.0 and upper_bound >= 0.0:
                return 0.0
            return np.min(
                [
                    self.flux_function(lower_bound, position),
                    self.flux_function(upper_bound, position),
                ]
            )

    def flux_function_max(self, lower_bound, upper_bound, position):
        if self.is_linearized:
            return np.max(
                [
                    self.flux_function(lower_bound, position),
                    self.flux_function(upper_bound, position),
                ]
            )
            pass
        else:
            return np.max(
                [
                    self.flux_function(lower_bound, position),
                    self.flux_function(upper_bound, position),
                ]
            )
