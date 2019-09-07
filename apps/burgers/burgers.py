import numpy as np


class Burgers:
    def __init__(self, max_wavespeed, initial_condition=None):
        if initial_condition is None:
            self.initial_condition = lambda x: np.sin(np.pi * x)
        else:
            self.initial_condition = initial_condition
        self.max_wavespeed = max_wavespeed

    def exact_solution(self, x, t):
        # solve characteristics
        pass

    def flux_function(self, u, position):
        return 0.5 * np.power(u, 2.0)

    def flux_function_derivative(self, u, position):
        return u

    def wavespeed_function(self, u, position):
        return u

    def flux_function_min(self, lower_bound, upper_bound, position):
        if lower_bound <= 0.0 and upper_bound >= 0.0:
            return 0.0
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
