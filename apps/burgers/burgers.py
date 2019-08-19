import numpy as np

class Burgers:
    def __init__(self, max_wavespeed):
        self.max_wavespeed = max_wavespeed

    def flux_function(self, u):
        return 0.5*np.power(u, 2.0)

    def wavespeed_function(self, u):
        return u

    def flux_function_min(self, lower_bound, upper_bound):
        if (lower_bound <= 0.0 and upper_bound >= 0.0):
            return 0.0
        return np.min([self.flux_function(lower_bound),
            self.flux_function(upper_bound)])

    def flux_function_max(self, lower_bound, upper_bound):
        return np.max([self.flux_function(lower_bound),
            self.flux_function(upper_bound)])