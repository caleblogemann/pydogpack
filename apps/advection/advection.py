import numpy as np


class Advection:
    def __init__(self, wavespeed=1.0):
        self.wavespeed = wavespeed
        self.max_wavespeed = wavespeed

    def flux_function(self, u):
        return self.wavespeed * u

    def wavespeed_function(self, u):
        return self.wavespeed

    def flux_function_min(self, lower_bound, upper_bound):
        return np.min(
            [self.flux_function(lower_bound), self.flux_function(upper_bound)]
        )

    def flux_function_max(self, lower_bound, upper_bound):
        return np.max(
            [self.flux_function(lower_bound), self.flux_function(upper_bound)]
        )
