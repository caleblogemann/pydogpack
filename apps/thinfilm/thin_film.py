import numpy as np


class ThinFilm:
    def __init__(self):
        pass

    def flux_function(self, q):
        return np.power(q, 2.0) - np.power(q, 3.0)

    def wavespeed_function(self, q):
        return 2.0 * q - 3.0 * np.power(q, 2.0)

    def flux_function_critical_points(self, lower_bound, upper_bound):
        critical_points = [lower_bound, upper_bound]
        if lower_bound <= 0.0 and upper_bound >= 0.0:
            critical_points.append(0.0)
        if lower_bound <= 2.0 / 3.0 and upper_bound >= 2.0 / 3.0:
            critical_points.append(2.0 / 3.0)
        return critical_points

    def flux_function_min(self, lower_bound, upper_bound):
        critical_points = self.flux_function_critical_points(lower_bound, upper_bound)
        return np.min(self.flux_function(critical_points))

    def flux_function_max(self, lower_bound, upper_bound):
        critical_points = self.flux_function_critical_points(lower_bound, upper_bound)
        return np.max(self.flux_function(critical_points))
