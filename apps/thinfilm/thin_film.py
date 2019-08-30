import pydogpack.math_utils as math_utils
import numpy as np


# represents q_t + (q^2 - q^3)_x = -(q^3 q_xxx)_x
# can more generally represent q_t + (q^2 - q^3)_x = -(f(q) q_xxx)_x
class ThinFilm:
    def __init__(self, f=None):
        if f is None:
            self.f = math_utils.Cube.function
        else:
            self.f = f
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

    # take in object that represents a function, f and function q
    # return function that represents exact expression of RHS
    @staticmethod
    def exact_operator(f, q):
        def exact_expression(x):
            first_term = f.function(q.function(x)) * q.fourth_derivative(x)
            second_term = (
                f.derivative(q.function(x)) * q.derivative(x) * q.third_derivative(x)
            )
            return -1.0 * (first_term + second_term)
        return exact_expression
