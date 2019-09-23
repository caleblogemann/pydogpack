from pydogpack.utils import flux_functions
from apps import app

from scipy import optimize


# TODO: think about adding a way to compute the time the exact_solution will shock
class Burgers(app.App):
    def __init__(self, max_wavespeed, source_function=None, initial_condition=None):
        flux_function = flux_functions.Polynomial([0.0, 0.0, 0.5])
        app.App.__init__(
            self, flux_function, source_function, initial_condition, max_wavespeed
        )

    # TODO: look more into how to linearize burgers equation about a dg_solution
    def linearized_wavespeed(self, dg_solution):
        pass

    def exact_solution(self, x, t):
        # solve characteristics
        # find xi that satisfies x = initial_condition(xi) * t + xi
        # then exact solution is u(x, t) = initial_condition(xi)
        def xi_function(xi):
            return self.initial_condition(xi) * t + xi - x

        # if exact solution has shocked, then newton will throw error
        # TODO: could catch exception
        xi = optimize.newton(xi_function, x)
        return self.initial_condition(xi)

    # TODO add time dependence
    def exact_operator(self, x, t):
        return -1.0 * self.initial_condition(x) * self.initial_condition.derivative(x)
